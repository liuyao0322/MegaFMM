#!/usr/bin/env python3
"""Export auditable local scientific/model inputs for the MegaFMM suite.

No network access or download is performed. NPZ uses flat numerical keys and a
Unicode metadata_json scalar. PT uses {'tensors': ..., 'metadata': ...}, safe to
read with torch.load(weights_only=True). Optional dependencies are loaded only
after argument parsing, so --help works without torch/PySCF/Transformers.

Examples:
  python scripts/export_suite_inputs.py --kind pyscf-overlap \
    --molecule 'O 0 0 0; H 0 .757 .587; H 0 -.757 .587' --basis sto-3g \
    --hamiltonian fock --output data/water.npz
  python scripts/export_suite_inputs.py --kind model-mlp \
    --checkpoint /local/Qwen2.5-0.5B-Instruct --text /local/fixture.txt \
    --tokens-output data/tokens.npz --output data/qwen_mlp.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.suite.model_eval import (DTYPES, checkpoint_manifest, load_local_model,
                                    sha256_file, token_tensors)


def write_archive(path, tensors, metadata):
    """Write only numeric arrays/tensors plus JSON metadata; never object arrays."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() not in (".npz", ".pt"):
        raise ValueError("Output extension must be .npz or .pt")
    if not tensors:
        raise ValueError("No tensors to export")
    # Also reject metadata that would require pickle, NaNs, or non-JSON classes.
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
    if path.suffix.lower() == ".npz":
        import numpy as np
        arrays = {}
        for key, value in tensors.items():
            if not isinstance(key, str) or key == "metadata_json":
                raise ValueError("Tensor keys must be strings distinct from metadata_json")
            if hasattr(value, "detach"):
                value = value.detach().cpu()
                # NumPy cannot represent BF16; preserve its exact stored values
                # in FP32 and record original dtype in metadata.
                if str(value.dtype) == "torch.bfloat16":
                    value = value.float()
                value = value.numpy()
            array = np.asarray(value)
            if array.dtype.kind not in "biufc":
                raise ValueError(f"{key} must be numerical, received {array.dtype}")
            if not np.isfinite(array).all():
                raise ValueError(f"{key} contains nonfinite values")
            arrays[key] = array
        arrays["metadata_json"] = np.asarray(encoded)
        np.savez(path, **arrays)
    else:
        import torch
        values = {key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor)
                  else torch.as_tensor(value).contiguous() for key, value in tensors.items()}
        if not all(torch.isfinite(value).all().item() for value in values.values()):
            raise ValueError("Exported tensors contain nonfinite values")
        torch.save({"tensors": values, "metadata": json.loads(encoded)}, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "keys": sorted(tensors), "metadata": metadata}


def export_pyscf(args):
    import numpy as np
    import pyscf
    from pyscf import gto, scf
    if not args.molecule:
        raise ValueError("pyscf-overlap requires --molecule (coordinates or local file)")
    if args.spin != 0:
        raise ValueError("This closed-shell exporter requires --spin 0")
    molecule_path = Path(args.molecule).expanduser()
    try:
        is_file = molecule_path.is_file()
    except OSError:
        is_file = False
    molecule = str(molecule_path.resolve()) if is_file else args.molecule
    mol = gto.M(atom=molecule, basis=args.basis, charge=args.charge, spin=args.spin,
                unit=args.unit, verbose=0)
    if mol.nelectron % 2:
        raise ValueError("Closed-shell nocc requires an even electron count")
    overlap = mol.intor_symmetric("int1e_ovlp")
    mean_field = scf.RHF(mol)
    hcore = mean_field.get_hcore()
    tensors = {"S": overlap, "H_ao": hcore}
    scf_metadata = {"executed": False}
    if args.hamiltonian == "fock":
        mean_field.conv_tol = args.scf_tolerance
        mean_field.max_cycle = args.scf_max_cycle
        energy = mean_field.kernel()
        if not mean_field.converged:
            raise ValueError("RHF did not converge; refusing to label an unconverged Fock matrix as converged")
        ao_hamiltonian = mean_field.get_fock()
        tensors["F"] = ao_hamiltonian
        scf_metadata = {"executed": True, "method": "RHF", "converged": True,
                        "total_energy_hartree": float(energy),
                        "convergence_tolerance": args.scf_tolerance,
                        "maximum_cycles": args.scf_max_cycle}
    else:
        ao_hamiltonian = hcore
    values, vectors = np.linalg.eigh(overlap)
    if float(values[0]) <= args.overlap_cutoff:
        raise ValueError(f"Overlap eigenvalue {values[0]:.4g} <= cutoff {args.overlap_cutoff}; "
                         "basis reduction is not silently applied")
    orthogonalizer = (vectors / np.sqrt(values)[None, :]) @ vectors.T
    hamiltonian = orthogonalizer.T @ ao_hamiltonian @ orthogonalizer
    tensors["H"] = (hamiltonian + hamiltonian.T) * 0.5
    source = {"molecule": args.molecule, "basis": args.basis, "charge": args.charge,
              "spin": args.spin, "unit": args.unit}
    if is_file:
        source.update({"molecule_file": str(molecule_path.resolve()),
                       "molecule_file_sha256": sha256_file(molecule_path)})
    metadata = {"schema_version": 1, "kind": "pyscf-overlap", "source": source,
                "input_origin": "pyscf_molecular_integrals", "pyscf_version": pyscf.__version__,
                "nao": int(overlap.shape[0]), "nocc": int(mol.nelectron // 2),
                "electron_count": int(mol.nelectron), "occupation_convention": "closed_shell_spinless_projector",
                "H_basis": "orthonormal", "H_role": args.hamiltonian,
                "H_transform": "S^-1/2 @ (F if fock else H_ao) @ S^-1/2",
                "H_ao_role": "core_hamiltonian", "F_basis": "AO_nonorthogonal" if "F" in tensors else None,
                "overlap_min_eigenvalue": float(values[0]), "overlap_max_eigenvalue": float(values[-1]),
                "overlap_condition_number": float(values[-1] / values[0]),
                "overlap_cutoff": args.overlap_cutoff, "effective_rank": int(overlap.shape[0]),
                "scf": scf_metadata,
                "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
                "scope_note": "Frozen molecular matrices; exporting them does not benchmark a complete SCF calculation."}
    return tensors, metadata


def load_model_batch(args):
    import torch
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint is None or not checkpoint.is_dir():
        raise ValueError("Model export requires --checkpoint pointing to an existing local directory")
    if bool(args.tokens) == bool(args.text):
        raise ValueError("Provide exactly one of --tokens and --text")
    if args.tokens:
        batch = token_tensors(args.tokens, args.device)
        if args.sequence_length:
            batch = {key: value[:, :args.sequence_length] for key, value in batch.items()}
        source = {"kind": "tokens", "path": str(Path(args.tokens).resolve()),
                  "sha256": sha256_file(args.tokens)}
    else:
        from transformers import AutoTokenizer
        text_path = Path(args.text).expanduser().resolve()
        text = text_path.read_text(encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                                  trust_remote_code=False)
        encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        ids = encoded["input_ids"]
        length = args.sequence_length or min(128, ids.shape[1])
        if args.batch_size > 1:
            if ids.numel() < length * args.batch_size:
                raise ValueError("Text does not contain enough tokens for the requested batch size and sequence length")
            ids = ids.reshape(-1)[:length * args.batch_size].reshape(args.batch_size, length)
        else:
            ids = ids[:, :length]
        batch = {"input_ids": ids.to(args.device),
                 "attention_mask": torch.ones_like(ids, device=args.device)}
        source = {"kind": "local_text", "path": str(text_path), "sha256": sha256_file(text_path),
                  "tokenizer_checkpoint": str(checkpoint), "add_special_tokens": True}
    if batch["input_ids"].shape[1] < 2:
        raise ValueError("At least two tokens are required")
    if args.tokens_output:
        source["tokens_archive"] = write_archive(args.tokens_output, batch,
                                                  {"schema_version": 1, "kind": "token_fixture", "source": source.copy()})
    model = load_local_model(checkpoint, args.dtype, args.device, args.attention)
    return model, batch, source


def _mlp_spec(module):
    """Return input projection, ordered GEMM weights and exact activation info."""
    if all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
        activation = getattr(module, "act_fn", None)
        if activation is not None and "silu" not in type(activation).__name__.lower():
            raise ValueError("Only SiLU-gated SwiGLU is supported for three-projection exports")
        return module.gate_proj, {"W1": module.gate_proj, "W2": module.down_proj,
                                  "W3": module.up_proj}, {"mlp_kind": "swiglu", "activation": "silu"}
    pairs = (("fc1", "fc2"), ("dense_h_to_4h", "dense_4h_to_h"), ("c_fc", "c_proj"))
    for first, second in pairs:
        if hasattr(module, first) and hasattr(module, second):
            activation = (getattr(module, "act", None) or getattr(module, "activation_fn", None)
                          or getattr(module, "act_fn", None))
            name = type(activation).__name__.lower() if activation is not None else ""
            function_name = getattr(activation, "__name__", "").lower()
            if not ("gelu" in name or "gelu" in function_name):
                raise ValueError("Two-projection export requires an identifiable GELU activation")
            if "quickgelu" in name or "clippedgelu" in name:
                raise ValueError("QuickGELU and clipped GELU are not standard GELU replay contracts")
            approximate = "tanh" if any(part in name for part in ("newgelu", "fastgelu", "gelutanh")) else "none"
            if getattr(activation, "approximate", None) == "tanh":
                approximate = "tanh"
            return getattr(module, first), {"W1": getattr(module, first), "W2": getattr(module, second)}, {
                "mlp_kind": "gelu", "activation": "gelu", "gelu_approximate": approximate,
                "activation_class": type(activation).__name__}
    raise ValueError("Unsupported MLP: expected gate_proj/up_proj/down_proj or a supported two-layer GELU module")


def resolve_mlp(model, requested=None):
    if requested:
        module = model.get_submodule(requested)
        first, weights, metadata = _mlp_spec(module)
        return requested, module, first, weights, metadata
    for name, module in model.named_modules():
        if not name or not any(part in {"mlp", "feed_forward", "ffn"} for part in name.split(".")):
            continue
        try:
            first, weights, metadata = _mlp_spec(module)
            return name, module, first, weights, metadata
        except ValueError:
            continue
    raise ValueError("No supported MLP found; specify its module path with --module")


def _weight_for_gemm(projection):
    import torch
    if isinstance(projection, torch.nn.Linear):
        return projection.weight.detach().T.contiguous()
    if type(projection).__name__ == "Conv1D":
        return projection.weight.detach().contiguous()
    raise ValueError(f"Unsupported projection class {type(projection).__name__}")


def export_model_mlp(args):
    import torch
    model, batch, token_source = load_model_batch(args)
    name, _, first, projections, activation = resolve_mlp(model, args.module)
    captured = []

    def capture(_module, inputs):
        if not captured:
            captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())

    hook = first.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    finally:
        hook.remove()
    if not captured:
        raise ValueError("Chosen MLP was not executed by the token batch")
    tensors = {"X": captured[0]}
    for key, projection in projections.items():
        tensors[key] = _weight_for_gemm(projection).cpu()
        if getattr(projection, "bias", None) is not None:
            tensors["b" + key[1:]] = projection.bias.detach().cpu()
    metadata = {"schema_version": 1, "kind": "model-mlp", "input_origin": "local_checkpoint_real_activation",
                "checkpoint": checkpoint_manifest(args.checkpoint), "tokens": token_source,
                "token_shape": list(batch["input_ids"].shape), "module": name,
                "weight_orientation": "input_output", "model_class": type(model).__name__,
                "dtype": args.dtype, "seed": args.seed, "eval_mode": True,
                "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
                "scope_note": "Single real MLP replay input; no complete-model quality or training claim.", **activation}
    return tensors, metadata


def export_muon_gradient(args):
    import torch
    model, batch, token_source = load_model_batch(args)
    named = dict(model.named_parameters())
    parameter_name = args.parameter
    if parameter_name is None:
        module_name, _, first, _, _ = resolve_mlp(model, args.module)
        parameter_name = next((name for name, value in named.items()
                               if value is first.weight), None)
        if parameter_name is None:
            raise ValueError(f"Could not resolve the first weight of {module_name}")
    if parameter_name not in named or named[parameter_name].ndim != 2:
        raise ValueError("--parameter must name a two-dimensional model parameter")
    model.train()
    # The recorded step index is actual local SGD work from the checkpoint.
    # It is never presented as checkpoint pretraining progress or as Muon training.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    losses = []
    gradient = None
    for step in range(args.training_step_index + 1):
        optimizer.zero_grad(set_to_none=True)
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[~batch["attention_mask"].bool()] = -100
        output = model(**batch, labels=labels, use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss).item():
            raise ValueError(f"Nonfinite training loss at local step {step}")
        loss.backward()
        losses.append(float(loss.detach().item()))
        if step == args.training_step_index:
            if named[parameter_name].grad is None:
                raise ValueError("Chosen parameter did not receive a gradient")
            gradient = named[parameter_name].grad.detach().cpu().clone()
        else:
            optimizer.step()
        del output, loss
    metadata = {"schema_version": 1, "kind": "muon-gradient", "input_origin": "local_checkpoint_backward",
                "checkpoint": checkpoint_manifest(args.checkpoint), "tokens": token_source,
                "token_shape": list(batch["input_ids"].shape), "parameter": parameter_name,
                "gradient_kind": "gradient", "momentum_accumulated": False,
                "training_step_index": args.training_step_index, "step_index_origin": "local_steps_since_checkpoint",
                "burn_in_optimizer": "SGD", "burn_in_updates": args.training_step_index,
                "learning_rate": args.learning_rate, "losses": losses,
                "seed": args.seed, "dtype": args.dtype, "model_class": type(model).__name__,
                "tensor_dtypes": {"G": str(gradient.dtype)},
                "scope_note": "Raw gradient snapshot for Muon NS replay; no Muon momentum or training-quality claim."}
    return {"G": gradient}, metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=("pyscf-overlap", "model-mlp", "muon-gradient"))
    parser.add_argument("--output", required=True, help="Output flat .npz or safe nested .pt archive")
    parser.add_argument("--molecule", "--atom", help="PySCF coordinates or an existing local XYZ file")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--unit", choices=("Angstrom", "Bohr"), default="Angstrom")
    parser.add_argument("--hamiltonian", choices=("core", "fock"), default="core")
    parser.add_argument("--overlap-cutoff", type=float, default=1e-10)
    parser.add_argument("--scf-tolerance", type=float, default=1e-10)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--checkpoint", help="Existing local Hugging Face checkpoint directory")
    parser.add_argument("--tokens", help="Local .npz/.pt token fixture, input_ids and optional attention_mask")
    parser.add_argument("--text", help="Existing UTF-8 text fixture to tokenize locally")
    parser.add_argument("--tokens-output", help="Optional .npz/.pt copy of the exact token batch for model evaluation")
    parser.add_argument("--sequence-length", type=int, help="Crop token sequence; text defaults to at most 128")
    parser.add_argument("--batch-size", type=int, default=1, help="Contiguous text chunks; token fixtures keep their stored batch size")
    parser.add_argument("--module", help="Exact MLP module path; default is the first supported MLP")
    parser.add_argument("--parameter", help="Exact 2D parameter name for gradient export; default first MLP input weight")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("sdpa", "eager", "default"), default="sdpa")
    parser.add_argument("--training-step-index", type=int, default=0,
                        help="Gradient at this actual local SGD step since checkpoint (0 means no burn-in updates)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sequence_length is not None and args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.training_step_index < 0 or args.batch_size < 1:
        parser.error("--training-step-index must be >=0 and --batch-size >=1")
    if args.learning_rate <= 0 or args.overlap_cutoff < 0:
        parser.error("learning rate must be positive and overlap cutoff nonnegative")
    try:
        if args.kind == "pyscf-overlap":
            tensors, metadata = export_pyscf(args)
        else:
            import torch
            torch.manual_seed(args.seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(args.seed)
            tensors, metadata = (export_model_mlp(args) if args.kind == "model-mlp"
                                 else export_muon_gradient(args))
        exported = write_archive(args.output, tensors, metadata)
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"input export failed: {error}\n")
    print(json.dumps({"path": exported["path"], "sha256": exported["sha256"],
                      "kind": args.kind, "keys": exported["keys"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
