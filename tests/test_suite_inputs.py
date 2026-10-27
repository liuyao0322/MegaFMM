"""Contracts for local input exports and model quality (CPU, no downloads)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite.model_eval import (causal_quality, checkpoint_manifest,
                                    load_token_arrays, replace_linears,
                                    select_linear_names)
from bench.suite import model_eval

spec = importlib.util.spec_from_file_location("suite_input_exporter", ROOT / "scripts/export_suite_inputs.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class DependencyFreeTests(unittest.TestCase):
    def test_cli_help_without_site_packages(self):
        for name in ("export_suite_inputs.py", "evaluate_suite_model.py"):
            result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / name), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--checkpoint", result.stdout)

    def test_checkpoint_hash_changes_with_weight_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"model_type":"fixture"}')
            weights = path / "model.safetensors"
            weights.write_bytes(b"first")
            first = checkpoint_manifest(path)
            weights.write_bytes(b"other")
            second = checkpoint_manifest(path)
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(len(first["files"]), 2)

    def test_activation_contract_preserves_gelu_approximation(self):
        for class_name, expected in (("GELUActivation", "none"), ("NewGELUActivation", "tanh"),
                                     ("PytorchGELUTanh", "tanh")):
            module = types.SimpleNamespace(fc1=object(), fc2=object(),
                                           act=type(class_name, (), {})())
            _, _, metadata = exporter._mlp_spec(module)
            self.assertEqual(metadata["gelu_approximate"], expected)
        with self.assertRaises(ValueError):
            exporter._mlp_spec(types.SimpleNamespace(fc1=object(), fc2=object(),
                               act=type("QuickGELUActivation", (), {})()))

    def test_quality_failures_have_nonzero_exit_unless_explicit_diagnostic(self):
        report = {"status": "completed", "application_quality_pass": False,
                  "backends": [{"backend": "megafmm", "status": "completed",
                                "application_quality_pass": False}]}
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["--checkpoint", directory, "--tokens", "unused.npz", "--output",
                         str(Path(directory) / "result.json")]
            with mock.patch.object(model_eval, "evaluate", return_value=report), mock.patch("builtins.print"):
                self.assertEqual(model_eval.main(arguments), 2)
                self.assertEqual(model_eval.main(arguments + ["--allow-quality-failures"]), 0)
                report["status"] = "partial_failure"
                self.assertEqual(model_eval.main(arguments + ["--allow-quality-failures"]), 1)


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed")
class ArchiveTests(unittest.TestCase):
    def test_npz_schema_round_trip_needs_no_pickle(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            metadata = {"kind": "pyscf-overlap", "nocc": 1, "source": "真实分子"}
            result = exporter.write_archive(path, {"S": np.eye(2), "H": np.diag([-1., 1.])}, metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"S", "H", "metadata_json"})
                self.assertEqual(archive["metadata_json"].dtype.kind, "U")
                self.assertEqual(json.loads(archive["metadata_json"].item()), metadata)
            self.assertEqual(len(result["sha256"]), 64)

    def test_archive_rejects_unsafe_arrays_and_nonfinite_values(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            for value in (np.array([object()], dtype=object), np.array([math.nan])):
                with self.assertRaises(ValueError):
                    exporter.write_archive(path, {"G": value}, {})

    def test_token_shape_and_mask(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npz"
            np.savez(path, input_ids=np.array([4, 5, 6]), attention_mask=np.array([1, 1, 0]))
            batch = load_token_arrays(path)
            self.assertEqual(batch["input_ids"].shape, (1, 3))
            self.assertEqual(batch["input_ids"].dtype, np.int64)

    def test_bad_tokens_are_rejected(self):
        import numpy as np
        bad = [dict(input_ids=np.array([1.])), dict(input_ids=np.array([1])),
               dict(input_ids=np.array([-1, 2])), dict(input_ids=np.array([True, False])),
               dict(input_ids=np.array([0, 2**64 - 1], dtype=np.uint64)),
               dict(input_ids=np.array([1, 2]), attention_mask=np.array([1, 2])),
               dict(input_ids=np.array([1, 2]), attention_mask=np.array([1, 0])),
               dict(input_ids=np.array([1, 2]), attention_mask=np.array([1, 1, 1]))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npz"
            for case in bad:
                np.savez(path, **case)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    load_token_arrays(path)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class ModelContractTests(unittest.TestCase):
    def test_safe_pt_tokens(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.pt"
            exporter.write_archive(path, {"input_ids": torch.tensor([[1, 2, 3]])}, {"kind": "tokens"})
            self.assertEqual(load_token_arrays(path)["input_ids"].tolist(), [[1, 2, 3]])

    def test_replacement_restores_modules_and_includes_bias(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 3))
                self.lm_head = torch.nn.Linear(3, 7)

        class Backend:
            def __init__(self):
                self.calls = []

            def gemm(self, left, right):
                self.calls.append((left.shape, right.shape))
                return left.float() @ right.float()

        model = Model()
        backend = Backend()
        names = select_linear_names(model)
        self.assertEqual(names, ["mlp.0", "mlp.1"])
        original = model.mlp[0]
        inputs = torch.randn(2, 4, 3)
        reference = model.mlp(inputs)
        with torch.no_grad(), replace_linears(model, names, backend):
            output = model.mlp(inputs)
            torch.testing.assert_close(output, reference)
        self.assertIs(model.mlp[0], original)
        self.assertEqual(backend.calls[0], (torch.Size([8, 3]), torch.Size([3, 5])))
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with replace_linears(model, names, backend):
                raise RuntimeError("fixture")
        self.assertIs(model.mlp[0], original)

    def test_nll_excludes_padded_targets(self):
        import torch
        logits = torch.zeros(1, 4, 2)
        logits[0, 2, :] = torch.tensor([100., -100.])
        batch = {"input_ids": torch.tensor([[0, 1, 0, 1]]),
                 "attention_mask": torch.tensor([[1, 1, 0, 0]])}
        metrics = causal_quality(logits, batch, logits.clone())
        self.assertEqual(metrics["target_tokens"], 1)
        self.assertAlmostEqual(metrics["nll"], math.log(2), places=6)
        self.assertAlmostEqual(metrics["perplexity"], 2., places=6)
        self.assertEqual(metrics["logits_relative_l2"], 0.)

    def test_end_to_end_model_eval_prepares_and_freezes(self):
        import numpy as np
        import torch

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(7, 3)
                self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 3))
                self.lm_head = torch.nn.Linear(3, 7)
                self.config = types.SimpleNamespace(_attn_implementation="fixture")

            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(logits=self.lm_head(self.mlp(self.embedding(input_ids))))

        class FakeBackend:
            instances = []

            def __init__(self, name, device, dtype, options=None):
                self.frozen = False
                self.shapes = set()
                self.calls = 0
                self.closed = False
                self.instances.append(self)

            def gemm(self, left, right):
                if torch.is_grad_enabled():
                    raise RuntimeError("model evaluation unexpectedly enabled gradients")
                shape = (tuple(left.shape), tuple(right.shape))
                if self.frozen and shape not in self.shapes:
                    raise RuntimeError("unprepared shape")
                self.shapes.add(shape)
                self.calls += 1
                return left.float() @ right.float()

            def freeze(self):
                self.frozen = True

            def metadata(self):
                return {"frozen": self.frozen, "calls": self.calls}

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"fixture")
            tokens = path / "tokens.npz"
            np.savez(tokens, input_ids=np.array([[1, 2, 3, 4]]))
            args = model_eval.build_parser().parse_args([
                "--checkpoint", str(path), "--tokens", str(tokens), "--output", str(path / "result.json"),
                "--dtype", "fp32", "--device", "cpu", "--warmup", "0", "--repeats", "1",
                "--backends", "torch", "cublaslt"])
            model = ToyModel().eval()
            original = model.mlp[0]
            with mock.patch.object(model_eval, "load_local_model", return_value=model), \
                 mock.patch("bench.suite.backends.Backend", FakeBackend):
                report = model_eval.evaluate(args)
            self.assertEqual(report["status"], "completed")
            self.assertTrue(all(row["application_quality_pass"] for row in report["backends"]))
            native = next(row for row in report["backends"] if row["backend"] == "cublaslt")
            self.assertTrue(native["metadata"]["frozen"])
            self.assertEqual(native["metadata"]["calls"], 3)  # dry run, quality, timing
            self.assertTrue(FakeBackend.instances[-1].closed)
            self.assertIs(model.mlp[0], original)


if __name__ == "__main__":
    unittest.main()
