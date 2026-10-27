# MegaFMM experiment suite

This suite measures matrix multiplication, numerical applications, and model
components under explicit quality requirements. A speedup is qualified only when
the application's quality checks pass. Synthetic algorithm tests, imported input
replay, and complete model evaluation are separate evidence scopes.

## What is measured

| Workload | Input | Quality checks | Evidence scope |
| --- | --- | --- | --- |
| GEMM | Existing shape catalog, BF16/FP16/FP32 | Original rounded-input FP64 reference and existing error gate | Computational primitive |
| LINPACK-style solve | FP32 dense square A and one RHS; synthetic or imported | FP64 residual of stored inputs, FP32-scaled backward error; forward error diagnostic | Single-device blocked LU; not standard FP64 HPL |
| Matrix functions | Controlled SPD matrices or imported overlap/covariance matrices | Square-root/orthogonalization residual, reference error, symmetry | Numerical algorithm or real input replay |
| Muon update | Synthetic matrices or captured gradients/momenta | Same-polynomial reference, update direction and norm, finite values | Optimizer component replay |
| MLP | Synthetic inputs or local checkpoint weights and activations | Output error and direction | Neural network component replay |
| Density purification | Controlled Hamiltonian or frozen scientific input | Idempotency, occupation, reference density and commutator | Numerical algorithm or real scientific input replay |
| Local model evaluation | Local checkpoint and fixed token/text data | NLL, perplexity and logits differences | Complete inference evaluation on the supplied data |

A gradient replay does not demonstrate unchanged training convergence. A frozen
Fock matrix does not demonstrate unchanged SCF energy or forces. These require
separate complete training/SCF experiments. The suite preserves that distinction
in metadata and reports.

## Backend contracts

The native MegaFMM API accepts BF16, FP16 or FP32 inputs and always writes FP32.
The matched cuBLASLt backend uses the same input/output contract and an
independently tuned Dense plan. The MegaFMM backend executes seven-product SW;
it does not replace SW with a complete dense AB call, add residual products, or
convert BF16 to FP16.

Applications explicitly cast FP32 outputs at the original algorithm's low
precision boundaries. Per-call input materialization, transposes, casts, SW
packing and reconstruction belong to the measured application. Changing those
boundaries defines another precision policy and must be labeled accordingly.

PyTorch is a practical framework baseline, not an alias for the matched native
cuBLASLt backend. In particular, a half-precision PyTorch matmul typically rounds
its result to the input dtype before conversion to FP32. Actual compute modes,
routes, workspace and library versions are recorded. FP32 pedantic comparisons
do not establish a win over TF32 or other quality-qualified implementations.

The current application runner compares GEMM replacements inside a shared
algorithm. An eigendecomposition used to construct an untimed reference is not
a timed algorithm-level baseline. Claims about the best scientific solver also
need timed cuSOLVER/EVD or established application-library competitors under the
same problem and quality requirements. Those competitors are not implemented in
this first suite. BF16 Muon follows the author's precision policy; FP16 Muon is
an explicitly tested precision variant.

`SWTuned` can select finite candidates that fail the historical local GEMM gate.
`implementation_pass`, `gemm_gate_pass`, `application_quality_pass` and
`performance_win` are distinct results. An unchecked quantity stays unchecked.
The local gate is retained as diagnostic evidence and is not silently relaxed.

## Build and environment

Use a PyTorch environment compatible with the target GPU and install the core
dependencies from `requirements-suite.txt`. Input export extras are optional.
No model or dataset is downloaded by a GPU benchmark or evaluation command.

On the current cluster, the existing allocation wrapper is:

```bash
bash scripts/on_gpu.sh nvidia-smi
```

The suite native bridge is built through CMake with the rest of MegaFMM. Build
on a compute node with CUDA loaded. Use a dedicated build/result directory to
preserve earlier benchmark artifacts. The shared bridge requires PIC objects;
the first build therefore rebuilds the static library.

```bash
MEGAFMM_BUILD_DIR=build-suite MEGAFMM_RESULT_DIR=results/suite_build \
  bash scripts/build.sh
```

The commands below describe supported workflows. They are usage examples;
execution evidence and performance conclusions must come from the resulting
artifacts, rather than from the presence of an example here.

### Choose and run a suite

The configurations in `bench/configs/` serve different purposes:

| Configuration | Coverage | Intended use |
| --- | --- | --- |
| `smoke.json` | One GEMM and four small synthetic applications, launch and graph modes, one independent process | Check integration and result production; one process cannot establish a performance confidence interval |
| `core.json` | Representative FP32/BF16/FP16 sizes, square and rectangular updates, fixed work and selected time-to-quality cases, three independent processes | Main controlled comparison with the matched native baseline |
| `stress.json` | Large condition numbers and dynamic ranges, narrow spectral gaps, longer iterations, odd and rectangular dimensions | Find numerical and coverage boundaries; failed quality checks remain results |
| `applications.json` | Imported scientific matrices, actual optimizer state and checkpoint MLP activations/weights, plus a practical PyTorch backend | Replay real inputs; missing files are recorded as skipped and never replaced with synthetic data |
| `linpack_smoke.json` | FP32 blocked LU at n=128/129, launch only, one process | Check native solve integration and block tails |
| `linpack.json` | FP32 blocked LU at n=512/1024/1025/2048/4096, block sizes 128/256, three processes | Measure complete solve wall time and equivalent FP32 GFLOPS; see [LINPACK suite](linpack_suite.md), not standard FP64 HPL |

Inspect the configuration without importing PyTorch or starting workers:

```bash
python scripts/run_suite.py --config bench/configs/core.json --list
python scripts/run_suite.py --config bench/configs/core.json --dry-run
```

Run the native smoke suite in an existing GPU allocation, using the active
PyTorch environment:

```bash
bash scripts/on_gpu.sh python scripts/run_suite.py \
  --config bench/configs/smoke.json \
  --library build-suite/libmegafmm_suite.so \
  --gemm-binary build-suite/megafmm_bench \
  --output results/suites/smoke-run

python scripts/report_suite.py results/suites/smoke-run
```

The runner creates the report automatically; the second command regenerates it
from saved observations and needs only the standard library. Use a new output
directory for each experiment. Replace the configuration with `core.json` or
`stress.json` when ready for broader coverage. `--case` is repeatable and selects
exact case IDs, for example `--case gemm_fp32_512 --case sqrt_fp32_512`.

Configuration schema version 1 uses an explicit `cases` list. Top-level
`backends`, `baseline`, `modes`, `measurement` and `backend_options` define the
shared protocol; cases can override their backends and modes. Dimensions,
dtype, iteration count, stopping rule, input path and quality thresholds belong
to the case. Relative `input_path` values resolve against the repository root,
including when the JSON configuration is saved elsewhere. `evaluation` is
`fixed_work` by default; `time_to_quality` supports launch mode and includes
its convergence checks. CUDA Graph timing requires fixed work.

### Export local scientific inputs

With the optional PySCF dependencies installed, export a small water example:

```bash
python scripts/export_suite_inputs.py --kind pyscf-overlap \
  --molecule 'O 0 0 0; H 0 .757 .587; H 0 -.757 .587' \
  --basis sto-3g --output data/suite/overlap.npz

python scripts/export_suite_inputs.py --kind pyscf-overlap \
  --molecule 'O 0 0 0; H 0 .757 .587; H 0 -.757 .587' \
  --basis sto-3g --hamiltonian fock --output data/suite/fock.npz
```

`--molecule` also accepts a local molecular coordinate file. These small inputs
exercise the pipeline; meaningful large-matrix performance studies need
appropriately sized molecules and basis sets. The Fock export executes RHF and
refuses an unconverged result. The default core-Hamiltonian export does not run
SCF.

Both exports contain the AO overlap `S` and core Hamiltonian `H_ao`. The Fock
export additionally stores AO `F`. The exported `H` is already in an orthonormal
basis, formed as `S^-1/2 @ F @ S^-1/2` for Fock or with `H_ao` for core.
Metadata sets `H_basis="orthonormal"` to prevent a second transformation.
`nocc=nelectron/2` is the rank of a closed-shell spinless projector. Imported
purification requires `occupied`/`nocc` in the case or asset metadata and does
not silently assume half filling. The exporter rejects overlap eigenvalues at
or below its cutoff; it does not silently discard basis directions.

### Export checkpoint activations and gradients

The following uses the existing local Qwen checkpoint and the repository's
short software fixture. It requires optional Transformers dependencies and
does not fetch a model, tokenizer or dataset:

```bash
bash scripts/on_gpu.sh python scripts/export_suite_inputs.py \
  --kind model-mlp \
  --checkpoint /share/home/liuyao/models/Qwen2.5-0.5B-Instruct \
  --text bench/fixtures/suite_sample.txt --sequence-length 32 \
  --dtype bf16 --device cuda \
  --tokens-output data/suite/model_tokens.npz \
  --output data/suite/mlp_replay.pt

bash scripts/on_gpu.sh python scripts/export_suite_inputs.py \
  --kind muon-gradient \
  --checkpoint /share/home/liuyao/models/Qwen2.5-0.5B-Instruct \
  --tokens data/suite/model_tokens.npz \
  --parameter model.layers.0.mlp.gate_proj.weight \
  --training-step-index 0 --dtype bf16 --device cuda \
  --output data/suite/muon_gradient.pt
```

MLP export defaults to the first supported module; `--module` chooses another
exact module path. `X` is an actual captured activation, with batch and sequence
flattened into the first dimension. Projection weights use mathematical
`[input, output]` orientation. Supported exports are two-layer GELU, preserving
the exact/tanh choice, or three-projection SiLU-gated SwiGLU.

The gradient command exports the raw gradient from a real backward pass. At
step index zero it applies no preceding update. Higher indices perform the
stated number of local SGD burn-in updates on the fixed batch, with the
configured learning rate; metadata records this history. It does not label
the result as accumulated Muon momentum or as the checkpoint's pretraining
step. To replay this archive, copy `applications.json` and change the Muon
case's ID, `input_path` and asset description to describe
`data/suite/muon_gradient.pt` as a gradient. The supplied
`muon_momentum_bf16` case is reserved for an actual momentum snapshot.

Once the scientific and MLP assets above exist, select their corresponding
application cases:

```bash
bash scripts/on_gpu.sh python scripts/run_suite.py \
  --config bench/configs/applications.json \
  --case lowdin_overlap_fp32 --case frozen_fock_purification_fp32 \
  --case prefill_mlp_bf16 \
  --library build-suite/libmegafmm_suite.so \
  --output results/suites/local-applications
```

### Evaluate a complete local model

Use the exact exported token fixture for a complete causal-LM prefill
comparison. Attention retains its model implementation; the default scope
replaces only MLP `nn.Linear` modules. `--scope projections` also selects
attention projections, while `--modules` selects exact names or patterns.
Required input materialization, FP32 output conversion and bias addition are
inside the replaced forward pass and its measured time.

```bash
bash scripts/on_gpu.sh python scripts/evaluate_suite_model.py \
  --checkpoint /share/home/liuyao/models/Qwen2.5-0.5B-Instruct \
  --tokens data/suite/model_tokens.npz --dtype bf16 --device cuda \
  --attention sdpa --scope mlp --backends torch cublaslt megafmm \
  --warmup 2 --repeats 5 --seed 0 \
  --max-nll-delta 0.01 --max-logits-relative-l2 0.05 \
  --backend-options '{"library":"build-suite/libmegafmm_suite.so","workspace_limit":268435456,"heuristic_candidates":4,"sw_tuning_candidates":4}' \
  --output results/suites/local-model-evaluation.json
```

The quality limits here are declared experimental settings, not general model
acceptance standards. The unchanged torch model supplies reference logits and
NLL; all requested backends receive the same evaluation tokens. The report
includes perplexity, logits error, checkpoint/input hashes, actual selected
modules, setup/tuning metadata and individual full-forward timing samples.
Plan discovery/tuning occurs before freeze and steady-state timing.

For held-out tuning inputs, add `--tuning-tokens` with another local fixture of
the same batch/sequence shape. The report states whether its input IDs differ;
without that option it explicitly records tuning/evaluation overlap. Multiple
seeds and representative held-out corpora are needed before broader quality
claims. The single-seed `suite_sample.txt` example establishes only behavior on
that fixture. It does not establish unchanged training convergence, corpus
perplexity or general language-model quality.

### Input and result formats

Application NPZ files contain flat, real numerical arrays and an optional
scalar Unicode `metadata_json` value. They are loaded with `allow_pickle=False`.
PT/PTH application files contain a flat tensor dictionary or
`{"tensors": tensor_dictionary, "metadata": json_dictionary}` and are loaded
with `weights_only=True`. Arbitrary Python objects, object arrays and
nonfinite tensors are rejected.

| Application | Tensor keys | Required conventions |
| --- | --- | --- |
| Matrix function | `S`, or accepted aliases `A`/`Sigma` | Symmetric positive-definite matrix after storage conversion; record regularization |
| Purification | `H`, or AO `F` together with `S` | Specify basis convention and `occupied`/`nocc`; an already orthogonal `H` must set `H_basis="orthonormal"` |
| Muon replay | `G` | State whether it is a gradient or momentum, checkpoint and actual step provenance |
| GELU MLP | `X`, `W1`, `W2`, optional `b1`, `b2` | `gelu(X @ W1 + b1) @ W2 + b2`; record GELU approximation |
| SwiGLU MLP | `X`, `W1`, `W2`, `W3`, optional `b1`, `b2`, `b3` | `(silu(X @ W1 + b1) * (X @ W3 + b3)) @ W2 + b2` |

Model token files are NPZ with integer `input_ids` and optional binary
`attention_mask`, or safe PT containing those tensors. A raw PT token tensor is
also accepted. Shape is `[batch, sequence]`; a single rank-one sequence becomes
one batch. At least two tokens and one valid next-token target are required.
Padded targets are excluded from NLL. NPZ cannot represent BF16 directly, so
the exporter preserves its represented values in FP32 and records the original
tensor dtype in metadata; the application configuration determines storage.

The suite output directory is an archive rather than only a timing table:

```text
manifest.json                 # source/binary/input hashes, environment, device, protocol
resolved_config.json          # validated configuration with defaults and resolved input paths
runs.jsonl                    # one backend/case/mode/independent-process record
samples.jsonl                 # raw timing-group observations
metrics.jsonl                 # quality observations with input/trial/phase identity
summary.json, summary.csv     # grouped process-level statistics and gate counts
report.html                  # statuses, speed/scale plots, per-run gates and metrics
report.md                    # compact speed/status summary; inspect HTML for gates
workers/<case>__<mode>__p<N>/
  payload.json, result.json   # authoritative worker request and terminal result
  stdout.log, stderr.log
  artifacts/                  # existing GEMM CSV/log artifacts when applicable
```

Schema version 1 run records identify `case_id`, `kind`, `backend`, `mode`,
`process_index`, `seed` and `dtype`. They keep `status`, `implementation_pass`,
`gemm_gate_pass`, `application_quality_pass`, `performance_win` and
`qualified_win` separate, alongside `steady_event_ms`, `steady_wall_ms`, cold
setup/capture costs, input/application provenance and backend route metadata.
JSON `null` denotes an unchecked gate, rather than a pass. `status="ok"`
means execution succeeded; quality may still fail.
`implementation_pass` records successful invocation under the worker's
execution contract; it is not a proof of general kernel correctness. Native
bridge tests, existing kernel tests and numerical/application checks supply
separate evidence.

Each timing sample stores its group, backend order, call count and per-call
`event_ms`/`wall_ms` group mean. Quality records retain `values` plus trial or
execution-phase identity; they include algorithm residuals and reference errors,
and local GEMM diagnostics when measured. The report pairs independent process
speed ratios for geometric means and bootstrap intervals. One process supplies
no interval; two processes are exploratory. Repeating timing groups does not
increase the independent process count.

The complete-model evaluator writes a separate JSON file, with `reference`,
`quality_contract`, `application_quality_pass` and per-backend `metrics`,
`timing`, `setup` and `metadata`. It is not automatically merged into the
application suite's JSONL files or report. Its `qualified_win` requires both
the measured backend and its comparison baseline to pass the quality contract.

### Exit codes, resume and checks

For `run_suite.py`, exit 0 means no execution or applicable quality failure was
recorded; slower results are valid experiments. Exit 1 indicates an execution
failure, unsupported execution, OOM or timeout. Exit 2 indicates configuration
errors or failed numerical/application quality gates. Missing external assets
are explicitly `skipped` and do not by themselves cause a nonzero exit; inspect
coverage and gate counts before treating an exit-0 run as evidence. Diagnostic
`--allow-quality-failures` suppresses the quality-related exit 2 while preserving
all failure records, and does not waive execution failures.

`evaluate_suite_model.py` likewise uses exit 1 for a backend execution failure
and exit 2 for failed quality gates or invalid input/setup. Its diagnostic flag
does not convert failed metrics to passes. A performance loss alone returns 0.

Resume an interrupted suite by repeating the same command and output directory
with `--resume`. The runner reuses saved terminal worker results, including
failures and skips, and executes only missing tasks. It rebuilds aggregate
JSONL files and reports from worker results. Resuming requires the same resolved
configuration, device, recorded source hashes, input hashes and binary/library
hashes. Changed code, newly available assets or a changed experimental protocol
require a new output directory; resume does not erase earlier failures or
silently rerun them.

Run the CPU suite checks inside the PyTorch environment:

```bash
python -m unittest discover -s tests -p 'test_suite_*.py' -v
```

Native bridge tests are opt-in and require a Hopper GPU and the built library:

```bash
bash scripts/on_gpu.sh env MEGAFMM_RUN_GPU_TESTS=1 \
  MEGAFMM_SUITE_LIBRARY="$PWD/build-suite/libmegafmm_suite.so" \
  python -m unittest discover -s tests -p 'test_suite_bridge.py' -v

bash scripts/on_gpu.sh ctest --test-dir build-suite --output-on-failure \
  -R '^fp32_pipeline_layouts$'
```

The CPU checks exercise configuration, safe input contracts, application state
and reference behavior, report handling, resume and model-evaluator substitution
on a toy model. The native bridge checks cover precision contracts, changed
bindings and CUDA Graph replay. The CTest example runs an existing FP32 layout
regression. Run the smoke suite as an additional application integration check;
running these commands is not a claim that the larger core, stress or complete
training/SCF experiments have passed.

## Measurement protocol

1. Resolve and save the complete configuration before executing it. Record
   source/binary hashes, device identity, software versions and input hashes.
2. Prepare input sets and independent references outside steady-state timing.
   Keep tuning and held-out quality inputs separate.
3. Create and tune plans before timing or capture. Freeze the plan set so a
   surprise shape cannot silently trigger tuning inside the measured call.
4. Every isolated algorithm invocation starts from its original input state.
   Replaying a graph does not continue a previously converged iteration.
5. Measure whole application invocations. Rotate input sets, alternate backend
   order, retain raw event/wall samples, and repeat in independent processes.
   Quantiles of group means are not single-request tail latency.
6. Report per-backend setup/tuning/capture cost separately from steady-state
   time. Include first-use cost and amortization when the evidence supports it.
7. Fixed-work comparisons use identical iteration counts. Time-to-quality
   comparisons include residual evaluation, convergence checks and their
   synchronization. Dynamic stopping is not treated as a fixed CUDA graph.
8. Preserve quality failures, nonconvergence, unsupported cases, OOM and slower
   results. Confidence intervals use independent repeats as the sampling unit.

## Input provenance and safety

The application loaders accept numeric NPZ files without pickle and tensor-only
PyTorch files via `weights_only=True`. The file hash and supplied metadata are
saved. A supplied source label is provenance information, not independent proof
that an input originated in a real experiment.

For synthetic cases, tuning, rotating slots and fresh quality trials use
different deterministic seeds. An imported application case currently replays
one fixed archive: new seeds do not create new external observations, and
rotating slots repeat that archive. Its hashes expose this reuse. Such a run
does not establish accuracy on held-out snapshots. The separate model evaluator
supports a distinct `--tuning-tokens` fixture and records whether token values
actually differ.

Scientific exports must state the basis convention, occupation convention,
regularization and any discarded directions. A nonorthogonal AO Fock matrix
must be transformed consistently before an orthogonal-basis SP2 calculation.
The occupation of a spinless projector is not automatically the electron count.

Muon follows a fixed low-precision polynomial. Exact SVD polar decomposition is
an auxiliary diagnostic, not its required output. Application-level competitors
such as Gram Newton-Schulz may change the graph and internal precision and
belong in a separately identified comparison.

## Initial validation on the current H100 allocation

The first validation executed the complete smoke suite, nine selected core
cases with three independent process pairs, five local-input cases, and one
local causal-LM fixture evaluation. It did not execute every core case or the
stress catalog. Exact commands are saved in
`results/suite_build/validation_commands.json`; source/artifact hashes and
preservation checks are in `results/suite_build/verified_build.json`.

| Validation | Observed result |
| --- | --- |
| Full PIC build | Successful; shared bridge and existing benchmark built |
| CPU suite | 58 tests: 57 passed, one explicitly opt-in GPU test skipped |
| Native bridge opt-in | All 10 tests passed, including the GPU dtype/layout/rebinding/Graph contract test |
| CMake host suite | 1/1 CTest passed |
| Smoke | 10 workers, 20 backend records; all executed; 16 application gates and both SW GEMM gates passed |
| Selected core | 48 workers, 96 records; all executed; all 60 application gates passed; six FP32 SW GEMM gates passed and 12 low-precision SW GEMM gates failed |
| Local input replay | Nine workers, 27 records; all executed; 18 application gates passed, nine failed |
| Complete model fixture | All three backends passed the stated fixture quality contract |
| Resume recovery | After deleting aggregate JSONL in a copy, resume reconstructed all three files byte-identically from saved workers |

Core SW/cuBLASLt speed ratios below use paired process geometric means. Values
above one indicate lower measured application latency. These are exploratory
observations on one allocation, with three independent process pairs per row
and per mode; they do not establish universal acceleration.

| Application | Ordinary launch | CUDA Graph | Application gate |
| --- | ---: | ---: | --- |
| BF16 Muon, 512 × 512 | 1.059× | 0.873× | Both passed |
| FP16 Muon variant, 1024 × 1024 | 1.071× | 0.868× | Both passed |
| BF16 SwiGLU, T=512 / D=512 / H=2048 | 1.032× | 0.734× | Both passed |
| FP32 square root, 512 × 512 | 1.014× | 0.872× | Both passed |
| FP32 SP2 time-to-quality, 512 × 512 | 1.009× | Dynamic stopping, not captured | Both passed |

The BF16 1024³ GEMM ordinary-launch speed ratio was 1.286×, but its original
local precision gate failed in all three processes. It is not a qualified GEMM
speedup. The inverse-square-root time-to-quality interval crossed one. Small
FP32 latency differences should be treated as initial observations. Full
process intervals, samples, gates and failures are in
`results/suites/core_validation/report.html`.

The real scientific fixture is a seven-orbital water/STO-3G RHF export. Its
Löwdin and frozen-Fock tests exercise real-input wiring. In this fixture, the
TF32 practical baseline failed three purification quality checks, while the
matched FP32 backends passed. The captured Qwen gradient failed the declared
Muon `relative_error <= 0.1` gate for all three backends in both modes; those
six failures were retained. Qwen MLP replay passed the application gate for SW
while its sampled local GEMM gates failed. The report preserves both verdicts.
See `results/suites/local_replay_validation/report.html`.

The model fixture used the existing Qwen2.5-0.5B-Instruct checkpoint, different
128-token tuning/evaluation text fixtures, and replacement of 72 MLP Linear
modules while retaining SDPA attention. With declared limits `abs(delta NLL)
<= 0.01` and logits relative L2 `<= 0.05`, SW measured delta NLL `+0.00343543`
and logits relative L2 `0.02340471`. Wall medians were 14.955 ms for unchanged
torch, 20.005 ms for matched cuBLASLt substitution and 20.012 ms for SW.
Five timing samples and one small fixture establish functional integration,
not a general language-model quality or performance result. Details are in
`results/suites/qwen_model_bf16.json`.

An initial bridge test used random BF16 inputs and an arbitrary pointwise
`rtol=atol=0.05`; it failed, and its log remains in
`results/suite_build/gpu_bridge_tests.log`. The corrected bridge contract test
uses exactly representable structured inputs with zero tolerance and checks
layout, pointer lifetime and Graph replay. It does not supersede random-input
precision evidence. Numerical acceptance remains with the explicit GEMM and
application gates.

Machine-readable final counts and verification are saved in
`results/suite_build/validation_summary.json`. Existing kernel sources and the
pre-existing dirty `.gitignore`, `results/build.log`, `results/current.csv`,
and frozen `archive/sw_specializations/solution/sw/sw1024.cu` were preserved. Complete SCF, full training,
algorithm-level EVD baselines and larger real datasets remain outside these
initial validation results.

## References

- [iSQRT-COV](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html)
- [PySCF Lowdin implementation](https://pyscf.org/_modules/pyscf/lo/orth.html)
- [Muon implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py)
- [Gram Newton-Schulz](https://dao-lab.ai/blog/2026/gram-newton-schulz/)
- [GAMBITs density matrix algorithms](https://github.com/tblite/gambits)
- [Mixed-precision GPU density matrix purification paper](https://doi.org/10.1021/acs.jctc.5c00262)

Design origin: academic-research-suite / experiment-agent, plan mode,
2026-10-21. Implementation and runtime evidence must be assessed separately
from this design; unexecuted configurations are not verified results.
