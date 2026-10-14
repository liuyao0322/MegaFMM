# Hints

## Current user directives — 2026-10-14 architecture reset

These directives supersede the previous experiment-specific instructions to preserve
all historical implementations or continue speed-only iterations. They come from
the user's requests in this conversation:

- Delete old code; redesign the SW architecture next. All old SW implementations,
  including Mega45 and Mega53, are removed. Do not silently restore them.
- Investigate a unified intermediate buffer and separate TMA, Tensor Core and
  CUDA Core warp roles. This is a proposed design, not a validated implementation.
- Numerical acceptance is a prerequisite for declaring a winner. Report numerical
  failures prominently; performance-only rankings do not establish a valid solution.
- First validate the new numerical representation and complete original-input
  precision gate, then optimize the accepted implementation's performance.

## Preserved baseline and contracts

- Current runtime exposes only Auto and Dense, both ordinary tuned cuBLASLt GEMM.
- A/B are BF16 or FP16; computation/accumulation and output are FP32.
- The reference computes ordinary AB in FP64 from the same original rounded A/B;
  never transform reference inputs to make an SW candidate pass.
- Preserve the fixed gate eF_SW <= 3*eF_dense + floor, BF16 floor=1e-3 and
  FP16 floor=1e-4; require all fresh trials and no nonfinite. Do not relax gates.
- Preserve explicit tuning, caller workspace, current pointer validation,
  alpha/beta semantics, and run() without allocation or host synchronization.
- Time complete invocations, including dynamic transformations. Compare matching
  dtype, shape, output, workspace and execution mode against independently tuned Dense.
- Future optimized SW enters Auto only after numerical and performance acceptance.
- Build and run CUDA on assigned compute resources, using cluster-llm-ops guidance.
  Existing scripts default to H100 allocation107488/CUDA12.4; verify allocation first.

## History and optimization workflow

- Prior committed directives are available in Git history; current directives are above.
- PLAN.md and ITERATIONS.md retain historical contracts and evidence. Architecture,
  route and resource descriptions there do not imply current source implementations.
- Keep formal historical results and trajectory snapshots. Avoid accumulating old
  candidates in solution/ merely to preserve history.
- When optimization resumes, follow AKO4ALL's bench -> iteration log -> Git commit
  protocol. The current reset is cleanup, not a new performance iteration.
- Prior session ncu probe failed; do not repeat profiling probes automatically just
  because of this cleanup. The user has not requested a new optimization loop.

## First implementation scope — latest user request

- Implement the proposed unified shared-buffer, TMA/CUDA-Core/WGMMA warp-specialized first version.
- Only test and benchmark M=N=K=1024 in this task, for BF16 and FP16.
- Compare complete invocations against tuned cuBLASLt Dense, with original FP64 reference and unchanged numerical gates.
- Deliver the first implementation and comparison; do not start an open-ended optimization loop.
