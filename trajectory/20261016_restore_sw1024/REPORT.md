# Restore specialized 1024 SW

Restored by explicit user request on 2026-10-16. The new `solution/sw1024.cu`
is byte-identical to `79dad0b:solution/sw_pipeline.cu` (iter313 restoring iter282).
This is the seven-product, uncompensated SW PDL/TMA/WGMMA implementation.

Explicit SWPipeline selects `sw1024_pdl` for contiguous, non-transposed 1024^3,
alpha=1, beta=0, workspace budget >=4 MiB. The plan reserves 4 MiB, and each
binding uses the specialization when A/B are 16-byte aligned. Other bindings
use general SW. Smaller budgets and other shapes/layouts retain the existing
zero-workspace general SW. No direct dense path is substituted for explicit SW.
Auto/Dense behavior and all precision research prototypes are preserved.

## Validation

- Full clean build on the existing H100 allocation: PASS.
- Selected CTest suite: 4/4 PASS (gpu_correctness, exact_winograd_dag,
  sw_pipeline_implementation, tuning_confirmation_policy).
- Dedicated BF16/FP16 integer diagonal inputs: exact original AB checks pass
  for fresh direct calls and Graph replay, with poisoned workspace. Low-budget
  and unaligned-binding general SW cases pass; missing workspace is rejected.
- Original FP64 numerical test: 72 checks, 48 failures, zero nonfinite.
  Each dtype has 36 checks / 24 failures. This is NOT numerical acceptance.
- Compute-sanitizer memcheck: quick scope, 0 errors. The target exits 2 due
  to the explicitly retained precision failures; no racecheck/synccheck run.
- No new performance measurement or optimization iteration. Historical latency
  must not be presented as a restoration measurement.

Exact source/binary hashes and error maxima: validation.json.
The pre-existing uncommitted unified SW edits were retained, not rolled back.
