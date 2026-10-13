# Hints

## User directives (2026-10-13)

- From the user's prompt: implement Strassen–Winograd according to `PLAN.md` and continuously optimize with AKO4ALL.
- `PLAN.md` defines the numerical, API and timing contract: BF16/FP16 inputs, FP32 sums before each final operand cast, FP32 accumulation/output; dynamic A/B transforms remain inside every timed invocation.
- Compare against tuned cuBLASLt with identical output dtype and timing mode. Fix the initial BF16/FP16 numerical gates before looking at candidates; report failures rather than widening gates.
- Use the existing H100 allocation 107488 only after verifying it, with CUDA 12.4 and CUTLASS 3.5.1. Downloads on login node; build, test and profile on assigned compute resources.
- Automatic dispatch selects ordinary GEMM unless a shape/dtype/mode has passed both numerical and measured end-to-end performance gates. Unsupported shapes/transpose/ranges fall back safely.
- Session profiler probe: ncu2024.1 failed with `InterprocessLockFailed` because `/tmp/nsight-compute-lock` is not writable. Per AKO4ALL, no repeated ncu probe this session; use end-to-end/stage event or Nsight Systems timing and compiler-resource analysis.
- User resumed from the final large-shape report: continue optimizing and focus on small-matrix speedup. Prioritize128/256/512/1024/2048/4096 squares and small rectangular controls; ordinary launch and CUDA Graph remain separate. Numerical failures must remain visible, but are not a reason to stop the requested execution-performance experiments.

<!-- User-supplied behavior directives. The skill reads this at session start
     and respects any constraint named here. Examples:
     - Optimization constraints or focus areas (e.g., "Prefer Triton over raw CUDA")
     - Strategies to try or avoid     (e.g., "Avoid shared memory")
     - Agent behavior controls         (e.g., "Stop after 5 iterations")
     - Dependency policies             (e.g., "Do not install new packages")
     - Environment constraints         (e.g., "ncu is unavailable on this host",
                                             "8GB device memory limit")

     The skill's own protocol (iteration steps, stall handling, ncu fallback,
     stopping rules) lives in SKILL.md — do not duplicate it here.
-->
