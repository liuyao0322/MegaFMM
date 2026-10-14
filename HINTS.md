# Hints

## Current user directives — 2026-10-14 architecture reset

These directives supersede the previous experiment-specific instructions to preserve
all historical implementations or continue speed-only iterations. They come from
the user's requests in this conversation:

- Delete old code; redesign the SW architecture next. All old SW implementations,
  including Mega45 and Mega53, are removed. Do not silently restore them.
- Investigate a unified intermediate buffer and separate TMA, Tensor Core and
  CUDA Core warp roles. Current implementation and its measured scope are recorded
  in README and the first-version report.
- Numerical acceptance is a prerequisite for declaring a winner. Report numerical
  failures prominently; performance-only rankings do not establish a valid solution.
- First validate the new numerical representation and complete original-input
  precision gate, then optimize the accepted implementation's performance.

## Preserved baseline and contracts

- Auto and Dense remain ordinary tuned cuBLASLt GEMM. Experimental SWPipeline is
  explicitly forced and supports only contiguous non-transposed 1024^3 / alpha=1 /
  beta=0. Passing the measured numerical cases does not promote it to Auto.
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
- Prior session ncu probe failed. Use analytical evidence and runtime statistics
  without repeating the known failed probe unless new evidence warrants it.

## First implementation scope — latest user request

- Implement the proposed unified shared-buffer, TMA/CUDA-Core/WGMMA warp-specialized first version.
- Only test and benchmark M=N=K=1024 in this task, for BF16 and FP16.
- Compare complete invocations against tuned cuBLASLt Dense, with original FP64 reference and unchanged numerical gates.
- The first implementation and comparison are complete. The latest user goal
  supersedes the previous stopping instruction: use AKO4ALL to continuously optimize
  the current kernel, while preserving 1024-only scope and numerical acceptance.

## Active optimization goal — latest user request

- User explicitly requested sustained AKO4ALL optimization. Continue beyond the first version.
- Rank actual SW complete-call latency, retain failures, and preserve all fresh-input gates.
- On every iteration: benchmark with iter-N label, immediately log, immediately commit.
- Keep Auto on Dense until numerical and end-to-end performance acceptance.

## NCU profiling — explicit user request after optimization pause

- User paused sustained optimization, then explicitly requested NCU profiling. Profiling does not resume optimization.
- NCU2025.1.1 from CUDA12.8 successfully profiled iter76 on allocation107488/g08,39 passes per dtype. Prior failed NCU probes no longer imply NCU is unavailable.
- The initial /tmp/nsight-compute-lock failure was resolved by a private TMPDIR after checking no other ncu process was active, following NVIDIA's documented workaround. Do not remove another user's lock file.
- Evidence: trajectory/20261014_ncu_iter76/REPORT.md, summary.json and .ncu-rep artifacts. Production source remains iter79; profiling used a separate iter76 snapshot.
- Key findings: Tensor elapsed activity~18.8percent; occupancy24.93percent; eligible warps~0.32/scheduler; DRAM active<1percent in this warm-cache capture. Long-scoreboard hotspots include product-ready mbarrier consumers, not just global input loads. P stores have8-way bank conflicts and401408 excessive wavefronts.
- Dense profiling must retain the actual independently tuned configuration. Under NCU, retuning picked a slower BF16 split-K algorithm; exclude that initial capture from comparisons. Fixed Dense id39/tile18/stage35/cluster10 (BF16) and id21/tile18/stage18 (FP16) were captured instead.
- Kernel replay, cache-control none, clock-control none. Short-kernel samples and unlocked clocks limit exact percentages; profiler timing is diagnostic and does not replace normal complete-call benchmarks.

## P shared bank conflict — latest narrow user request

- User explicitly asked to fix the eight-way bank conflict when storing P to shared. This authorizes this fix and its validation; sustained optimization remains paused.
- Final iter81 (58743df) starts from profiled iter76 (456ce9d), uses row-local XOR swizzle `row*32+(col^((row&1)|((row&6)<<2)))` for both P producer and consumer, retaining row-major global output. Other arithmetic and barriers match iter76.
- NCU confirms all16 P STS and42 P LDS per dtype are1-way,0 excessive wavefront. P store excessive wavefront drops401408->0. Kernel-wide hardware bank-conflict counters are not all zero; do not conflate those counters with P instruction evidence.
- Full invocation BF16/FP16=21.216/21.661us versus historical iter76=21.245/21.883us: no significant overall speedup established. Still slower than tuned Dense12.381/9.992us. Auto remains Dense.
- Original-input numerical gates pass, independent72-check launch/Graph stress passes, memcheck/racecheck/synccheck quick tests pass. Only1024^3 tested.
- Evidence: trajectory/20261014_validation-iter81-p-banks/REPORT.md and NCU artifacts.

## Streaming recombination — latest user request

- User asked to try the proposed next version. Implement and validate the first step: P1,P2,P6,P7,P5,P3,P4 scheduling with streaming recombination and early output.
- Keep BK256, two operand slots, two conflict-free P slots and current warp roles for this experiment; do not conflate it with deeper WGMMA pipelining or warp-count changes.
- Advance operand/product slot phases by execution sequence, not logical P number. Retain original recombination operation order and original-input precision gates.
- Only1024^3 BF16/FP16; measure full invocations against tuned Dense, independent launch/Graph correctness and NCU evidence. This request does not resume an indefinite optimization goal.
