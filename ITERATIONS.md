# Iteration Log

Prior history remains in Git. Current optimization resumes at iteration 51.

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|---|---|---|---|---|

| 51 | Publish shared stages once per warp | 0.0282 | 0.392295 | measured |

| 52 | Test independently aligned 32-byte swizzle operands | 0.0270 | 0.404168 | failed |

| 53 | Write swizzled operands through typed CuTe tensor views | 0.0248 | 0.438394 | measured |

| 54 | Process BK64 with on-demand FP32 transforms | 0.0330 | 0.335406 | measured |

| 55 | Poll barrier phase without suspend timer | 0.0319 | 0.346034 | measured |

| 56 | Align right operand with natural row-major shared access | 0.0351 | 0.314660 | measured |

| 57 | Keep four output tiles in CUDA worker registers | 0.0349 | 0.317959 | measured |

| 58 | Batch BK128 using one operand stage and register outputs | 0.0311 | 0.356526 | measured |

| 59 | Skip preparation of unused residual planes | 0.0317 | 0.349697 | measured |

| 60 | Enable two-CTA capacity with BN16 and register output | 0.0338 | 0.319083 | measured |

| 61 | Increase CUDA producer parallelism based on stage clocks | 0.0719 | 0.152644 | measured |

| 62 | Scale producer population to twenty-seven warps | 0.0874 | 0.126517 | measured |

| 63 | Compute shared global transforms once within a cooperative invocation | 0.2507 | 0.044046 | failed |

| 64 | Correct original A22 coordinates in cooperative pipeline | 0.2517 | 0.044002 | passed |

| 65 | Batch all leaf operands per K slice | 0.2471 | 0.044803 | passed |

| 66 | Issue seven independent WGMMA products before waiting | 0.1933 | 0.057705 | passed |

| 67 | Retain CUDA output accumulators in registers | 0.2419 | 0.045762 | passed |

| 68 | Double reduction tile with split swizzled A loads | 0.3680 | 0.030149 | passed |

| 69 | Probe single-buffer BK256 geometry | 0.0000 | 0.000000 | failed |

| 70 | Measure BK256 with correct aligned shared budget | 0.3132 | 0.035336 | passed |

| 71 | Increase CUDA recombination parallelism on BK128 | 0.3920 | 0.028272 | passed |

| 72 | Accumulate full-K products and recombine once | 0.5341 | 0.023643 | passed |

| 73 | Pipeline two WGMMA K groups before releasing operands | 0.3639 | 0.030428 | passed |

| 74 | Release operand slots with four warp leaders | 0.4706 | 0.023473 | passed |

| 75 | Fit two cooperative CTAs per SM using BN16 | 0.3649 | 0.030256 | passed |

| 76 | Stream complete products into CUDA registers and double BK | 0.5152 | 0.021564 | passed |

| 77 | Load complete reduction operands before each leaf | 0.3903 | 0.028540 | passed |

| 78 | Reduce per-worker P register footprint with27 CUDA warps | 0.3839 | 0.028947 | passed |

| 79 | Four operand slots with a single WGMMA wait per P | 0.4464 | 0.024899 | passed |

| 80 | Store and consume P in native WGMMA fragment order | 0.4634 | 0.024033 | passed |

| 81 | XOR-swizzle P rows for conflict-free stores and loads | 0.5224 | 0.021439 | passed |

| 82 | Stream Winograd recombination and publish output early | 0.5180 | 0.021406 | passed |

| 83 | Reuse canonical GMMA descriptors across K tiles | 0.5328 | 0.020759 | passed |

| 84 | Restrict GMMA K offsets to the low descriptor word | 0.0000 | 0.000000 | failed |

| 85 | Keep upper GMMA descriptor fields invariant | 0.5382 | 0.020929 | passed |

| 86 | Expose fixed operand slot addresses to GMMA lowering | 0.5448 | 0.020372 | passed |

| 87 | Publish high operands before residual operands | 0.5764 | 0.019300 | passed |

| 88 | Reduce P consumers to five warps while preserving input parallelism | 0.5520 | 0.020125 | passed |

| 89 | Use eight evenly packed P-consumer warps | 0.5673 | 0.019577 | passed |

| 90 | Vectorize input transformation IO in adjacent pairs | 0.5825 | 0.019053 | passed |

| 91 | Transform four adjacent elements with64bit IO | 0.5703 | 0.019613 | passed |

| 92 | Combine A chunk transfers with higher-rank TMA maps | 0.5813 | 0.019093 | passed |

| 93 | Expose the existing no-write-alias binding contract to NVCC | 0.5835 | 0.019041 | passed |

| 94 | Overlap the two K-block WGMMA groups with proven slot retirement | 0.9225 | 0.020434 | passed |

| 95 | Prefetch tensor-map descriptors during input transformation | 0.5804 | 0.019156 | passed |

| 96 | Restore the validated small-gain iter92 candidate | 0.5822 | 0.019069 | passed |

| 97 | Seven-product SW without residual compensation | 0.7728 | 0.014410 | experiment |

| 98 | Compact two-stage operand shared storage | 0.7745 | 0.014399 | experiment |

| 99 | Three compact operand stages with immediate retirement | 0.8297 | 0.013506 | experiment |

| 100 | Four compact operand stages | 0.8573 | 0.013169 | experiment |

| 101 | K128 with six operand prefetch stages | 0.9326 | 0.013673 | experiment |

| 102 | K512 one completion group per product | 0.8022 | 0.013924 | experiment |

| 103 | Two outstanding WGMMA groups on four-stage operand ring | 0.8453 | 0.013214 | experiment |

| 104 | Four-element64bit high-only transform IO | 0.8815 | 0.012677 | experiment |

| 105 | Correct64bit transform output pointer width | 0.8444 | 0.013184 | experiment |

| 106 | Separate transform kernel instead of cooperative grid barrier | 0.7748 | 0.014437 | experiment |

| 107 | Use grid synchronization to order barrier initialization | 0.8478 | 0.013166 | experiment |

| 108 | Specialize all seven TC product steps for fixed four-slot addresses | 0.7450 | 0.015039 | experiment |

## Iterations

### Iter 51 — Publish shared stages once per warp

- **Hypothesis:** Reduce per-thread barrier traffic with warp-leader acquire/release and warp sync.
- **Changes:** Matching 3/4 arrival counts; async proxy fences and WGMMA wait0 unchanged.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.392295 ms; mean speedup=0.0282x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x32 v0 | launch | 0.012356 | 0.401572 | 0.0308 | 7.15225e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x32 v0 | launch | 0.009790 | 0.383018 | 0.0256 | 3.09536e-07 | 1 |

- **Analysis:** Full launch 1024 both dtype,20warm100calls5groups3fresh. Prior logger attempt failed because ITERATIONS.md had disappeared during the run; continue current log without restoring old content.
- **Next:** Inspect latency and correctness; test shared operand bank layout next.
- **Archive:** trajectory/20261014_185923_iter-51/

### Iter 52 — Test independently aligned 32-byte swizzle operands

- **Hypothesis:** Interleaved operand stores scatter K groups across shared banks; SW32 gives aligned K16 blocks without the SW64 origin issue.
- **Changes:** Restore iter50 core verbatim then change only operand layouts to Layout_K_SW32_Atom, with matching descriptor and physical indexing.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.404168 ms; mean speedup=0.0270x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x32 v0 | launch | 0.012122 | 0.411769 | 0.0294 | 1.32243 | 0 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x32 v0 | launch | 0.009768 | 0.396566 | 0.0246 | 1.32243 | 0 |

- **Analysis:** Benchmark exited2; retain full output and numerical failure before investigating. Timings are diagnostic until precision passes.
- **Next:** Inspect the numerical results and compiled layout mapping; retain only a correct faster implementation.
- **Archive:** trajectory/20261014_190111_iter-52/

### Iter 53 — Write swizzled operands through typed CuTe tensor views

- **Hypothesis:** Layout with smem_ptr_flag encodes byte-address swizzling; plain pointer[layout(element)] is not the physical address used by the descriptor. Typed tensor views must be used for stores.
- **Changes:** Store hi/lo A/B through the same CuTe tensors consumed by WGMMA, keeping SW32 and all numerical operations unchanged.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.438394 ms; mean speedup=0.0248x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x32 v0 | launch | 0.012085 | 0.442599 | 0.0273 | 7.15225e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x32 v0 | launch | 0.009724 | 0.434189 | 0.0224 | 3.09536e-07 | 1 |

- **Analysis:** Complete benchmark exited0; both dtypes pass all three fresh FP64 checks. Latency is recorded below; compare against baseline before claiming a gain.
- **Next:** Inspect current assembly/resource use and elapsed latency; continue reducing real critical-path work.
- **Archive:** trajectory/20261014_190221_iter-53/

#### Reassessment after three non-improving candidates (51–53)
51 regressed despite fewer arrivals;52 invalid swizzled stores failed;53 fixes typed
byte-address swizzling and passes but is slower (443/434us). No new ncu probe per
known prior failure. SASS resource inspection shows no local spills or CALL in the
SW kernels; static operand/store/control work remains substantial. CUTLASS primary
pipeline documentation https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md
supports organizing real producer/consumer overlap; it does not prove our bottleneck.
Next direction: larger K slices and remove global shared ST table, evaluating total
work/handshake tradeoff rather than continuing the unsuccessful layout-only path.

### Iter 54 — Process BK64 with on-demand FP32 transforms

- **Hypothesis:** Halve the K-slice handshake/recombination count; remove shared ST table to fit larger slices, computing each final operand through the same RN FP32 DAG.
- **Changes:** Restore iter50 core, BK64, one raw stage and two operand stages, on-demand ST, dynamic shared188544 bytes, matching A/B TMA box dimensions. Keep hi+lo and 15 leaf terms.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.335406 ms; mean speedup=0.0330x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012383 | 0.342049 | 0.0362 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009782 | 0.328763 | 0.0298 | 4.79212e-07 | 1 |

- **Analysis:** Complete benchmark exited0. Both dtypes pass all three fresh original-input checks; timings below determine actual improvement. Transform operation count rises because shared ST reuse is removed.
- **Next:** Inspect measured gain and validate stress; explore larger K slices if critical-path reduction is effective.
- **Archive:** trajectory/20261014_190538_iter-54/

### Iter 55 — Poll barrier phase without suspend timer

- **Hypothesis:** Short producer/consumer stages may be limited by suspending try-wait wakeup latency; active acquire-phase checks preserve ordering.
- **Changes:** Replace ClusterBarrier wait with repeated test_wait phase checks; keep layout, roles, transforms, tile and precision identical to iter54.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.346034 ms; mean speedup=0.0319x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012334 | 0.351558 | 0.0351 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009809 | 0.340510 | 0.0288 | 4.79212e-07 | 1 |

- **Analysis:** Full original launch benchmark exited0; compare measured rows to54 before attributing any timing change.
- **Next:** Validate improvement and stress; use larger compute batches or register output accumulation next.
- **Archive:** trajectory/20261014_190704_iter-55/

### Iter 56 — Align right operand with natural row-major shared access

- **Hypothesis:** B K-major storage transposes contiguous source lanes into strided shared writes; MN-major B and swizzled A should improve physical access.
- **Changes:** From best54: A K-major SW128, B MN-major SW64, matched typed CuTe stores/descriptors and WGMMA transpose mode. Same FP32 operations and high-low products.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.314660 ms; mean speedup=0.0351x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012325 | 0.319603 | 0.0386 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009800 | 0.309717 | 0.0316 | 4.79212e-07 | 1 |

- **Analysis:** Full 1024 launch benchmark exited0, all fresh original gates pass. Actual latencies below; layout cost attribution awaits measured comparison.
- **Next:** Inspect measured runtime; retain best source and continue work batching/accumulator ownership improvements.
- **Archive:** trajectory/20261014_190904_iter-56/

### Iter 57 — Keep four output tiles in CUDA worker registers

- **Hypothesis:** Repeated output shared loads/stores can be removed without changing each element arithmetic order; static unrolled chunks retain values in registers.
- **Changes:** From56, move four output accumulators to per-worker register arrays, remove32KiB shared output region, dynamic shared155776 bytes. Preserve P handoff and numerical operations.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.317959 ms; mean speedup=0.0349x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012443 | 0.317511 | 0.0392 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009773 | 0.318407 | 0.0307 | 4.79212e-07 | 1 |

- **Analysis:** Full 1024 launch benchmark exited0 with both dtype fresh gates passing. Assess runtime and register/local-memory resources before retaining.
- **Next:** Inspect resource usage and stress; exploit reduced shared footprint to increase K batching if faster.
- **Archive:** trajectory/20261014_191045_iter-57/

### Iter 58 — Batch BK128 using one operand stage and register outputs

- **Hypothesis:** Four rather than eight K slices reduce seven-P publication/recombination boundaries; one operand stage keeps the larger working set within Hopper shared capacity.
- **Changes:** From57, BK128, one raw stage, one operand stage, dynamic shared204928 bytes, unchanged15 high-low leaf terms and FP32 RN DAG.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.356526 ms; mean speedup=0.0311x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x128 v0 | launch | 0.012230 | 0.355055 | 0.0344 | 7.17661e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x128 v0 | launch | 0.009918 | 0.357996 | 0.0277 | 8.25634e-07 | 1 |

- **Analysis:** Full 1024 launch benchmark exited0; both dtypes pass fresh numerical gates. No claim about actual occupancy; register/shared costs must be assessed against measured runtime.
- **Next:** Measure gain; remove unused residual-plane preparation for original operands and validate accepted candidate.
- **Archive:** trajectory/20261014_191209_iter-58/

### Iter 59 — Skip preparation of unused residual planes

- **Hypothesis:** Original input operands need no low plane and their consumers never read one; avoid these conversions and shared stores.
- **Changes:** From best56, prepare A-low only forP3/P5/P6/P7 and B-low only forP4/P5/P6/P7. All nonzero correction terms retained.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.349697 ms; mean speedup=0.0317x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012389 | 0.355324 | 0.0349 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009826 | 0.344070 | 0.0286 | 4.79212e-07 | 1 |

- **Analysis:** Full launch exited0; no numerical approximation was removed. Record all rows and compare complete-call own latency to56.
- **Next:** Assess gain; explore BN16 with register outputs to enable two CTA capacity per SM.
- **Archive:** trajectory/20261014_191357_iter-59/

#### Reassessment after non-improving57–59
Register Q alone did not improve both dtypes; BK128 with one operand slot regressed,
showing the larger batch does not compensate for lost operand-stage overlap here.
Skipping unused lows introduced uniform control and regressed. Best remains56.
No new ncu probe. Next distinct axis is residency capacity: BN16 plus register Q
reduces shared to approximately108KiB, allowing two CTA capacity on H100; actual
performance and compiler resources must verify whether it helps, not assume it.

### Iter 60 — Enable two-CTA capacity with BN16 and register output

- **Hypothesis:** BN16 reduces shared sufficiently for two resident CTA capacity, potentially hiding producer-consumer stage latency.
- **Changes:** From57, BN16/BK64, eight accumulator registers per WGMMA thread, MN32 B layout, shared110720 bytes, launch bounds256/2. Same original numerical gate and15 leaf terms.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.319083 ms; mean speedup=0.0338x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x16x64 v0 | launch | 0.011884 | 0.323156 | 0.0368 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x16x64 v0 | launch | 0.009708 | 0.315010 | 0.0308 | 4.79212e-07 | 1 |

- **Analysis:** Complete launch benchmark exited0; do not infer measured occupancy from capacity. Use latency and resources for assessment.
- **Next:** Inspect performance and resource use; instrument stage clocks if low occupancy or producer waits remain ambiguous.
- **Archive:** trajectory/20261014_191528_iter-60/

### Iter 61 — Increase CUDA producer parallelism based on stage clocks

- **Hypothesis:** Instrumented56 shows operand preparation about94percent of CUDA-worker cycles while Tensor waits; increase producer warps to hide load/convert/address dependencies.
- **Changes:** Restore56,512-thread CTA with11 SIMT producer warps,4 WGMMA warps,1 TMA warp; matching352 arrivals and disjoint worker indices. Precision and shared layout unchanged.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.152644 ms; mean speedup=0.0719x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012254 | 0.155966 | 0.0786 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009736 | 0.149323 | 0.0652 | 4.79212e-07 | 1 |

- **Analysis:** Full original 1024 launch benchmark exited0. Diagnostic clocks are from a separate instrumented clone, not the formally timed kernel; no source caching or input-dependent work removal.
- **Next:** Inspect actual speedup and verify stress if improved; tune producer population and operand generation based on evidence.
- **Archive:** trajectory/20261014_191901_iter-61/

### Iter 62 — Scale producer population to twenty-seven warps

- **Hypothesis:** More independent producer warps can hide operand load/convert/address dependency latency identified by stage clocks.
- **Changes:** From61,1024-thread CTA,864 CUDA workers,128 WGMMA consumers,one TMA warp; matching arrival counts. Same shared geometry and arithmetic.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.126517 ms; mean speedup=0.0874x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.012330 | 0.129250 | 0.0954 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_unified_v1_hilo_64x32x64 v0 | launch | 0.009817 | 0.123784 | 0.0793 | 4.79212e-07 | 1 |

- **Analysis:** Full launch benchmark exited0; inspect actual complete-call latency and resource limits before choosing between61 and62.
- **Next:** Validate faster source and inspect resource usage; next investigate eliminating repeated CTA-local transformations through cooperative once-per-invocation packing.
- **Archive:** trajectory/20261014_192041_iter-62/

### Iter 63 — Compute shared global transforms once within a cooperative invocation

- **Hypothesis:** Remove repeated CTA-local operand construction by packing ST once into caller workspace, grid-syncing, then using TMA directly into WGMMA layouts.
- **Changes:** Single cooperative128-CTA launch,8MiB caller workspace,all threads pack FP32 DAG high-low ST,async global proxy fence+grid sync,swizzled2D/3D TMA directly to double-buffered operands; same15 leaf terms.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.044046 ms; mean speedup=0.2507x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.012207 | 0.044301 | 0.2755 | 1.00221 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.009887 | 0.043790 | 0.2258 | 1.0022 | 0 |

- **Analysis:** Benchmark exited2; failed numerical rows are diagnostic and cannot be promoted. Full input-dependent packing and cooperative barrier are inside every timed invocation.
- **Next:** Inspect correctness and map/layout consistency, then repair or restore the best62 source.
- **Archive:** trajectory/20261014_192502_iter-63/

### Iter 64 — Correct original A22 coordinates in cooperative pipeline

- **Hypothesis:** P4 must use A22; the missing column offset caused unit relative error in63.
- **Changes:** Correct p3 original A tensor-map column offset to512; retain once-per-invocation global high-low packing and complete-call timing.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.044002 ms; mean speedup=0.2517x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.012275 | 0.044221 | 0.2776 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.009884 | 0.043783 | 0.2258 | 4.79212e-07 | 1 |

- **Analysis:** Complete frozen benchmark exited0; all original numerical gates passed. Inspect measured timings and stress before accepting.
- **Next:** Validate72 stress cases and graph execution; assess complete latency against Dense.
- **Archive:** trajectory/20261014_192807_iter-64/

### Iter 65 — Batch all leaf operands per K slice

- **Hypothesis:** Eight shared operand publications instead of56 can reduce serial TMA and barrier cost.
- **Changes:** 225408-byte shared arena retains11 A and11 B planes per K slice, reusing originals and transforms across7 products; one ready/free pair per slice.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.044803 ms; mean speedup=0.2471x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.012284 | 0.044964 | 0.2732 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.009862 | 0.044642 | 0.2209 | 4.79212e-07 | 1 |

- **Analysis:** Full unchanged benchmark exited0. All numerical gates passed; compare actual complete invocation timings.
- **Next:** Inspect latency and validate faster shared-buffer arrangement.
- **Archive:** trajectory/20261014_192948_iter-65/

### Iter 66 — Issue seven independent WGMMA products before waiting

- **Hypothesis:** Independent P accumulators remove per-product completion latency bubbles in the bulk operand design.
- **Changes:** Seven16-register accumulator fragments and one WGMMA commit/wait per K slice; preserve fixed leaf order and CUDA output recombination.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.057705 ms; mean speedup=0.1933x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.012445 | 0.058099 | 0.2142 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.009875 | 0.057312 | 0.1723 | 4.79212e-07 | 1 |

- **Analysis:** Original full benchmark exited0 and numerical gates passed. Compare performance before selecting bulk batching.
- **Next:** Inspect timings and register use; validate stress on a performance improvement.
- **Archive:** trajectory/20261014_193108_iter-66/

### Iter 67 — Retain CUDA output accumulators in registers

- **Hypothesis:** Stage-clock64 finds significant output recombination stalls; static register accumulators remove repeated shared output reads and writes.
- **Changes:** Restore64 and replace four shared Q planes with statically unrolled22-element per-worker register accumulators; shared106624 bytes.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.045762 ms; mean speedup=0.2419x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.012280 | 0.045980 | 0.2671 | 7.15726e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x64 v0 | launch | 0.009875 | 0.045543 | 0.2168 | 4.79212e-07 | 1 |

- **Analysis:** Frozen complete benchmark exited0; all numerical gates passed. Instrumented64 timings are diagnostic rather than native latency claims.
- **Next:** Inspect speed and full stress, then reassess recent batching alternatives.
- **Archive:** trajectory/20261014_193329_iter-67/

### Iter 68 — Double reduction tile with split swizzled A loads

- **Hypothesis:** Three recent batching/recombination changes did not improve64; stage clocks show more than half time in MMA/store. Larger K tiles halve product barriers and short completion waits.
- **Changes:** Restore64 with BK128 double buffers,188544-byte shared, and two64-column TMA boxes per A plane to respect SW128 inner-box bound; B loads full128 rows.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.030149 ms; mean speedup=0.3680x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012297 | 0.030261 | 0.4064 | 7.17661e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.009898 | 0.030036 | 0.3295 | 8.25634e-07 | 1 |

- **Analysis:** Complete unchanged benchmark exited0. Reassessment reviewed65-67, clock64 diagnostics, zero spills in66 and NVIDIA CUDA TMA swizzle restrictions plus CUTLASS warp-specialized pipeline docs.
- **Next:** Inspect complete latency and validate stress/Graph for the selected candidate.
- **Archive:** trajectory/20261014_193457_iter-68/

### Iter 69 — Probe single-buffer BK256 geometry

- **Hypothesis:** Larger K amortizes product completion and recombination barriers.
- **Changes:** BK256,one operand slot; phase alternates per product and A is split into64-column TMA boxes.
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Benchmark exited2 during build; inspect shared-budget static assertion before execution. No runtime or correctness evidence.
- **Next:** Correct shared-budget accounting and rebuild.
- **Archive:** trajectory/20261014_193556_iter-69/

### Iter 70 — Measure BK256 with correct aligned shared budget

- **Hypothesis:** BK256 halves K slices again while retaining fixed precision and within188544-byte shared capacity.
- **Changes:** Correct sizeof assertion and metadata for128-byte struct alignment; execute69 design unchanged.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.035336 ms; mean speedup=0.3132x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012284 | 0.035601 | 0.3451 | 7.25171e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009864 | 0.035071 | 0.2813 | 1.52049e-06 | 1 |

- **Analysis:** Frozen full benchmark exited0 and numerical gates passed. No size other than1024 was executed.
- **Next:** Select faster candidate, stress fresh inputs and Graph, and check sanitizer.
- **Archive:** trajectory/20261014_193634_iter-70/

### Iter 71 — Increase CUDA recombination parallelism on BK128

- **Hypothesis:** More CUDA output workers reduce shared product consumption and Q accumulation latency without increasing the15-term Tensor work.
- **Changes:** Restore68;512 threads with352 CUDA workers,128 WGMMA threads andone TMA warp, matched352 product-free arrivals; shared188544.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.028272 ms; mean speedup=0.3920x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012267 | 0.028326 | 0.4330 | 7.17661e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.009905 | 0.028217 | 0.3510 | 8.25634e-07 | 1 |

- **Analysis:** Full original benchmark exited0 with all numerical gates passing; BK256 single-buffer70 was slower than68 and is not selected.
- **Next:** Validate best latency source with72 stress, Graph timing and memory checks.
- **Archive:** trajectory/20261014_193739_iter-71/

## Validated milestone after iter71

- Selected source: iter71, commit8b1e8ca, BK128/512 threads/11 CUDA warps,8MiB workspace.
- Matching launch Dense/SW BF16=12.2666/28.3264us,FP16=9.9053/28.2166us. SW remains slower.
- Same source Graph Dense/SW BF16=11.5149/27.5709us,FP16=9.4352/27.5062us.
- All12 fresh standard numerical checks,72 stress checks,2 scoped CTests passed; no other size tested.
- CUDA12.8 memcheck fullquick=0 errors; SW-filtered racecheck=0 hazards/errors/warnings; synccheck=0 errors. Tools exited0.
- Added rejection checks for missing, misaligned, overlapping and under-budget workspace; full CTest and sanitizer after rebuild passed.
- Details and raw logs: trajectory/20261014_193836_validation-iter71-graph/REPORT.md and validation.json.
- The optimization goal remains active; no physical-floor or exhaustion claim. Next evaluate reducing WGMMA/P handoff stalls and global packing cost.

### Iter 72 — Accumulate full-K products and recombine once

- **Hypothesis:** Remove per-K CUDA output accumulation and product handshakes by making each P complete over all512 reduction elements before one U recombination.
- **Changes:** P-major TMA/TC loops; persistent16-register accumulator per leaf, seven shared products written once, no shared Q or product-free barrier;155776 bytes dynamic shared.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.023643 ms; mean speedup=0.5341x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012275 | 0.023541 | 0.5214 | 7.32914e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012984 | 0.023744 | 0.5468 | 2.11007e-06 | 1 |

- **Analysis:** Complete unchanged1024 benchmark exited0; original numerical gates passed. FP32 summation order changes and requires stress verification.
- **Next:** Inspect latency, validate cancellation and fresh Graph inputs, then overlap sequential WGMMA K slices.
- **Archive:** trajectory/20261014_194401_iter-72/

### Iter 73 — Pipeline two WGMMA K groups before releasing operands

- **Hypothesis:** Keep two independent operand slots busy while serial K updates of one leaf accumulator execute asynchronously, reducing per-slice completion bubbles.
- **Changes:** Fence accumulators once per P, commit each K group, wait_group1 then release previous slot, drain wait_group0 before reading accumulators and releasing final slot.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.030428 ms; mean speedup=0.3639x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012260 | 0.030550 | 0.4013 | 7.32914e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.009897 | 0.030306 | 0.3266 | 2.11007e-06 | 1 |

- **Analysis:** Complete1024 benchmark exited0 with original numerical gates passed. Asynchronous reuse needs targeted racecheck on selected source.
- **Next:** Inspect complete latency and correctness stress; tune CTA roles or operand stages from evidence.
- **Archive:** trajectory/20261014_194513_iter-73/

### Iter 74 — Release operand slots with four warp leaders

- **Hypothesis:** Full-K recombination removes CUDA handshakes; remaining128 arrivals per product slice may dominate operand release.
- **Changes:** Restore72, initialize each free barrier to4 and publish one arrival per Tensor warp after warp sync and completed WGMMA wait.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.023473 ms; mean speedup=0.4706x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012285 | 0.023495 | 0.5229 | 7.32914e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.009811 | 0.023452 | 0.4183 | 2.11007e-06 | 1 |

- **Analysis:** Complete benchmark exited0, original gates passed. Two-inflight73 regressed to30us and is not retained.
- **Next:** Inspect timings and test smaller tiles with two resident cooperative CTAs per SM.
- **Archive:** trajectory/20261014_194632_iter-74/

### Iter 75 — Fit two cooperative CTAs per SM using BN16

- **Hypothesis:** Smaller P arena and N tile permit two independent Tensor warpgroups per SM to cover short WGMMA/TMA latencies.
- **Changes:** BN16,BK128,256 threads,110720-byte shared,MN32 B layout,8-register fragments;256 cooperative CTAs with launch bounds256/2 and explicit resident-capacity check.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.030256 ms; mean speedup=0.3649x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x16x128 v0 | launch | 0.012161 | 0.030385 | 0.4002 | 7.32914e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x16x128 v0 | launch | 0.009926 | 0.030126 | 0.3295 | 2.11007e-06 | 1 |

- **Analysis:** Original full benchmark exited0 and numerical gates passed. Resource capacity is checked at plan creation, not claimed as measured occupancy.
- **Next:** Inspect latency and stress; reassess three recent non-improving directions if75 does not beat72 by3percent.
- **Archive:** trajectory/20261014_194738_iter-75/

### Iter 76 — Stream complete products into CUDA registers and double BK

- **Hypothesis:** Reassessment73-75 finds no3percent improvement; shrink shared P to two slots so BK256 double operands fit without losing TMA overlap.
- **Changes:** Restore74, BK256,213120-byte arena, two shared P slots with ready/free barriers, statically unrolled CUDA registers retain seven P before fixed U recombination.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.021564 ms; mean speedup=0.5152x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012272 | 0.021245 | 0.5776 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009908 | 0.021883 | 0.4528 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 benchmark exited0 and frozen numerical gates passed. Reviewed recent latency/stress and NVIDIA TMA box limits plus CUTLASS register/pipeline docs before changing direction.
- **Next:** Check actual improvement, compiler spills and fresh-input stress; test full-K single-slot geometry if warranted.
- **Archive:** trajectory/20261014_194938_iter-76/

### Iter 77 — Load complete reduction operands before each leaf

- **Hypothesis:** Full-K BK512 removes the remaining per-slice completion boundaries; compare single operand slot against overlapping BK256.
- **Changes:** Single full-K operand arena with two P slots, A split64-column boxes, B split256-row boxes, seven product waits; same213120 shared and CUDA register U DAG.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.028540 ms; mean speedup=0.3903x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x512 v0 | launch | 0.012353 | 0.028465 | 0.4340 | 7.55609e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x512 v0 | launch | 0.009918 | 0.028615 | 0.3466 | 2.89265e-06 | 1 |

- **Analysis:** Full original benchmark exited0 with all numerical gates passed;76 passed72 stress checks and had110 registers with zero local/stack.
- **Next:** Inspect latency, then focus on lower-cost packed ST generation and mixed operand layouts.
- **Archive:** trajectory/20261014_195039_iter-77/

### Iter 78 — Reduce per-worker P register footprint with27 CUDA warps

- **Hypothesis:** More CUDA workers halve live P registers per thread and raise packing parallelism; test whether fewer registers and faster handoffs improve76.
- **Changes:** Restore76 with1024 threads,864 CUDA workers,three register chunks per worker and matched864 product-free arrivals.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.028947 ms; mean speedup=0.3839x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012250 | 0.028840 | 0.4248 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009966 | 0.029053 | 0.3430 | 2.37136e-06 | 1 |

- **Analysis:** Complete unchanged benchmark exited0; numerical gates passed. Full-K single-buffer77 regressed, retaining BK256 double buffering.
- **Next:** Inspect latency and spills, then evaluate four BK128 operand slots with one leaf completion wait.
- **Archive:** trajectory/20261014_195223_iter-78/

### Iter 79 — Four operand slots with a single WGMMA wait per P

- **Hypothesis:** Keep per-slice TMA/compute overlap while removing intermediate WGMMA completion waits using four BK128 slots.
- **Changes:** Restore76; four BK128 operands, one WGMMA group per complete P and all four slots released after wait0; same213120 shared.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.024899 ms; mean speedup=0.4464x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.012319 | 0.024951 | 0.4937 | 7.32914e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x128 v0 | launch | 0.009914 | 0.024847 | 0.3990 | 2.11007e-06 | 1 |

- **Analysis:** Benchmark completed with exit0 before user paused optimization. No further optimization is authorized while paused; user separately requested NCU profiling.
- **Next:** Inspect candidate performance, then profile the best accepted source as explicitly requested.
- **Archive:** trajectory/20261014_195354_iter-79/

## User-requested NCU profiling while optimization is paused

- Profiled best accepted iter76 in an isolated snapshot; production solution remains iter79. No new optimization iteration.
- NCU2025.1.1,39 passes per kernel, both1024 dtypes; complete SW .ncu-rep plus lineinfo capture and fixed-configuration Dense comparison saved in trajectory/20261014_ncu_iter76/.
- Confirmed P store8-way bank conflicts at iter76 sw_pipeline.cu:154,401408 excessive shared wavefronts. Long-scoreboard hotspots include product-ready mbarrier consumers. HBM not saturated in this warm-cache capture; Tensor elapsed activity~18.8percent.
- Occupancy24.93percent limited independently by shared and register capacity; Dense lower occupancy still runs faster. Original performance measurements remain authoritative; NCU durations are diagnostic.
- Initial lock failure resolved with documented private TMPDIR after checking no concurrent ncu process. Initial NCU-retuned Dense picked the wrong comparison config and was excluded; fixed ids39/21 captured successfully.
- Details: trajectory/20261014_ncu_iter76/REPORT.md. Optimization remains paused by user.

### Iter 80 — Store and consume P in native WGMMA fragment order

- **Hypothesis:** Consecutive TC lanes writing consecutive floats remove the measured eight-way P shared-store conflict.
- **Changes:** Use profiled iter76 BK256 baseline; store P at fragment_element*128+tc_thread; CUDA workers consume physical order and map output through CLayout_64x32. Arithmetic and barriers match iter76.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.024033 ms; mean speedup=0.4634x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012250 | 0.023930 | 0.5119 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010013 | 0.024135 | 0.4149 | 2.37136e-06 | 1 |

- **Analysis:** Standard 1024-only BF16 and FP16 benchmark completed; frozen original-input numerical gate checked by harness. NCU shared-conflict verification and independent stress checks follow this commit.
- **Next:** Confirm P store wavefronts and both shared load/store conflicts with NCU; run fresh-input stress and sanitizers.
- **Archive:** trajectory/20261014_202938_iter-80/

### Iter 81 — XOR-swizzle P rows for conflict-free stores and loads

- **Hypothesis:** Map fragment row bits into the unused shared-bank bits, eliminating eight-way stores while retaining row-major global output.
- **Changes:** Relative to profiled iter76 only P addressing changes: physical=row*32+(col XOR ((row AND 1) OR ((row AND 6)<<2))); both producer and consumer use the mapping. Restore row-major CUDA output traversal after iter80 regression.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.021439 ms; mean speedup=0.5224x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012381 | 0.021216 | 0.5836 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009992 | 0.021661 | 0.4613 | 2.37136e-06 | 1 |

- **Analysis:** Exhaustive layout check proves bijection and 32 distinct banks for every producer warp/fragment and every consumer row. Full standard 1024 BF16/FP16 bench completed with frozen precision gate. Iter80 native fragment traversal passed 72 stress checks but regressed complete-call time to 23.93/24.14 us, so use row swizzle instead. NCU and final stress follow.
- **Next:** Measure actual P store/load wavefronts with NCU and run stress plus shared-memory sanitizers.
- **Archive:** trajectory/20261014_203116_iter-81/

- **Iter81 post-commit verification:** NCU2025.1.1 confirms all16 P STS and42 P LDS per dtype are1-way with0 excessive wavefronts. P store actual/ideal wavefronts=57344/57344, versus iter76=458752/57344 (401408 excessive). Independent72-check1024 launch/Graph stress passes; quick memcheck/racecheck/synccheck pass;3 CPU DAG tests pass. Launch BF16/FP16=21.216/21.661us, tuned Dense=12.381/9.992us; no significant overall speedup over historical iter76=21.245/21.883us established. Full evidence: trajectory/20261014_validation-iter81-p-banks/REPORT.md.

### Iter 82 — Stream Winograd recombination and publish output early

- **Hypothesis:** Schedule P1,P2,P6,P7,P5,P3,P4 so SIMT workers recombine and write completed quadrants while subsequent leaves execute, reducing live P state and epilogue tail.
- **Changes:** Keep iter81 BK256, two operand slots, two XOR-swizzled P slots and 512-thread roles. Advance all barrier phases by execution step, not logical P ID. Consumers retain two state arrays plus current P, release the P slot after reading, and write D11,D22,D12,D21 as soon as available.
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.021406 ms; mean speedup=0.5180x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012194 | 0.021350 | 0.5711 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009978 | 0.021462 | 0.4649 | 2.37136e-06 | 1 |

- **Analysis:** Full standard 1024-only BF16/FP16 benchmark completed with original-input FP64 precision checks and full counts. Independent fresh-input launch/Graph stress, sanitizers and NCU follow the iteration commit; no speedup certification inferred yet.
- **Next:** Compare complete-call latency to iter81; verify streaming synchronization and profile P accesses plus dynamic instruction and register changes.
- **Archive:** trajectory/20261014_212022_iter-82/

- **Iter82 post-commit verification:** Original-input gates and independent72-check1024 launch/Graph stress pass, quick memcheck/racecheck/synccheck pass,3 CPU DAG tests pass. NCU confirms all16 P STS and42 P LDS per dtype remain1-way,0 excessive wavefront. Registers110/thread,stack/local0,shared213120. Standard launch BF16/FP16=21.350/21.462us versus Dense12.194/9.978us. Contemporaneous iter81/iter82 ABBA launch comparison gives1.60%/1.49% lower mean of two CSV medians; Graph0.84%/1.44% lower. Below3% improvement threshold; retain correct streaming candidate, not a performance winner. Tensor activity~19percent unchanged, dynamic instructions~7percent higher. Evidence: trajectory/20261014_validation-iter82-streaming/REPORT.md.

### iter82 production-chain diagnosis (no solution edit)

- Isolated sparse and cumulative timing on1024^3 BF16/FP16. Median TC production loop13.344/13.408us; operand waits2.432/2.464us; prepare+issue8.544/8.576us; explicit wait0 residual0.352/0.352us (sparse cross-check~0.448us).
- Main measured interval is prepare/issue, not the residual full wait. Operand waits are~7x residual wait0. Execution overlaps instruction issue, so prepare/issue is not all software overhead. This revises the earlier priority assumption.
- Baseline SASS identical to production; diagnostic output bitwise equal on3fresh inputs; no spill/occupancy change; aggregate instrumentation event overhead averages0.53%/0.73%. Original-input gates pass on fresh formal Graph benchmark. Production code unchanged.
- Full evidence and reproduction: trajectory/20261014_profile-production-chain/REPORT.md.

### Iter 83 — Reuse canonical GMMA descriptors across K tiles

- **Hypothesis:** CuTe descriptor iteration reduces repeated address-bitfield preparation in the dominant WGMMA issue interval
- **Changes:** Construct A/B descriptor tensors once per matrix term and index K tiles with the official CuTe descriptor iterator; preserve all arithmetic and synchronization
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.020759 ms; mean speedup=0.5328x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012234 | 0.020741 | 0.5898 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009887 | 0.020777 | 0.4759 | 2.37136e-06 | 1 |

- **Analysis:** Full 1024 cubed BF16/FP16 benchmark completed; recorded numerical gates and complete-call timings in this archive. Generated instruction and independent correctness validation follow.
- **Next:** Inspect timings, generated instructions and independent stress validation before retaining the candidate
- **Archive:** trajectory/20261014_215354_iter-83/

### Iter 84 — Restrict GMMA K offsets to the low descriptor word

- **Hypothesis:** Shared addresses below 256 KiB allow eliminating 64-bit descriptor add/carry while preserving upper configuration fields
- **Changes:** Read canonical CuTe descriptor bases and add K layout offsets only to reg32_[0]
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Build or benchmark failed; preserve this attempt before investigating the failure
- **Next:** Inspect compiler output and correct descriptor representation access if needed
- **Archive:** trajectory/20261014_215806_iter-84/

### Iter 85 — Keep upper GMMA descriptor fields invariant

- **Hypothesis:** Low-word-only K offsets avoid the generic 64-bit descriptor carry path
- **Changes:** Fix iter84 descriptor access by indexing the zero-offset CuTe descriptor, then add each K offset to the low word only
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.020929 ms; mean speedup=0.5382x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012506 | 0.020847 | 0.5999 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010012 | 0.021012 | 0.4765 | 2.37136e-06 | 1 |

- **Analysis:** Full 1024 cubed BF16/FP16 benchmark completed; independent performance and generated-code validation follow
- **Next:** Compare with iter82 and iter83, validate precision and synchronization, and retain the fastest validated version
- **Archive:** trajectory/20261014_215844_iter-85/

### Iter83–85 independent validation

Current iter85 passes original-input FP64 gates, independent72-check launch/Graph stress and quick memcheck/racecheck/synccheck. Iter84 build failure is retained and fixed in85. Paired full-call launch improves3.04% BF16/2.13% FP16 versus82; Graph3.47%/1.59%. Versus83, FP16Graph regresses1.20%; gains are small and mode dependent. WGMMA prepare/issue median8.608us->7.776/7.744us; R2UR unchanged, P remains1-way0excess. Dense remains faster and Auto remains Dense. Full evidence: [iter85 report](trajectory/20261014_validation-iter85-descriptors/REPORT.md), [iter83 report](trajectory/20261014_validation-iter83-descriptors/REPORT.md).

### Iter 86 — Expose fixed operand slot addresses to GMMA lowering

- **Hypothesis:** Unrolling the two K blocks exposes static slot offsets and permits descriptor constant folding or uniform reuse
- **Changes:** Explicitly unroll only the TC two-slot K-block loop; preserve arithmetic and all barriers
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.020372 ms; mean speedup=0.5448x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012261 | 0.020378 | 0.6017 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009938 | 0.020367 | 0.4880 | 2.37136e-06 | 1 |

- **Analysis:** Full 1024 cubed BF16/FP16 benchmark completed; inspect SASS and paired complete-call measurements next
- **Next:** Measure whether actual R2UR and descriptor preparation decrease before advancing to staged readiness
- **Archive:** trajectory/20261014_220736_iter-86/

### Iter 87 — Publish high operands before residual operands

- **Hypothesis:** Split high/low TMA readiness lets the high-high WGMMA issue while residual transfers finish
- **Changes:** Separate high and low transaction barriers; issue all high TMA loads before lows; wait for lows after issuing high-high and before correction terms; release slots after the unchanged full WGMMA wait
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019300 ms; mean speedup=0.5764x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012303 | 0.019316 | 0.6369 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009949 | 0.019283 | 0.5160 | 2.37136e-06 | 1 |

- **Analysis:** Full 1024 cubed BF16/FP16 benchmark completed; retain exact accumulation order and phase advancement even for zero-byte low stages. Independent synchronization and comparative validation follow.
- **Next:** Compare against iter86 and measure total high+low waits separately from WGMMA issue; retain only with complete-call benefit and sanitizer passes
- **Archive:** trajectory/20261014_221021_iter-87/

### Iter86–87 final independent validation

Retain iter87 (c9380c3). Fixed slots cut R2UR995968->487424/dtype; staged readiness further changes generated-code reuse to178944. RealWGMMA245760/dtype unchanged. Three-version A/B/C/C/B/A comparison against85: launch6.32%BF16/6.76%FP16 lower; Graph7.43%/8.74% lower. Against86: launch5.61%/5.79%, Graph4.74%/5.02% lower. Operand cumulative waits(high+low)~2.95->2.37us; TCloop~12.4->11.74us. Do not attribute all gains to TMA overlap because generated descriptor instructions changed too.
Original-input FP64 gates, independent72-check stress and final quick memcheck/racecheck/synccheck pass. Formal SW19.316/19.283us versus tunedDense12.303/9.949us; Auto remains Dense. Registers79,local/stack0,shared213120,allPSTS/LDS1-way0excess. Scope1024^3 only. [Final iter87 report](trajectory/20261014_validation-iter87-staged-ready/REPORT.md), [iter86 report](trajectory/20261014_validation-iter86-fixed-slots/REPORT.md). The two approved experiments are complete.

### Iter 88 — Reduce P consumers to five warps while preserving input parallelism

- **Hypothesis:** The producer reaches its final barrier without waiting for recombination; fewer polling consumer warps may reduce scheduling pressure
- **Changes:** Use160 P-consumer threads and adjust chunk distribution/free-barrier arrivals; keep512 input-transform threads and unchanged TC/TMA roles
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.020125 ms; mean speedup=0.5520x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012262 | 0.020168 | 0.6080 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009960 | 0.020081 | 0.4960 | 2.37136e-06 | 1 |

- **Analysis:** Full 1024 cubed BF16/FP16 benchmark completed; inspect performance and resource changes next
- **Next:** Compare complete-call latency against87 and check whether recombination or register pressure becomes limiting
- **Archive:** trajectory/20261014_222749_iter-88/

### Iter 89 — Use eight evenly packed P-consumer warps

- **Hypothesis:** 256 consumer threads divide2048 exactly and keep row-XOR offsets regular across chunks, avoiding tails and reducing address work
- **Changes:** Increase Consumers160->256 while preserving512 transform threads, unchanged TC/TMA paths and original arithmetic
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019577 ms; mean speedup=0.5673x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012241 | 0.019487 | 0.6282 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009958 | 0.019666 | 0.5064 | 2.37136e-06 | 1 |

- **Analysis:** Full 1024 cubed BF16/FP16 benchmark completed; compare against87 and the five-warp failed-performance candidate88
- **Next:** Retain only with complete-call improvement; next investigate vectorized input transformation based on the measured4us phase
- **Archive:** trajectory/20261014_223048_iter-89/

### Iter 90 — Vectorize input transformation IO in adjacent pairs

- **Hypothesis:** The measured4us transform/grid phase can shrink by halving global load/store instruction counts with aligned32bit pairs
- **Changes:** Restore validated iter87 consumer layout; load pairs of input elements with32bit IO, preserve scalar FP32 transformation/rounding, and pack adjacent high/low outputs into32bit stores
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019053 ms; mean speedup=0.5825x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012241 | 0.018909 | 0.6474 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009938 | 0.019196 | 0.5177 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 cubed BF16/FP16 benchmark completed; independently compare output bitwise and profile transform timing plus generated IO widths
- **Next:** Verify original gates, paired full invocation, vectorized SASS and synchronization before retaining
- **Archive:** trajectory/20261014_223240_iter-90/

### Reassessment after iter88–90

Three candidates did not establish a >=3percent full-call gain over87. Five/eight consumer warp allocations regress;32bit input IO halves load/store instructions and reduces measured transform/grid phase~4.0->3.68us, but full-call gain is only~1percent and BF16Graph regresses~1percent. Reread current NCU and paired CSVs; no performance floor is claimed. NVIDIA CUDA programming guide supports naturally aligned8byte global IO; all current input/workspace alignment constraints satisfy it. Next isolate64bit/four-element IO so every original thread handles exactly one input group, preserving full CTA transform parallelism.

### Iter 91 — Transform four adjacent elements with64bit IO

- **Hypothesis:** Four-element aligned IO reduces instruction count further and removes the per-thread transform loop without reducing CTA parallelism
- **Changes:** Widen input/output vectors32->64bits and process four scalar lanes with unchanged FP32 arithmetic and rounding; keep iter87 producer/consumer roles
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019613 ms; mean speedup=0.5703x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012427 | 0.019552 | 0.6356 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009935 | 0.019673 | 0.5050 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 cubed BF16/FP16 benchmark completed after documented stall reassessment; compare against87 and90 and inspect resource changes
- **Next:** Profile transform/grid timing, bitwise equivalence and paired complete-call latency
- **Archive:** trajectory/20261014_223537_iter-91/

### Iter 92 — Combine A chunk transfers with higher-rank TMA maps

- **Hypothesis:** One3D/4D A transfer can replace four64-column commands, reducing TMA issue overhead without changing shared layout
- **Changes:** Restore32bit vectorized transform from90; encode A/S with split-K tensor dimensions and load fullBK tile in oneTMA command per high/low plane; B/T and transaction bytes unchanged
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019093 ms; mean speedup=0.5813x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012228 | 0.019151 | 0.6385 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009975 | 0.019035 | 0.5240 | 2.37136e-06 | 1 |

- **Analysis:** Driver map probe accepts the permuted strides; full1024 cubed BF16/FP16 benchmark completed. Check numerical gates and independent bitwise plus sanitizer validation.
- **Next:** Profile TMA instruction reduction and operand waits, compare against87/90 and retain only after full-call validation
- **Archive:** trajectory/20261014_223916_iter-92/

### Iter 93 — Expose the existing no-write-alias binding contract to NVCC

- **Hypothesis:** Restricted packed/output pointers allow more input-load scheduling freedom across transform stores without changing accepted bindings
- **Changes:** Add restrict qualifiers to kernel input/workspace/output pointers; existing API rejects writable overlap and permits read-only A/B sharing
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019041 ms; mean speedup=0.5835x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012261 | 0.018975 | 0.6462 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009952 | 0.019107 | 0.5209 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 cubed BF16/FP16 benchmark completed; evaluate generated code and complete-call measurements rather than assuming a benefit
- **Next:** Compare against92 and87; inspect whether load ordering changes before trying explicit transform software pipelining
- **Archive:** trajectory/20261014_224418_iter-93/

### Reassessment after iter91–93

No >=3percent all-mode gain over87 is established. Four-element64bit IO regresses despite fewer memory instructions;32bit pairs remain the better transform candidate. Higher-rank A maps reduce TMA instructions110->44/CTA but only slightly reduce total operand waits (~2.34us). Sparse profiling shows most second-K high-ready checks already short (~64ns), so prioritizing their TMA commands is not justified. TMA instead waits~6.6us total for free slots; CUDA consumers wait~6.8us for P, while TC product-free waits~0.384us. Current wait0 tail is~0.99us, larger than before descriptor improvements. Consulted NVIDIA CUTLASS multistage GMMA implementation for outstanding-group retirement and buffer release. Next test two-group K overlap with retirement of slot0 only after wait_group1 proves its reads complete; preserve arithmetic order and do not release slot1 before wait_group0.

### Iter 94 — Overlap the two K-block WGMMA groups with proven slot retirement

- **Hypothesis:** Deferring q0 full wait until q1 high issue can overlap outstanding tensor work and operand readiness while preserving safe reuse
- **Changes:** Start from92; commit q0 without draining, commit q1 high then wait_group1 and release slot0; issue corrections, wait_group0 and release slot1
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.020434 ms; mean speedup=0.9225x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.027670 | 0.020368 | 1.3585 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009972 | 0.020500 | 0.4864 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 cubed BF16/FP16 benchmark completed; numerical gates and independent sanitizer validation determine correctness, and full-call comparison determines retention
- **Next:** Compare against92 and87, inspect WGMMA waits and TMA free-slot effects; keep the fastest independently validated candidate
- **Archive:** trajectory/20261014_224630_iter-94/

### Iter 95 — Prefetch tensor-map descriptors during input transformation

- **Hypothesis:** Four early descriptor prefetches may hide the first-stage descriptor startup within the independent transform phase
- **Changes:** Restore92 and prefetch all four tensor maps with the TMA leader before input transformation and grid synchronization
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019156 ms; mean speedup=0.5804x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012315 | 0.019031 | 0.6471 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009903 | 0.019280 | 0.5136 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 cubed BF16/FP16 benchmark completed; compare against92/87 to avoid attributing unlocked-clock variation to prefetch
- **Next:** Retain only with full-call improvement; finalize the best validated source and archive rejected consumer/vector/pipeline candidates
- **Archive:** trajectory/20261014_224940_iter-95/

### Iter 96 — Restore the validated small-gain iter92 candidate

- **Hypothesis:** The32bit transform plus combined A TMA gives the most consistent ordinary full-call improvement without the mode-specific regressions of alias/prefetch candidates
- **Changes:** Verbatim git checkout of bf34e5a solution; discard reduced consumers,64bit transform,alias hint,two-group delayed retirement and descriptor prefetch
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019069 ms; mean speedup=0.5822x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012252 | 0.018952 | 0.6465 | 7.39459e-06 | 1 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009939 | 0.019187 | 0.5180 | 2.37136e-06 | 1 |

- **Analysis:** Full1024 cubed BF16/FP16 restore benchmark completed. Gains over87 remain small (~1-2percent ordinary launch); do not certify a >=3percent universal win or Dense speedup.
- **Next:** Run final independent stress/sanitizers, freeze best source and archive all profiling plus rejected candidates
- **Archive:** trajectory/20261014_225150_iter-96/

### Final validation — retain iter92 as iter96

- Verbatim restore already committed in iter96; final labeled full benchmark passed original FP64 gates, both dtypes, nonfinite=0.
- 72 independent stress checks, quick memcheck/racecheck/synccheck and 3 DAG tests passed.
- Explored six distinct directions in88–95; retain32bit input IO + combined A TMA. No physical-floor claim. Ordinary-call gains are small (~1–2%), not a universal >=3% win and not faster than Dense.
- Complete comparison and profiling evidence: trajectory/20261014_validation-iter96-supply/REPORT.md.

| bf16 | 18.96224 | 12.26752 | 7.39458724362e-06 | 0.00100255679669 | PASS |
| fp16 | 18.95360 | 9.94816 | 2.37135524364e-06 | 0.000103752760587 | PASS |

### Iter 97 — Seven-product SW without residual compensation

- **Hypothesis:** Removing eight residual products may improve full-call speed at a measurable accuracy cost
- **Changes:** Remove low-part generation, global stores, TMA loads, low waits and eight correction products; preserve high transforms, shared budget and roles
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014410 ms; mean speedup=0.7728x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012312 | 0.014464 | 0.8512 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009970 | 0.014356 | 0.6944 | 0.000754626 | 0 |

- **Analysis:** User-requested performance ablation; original numerical gates remain authoritative. Benchmark exit may indicate numerical failure, not compilation failure; compare both dtype rows and fresh same-session baseline next.
- **Next:** Run interleaved full-call comparisons versus rebuilt iter96 and report accuracy even if rejected
- **Archive:** trajectory/20261015_010804_iter-97/

### Iter97 follow-up and final restore

- A/B/B/A rebuilt96/97 full-call comparison,20warmup,5x100calls,3freshaccuracy inputs; unchanged original gates.

| mode | dtype | compensated us | seven-product us | latency reduction | concurrent Dense us |
|---|---|---:|---:|---:|---:|
| launch | bf16 | 18.92112 | 14.17568 | 25.08% | 12.25952 |
| launch | fp16 | 19.15296 | 14.40080 | 24.81% | 9.95504 |
| graph | bf16 | 18.60288 | 13.56640 | 27.07% | 11.27648 |
| graph | fp16 | 18.81120 | 14.06144 | 25.25% | 9.54912 |

- Seven-product BF16 eF=.00601520 vs gate.00100256; FP16eF=.000754626 vs gate.000103753: both FAIL,nonfinite0. Benchmark exit2 expected.
- Verbatim restored solution from5f5327d and final full benchmark passed both dtypes. Experimental source remains503c7de and trajectory/20261015_010804_iter-97/.
- Report: trajectory/20261015_validation-iter97-no-compensation/REPORT.md. No winner certification for no-compensation candidate.

### User-directed selection — keep iter97

- User explicitly requested retaining the seven-product no-compensation version after measurement. Restored solution verbatim from503c7de, superseding the interim compensated restoration. No implementation edits beyond the exact already-benchmarked source; no redundant benchmark required.
- Current source:7products, ordinarylaunch BF16/FP16 14.17568/14.40080us; Graph13.56640/14.06144us in paired trials. Both original numerical gates FAIL, nonfinite0. Auto remainsDense.
- README and HINTS now identify this as the current experimental source; previous compensated validation is historical.

### Profiling current seven-product version — no solution change

- User requested a fresh bottleneck conclusion. Currentsource still503c7de, selected0d5d74d. BothdtypeNCU39passes each,plusallrole3freshinputdiagnostics.
- Main exposed interval nowoperandready~2.9us,issue2.37us,wait01.41us;TCloop7.94us. Transform+grid2.62us. Tensoractivity~13%,HBM<.4%,Pbankconflict0,regs56nospill. Shared213120stilllimitsoneCTA/SM,unusedlowspace98304B.
- Formal launchBF16/FP16=14.16448/14.31584us,Graph13.85632/14.10080us. OriginalFP64gates remainFAIL; source retained per user instruction.
- Recommendations only: improve slotreuse and dataoverlap, reconsider tile/Kpipeline, reduce transform/gridcost. No optimization implemented. Report trajectory/20261015_profile-iter97-seven-products/REPORT.md.

### Iter 98 — Compact two-stage operand shared storage

- **Hypothesis:** Reclaim unused residual slots without changing seven-product arithmetic or release timing
- **Changes:** Compact operand A/B into two high-only stages,remove unused low barriers; shared213120 to114816bytes
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014399 ms; mean speedup=0.7745x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012324 | 0.014318 | 0.8607 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009964 | 0.014479 | 0.6882 | 0.000754626 | 0 |

- **Analysis:** Expected original numerical failure remains; correctness of scheduling will be checked bitwise against97. Full-call retention requires paired measurement.
- **Next:** Measure paired performance then test deeper prefetch in reclaimed space
- **Archive:** trajectory/20261015_012249_iter-98/

### Iter 99 — Three compact operand stages with immediate retirement

- **Hypothesis:** A third slot lets TMA prefetch farther while TC still releases each K tile immediately after wait0
- **Changes:** Stages3 with modulo phase indexing,163968shared; retain existing BK256,math and wait0 release rhythm
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013506 ms; mean speedup=0.8297x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012299 | 0.013423 | 0.9162 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010098 | 0.013589 | 0.7431 | 0.000754626 | 0 |

- **Analysis:** Original FP64gate expected FAIL in this no-compensation experiment; compare exact output and full-call timing against97/98.
- **Next:** Test fourth stage and compare contemporaneous complete invocations
- **Archive:** trajectory/20261015_012441_iter-99/

### Iter 100 — Four compact operand stages

- **Hypothesis:** Use recovered low-part space for four high-only prefetch stages without delaying slot release
- **Changes:** Stages4,BK256,213120shared; producer free phases and TC ready phases follow four-stage ring
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013169 ms; mean speedup=0.8573x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012515 | 0.013119 | 0.9539 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010055 | 0.013219 | 0.7607 | 0.000754626 | 0 |

- **Analysis:** Seven main products and original gates unchanged. Deeper buffering may add address/phase overhead; full-call comparisons decide retention.
- **Next:** Reassess98-100 using paired timings, generated code and fresh diagnostics before K-block changes
- **Archive:** trajectory/20261015_012528_iter-100/

### Iter 101 — K128 with six operand prefetch stages

- **Hypothesis:** Smaller K chunks and six prefetched tiles may trade shorter individual completion waits for better data readiness
- **Changes:** BK128,Stages6,shared163968; preserve math order and immediate per-tile retirement
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013673 ms; mean speedup=0.9326x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012305 | 0.013526 | 0.9097 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.013205 | 0.013820 | 0.9555 | 0.000754626 | 0 |

- **Analysis:** Both dtype full benchmarks completed; original accuracy failures remain expected. More K stages also increase synchronization frequency.
- **Next:** Compare withK256/4stages; testK512 with split B TMA to reduce completion frequency
- **Archive:** trajectory/20261015_012641_iter-101/

### Iter 102 — K512 one completion group per product

- **Hypothesis:** A full512-wide product tile halves wait0 and barrier frequency while two full-K slots prefetch the next product
- **Changes:** BK512,Stages2,shared213120; split B TMA into two256-wide chunks, keep A combined higher-rank TMA and accumulation order
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013924 ms; mean speedup=0.8022x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012331 | 0.013880 | 0.8884 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010001 | 0.013968 | 0.7160 | 0.000754626 | 0 |

- **Analysis:** Full benchmark completed with original expected no-compensation accuracy failures. Reduced synchronization may be offset by longer individual MMA groups or TMA latency.
- **Next:** Compare K128/K256/K512 variants then optimize front-end transform and grid synchronization
- **Archive:** trajectory/20261015_012733_iter-102/

### Iter 103 — Two outstanding WGMMA groups on four-stage operand ring

- **Hypothesis:** With four prefetched slots, defer q0 completion until q1 commit and release q0 at wait1 to overlap arithmetic without starving TMA
- **Changes:** Basedon100; q0commit only,q1commit wait1 releaseprevious then wait0 releasecurrent; no accumulator reorder
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013214 ms; mean speedup=0.8453x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012321 | 0.013181 | 0.9347 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010013 | 0.013246 | 0.7560 | 0.000754626 | 0 |

- **Analysis:** Original no-compensation precision gates remain unchanged and fail as expected; compare complete-call performance and validate ring safety independently.
- **Next:** Select issue rhythm by paired timings, then test transform vector width and separate-transform launch
- **Archive:** trajectory/20261015_012836_iter-103/

### Iter 104 — Four-element64bit high-only transform IO

- **Hypothesis:** One four-element transform iteration per thread may reduce loop/address overhead versus two packed-pair iterations
- **Changes:** Basedon100; aligned64bit A/B reads and S/T stores,four independent scalar FP32 transforms with unchanged rounding
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012677 ms; mean speedup=0.8815x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012353 | 0.012644 | 0.9770 | 6.42534e+35 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009990 | 0.012710 | 0.7860 | 3.94678 | 0 |

- **Analysis:** Bothdtype benchmarks finished; distinguish input-stage savings from potential register/codegen cost using diagnostics and full-call comparison.
- **Next:** Reassess no-improvement candidates using updated profiling before separate-transform/grid experiment
- **Archive:** trajectory/20261015_013006_iter-104/

### Iter 105 — Correct64bit transform output pointer width

- **Hypothesis:** The64bit IO candidate must preserve every high-part element before its timing can be considered
- **Changes:** Fix104 packed store pointer fromuint32_t to uint64_t; retain64bit input and output indexing
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013184 ms; mean speedup=0.8444x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012310 | 0.013147 | 0.9363 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009948 | 0.013220 | 0.7525 | 0.000754626 | 0 |

- **Analysis:** Iter104 is INVALID: truncated stores caused errors unrelated to accepted seven-product approximation; exclude all104 timings. Corrected105 must match97 output independently.
- **Next:** Bitwise-check candidates,compare105vs100,profile best ring before split-transform experiment
- **Archive:** trajectory/20261015_013122_iter-105/

### Reassessment after101–105

- K128/6slots andK512/2slots are slower than100; two outstanding groups103 ties or regresses100;104 invalid due truncated store,105corrected matches original approximation but is slower than100. No >=3% improvement over100 across these candidates.
- Fresh100NCU and all-role diagnostics completed. Same-invocation BF16/FP16 operandready1.440/1.472us,issue3.648/3.680us,wait0 .512/.512us,TCloop7.072/7.104us; fronttransform+grid2.592us. Compare historical97ready2.880/2.912,issue2.368,wait01.408,loop7.936. Changes affect codegen too; do not attribute entire gain only to prefetch.
- Reviewed NVIDIA Hopper pipeline guidance and CUTLASS TMA/WGMMA examples. Next distinct direction: separate transform kernel to replace cooperative grid synchronization with stream-order dependency; ordinarylaunch overhead may offset Graph benefit. Keep100 as baseline and measure bothmodes.

### Iter 106 — Separate transform kernel instead of cooperative grid barrier

- **Hypothesis:** Replacing in-kernel grid sync with stream-ordered transform and compute kernels may shorten the front phase, especially under Graph
- **Changes:** Basedon100; launch standalone high-only transform then ordinary math kernel,no cooperative grid barrier; preserve transform/accumulation arithmetic and shared ring
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014437 ms; mean speedup=0.7748x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012356 | 0.014228 | 0.8684 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009978 | 0.014645 | 0.6813 | 0.000754626 | 0 |

- **Analysis:** Full two-kernel invocation measured with unchanged trial counts and original accuracy gates. Extra launch cost must be included; do not time compute kernel alone.
- **Next:** Compare complete call versus100; retain only independent bitwise-safe improvement and finalize validated selection
- **Archive:** trajectory/20261015_013336_iter-106/

### Iter 107 — Use grid synchronization to order barrier initialization

- **Hypothesis:** The grid sync already orders all initialization before operand pipeline use, making the preceding CTA barrier redundant
- **Changes:** Basedon100; remove only the initial __syncthreads before input transformation,keep barrier init fence and cooperative grid sync
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013166 ms; mean speedup=0.8478x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012340 | 0.013064 | 0.9446 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009963 | 0.013267 | 0.7509 | 0.000754626 | 0 |

- **Analysis:** Original approximation gates unchanged; compare against100 and validate synchronization with independent bitwise checks plus sanitizers before retaining.
- **Next:** Finish paired comparisons and select the best independently checked seven-product version
- **Archive:** trajectory/20261015_013455_iter-107/

### Reassessment after105–107

- Corrected64bit transform105,split-transform106,andbarrier-removal107 provide no >=3% gain over100 in paired full calls.106 ordinarylaunch regresses~9-10%;107 is a mode-dependent noise-sized tradeoff.
- Re-read best100NCU:dynamicwarp instructions~3.921M vs97~3.204M despite fasterruntime,regs53,no spill,Tensoractivity~14.1%,issueactive~38.5%. Thefour-stage ring changes descriptor/address lowering. Fresh cumulativeissue~3.65us is now larger thanready~1.45us.
- NVIDIA Hopper/CUTLASS pipeline references already consulted this round. Next targeted direction: unroll seven TC product steps so all four operand slots are compile-time addresses, testing whether descriptor lowering can recover fixed-slot issue efficiency without losing deeper prefetch. Do not infer every issue interval is removable software overhead.

### Iter 108 — Specialize all seven TC product steps for fixed four-slot addresses

- **Hypothesis:** Outer product unrolling can reduce runtime slot-address and descriptor work while retaining deeper prefetch and prompt retirement
- **Changes:** Basedon100; explicitly unrollTC step0..6 in addition to twoK chunks, preserve accumulation/recombination order and synchronization
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.015039 ms; mean speedup=0.7450x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012355 | 0.014948 | 0.8265 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010038 | 0.015130 | 0.6634 | 0.000754626 | 0 |

- **Analysis:** Both original dtype gates remain FAIL by design of uncompensated route; numerical agreement with97 and independent synchronization checks determine implementation validity.
- **Next:** Compare108with100/97,retain fastest validated source and complete final sanitizer/profiling evidence
- **Archive:** trajectory/20261015_013630_iter-108/
