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
