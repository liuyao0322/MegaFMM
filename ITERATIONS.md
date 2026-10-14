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
