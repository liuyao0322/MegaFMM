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
