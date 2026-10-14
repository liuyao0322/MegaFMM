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

| 109 | Two Tensor Core warpgroups per CTA | 1.0033 | 0.011174 | experiment |

| 110 | Measure dual-warpgroup seven-product pipeline | 0.8059 | 0.013946 | experiment |

| 111 | Hoist canonical operand descriptors outside product loop | 0.8557 | 0.013164 | experiment |

| 112 | N16 K128 tiles with two CTA residency | 0.0000 | 0.000000 | experiment |

| 113 | Correct small-tile shared budget and measure | 0.6347 | 0.017593 | experiment |

| 114 | Overlap S/T transform with first two original-input products | 0.7973 | 0.013971 | experiment |

| 115 | More CUDA transform workers in overlapping pipeline | 0.8033 | 0.013890 | experiment |

| 116 | Continuous two-accumulator WGMMA pipeline across product boundaries | 0.6124 | 0.018222 | experiment |

| 117 | Register recombination and vector output from Tensor Core warpgroup | 0.4363 | 0.025648 | experiment |

| 118 | Fence accumulator operands for register recombination | 0.4359 | 0.025522 | experiment |

| 119 | Specialize register recombination by unrolling leaf products | 0.6906 | 0.016136 | experiment |

| 120 | TC-owned shared recombination state | 0.5416 | 0.020784 | experiment |

| 121 | Uniform parameter descriptors and fixed slot product pairs | 0.4638 | 0.024103 | experiment |

| 122 | Include driver-reserved shared region in descriptor bases | 0.4609 | 0.024365 | experiment |

| 123 | Uniform descriptors with compact single-product loop | 0.8408 | 0.013301 | experiment |

| 124 | Distributed wide leaf GEMMs and global product recombination | 0.0000 | 0.000000 | experiment |

| 125 | Bound registers for two-block distributed leaf residency | 0.7146 | 0.015646 | experiment |

| 126 | N256 leaf tiles with one product per resident CTA | 0.6655 | 0.016752 | experiment |

| 127 | Direct vector operand producers without global S/T or grid barriers | 0.0000 | 0.000000 | experiment |

| 128 | Correct direct-producer recombination extraction | 0.0941 | 0.118200 | experiment |

| 129 | TMA-leader publication while TC and CUDA roles progress independently | 0.8641 | 0.012951 | experiment |

| 130 | 608 workers for asynchronous S/T publication | 0.8952 | 0.012499 | experiment |

| 131 | Reduce recombination consumers to four warps | 0.9010 | 0.012435 | experiment |

| 132 | Vectorized shared product exchange and output | 0.8942 | 0.012557 | experiment |

| 133 | Pair-preserving swizzle for64-bit P exchange | 0.8902 | 0.012504 | experiment |

| 134 | Warp-leader product barrier publication and release | 0.8916 | 0.012520 | experiment |

| 135 | Two recombination consumer warps | 0.7529 | 0.014806 | experiment |

| 136 | Partially unroll TC product loop by two | 0.8900 | 0.012540 | experiment |

| 137 | Bind static TMA address metadata during explicit tuning | 0.8984 | 0.012450 | experiment |

| 138 | Combine asynchronous transform publication with four consumers | 0.9186 | 0.012177 | experiment |

| 139 | Simplify four-consumer shared and output address generation | 0.9102 | 0.012211 | experiment |

| 140 | More transform workers with balanced per-CTA ownership | 0.9646 | 0.011552 | experiment |

| 141 | 1024-thread balanced asynchronous transform CTA | 0.9441 | 0.011814 | experiment |

| 142 | Eight K128 operand slots on the balanced asynchronous pipeline | 0.9104 | 0.012285 | experiment |

| 143 | Two outstanding WGMMA groups with eight K128 slots | 0.9153 | 0.012092 | experiment |

| 144 | Statically validated GMMA start fields and partial product unrolling | 0.9607 | 0.011595 | experiment |

| 145 | K512 two-slot asynchronous input publication | 0.8824 | 0.012664 | experiment |

| 146 | Two-CTA DSM cluster with split seven-product ownership | 0.7066 | 0.015789 | experiment |

| 147 | Clustered products with asynchronous transform publication | 0.7496 | 0.014837 | experiment |

| 148 | Balanced four-element64bit input transformation IO | 0.9609 | 0.011656 | experiment |

| 149 | Return independently after final role work | 0.9775 | 0.011387 | experiment |

| 150 | Multicast A across adjacent output CTAs | 0.7559 | 0.014813 | experiment |

| 151 | Local shared transforms and seven persistent register accumulators | 0.4998 | 0.022367 | experiment |

| 152 | Position-independent shared swizzle indexing | 0.5016 | 0.022383 | experiment |

| 153 | Correct local P3 P4 accumulator ownership | 0.4981 | 0.022374 | experiment |

| 154 | Trade transform warps for local-fusion register capacity | 0.4319 | 0.025897 | experiment |

| 155 | Compact uncompensated operand planes into4MiB workspace | 0.9687 | 0.011534 | experiment |

| 156 | Explicit paired16bit conversion for transformed operands | 0.9620 | 0.011586 | experiment |

| 157 | One-shot release reduction publication counter | 0.9667 | 0.011551 | experiment |

| 158 | Two-accumulator interleaved K issue with explicit retirement | 0.6338 | 0.017603 | experiment |

| 159 | Reduce CTA threads for dual-accumulator register capacity | 0.8036 | 0.013897 | experiment |

| 160 | Warp-level MMA sync with LDSM operand fragments | 0.3821 | 0.030048 | experiment |

| 161 | Bound warp-MMA K fragment lifetime | 0.3814 | 0.029243 | experiment |

| 162 | Compact workspace with independent role returns | 0.9706 | 0.011437 | experiment |

| 163 | Eight consumer warps with constant eight-row chunk offsets | 0.9714 | 0.011482 | experiment |

| 164 | Choose P consumers from shorter transform workers | 0.9745 | 0.011466 | experiment |

| 165 | Four independent TMA slot owners | 0.8825 | 0.012623 | experiment |

| 166 | Static sequential product pair helpers | 1.2358 | 0.013534 | experiment |

| 167 | Three product slots | 0.9635 | 0.011551 | experiment |

| 168 | Barrier test-wait fast path | 0.9697 | 0.011486 | experiment |

| 169 | Warp-aggregated pack publication | 0.9762 | 0.011403 | experiment |

| 170 | Immutable tensor-map geometry templates | 0.9701 | 0.011466 | experiment |

| 171 | Two resident CTA N16 with current asynchronous frontend | 0.0000 | 0.000000 | experiment |

| 172 | N16 shared budget correction | 0.0000 | 0.000000 | experiment |

| 173 | Measure corrected N16 asynchronous two-CTA pipeline | 0.6760 | 0.016503 | experiment |

| 174 | Short barrier suspension hint | 0.9717 | 0.011429 | experiment |

| 175 | Single operand-release leader | 0.9805 | 0.011448 | experiment |

| 176 | Global U2 U3 split-product pair | 0.7316 | 0.015313 | experiment |

| 177 | Three operand slots in global split-product pair | 0.7400 | 0.014998 | experiment |

| 178 | Hide global pair exchange behind fourth consumer product | 0.6285 | 0.017677 | experiment |

| 179 | Single PTX block for sixteen WGMMA instructions | 0.9729 | 0.011512 | experiment |

| 180 | PTX descriptor low-word arithmetic | 0.9754 | 0.011437 | experiment |

| 181 | Template-constant descriptors via uniform slot dispatch | 0.9729 | 0.011454 | experiment |

| 182 | Fence only before first K group | 0.9741 | 0.011436 | experiment |

| 183 | Independent A and B TMA producer warps | 0.9600 | 0.011568 | experiment |

| 184 | Role-specific initialization and pack named barriers | 0.9564 | 0.011607 | experiment |

| 185 | TMA L2 promotion256B | 0.9161 | 0.012146 | experiment |

| 186 | One leaf product per CTA with fused last-CTA recombination | 0.3470 | 0.032147 | experiment |

| 187 | 384 threads for parallel leaf register headroom | 0.3521 | 0.031880 | experiment |

| 188 | Coalesced row-major last-CTA recombination | 0.5224 | 0.021359 | experiment |

| 189 | Reuse waiting TC warps for parallel-leaf input transformation | 0.5797 | 0.019352 | experiment |

| 190 | Direct atomic leaf recombination | 0.3647 | 0.030709 | experiment |

| 191 | TMA tensor reduction of shared leaf tiles | 0.0000 | 0.000000 | experiment |

| 192 | Aligned TMA tensor reduction | 0.7058 | 0.015783 | experiment |

| 193 | Reuse retired operands for full TMA reduction tiles | 0.7107 | 0.015616 | experiment |

| 194 | Vectorize P exchange and fixed-offset recombination | 0.9725 | 0.011436 | experiment |

| 195 | Single CTA N64 with three operand slots | 0.0000 | 0.000000 | experiment |

| 196 | Correct and measure N64 single-P pipeline | 0.7840 | 0.014195 | experiment |

| 197 | Two tensor warpgroups with asynchronous480-worker pack | 0.9221 | 0.012164 | experiment |

| 198 | Dual tensor warpgroups with608 transform workers | 0.9651 | 0.011685 | experiment |

| 199 | Cluster-local seven-leaf DSM combination | 0.0000 | 0.000000 | experiment |

| 200 | Four-WG leaf CTAs and explicit cluster capacity check | 0.0000 | 0.000000 | experiment |

| 201 | Measure fifteen persistent eight-CTA leaf clusters | 0.3261 | 0.034099 | experiment |

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

### Final selection — restore100 after98–108 experiments

- Selected100(76b4d09),compact4slotBK256 with timely per-K release; verbatimrestore and finalfullbenchmark completed. User-requested7products retained; originalFP64gates FAILboth,nonfinite0.
- Finalpaired97/100/108:

| mode | dtype |97us|100us| reduction | concurrentDenseus |
|---|---|---:|---:|---:|---:|
| launch | bf16 | 14.23776 | 13.05968 | 8.27% | 12.28576 |
| launch | fp16 | 14.38592 | 13.24832 | 7.91% | 9.97968 |
| graph | bf16 | 13.56512 | 12.27312 | 9.52% | 11.30288 |
| graph | fp16 | 14.06352 | 12.53360 | 10.88% | 9.52576 |

- 72bitwise-vs97checks,270diagnostictimingrowsbitexact,all3quicksanitizers0errors/hazards/warnings,DAG3tests PASS. These do not certify original numerical acceptance.
- Shared space recovered then reused for4prefetchslots,totalstill213120B. K128/K512,twoMMAgroups,64bittransform,splittransform,initialbarrierremoval,fullTCunroll rejected;104invalid andexcluded. Fronttransform/gridno reliableimprovement.
- Report:trajectory/20261015_validation-seven-prefetch/REPORT.md. No physical-floor claim.

### Iter 109 — Two Tensor Core warpgroups per CTA

- **Hypothesis:** Overlap independent leaf products to shorten the serial seven-product critical path
- **Changes:** Assign two 128-thread tensor warpgroups alternating products, three shared product slots and 224 CUDA consumers; four operand slots unchanged; shared221312B
- **Bench:** Compiled=True; frozen numerical gate=PASS. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011174 ms; mean speedup=1.0033x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | dense_cublaslt v0 | launch | 0.012455 | 0.012428 | 1.0021 | 8.52266e-07 | 1 |
| 1024,1024,1024 | fp16 | dense_cublaslt v0 | launch | 0.009965 | 0.009920 | 1.0045 | 1.25092e-06 | 1 |

- **Analysis:** Full benchmark recorded; original FP64 gates remain unchanged. Inspect complete-call measurements and independent numerical behavior after commit.
- **Next:** Compare with iter100 and validate pipeline synchronization before retention
- **Archive:** trajectory/20261015_014634_iter-109/

### Iter 110 — Measure dual-warpgroup seven-product pipeline

- **Hypothesis:** Two independent tensor warpgroups overlap leaf execution
- **Changes:** Accurate dual-warpgroup configuration metadata; explicitly select SW benchmark path; iter109 accidentally measured Dense and is invalid as kernel performance evidence
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013946 ms; mean speedup=0.8059x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012408 | 0.013854 | 0.8956 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010054 | 0.014038 | 0.7162 | 0.000754626 | 0 |

- **Analysis:** Full SW benchmark and unchanged FP64 gates recorded. Compare candidate timing and numerical errors with100 after commit.
- **Next:** Evaluate dual-warpgroup latency and tune operand supply if justified
- **Archive:** trajectory/20261015_014728_iter-110/

### Iter 111 — Hoist canonical operand descriptors outside product loop

- **Hypothesis:** Invariant descriptor construction plus low-word slot offsets reduce issue overhead
- **Changes:** Build slot-zero CuTe GMMA descriptors once per tensor warpgroup and add aligned slot offsets before K issue; restore single-warpgroup iter100 architecture
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013164 ms; mean speedup=0.8557x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012516 | 0.013083 | 0.9567 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009997 | 0.013245 | 0.7548 | 0.000754626 | 0 |

- **Analysis:** Complete SW benchmark archived with unchanged original FP64 gates; inspect measurements after commit.
- **Next:** Compare latency and generated instructions, then explore CTA concurrency
- **Archive:** trajectory/20261015_014859_iter-111/

### Iter 112 — N16 K128 tiles with two CTA residency

- **Hypothesis:** 256 smaller CTAs and two resident blocks per SM hide pipeline latency
- **Changes:** 64x16 output tile, K128 four operand slots, 8-register accumulators, SW32 B layout and adjusted conflict-free P permutation; 90368B shared
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Full SW benchmark archived; unchanged original FP64 gates. Need inspect compile/runtime and compare with100.
- **Next:** Assess achieved concurrency and full-call latency before choosing architecture
- **Archive:** trajectory/20261015_014955_iter-112/

### Iter 113 — Correct small-tile shared budget and measure

- **Hypothesis:** Two resident smaller CTAs hide operand and issue latency
- **Changes:** Correct shared size assertion to90240B for N16 K128 four slots;112 failed build at assertion and has no timing evidence
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.017593 ms; mean speedup=0.6347x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012340 | 0.017434 | 0.7078 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009969 | 0.017751 | 0.5616 | 0.000754626 | 0 |

- **Analysis:** Full benchmark archived with original FP64 gates unchanged; inspect results after commit.
- **Next:** Evaluate CTA concurrency, then overlap original-input P1/P2 with input transformation
- **Archive:** trajectory/20261015_015025_iter-113/

### Iter 114 — Overlap S/T transform with first two original-input products

- **Hypothesis:** P1/P2 can execute before S/T publication, hiding transform and grid synchronization behind useful tensor work
- **Changes:** Split cooperative pipeline into P1/P2 and remaining five products; CUDA consumers transform inputs before consuming P1/P2, then all roles grid-sync once; keep slot phase continuity and arithmetic
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013971 ms; mean speedup=0.7973x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012260 | 0.013852 | 0.8850 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009996 | 0.014090 | 0.7095 | 0.000754626 | 0 |

- **Analysis:** Full benchmark archived; original FP64 gates unchanged. Need independent launch/Graph stress before retaining synchronization change.
- **Next:** Assess overlap gain and tune remaining role allocation and pipeline issue
- **Archive:** trajectory/20261015_015132_iter-114/

### Iter 115 — More CUDA transform workers in overlapping pipeline

- **Hypothesis:** 608 transform workers per CTA shorten the pack phase while TC executes P1/P2
- **Changes:** Increase CTA from512 to768threads; consumers and transform workers352->608 with matching barrier counts and indexing
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013890 ms; mean speedup=0.8033x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012330 | 0.013761 | 0.8960 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009962 | 0.014019 | 0.7106 | 0.000754626 | 0 |

- **Analysis:** Full SW measurements recorded, unchanged original accuracy gates;114 uses53regs and zero spills so resource headroom permits this experiment.
- **Next:** Compare complete-call latency and reprofile if no gain
- **Archive:** trajectory/20261015_015248_iter-115/

### Iter 116 — Continuous two-accumulator WGMMA pipeline across product boundaries

- **Hypothesis:** Keep one WGMMA group pending across leaf boundaries rather than draining after each P
- **Changes:** Two register accumulator sets; issue next product before publishing previous product; wait1 retires immediately previous operand slot, final wait0 drains last product
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.018222 ms; mean speedup=0.6124x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012303 | 0.018166 | 0.6773 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010009 | 0.018278 | 0.5476 | 0.000754626 | 0 |

- **Analysis:** Full SW benchmark archived with unchanged FP64 gates; independent synchronization validation required if retained.
- **Next:** Compare performance and error with100, then consider direct register recombination
- **Archive:** trajectory/20261015_015523_iter-116/

### Iter 117 — Register recombination and vector output from Tensor Core warpgroup

- **Hypothesis:** Eliminate shared product exchange and CUDA consumer polling by recombining register fragments directly
- **Changes:** Remove P shared buffers and barriers; TC retains first/second fragments, preserves FP32 arithmetic order and writes adjacent output pairs as float2; shared196736B
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.025648 ms; mean speedup=0.4363x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012424 | 0.025748 | 0.4825 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009965 | 0.025547 | 0.3901 | 0.000754626 | 0 |

- **Analysis:** Complete SW benchmark archived, original FP64 gates unchanged. Validate bitwise behavior and register resources before retaining.
- **Next:** Assess direct recombination and tune issue and front-end for this architecture
- **Archive:** trajectory/20261015_015635_iter-117/

### Iter 118 — Fence accumulator operands for register recombination

- **Hypothesis:** Explicit compiler operand fences prevent ptxas C7515 serialization seen in117
- **Changes:** Apply warpgroup_fence_operand to all accumulator registers before WGMMA fence and after wait0
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.025522 ms; mean speedup=0.4359x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012343 | 0.025493 | 0.4842 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009906 | 0.025551 | 0.3877 | 0.000754626 | 0 |

- **Analysis:** 117 emitted C7515 and ran25.5us; complete118 SW benchmark archived with unchanged FP64 gates. Inspect warning elimination and timing after commit.
- **Next:** Validate corrected compiler scheduling and compare with100
- **Archive:** trajectory/20261015_015739_iter-118/

### Iter 119 — Specialize register recombination by unrolling leaf products

- **Hypothesis:** Remove loop-carried accumulator aliases that cause ptxas to serialize every WGMMA
- **Changes:** Unroll seven TC product steps so register recombination branches and accumulator lifetimes are static; keep operand fences from118
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.016136 ms; mean speedup=0.6906x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012328 | 0.016290 | 0.7568 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009980 | 0.015982 | 0.6244 | 0.000754626 | 0 |

- **Analysis:** 118 fences alone did not remove C7515; SASS shows wait0 after every HGMMA despite no spills. Full119 benchmark archived for warning/timing inspection.
- **Next:** Check whether specialization restores asynchronous WGMMA and evaluate full-call gain
- **Archive:** trajectory/20261015_015915_iter-119/

### Iter 120 — TC-owned shared recombination state

- **Hypothesis:** Keep compact product loop and avoid inter-warpgroup P exchange without register aliasing
- **Changes:** Replace carried first/second register arrays with TC-owned swizzled shared state; direct float2 output; no product consumer barriers
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.020784 ms; mean speedup=0.5416x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012508 | 0.020787 | 0.6017 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010005 | 0.020780 | 0.4815 | 0.000754626 | 0 |

- **Analysis:** 119 unrolling removed C7515 but still16us; full120 SW benchmark archived with unchanged precision gates.
- **Next:** Check scheduling diagnostics and pursue uniform descriptor issue if needed
- **Archive:** trajectory/20261015_020022_iter-120/

### Iter 121 — Uniform parameter descriptors and fixed slot product pairs

- **Hypothesis:** Host static-layout descriptors loaded from grid-constant parameters remove repeated lane-register descriptor construction
- **Changes:** Derive shared base from cudaFuncGetAttributes, encode canonical A/B descriptors at setup, pass four descriptor pairs by grid-constant value, specialize TC even/odd products to slots0/1 and2/3
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.024103 ms; mean speedup=0.4638x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012336 | 0.024061 | 0.5127 | 1.64825e+36 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010016 | 0.024145 | 0.4148 | 639.111 | 0 |

- **Analysis:** Full SW benchmark archived; exact numerical and SASS uniformization checks needed. Static setup uses no input contents, all transforms and arithmetic remain timed per invocation.
- **Next:** Inspect descriptor correctness, latency and actual R2UR elimination
- **Archive:** trajectory/20261015_020220_iter-121/

### Iter 122 — Include driver-reserved shared region in descriptor bases

- **Hypothesis:** Correct descriptors must include cudaDeviceProp reservedSharedMemPerBlock
- **Changes:** Add queried1024B driver shared reservation to reported static shared base;121 was invalid from wrong start fields and all its timings excluded
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.024365 ms; mean speedup=0.4609x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012431 | 0.024354 | 0.5104 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010029 | 0.024377 | 0.4114 | 0.000754626 | 0 |

- **Analysis:** Diagnostic shows actual start0x40 versus supplied0; attributes static0 reserved1024. Complete122 benchmark archived with original FP64 gates unchanged.
- **Next:** Check corrected errors and instruction savings, then consider wider distributed leaf tiles
- **Archive:** trajectory/20261015_020448_iter-122/

### Iter 123 — Uniform descriptors with compact single-product loop

- **Hypothesis:** Remove paired-body register pressure and stack traffic introduced by121/122
- **Changes:** Restore iter100 TC loop topology while loading per-slot descriptors from grid-constant parameters; corrected driver reservation retained
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013301 ms; mean speedup=0.8408x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012333 | 0.013209 | 0.9337 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010016 | 0.013393 | 0.7478 | 0.000754626 | 0 |

- **Analysis:** 122 numerical error restored to baseline but128regs and552B stack with STL/LDL explain regression; full123 benchmark archived.
- **Next:** Assess compact descriptor path, then widen and distribute leaf GEMM tiles
- **Archive:** trajectory/20261015_020626_iter-123/

### Iter 124 — Distributed wide leaf GEMMs and global product recombination

- **Hypothesis:** 64x128 tiles reduce repeated A reads and WGMMA issue count while224 CTAs permit two-block residency
- **Changes:** Assign one P tile per CTA, K64 four slots,256threads,98304B operand data, second cooperative grid barrier before vector global recombination; workspace15MiB
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Benchmark attempt archived; original gates unchanged. Inspect any build/runtime failure before interpreting performance.
- **Next:** Debug distributed architecture and compare complete-call latency after correctness checks
- **Archive:** trajectory/20261015_020828_iter-124/

### Iter 125 — Bound registers for two-block distributed leaf residency

- **Hypothesis:** Two-block launch bounds permit224-CTA cooperative grid
- **Changes:** Set minimum two blocks per SM in launch bounds;124 compiled but rejected at setup because168regs allowed onlyoneblock
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.015646 ms; mean speedup=0.7146x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_distributed_64x128x64 v0 | launch | 0.012387 | 0.015708 | 0.7886 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_distributed_64x128x64 v0 | launch | 0.009983 | 0.015584 | 0.6406 | 0.000754626 | 0 |

- **Analysis:** Full125 benchmark archived with unchanged accuracy gate; inspect potential spills and output validity before performance conclusions.
- **Next:** Tune thread count to preserve two-block residency without spilling if required
- **Archive:** trajectory/20261015_020916_iter-125/

### Iter 126 — N256 leaf tiles with one product per resident CTA

- **Hypothesis:** 112 wide CTAs use all SMs while further reducing WGMMA issue and operand duplication
- **Changes:** 64x256x64, three operand slots,128 accumulator registers,256threads,112residentCTAs,shared123008B; global products and second grid recombination
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.016752 ms; mean speedup=0.6655x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_distributed_64x256x64 v0 | launch | 0.012246 | 0.016682 | 0.7341 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_distributed_64x256x64 v0 | launch | 0.010040 | 0.016822 | 0.5969 | 0.000754626 | 0 |

- **Analysis:** 125 is numerically equivalent but15.7us and16B stack; complete126 benchmark archived with unchanged gates.
- **Next:** Measure wide-tile critical phases and determine whether global synchronization costs dominate
- **Archive:** trajectory/20261015_021021_iter-126/

### Iter 127 — Direct vector operand producers without global S/T or grid barriers

- **Hypothesis:** Generate high operands directly into shared and trade duplicated transforms for removal of global pack and cooperative synchronization
- **Changes:** Two ordinary kernels:128 CUDA producer threads and128 TC threads per leaf CTA,64x128x64 four slots, FP32 transforms with original rounding; second vector recombination kernel,8MiB workspace
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Full127 benchmark archived; check original error level and resource scheduling before interpreting latency.
- **Next:** Profile operand generation versus tensor compute and compare two-kernel full-call overhead
- **Archive:** trajectory/20261015_021256_iter-127/

### Iter 128 — Correct direct-producer recombination extraction

- **Hypothesis:** Direct shared operand generation may offset removal of global transform and cooperative barriers
- **Changes:** Fix127 generator which copied transform/mainloop into recombination kernel; recombination now only reads seven global products and writes four output quadrants
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.118200 ms; mean speedup=0.0941x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_direct_64x128x64 v0 | launch | 0.012337 | 0.115803 | 0.1065 | 1.37006 | 0 |
| 1024,1024,1024 | fp16 | sw_direct_64x128x64 v0 | launch | 0.009858 | 0.120598 | 0.0817 | 1.37005 | 0 |

- **Analysis:** 127 failed compile and has no timings; full128 benchmark archived with original numerical gates unchanged.
- **Next:** Assess full-call cost of direct producers and profile retained baseline if it regresses
- **Archive:** trajectory/20261015_021356_iter-128/

### Iter 129 — TMA-leader publication while TC and CUDA roles progress independently

- **Hypothesis:** Overlap P1/P2 with S/T generation without draining TC at an all-thread grid barrier
- **Changes:** CUDA workers transform and arrive shared pack barrier; TMA leader issues first four K tiles then joins per-invocation driver-workspace release/acquire barrier; TC and consumers do not join grid barrier; CUDA12.4 internal ABI dependency explicit
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012951 ms; mean speedup=0.8641x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012358 | 0.012890 | 0.9587 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010013 | 0.013013 | 0.7695 | 0.000754626 | 0 |

- **Analysis:** Full benchmark archived with original FP64 gates. Independent fresh-input launch/Graph/concurrency and sanitizer validation required before retention; no persistent computed data or host counter.
- **Next:** Evaluate complete-call improvement and tune transform worker allocation if promising
- **Archive:** trajectory/20261015_021723_iter-129/

### Iter 130 — 608 workers for asynchronous S/T publication

- **Hypothesis:** Two transform iterations per worker reduce publication delay while TC computes P1/P2
- **Changes:** Increase asynchronous129 CTA512->768threads, CUDA workers352->608 with matching pack and product barriers
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012499 ms; mean speedup=0.8952x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012420 | 0.012435 | 0.9988 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009944 | 0.012563 | 0.7915 | 0.000754626 | 0 |

- **Analysis:** 129 paired launch improves~2% but Graph is flat; NCU53regs,3.84M instructions,Tensor14.7–14.9%. Full130 benchmark archived.
- **Next:** Assess transform scaling and test consumer or product vectorization independently
- **Archive:** trajectory/20261015_021918_iter-130/

### Iter 131 — Reduce recombination consumers to four warps

- **Hypothesis:** Fewer polling CUDA warps reduce scheduler pressure on the TC issue chain
- **Changes:** From100 retain all512 transform threads but use128 product consumers; adjust P-free counts and16chunks per consumer; other workers idle after pack
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012435 ms; mean speedup=0.9010x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012390 | 0.012384 | 1.0004 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010007 | 0.012485 | 0.8015 | 0.000754626 | 0 |

- **Analysis:** 130 launch12.435/12.563us with original error level, promising versus100; full131 benchmark archived independently to isolate consumer allocation.
- **Next:** Inspect consumer timing and test vector product exchange
- **Archive:** trajectory/20261015_022028_iter-131/

### Iter 132 — Vectorized shared product exchange and output

- **Hypothesis:** Float2 exchange halves shared and output instruction count while preserving the conflict-free permutation
- **Changes:** From131 use float2 P stores/loads with row-parity pair swapping,8vector chunks per128CUDAconsumers, vector output and exact FP32 lane arithmetic
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012557 ms; mean speedup=0.8942x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012448 | 0.012484 | 0.9971 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009994 | 0.012630 | 0.7913 | 0.000754626 | 0 |

- **Analysis:** 131 launch12.384/12.485us,130 paired12.321/12.526us with Graph11.934/12.091; full132 benchmark archived, original gates unchanged.
- **Next:** Check numerical equivalence, latency and shared bank metrics if retained
- **Archive:** trajectory/20261015_022158_iter-132/

### Iter 133 — Pair-preserving swizzle for64-bit P exchange

- **Hypothesis:** A float2-specific shared permutation avoids row-parity pair swap instructions
- **Changes:** Map row low bits to column bits3/4 for64-bit accesses, keep adjacent fragment values in natural order, remove per-pair swaps
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012504 ms; mean speedup=0.8902x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012260 | 0.012428 | 0.9865 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009988 | 0.012580 | 0.7940 | 0.000754626 | 0 |

- **Analysis:** 132 vectorization with scalar swizzle was slightly slower; full133 benchmark archived, original FP64 gates unchanged. Need NCU excessive-wavefront verification.
- **Next:** Assess bank metrics and complete-call comparison, then tune consumer count or architecture
- **Archive:** trajectory/20261015_022317_iter-133/

### Iter 134 — Warp-leader product barrier publication and release

- **Hypothesis:** Reduce product barrier atomic arrivals128->4 while warp synchronization preserves store/load ordering
- **Changes:** From scalar131 use four-arrival P-ready/P-free barriers; sync each participating warp before its lane0 arrives; operand slots unchanged
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012520 ms; mean speedup=0.8916x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012348 | 0.012476 | 0.9897 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009971 | 0.012565 | 0.7935 | 0.000754626 | 0 |

- **Analysis:** 133 vector swizzle preserves errors but no gain; full134 benchmark archived. New synchronization requires stress/sanitizers and fresh launch/Graph checks before retention.
- **Next:** Compare complete-call timings and verify shared publication ordering
- **Archive:** trajectory/20261015_022513_iter-134/

### Iter 135 — Two recombination consumer warps

- **Hypothesis:** Further reduce CUDA polling warps while retaining all512 transform threads
- **Changes:** From131 consumers128->64,32scalar chunks each and matching64-count P-free barriers; no numerical or operand changes
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014806 ms; mean speedup=0.7529x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012300 | 0.014762 | 0.8332 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009988 | 0.014850 | 0.6725 | 0.000754626 | 0 |

- **Analysis:** 134 warp-leader product barriers did not improve first-run latency. Full135 benchmark archived for consumer bottleneck and resource assessment.
- **Next:** Compare two versus four consumers and test partial TC product unrolling
- **Archive:** trajectory/20261015_022616_iter-135/

### Iter 136 — Partially unroll TC product loop by two

- **Hypothesis:** Specialize repeating four-slot parity without fully unrolling seven products
- **Changes:** Add pragma unroll2 to TC outer product loop on131; all arithmetic, warp allocation and shared synchronization unchanged
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012540 ms; mean speedup=0.8900x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012315 | 0.012486 | 0.9863 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009995 | 0.012594 | 0.7937 | 0.000754626 | 0 |

- **Analysis:** 131 NCU2.700M instructions,110regs,Tensor~15%; two-consumer135 regressed14.8us. Full136 benchmark archived for partial-unroll/code-size assessment.
- **Next:** Compare with131 and collect same-invocation phase timing to guide remaining work
- **Archive:** trajectory/20261015_022816_iter-136/

### Iter 137 — Bind static TMA address metadata during explicit tuning

- **Hypothesis:** Reuse descriptor metadata for identical current pointers to reduce host launch gaps without caching input values
- **Changes:** Explicit tune prepares four tensor maps; const run compares allA/B/workspace pointers and re-encodes on any mismatch; no run allocation, GPU synchronization or mutable cache
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012450 ms; mean speedup=0.8984x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012328 | 0.012400 | 0.9942 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010032 | 0.012499 | 0.8027 | 0.000754626 | 0 |

- **Analysis:** 131 independent72bitwise checks and memcheck/racecheck/synccheck all pass. Full137 benchmark archived; Graph expected largely unchanged because device work is identical.
- **Next:** Validate bound and changed-pointer paths independently and compare full launch/Graph latency
- **Archive:** trajectory/20261015_023111_iter-137/

### Iter 138 — Combine asynchronous transform publication with four consumers

- **Hypothesis:** Overlap original-input products and retain low consumer polling pressure simultaneously
- **Changes:** From129 keep352 transform workers and TMA-leader driver-workspace barrier, but only128 CUDA consumers with16chunks and matching P-free count
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012177 ms; mean speedup=0.9186x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012357 | 0.012054 | 1.0252 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009988 | 0.012300 | 0.8121 | 0.000754626 | 0 |

- **Analysis:** 137 address-metadata reuse has no material timing gain. Full138 benchmark archived; synchronization changes require independent stress before retention.
- **Next:** Assess combined full-call gain and simplify consumer addresses to reduce register pressure
- **Archive:** trajectory/20261015_023244_iter-138/

### Iter 139 — Simplify four-consumer shared and output address generation

- **Hypothesis:** Fixed chunk offsets and unsigned row/column arithmetic reduce consumer registers and issue instructions
- **Changes:** Explicit16valid chunks per128consumers, two precomputed P bases for even/odd chunks, one global output base with constant offsets; preserve early slot release and async138 publication
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012211 ms; mean speedup=0.9102x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012254 | 0.012140 | 1.0093 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009962 | 0.012282 | 0.8111 | 0.000754626 | 0 |

- **Analysis:** 138 first launch12.054/12.300us, BF16 narrowly ahead of current Dense but not certified; full139 benchmark archived with original gates unchanged.
- **Next:** Inspect register reduction and independently stress retained asynchronous candidate
- **Archive:** trajectory/20261015_023425_iter-139/

### Iter 140 — More transform workers with balanced per-CTA ownership

- **Hypothesis:** 608workers hide input preparation while fixed2048elements per CTA avoid global-barrier stragglers
- **Changes:** From139 increase CTA512->768,352->608 transform workers but retain128consumers; each CTA transforms exactly2048planeelements; shared pack count608
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011552 ms; mean speedup=0.9646x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012259 | 0.011451 | 1.0705 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010006 | 0.011654 | 0.8586 | 0.000754626 | 0 |

- **Analysis:** 139 simplified addresses reduce110->64regs with zero stack, latency~12.14/12.28us. Full140 benchmark archived with unchanged accuracy gates.
- **Next:** Measure balanced worker scaling and validate retained asynchronous pipeline
- **Archive:** trajectory/20261015_023549_iter-140/

### Iter 141 — 1024-thread balanced asynchronous transform CTA

- **Hypothesis:** 864 transform workers further hide S/T preparation while retaining128 consumers and64-register bound
- **Changes:** Increase140 CTA768->1024, pack workers608->864, keep2048elements per CTA and product consumers128
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011814 ms; mean speedup=0.9441x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012310 | 0.011766 | 1.0462 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009986 | 0.011861 | 0.8419 | 0.000754626 | 0 |

- **Analysis:** 140 paired launch11.422/11.584us, Graph10.965/11.308;72bitwise checks and3sanitizers pass. Full141 benchmark archived with original gates unchanged.
- **Next:** Select worker count, then test deeper small-K prefetch and issue/wait cadence
- **Archive:** trajectory/20261015_023840_iter-141/

### Iter 142 — Eight K128 operand slots on the balanced asynchronous pipeline

- **Hypothesis:** Double slot count with smaller K tiles enables a different latency-overlap regime
- **Changes:** From140 K256/four slots ->K128/eight slots,shared213248B; publication starts after both original-input products have been prefetched; immediate release remains
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012285 ms; mean speedup=0.9104x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012346 | 0.012172 | 1.0143 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009997 | 0.012397 | 0.8064 | 0.000754626 | 0 |

- **Analysis:** 1411024threads regressed11.77/11.86us,140768retained as validated best. Full142 benchmark archived for K/prefetch tuning.
- **Next:** Try overlapping WGMMA groups on deeper slots if numerical behavior is unchanged
- **Archive:** trajectory/20261015_023949_iter-142/

### Iter 143 — Two outstanding WGMMA groups with eight K128 slots

- **Hypothesis:** Smaller K tiles need overlapping WGMMA groups to hide issue/completion latency
- **Changes:** On142 issue next K group before wait1 releases previous slot, drain wait0 only at product end; keep up to two groups in flight
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012092 ms; mean speedup=0.9153x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012190 | 0.012010 | 1.0150 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009928 | 0.012173 | 0.8156 | 0.000754626 | 0 |

- **Analysis:** 142 immediate waits regressed12.17/12.40us. Full143 benchmark archived; no arithmetic changes or gate relaxation.
- **Next:** Assess issue/wait overlap, reprofile best and try statically validated descriptor bases if needed
- **Archive:** trajectory/20261015_024055_iter-143/

### Iter 144 — Statically validated GMMA start fields and partial product unrolling

- **Hypothesis:** Compile-time shared start fields remove runtime descriptor address work
- **Changes:** From140 preserve CuTe layout bits, replace start fields with1024B driver-reserved base plus slot offset; host validates static shared0 and reserved1024; TC product loop unroll2
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011595 ms; mean speedup=0.9607x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012274 | 0.011529 | 1.0646 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009991 | 0.011660 | 0.8568 | 0.000754626 | 0 |

- **Analysis:** 143 eight-slot K128 wait1 regressed~12.01/12.17us; full144 benchmark archived, original gates unchanged. Numerical and SASS validation needed before retention.
- **Next:** Inspect timing, errors and generated uniform descriptor instructions
- **Archive:** trajectory/20261015_024310_iter-144/

### Iter 145 — K512 two-slot asynchronous input publication

- **Hypothesis:** Halving K-stage waits with asynchronous packing can reduce leaf critical path
- **Changes:** From140 K512 and two compact slots; two B TMA256 transfers; publish transformed operands before sequence2
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012664 ms; mean speedup=0.8824x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012338 | 0.012660 | 0.9746 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010011 | 0.012669 | 0.7902 | 0.000754626 | 0 |

- **Analysis:** Numerically matches seven-product baseline, both original gates fail as expected. Launch12.66/12.67us regresses versus14011.42/11.58us; longer stages dominate saved waits.
- **Next:** Explore clustered wider tiles to reduce seven-product instruction and exchange overhead
- **Archive:** trajectory/20261015_024706_iter-145/

### Iter 146 — Two-CTA DSM cluster with split seven-product ownership

- **Hypothesis:** Wider WGMMA reduces instructions while DSM exchanges only recombination state
- **Changes:** 64x64x256 tile, two operand slots, cluster2; CTA0 P1 P2 P6 P7, CTA1 P5 P3 P4; 256 consumers, remote mbarrier protects DSM state lifetime; full grid input publication
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.015789 ms; mean speedup=0.7066x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cluster_64x64x256 v0 | launch | 0.012315 | 0.015717 | 0.7835 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cluster_64x64x256 v0 | launch | 0.009989 | 0.015862 | 0.6297 | 0.000754626 | 0 |

- **Analysis:** All six fresh checks reproduce baseline error and no nonfinite; gates still FAIL. BF16/FP16 launch15.72/15.86us, materially slower than140. Needs profiling before further cluster tuning.
- **Next:** Profile instruction resource and waiting overhead of DSM candidate
- **Archive:** trajectory/20261015_024843_iter-146/

### Iter 147 — Clustered products with asynchronous transform publication

- **Hypothesis:** Overlap raw P1 P2 with transforms while peer starts transformed products after publication
- **Changes:** From146 use768 threads and608 balanced CUDA pack workers; only TMA leaders publish through invocation-private cooperative workspace; remote readiness protects DSM lifetime
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014837 ms; mean speedup=0.7496x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cluster_64x64x256 v0 | launch | 0.012248 | 0.014718 | 0.8322 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cluster_64x64x256 v0 | launch | 0.009976 | 0.014957 | 0.6670 | 0.000754626 | 0 |

- **Analysis:** Expected original numerical FAIL, baseline eF reproduced. Cluster improves15.72/15.86 to14.74/14.97us but remains slower than140. NCU146 full set failed LaunchFailed, resource probe103regs zero spills.
- **Next:** Evaluate four-stage K128 clustered supply then return to best architecture if slower
- **Archive:** trajectory/20261015_025003_iter-147/

### Iter 148 — Balanced four-element64bit input transformation IO

- **Hypothesis:** Reduce repeated transform IO instructions and loop trips on asynchronous608-worker packing
- **Changes:** From140 vectorize four input elements per worker using uint64 loads/stores; preserve FP32 transform arithmetic and high rounding
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011656 ms; mean speedup=0.9609x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012418 | 0.011510 | 1.0788 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009949 | 0.011801 | 0.8431 | 0.000754626 | 0 |

- **Analysis:** Baseline approximation errors unchanged and gates FAIL. Launch11.51/11.80us, no improvement over14011.42/11.58us; not retained.
- **Next:** Remove terminal CTA barrier while retaining initial barrier and all per-slot completion dependencies
- **Archive:** trajectory/20261015_025128_iter-148/

### Iter 149 — Return independently after final role work

- **Hypothesis:** Remove final CTA barrier overhead without shortening CTA shared storage lifetime
- **Changes:** From140 remove terminal syncthreads; keep initial barrier, all WGMMA waits and product consumer dependencies
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011387 ms; mean speedup=0.9775x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012289 | 0.011290 | 1.0885 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009951 | 0.011483 | 0.8666 | 0.000754626 | 0 |

- **Analysis:** Fresh outputs reproduce baseline errors, both original gates FAIL. FP16 full launch11.48us is modest improvement versus previous14011.58us; requires paired comparison and independent synchronization validation before retention.
- **Next:** Profile current role completion and test targeted barrier suspension
- **Archive:** trajectory/20261015_025205_iter-149/

### Iter 150 — Multicast A across adjacent output CTAs

- **Hypothesis:** Reduce repeated L2 reads by sharing A while preserving balanced seven-product ownership
- **Changes:** From140 cluster2x1, multicast A with two-arrival transaction barriers; rank0 slot reuse waits both TC groups; initial and terminal cluster barriers protect remote accesses
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014813 ms; mean speedup=0.7559x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012345 | 0.014675 | 0.8412 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010026 | 0.014952 | 0.6706 | 0.000754626 | 0 |

- **Analysis:** Numerical baseline errors reproduced and original gates FAIL. Full launch14.67/14.95us regresses from140, cluster coordination overwhelms saved A reads. No bandwidth-floor claim: best140 L2 throughput~50percent.
- **Next:** Re-profile best kernel and inspect remaining launch/role completion limits
- **Archive:** trajectory/20261015_025403_iter-150/

### Iter 151 — Local shared transforms and seven persistent register accumulators

- **Hypothesis:** Eliminate global S T publication and product shared exchange while halving raw input traffic
- **Changes:** Four raw quadrants in two K128 shared slots plus A22 B22 backup;224 CUDA workers transform shared in place; TC keeps seven accumulator tiles and issues two then five groups; single ordinary launch
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.022367 ms; mean speedup=0.4998x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_local_transform_64x32x128 v0 | launch | 0.012443 | 0.022893 | 0.5435 | 1.58558 | 0 |
| 1024,1024,1024 | fp16 | sw_local_transform_64x32x128 v0 | launch | 0.009959 | 0.021841 | 0.4560 | 1.58553 | 0 |

- **Analysis:** INVALID numerical eF~1.58 for both dtypes, timing22.90/21.84us is not valid performance evidence. No C7515 warnings observed. Must diagnose local transform or accumulator layout before evaluating direction.
- **Next:** Check shared vector layout and in-place transform race with independent product validation
- **Archive:** trajectory/20261015_025705_iter-151/

### Iter 152 — Position-independent shared swizzle indexing

- **Hypothesis:** Use element-scaled CuTe swizzle and inline transform to eliminate layout and call-stack concerns
- **Changes:** From151 convert flagged layout to position-independent swizzle layout and force inline local shared transform
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.022383 ms; mean speedup=0.5016x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_local_transform_64x32x128 v0 | launch | 0.012521 | 0.023043 | 0.5434 | 1.58558 | 0 |
| 1024,1024,1024 | fp16 | sw_local_transform_64x32x128 v0 | launch | 0.009990 | 0.021724 | 0.4599 | 1.58553 | 0 |

- **Analysis:** INVALID errors unchanged~1.58, so layout correction did not address numerical cause. Timing23.04/21.72us excluded. Investigate raw product accuracy and per-quadrant mismatches before more performance changes.
- **Next:** Independent raw-product diagnostic without changing production benchmark contract
- **Archive:** trajectory/20261015_025838_iter-152/

### Iter 153 — Correct local P3 P4 accumulator ownership

- **Hypothesis:** Quadrant-specific diagnostic isolates swapped off-diagonal leaf IDs
- **Changes:** From152 assign A22*T4 to P4 and S4*B22 to P3, preserving original recombination order
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.022374 ms; mean speedup=0.4981x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_local_transform_64x32x128 v0 | launch | 0.012336 | 0.022937 | 0.5378 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_local_transform_64x32x128 v0 | launch | 0.009997 | 0.021812 | 0.4584 | 0.000754626 | 0 |

- **Analysis:** Fresh original-reference errors now match seven-product baseline exactly; both gates still FAIL. Debug152 D11 D22 bitwise exact and only off-diagonals wrong. Correct153 launch~22.96/21.81us slower than140; need generated-code and local-memory analysis before discarding local fusion.
- **Next:** Inspect call-stack, spills and shared traffic in correct local-transform kernel
- **Archive:** trajectory/20261015_025958_iter-153/

### Iter 154 — Trade transform warps for local-fusion register capacity

- **Hypothesis:** Fewer total threads permits larger TC register allocation and fewer spills
- **Changes:** From153 reduce384threads to320, transform workers224 to160; preserve all raw pipeline and arithmetic
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.025897 ms; mean speedup=0.4319x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_local_transform_64x32x128 v0 | launch | 0.012409 | 0.026673 | 0.4652 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_local_transform_64x32x128 v0 | launch | 0.010014 | 0.025121 | 0.3986 | 0.000754626 | 0 |

- **Analysis:** Baseline seven-product errors preserved, gates FAIL. Launch26.67/25.12us regresses despite anticipated register capacity relief.153 NCU showed168regs,96B stack with STL/LDL and~6.4M instructions,8.3percent Tensor; inspect154 generated code to establish actual spill response.
- **Next:** Return to validated fast architecture; reassess measured bottlenecks rather than persist slower local fusion
- **Archive:** trajectory/20261015_030106_iter-154/

### Iter 155 — Compact uncompensated operand planes into4MiB workspace

- **Hypothesis:** Remove obsolete residual-plane gaps to improve transformed-input locality
- **Changes:** From140 map S/T to four contiguous high planes each, halve caller workspace8MiB to4MiB, preserve FP32 transforms and full shared pipeline
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011534 ms; mean speedup=0.9687x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012334 | 0.011446 | 1.0776 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009993 | 0.011622 | 0.8599 | 0.000754626 | 0 |

- **Analysis:** Fresh baseline errors preserved and original gates FAIL. Launch11.446/11.622us is flat versus140; workspace reduction useful but no performance claim pending paired validation.
- **Next:** Test packed two-result dtype conversion and inspect emitted conversion instructions
- **Archive:** trajectory/20261015_030308_iter-155/

### Iter 156 — Explicit paired16bit conversion for transformed operands

- **Hypothesis:** Pack two FP32 transformed results in one dtype-conversion instruction
- **Changes:** From155 compute two lanes in FP32 and use cvt.rn.f16x2.f32 or bf16x2 packing with documented high/low operand order
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011586 ms; mean speedup=0.9620x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012310 | 0.011494 | 1.0710 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009961 | 0.011678 | 0.8530 | 0.000754626 | 0 |

- **Analysis:** Original seven-product errors preserved, gates FAIL. Launch11.494/11.678us does not improve155 or140; do not retain purely from static instruction expectations.
- **Next:** Test non-returning reduction for one-shot invocation-local publication counter
- **Archive:** trajectory/20261015_030407_iter-156/

### Iter 157 — One-shot release reduction publication counter

- **Hypothesis:** Non-returning atomic reduction avoids returned atomic serialization and phase bookkeeping
- **Changes:** From155 RED release increments private invocation workspace counter and leaders acquire-wait for128; same CUDA-to-TMA shared publication and no cross-call state
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011551 ms; mean speedup=0.9667x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012348 | 0.011477 | 1.0758 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009968 | 0.011625 | 0.8575 | 0.000754626 | 0 |

- **Analysis:** Baseline numerical errors reproduced, both original gates FAIL. Launch11.477/11.625us remains flat versus155. Compact workspace and counter changes do not resolve FP16 gap.
- **Next:** Reassess best NCU and test direct two-accumulator interleaving with immediate retire
- **Archive:** trajectory/20261015_030453_iter-157/

### Iter 158 — Two-accumulator interleaved K issue with explicit retirement

- **Hypothesis:** Interleave independent products on slots0 2 1 3 to overlap GMMA groups without lambda accumulator aliases
- **Changes:** From155 explicit fixed-slot helpers, acc0 acc1; wait1 frees each completed operand, publish P0 before P1 drain; tail seventh product ordinary
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.017603 ms; mean speedup=0.6338x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012324 | 0.017640 | 0.6986 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009993 | 0.017565 | 0.5689 | 0.000754626 | 0 |

- **Analysis:** Baseline errors preserved and original gates FAIL. No C7515 compiler warning, but launch17.65/17.57us is slower than155 and140. Inspect generated spills before treating independent-accumulator approach exhausted.
- **Next:** Check resource usage and then target remaining best-kernel wait/descriptor overhead
- **Archive:** trajectory/20261015_030630_iter-158/

### Iter 159 — Reduce CTA threads for dual-accumulator register capacity

- **Hypothesis:** 512threads allow more registers than80-register768thread dual-accumulator implementation
- **Changes:** From158 use512threads and352 balanced pack workers, preserving interleaved leaf issue and retirement
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013897 ms; mean speedup=0.8036x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012329 | 0.013709 | 0.8993 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009970 | 0.014085 | 0.7078 | 0.000754626 | 0 |

- **Analysis:** Baseline approximation errors preserved, gates FAIL. Launch13.71/14.09us improves15817.65/17.57 but remains slower than140.158 resource80regs STACK264B shows actual local spill traffic despite LOCAL0 field.
- **Next:** Test warp-level MMA alternative for small N32 tile, then compare retained low-overhead candidate
- **Archive:** trajectory/20261015_030754_iter-159/

### Iter 160 — Warp-level MMA sync with LDSM operand fragments

- **Hypothesis:** Independent warp scheduling may suit small N32 tiles better than warpgroup synchronization
- **Changes:** From155 replace WGMMA with four independent16x32 warp tiles using m16n8k16 and ldmatrix x4 A plus x2 trans B; preserve shared TMA ring and P layout
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.030048 ms; mean speedup=0.3821x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012980 | 0.030125 | 0.4309 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009992 | 0.029971 | 0.3334 | 0.000754626 | 0 |

- **Analysis:** All fresh errors match seven-product baseline, gates FAIL. Complete calls30.13/29.97us materially slower, and initial Dense BF16 groups varied12.4-13.8us; no performance win. Explicit operand loads and increased instructions negate scheduling benefit.
- **Next:** Profile HMMA instruction and shared-bank behavior; retain best WGMMA source and continue targeted latency reductions
- **Archive:** trajectory/20261015_030926_iter-160/

### Iter 161 — Bound warp-MMA K fragment lifetime

- **Hypothesis:** Avoid fully unrolled fragment-load scheduling and reduce instruction footprint/spills
- **Changes:** From160 do not unroll K16 fragment loop; preserve all MMA order and warp mapping
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.029243 ms; mean speedup=0.3814x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012380 | 0.029266 | 0.4230 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009927 | 0.029219 | 0.3397 | 0.000754626 | 0 |

- **Analysis:** Baseline numerical errors preserved and gates FAIL. Launch29.27/29.22us remains far slower than WGMMA best.160 NCU~6.55M instructions,80regs STACK96B,Tensor9percent; instruction/operand loading path remains costly.
- **Next:** Combine independently plausible workspace reduction and terminal-barrier removal, validate paired against140 before further optimization
- **Archive:** trajectory/20261015_031038_iter-161/

### Iter 162 — Compact workspace with independent role returns

- **Hypothesis:** Keep four-plane operands while removing redundant final CTA barrier
- **Changes:** From155 remove final syncthreads as149; retain initialization barrier, all TMA readiness and WGMMA completion, shared P consumption dependencies
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011437 ms; mean speedup=0.9706x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012238 | 0.011359 | 1.0774 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009948 | 0.011516 | 0.8639 | 0.000754626 | 0 |

- **Analysis:** Baseline seven-product errors preserved and original gates FAIL. FP16 launch11.516us, still slower than Dense~9.95us. Requires full paired140comparison and72-check/sanitizer validation before retaining as checkpoint.
- **Next:** Paired complete-call launch Graph comparison and independent fresh-input validation
- **Archive:** trajectory/20261015_031116_iter-162/

### Iter 163 — Eight consumer warps with constant eight-row chunk offsets

- **Hypothesis:** 256 consumers reduce per-warp recombination work and P-free backpressure without expensive general indexing
- **Changes:** From162 use256 P-consumer threads and eight chunks, base swizzle invariant under eight-row increments;608packworkers unchanged, P-free count256
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011482 ms; mean speedup=0.9714x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012304 | 0.011383 | 1.0810 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009981 | 0.011582 | 0.8618 | 0.000754626 | 0 |

- **Analysis:** Baseline numerical errors preserved and original gates FAIL. Launch11.383/11.582us does not improve selected16211.337/11.542us. More consumer warps alone do not recover remaining FP16 gap.
- **Next:** Assign128consumers to pack workers with shorter one-trip input transformation
- **Archive:** trajectory/20261015_031605_iter-163/

### Iter 164 — Choose P consumers from shorter transform workers

- **Hypothesis:** Workers with one transform loop trip can begin raw-product consumption sooner
- **Changes:** From162 select worker480..607 as128 P consumers; their pack loop executes once instead of twice; preserve ownership, arithmetic and barriers
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011466 ms; mean speedup=0.9745x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012337 | 0.011366 | 1.0854 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009989 | 0.011567 | 0.8636 | 0.000754626 | 0 |

- **Analysis:** Baseline errors preserved and gates FAIL. Launch11.366/11.567us remains flat versus162; early consumer arrival is not a measured improvement.
- **Next:** Refresh cumulative phase diagnostics on162 and test statically specialized sequential product pairs
- **Archive:** trajectory/20261015_031646_iter-164/

### Iter 165 — Four independent TMA slot owners

- **Hypothesis:** Issuing independent operand slots from four lanes reduces serial producer instruction and head-of-line overhead.
- **Changes:** Restore162, distribute sequence modulo4 across tid0..3, each waits local pack and lane0 publishes grid with syncwarp0xf before transformed loads.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012623 ms; mean speedup=0.8825x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012291 | 0.012520 | 0.9817 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009969 | 0.012726 | 0.7833 | 0.000754626 | 0 |

- **Analysis:** Full-call BF16 about12.52us and FP16 about12.73us regress versus16211.34/11.54us. Original FP64 gates still fail with expected baseline errors and nonfinite0. Divergent lane scheduling adds overhead; reject.
- **Next:** Restore162 and try static two-product sequential helpers without dual live accumulators.
- **Archive:** trajectory/20261015_032205_iter-165/

### Iter 166 — Static sequential product pair helpers

- **Hypothesis:** Compile-time operand and product slots remove descriptor address overhead without dual live accumulator arrays.
- **Changes:** Restore162; three runtime pairs call forced-inline SlotBase0/P0 and SlotBase2/P1 helpers sequentially; tail static0/P0.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.013534 ms; mean speedup=1.2358x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.023324 | 0.013431 | 1.7365 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010023 | 0.013636 | 0.7350 | 0.000754626 | 0 |

- **Analysis:** BF16 about13.43us and FP16 about13.64us are materially slower than selected162. Expected uncompensated FP64 gate failures preserved. Dense BF16 measured22-29us anomalously vs normal12us, so apparent speedup invalid as verdict; reject candidate based on own latency.
- **Next:** Inspect generated resources, restore162 and test a third product buffer to reduce consumer backpressure.
- **Archive:** trajectory/20261015_032253_iter-166/

### Iter 167 — Three product slots

- **Hypothesis:** One extra product slot absorbs consumer jitter and reduces TC backpressure.
- **Changes:** Restore162 and expand P buffers and barriers from2 to3; update all slot and phase expressions; shared221312bytes.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011551 ms; mean speedup=0.9635x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012324 | 0.011472 | 1.0743 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009916 | 0.011629 | 0.8526 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.45us FP16 about11.63us, slightly slower than162. Expected errors and nonfinite0 retained. Extra buffer provides no useful gain on complete calls; config string stale claims2 slots so source sizes are authoritative.
- **Next:** Restore162 and inspect host tensor-map construction and barrier wait implementation.
- **Archive:** trajectory/20261015_032335_iter-167/

### Iter 168 — Barrier test-wait fast path

- **Hypothesis:** Nonblocking test before suspending wait avoids overhead for already-ready barriers.
- **Changes:** Restore162; wait helper tests phase first then uses CUTLASS wait only if incomplete, preserving acquire and compiler memory fence.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011486 ms; mean speedup=0.9697x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012262 | 0.011439 | 1.0720 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010005 | 0.011534 | 0.8674 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.44us and FP16 about11.53us, flat versus162 with no substantial reduction. Expected original numerical failures preserved. NVIDIA PTX documents try_wait suspension and test_wait nonblocking; reference https://docs.nvidia.com/cuda/archive/12.0.0/parallel-thread-execution/index.html.
- **Next:** Restore162; aggregate pack publication to19 warp leaders with full-warp ordering.
- **Archive:** trajectory/20261015_032426_iter-168/

### Iter 169 — Warp-aggregated pack publication

- **Hypothesis:** Reducing608 pack barrier arrivals to19 warp-leader arrivals reduces publication contention.
- **Changes:** Restore162; every pack lane retains proxy fence, then syncwarp and only lane0 arrives; pack barrier count19.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011403 ms; mean speedup=0.9762x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012276 | 0.011304 | 1.0860 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009966 | 0.011502 | 0.8665 | 0.000754626 | 0 |

- **Analysis:** FP16 about11.50us versus selected16211.54us, negligible single-run change. BF16 retains expected error; both original numerical gates fail and nonfinite0. Full validation required before adopting synchronization change.
- **Next:** Archive binary for later paired comparison; investigate tensor-map host construction.
- **Archive:** trajectory/20261015_032500_iter-169/

### Iter 170 — Immutable tensor-map geometry templates

- **Hypothesis:** Replacing only addresses instead of re-encoding fixed layouts reduces complete-call host overhead.
- **Changes:** Restore162; plan stores four address-free geometry templates; each call copies descriptors and uses official cuTensorMapReplaceAddress with current pointers, preserving thread safety and alignment checks.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011466 ms; mean speedup=0.9701x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012300 | 0.011391 | 1.0798 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009929 | 0.011540 | 0.8604 | 0.000754626 | 0 |

- **Analysis:** FP16 about11.54us, no meaningful gain versus162. Original FP64 gates still fail with unchanged expected errors and nonfinite0. Native driver API accepted null-address templates. No cached tensor values or cross-call compute state. Reference https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html.
- **Next:** Return to device supply latency; test short suspension hints in barrier wait.
- **Archive:** trajectory/20261015_032550_iter-170/

### Iter 171 — Two resident CTA N16 with current asynchronous frontend

- **Hypothesis:** N16 BK256 two stages permits256 CTA grid and two resident CTAs per SM, unlike current128CTA one perSM.
- **Changes:** Restore162; N16, BK256 stages2,512threads352pack,8accumulators,128consumer8chunks,new bijective XOR layout,256leader publication and grid.
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Compilation failed only at incorrect static sizeof assertion90368; actual aligned allocation should90240. No timing or correctness result, excluded. Revisits113 with modern async frontend and twice K block, not identical113.
- **Next:** Correct static assertion90240 and measure unchanged candidate.
- **Archive:** trajectory/20261015_032702_iter-171/

### Iter 172 — N16 shared budget correction

- **Hypothesis:** Correct static allocation then measure N16 two-resident-CTA candidate.
- **Changes:** Correct sizeof assertion to90240bytes.
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Compile failed at MN SW64 atom minimum32 rows versus N16. No performance data; must use MN SW32 atom and matching TMA32B swizzle, as113.
- **Next:** Correct B shared layout and tensor-map swizzle then run same candidate.
- **Archive:** trajectory/20261015_032724_iter-172/

### Iter 173 — Measure corrected N16 asynchronous two-CTA pipeline

- **Hypothesis:** Two resident CTA pipelines hide latency with N16 and BK256.
- **Changes:** Correct MN shared atom toSW32 and B tensor map swizzle32B; all171architecture otherwise unchanged.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.016503 ms; mean speedup=0.6760x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012288 | 0.016409 | 0.7489 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010011 | 0.016597 | 0.6032 | 0.000754626 | 0 |

- **Analysis:** BF16 about16.41us FP16 about16.60us, slower than162 despite correct baseline errors/nonfinite0. Smaller N doubles global CTA supply traffic and publication participation; no occupancy-based win. Config string still stale162geometry, actual source is N16 and shared90240.
- **Next:** Restore162; test shortened barrier suspension hint.
- **Archive:** trajectory/20261015_032756_iter-173/

### Iter 174 — Short barrier suspension hint

- **Hypothesis:** A256ns try_wait suspension hint may reduce wakeup latency on short operand-slot dependencies.
- **Changes:** Restore162 and use inline PTX parity try_wait with256ns hint, same phase semantics and memory clobber.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011429 ms; mean speedup=0.9717x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012218 | 0.011366 | 1.0750 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009980 | 0.011492 | 0.8684 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.37us FP16 about11.49us, small single-run variation versus162, no3percent improvement. Expected original precision failures remain. Does not prove wakeup latency changed.
- **Next:** Archive binary and restore162; single warpgroup leader releases operands after full WGMMA wait0.
- **Archive:** trajectory/20261015_032840_iter-174/

### Iter 175 — Single operand-release leader

- **Hypothesis:** WGMMA wait0 completes all shared-operand reads, allowing one leader to free slots without four arrivals.
- **Changes:** Restore162; operand free barrier count1, only tid128 arrives after unchanged collective WGMMA wait0 and syncwarp.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011448 ms; mean speedup=0.9805x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012518 | 0.011357 | 1.1023 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009910 | 0.011540 | 0.8587 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.36us FP16 about11.54us, no gain against162; original precision failures expected and nonfinite0. Reject as optimization; collective completion correctness needs validation if retained later.
- **Next:** Review prior split-product DSM experiment; test global publication to remove cluster synchronization overhead.
- **Archive:** trajectory/20261015_032937_iter-175/

### Iter 176 — Global U2 U3 split-product pair

- **Hypothesis:** Replacing DSM exchange and cluster lifetime waits with per-consumer global release/acquire enables parallel4plus3 products without cluster overhead.
- **Changes:** Base147 N64two-stage async608; compact S/T4MiB; addU2/U3global2MiB and64KiB current-invocation flags inside8MiB workspace; rank1 initializes flags before global pack publication; rank0 per-consumer releases after stores; no clusters or terminal CTA barrier.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.015313 ms; mean speedup=0.7316x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cluster_64x64x256 v0 | launch | 0.012504 | 0.015481 | 0.8077 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cluster_64x64x256 v0 | launch | 0.009927 | 0.015146 | 0.6554 | 0.000754626 | 0 |

- **Analysis:** BF16 about15.48us FP16 about15.15us versus16211.34/11.54us, no win. Expected uncompensated errors and nonfinite0 retained. Global publication did not recover split-product cost; inspect registers and local spills before further variants.
- **Next:** Inspect resource usage, then eliminate shared rank1 staging and attempt deeper operand pipeline only if resources support it.
- **Archive:** trajectory/20261015_033052_iter-176/

### Iter 177 — Three operand slots in global split-product pair

- **Hypothesis:** Removing shared local partial staging permits deeper operand prefetch for paired product CTAs.
- **Changes:** Base176; rank1 keeps P5/P3 in first/second registers and P4 current, no shared state; three operand slots and one product slot fit213120bytes.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014998 ms; mean speedup=0.7400x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cluster_64x64x256 v0 | launch | 0.012269 | 0.014841 | 0.8267 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cluster_64x64x256 v0 | launch | 0.009901 | 0.015155 | 0.6533 | 0.000754626 | 0 |

- **Analysis:** BF16 about14.84us and FP16 about15.15us, still far slower162; expected baseline errors retained and nonfinite0. No win from current deeper pair pipeline.
- **Next:** Move P2 to the output-consuming CTA so U2/U3 producer finishes three products and exchange may overlap consumers fourth product.
- **Archive:** trajectory/20261015_033152_iter-177/

### Iter 178 — Hide global pair exchange behind fourth consumer product

- **Hypothesis:** Move P2 to receiving CTA so sender publishes P1/U2/U3 after only three products while receiver computes four.
- **Changes:** Base177; rank0 P1/P6/P7,rank1 P2/P5/P3/P4; addP1 plane at4MiB,shiftU2/U3 and flags7MiB; both leaders publish transform barrier atsequence2; exact FP32 recombination order preserved.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.017677 ms; mean speedup=0.6285x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cluster_64x64x256 v0 | launch | 0.012326 | 0.017665 | 0.6978 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cluster_64x64x256 v0 | launch | 0.009894 | 0.017690 | 0.5593 | 0.000754626 | 0 |

- **Analysis:** BF16 about17.67us FP16 about17.69us, regresses177 and far slower162. Expected numerical errors/nonfinite0 retained. Splitting early raw products reduces overlap available before global transformed-input publication, and extra exchange does not help.
- **Next:** Re-profile178 supply issue and local spill counters; restore162 before next independent direction.
- **Archive:** trajectory/20261015_033239_iter-178/

### Iter 179 — Single PTX block for sixteen WGMMA instructions

- **Hypothesis:** Keeping descriptors and accumulator references within one assembly block permits ptxas to reduce uniform-register transfer overhead.
- **Changes:** Restore162; one asm block per K256 tile, shared descriptor bases passed once, fixed CuTe K offsets as immediates, sixteen same-order WGMMA operations and unchanged transpose/scales.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011512 ms; mean speedup=0.9729x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012442 | 0.011406 | 1.0909 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009933 | 0.011618 | 0.8550 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.41us FP16 about11.62us, no improvement versus162. Expected numerical errors/nonfinite0. Must inspect SASS before making any claim about descriptor R2UR reduction.
- **Next:** Inspect dynamic descriptor code generation; test immutable descriptor low-word construction if transfer count unchanged.
- **Archive:** trajectory/20261015_033439_iter-179/

### Iter 180 — PTX descriptor low-word arithmetic

- **Hypothesis:** Avoid64bit descriptor carry and retain invariant high words inside the WGMMA assembly block.
- **Changes:** Base179; unpack base descriptor into low/high32, add fixed K offsets only to low words, repack for each same-order WGMMA.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011437 ms; mean speedup=0.9754x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012325 | 0.011346 | 1.0863 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009965 | 0.011528 | 0.8644 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.35us FP16 about11.53us, flat162. Expected original precision failures remain. 179SASS had64bit carry and four R2UR around each WGMMA; this removes carry but has no substantial complete-call gain.
- **Next:** Restore162 and use uniform slot dispatch with template-constant start descriptors, shared accumulator array and unchanged perK waits.
- **Archive:** trajectory/20261015_033526_iter-180/

### Iter 181 — Template-constant descriptors via uniform slot dispatch

- **Hypothesis:** Four fixed slot cases allow constant shared descriptor start addresses and less R2UR work without multiple live accumulator arrays.
- **Changes:** Restore162; template Slot in mma_term, explicit uniform switch, constant1024B shared start and constant slot offsets, host verifies static shared0/reservation1024.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011454 ms; mean speedup=0.9729x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012316 | 0.011373 | 1.0829 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009953 | 0.011534 | 0.8629 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.37us FP16 about11.53us, flat162. Original FP64 failures and nonfinite0 retained. No claim of R2UR savings without SASS inspection.
- **Next:** Inspect SASS and test avoiding redundant fence between two WGMMA K groups, preserving operand waits and release.
- **Archive:** trajectory/20261015_033624_iter-181/

### Iter 182 — Fence only before first K group

- **Hypothesis:** Consecutive same-shape WGMMA accumulator accesses require no additional register fence between q0 and q1 when generic code does not access accumulators.
- **Changes:** Restore162; retain warpgroup fence before q0 only, unchanged perK ready/wait0/free and output stores.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011436 ms; mean speedup=0.9741x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012285 | 0.011329 | 1.0844 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009970 | 0.011543 | 0.8638 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.33us FP16 about11.54us, no material improvement. Expected original FP64 failures retained. NVIDIA PTX permits same-shape consecutive WGMMA accumulator ordering without extra fence; reference https://docs.nvidia.com/cuda/pdf/ptx_isa_8.8.pdf.
- **Next:** Try separate A and B TMA producer warps with shared per-slot transaction completion and explicit local publication handoff.
- **Archive:** trajectory/20261015_033722_iter-182/

### Iter 183 — Independent A and B TMA producer warps

- **Hypothesis:** Separate warp leaders issue A and B independently, reducing serial TMA producer work.
- **Changes:** Restore162; tid0 A producer andtid32 B producer; ready count2 with separate expected transaction bytes; A leader global publication handed to B by local barrier;576pack workers, same shared budget.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011568 ms; mean speedup=0.9600x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012273 | 0.011534 | 1.0641 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009931 | 0.011603 | 0.8559 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.53us FP16 about11.60us, slower162. Expected precision errors retained and nonfinite0. Separate producer warps do not improve full-call latency.
- **Next:** Restore162; allow pack warps to start before initial CTA synchronization, using explicit role-specific named barriers.
- **Archive:** trajectory/20261015_033818_iter-183/

### Iter 184 — Role-specific initialization and pack named barriers

- **Hypothesis:** Allow pack warps to compute immediately while only TMA and TC wait for shared barrier initialization.
- **Changes:** Restore162; initial named barrier1 has160TMA/TC threads; warp0 and608pack threads rendezvous named barrier2 count640 before global transform publication; no pack mbarrier arrivals, TC rawP1/P2 overlap preserved.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011607 ms; mean speedup=0.9564x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012258 | 0.011641 | 1.0529 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009950 | 0.011573 | 0.8598 | 0.000754626 | 0 |

- **Analysis:** FP16 about11.57us, no gain162; expected original numerical errors and nonfinite0. Stronger role-specific ordering is valid conceptually but offers no complete-call benefit; reject.
- **Next:** Inspect earlier vector consumer experiments and test only materially distinct mapping; otherwise return to operand map policy.
- **Archive:** trajectory/20261015_033924_iter-184/

### Iter 185 — TMA L2 promotion256B

- **Hypothesis:** Larger promotion granularity may improve reuse across neighboring CTA tiles and reduce supply overhead.
- **Changes:** Restore162; both raw and transformed tensor maps use official L2 promotion256B instead ofNONE.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012146 ms; mean speedup=0.9161x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012245 | 0.011943 | 1.0252 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009965 | 0.012349 | 0.8070 | 0.000754626 | 0 |

- **Analysis:** BF16 about11.94us FP16 about12.35us, slower162; expected original precision failures retained/nonfinite0. Larger L2 promotion is counterproductive for current narrow tiles.
- **Next:** Restore162 and reassess coarse parallelization: one product per CTA with fused last-CTA recombination, while preserving original precision acceptance.
- **Archive:** trajectory/20261015_034008_iter-185/

### Iter 186 — One leaf product per CTA with fused last-CTA recombination

- **Hypothesis:** 224cooperative CTAs, each computing one128x64 leaf tile using two TC groups, increase parallelism and remove seven-product serialization.
- **Changes:** New512thread kernel, twoK128operand slots98432sharedbytes,224pack workers perCTA, two64x64WGMMAgroups,7MiB global FP32 leaf results,32per-tile counters initialized before private invocation publication; last acqrel counter CTA recombines all outputs,12MiBworkspace. Original transform and FP32 recombination order retained.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.032147 ms; mean speedup=0.3470x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012326 | 0.032000 | 0.3852 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009975 | 0.032293 | 0.3089 | 0.000754626 | 0 |

- **Analysis:** BF16 about32.0us FP16 about32.29us, much slower162 though original approximation errors retained/nonfinite0. Cannot infer source of regression without spill and instruction profiling. No numerical pass claimed.
- **Next:** Profile register spills and instructions; compare forced min2 launch bound to a resource-compatible reduction in threads.
- **Archive:** trajectory/20261015_034233_iter-186/

### Iter 187 — 384 threads for parallel leaf register headroom

- **Hypothesis:** Reducing512 to384threads raises register budget while preserving two-CTA residency and both TC groups.
- **Changes:** Base186;96pack workers instead of224, full input transformation coverage retained with new global stride;384threads launch bound min2.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.031880 ms; mean speedup=0.3521x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012459 | 0.031867 | 0.3910 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009992 | 0.031893 | 0.3133 | 0.000754626 | 0 |

- **Analysis:** BF16 and FP16 about31.87/31.89us, essentially unchanged and much slower162. Expected original precision failures retained/nonfinite0. Removing register pressure alone does not rescue this direction; inspect actual spill counters before conclusions.
- **Next:** Profile187 actual resources and phase tail; investigate last-CTA epilogue code-size overhead before rejecting coarse leaf parallelism.
- **Archive:** trajectory/20261015_034338_iter-187/

### Iter 188 — Coalesced row-major last-CTA recombination

- **Hypothesis:** 187trace shows9-14us epilogue; use row-major contiguous float2 loads/stores and a compact loop instead of unrolled fragment-order reads.
- **Changes:** Base187;256TCthreads remap to8rows64columns per chunk,16rowchunks with fixed8row stride, loop kept compact; all seven values and FP32 arithmetic order unchanged.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.021359 ms; mean speedup=0.5224x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012336 | 0.021388 | 0.5768 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009982 | 0.021329 | 0.4680 | 0.000754626 | 0 |

- **Analysis:** BF16 about21.39us FP16 about21.33us, roughly10us faster187 but still much slower162. Original approximation errors retained/nonfinite0. Diagnostic187 had~10percent time perturbation and16Bspill; use it only as localization evidence, production187fullNCU79regs/local0.
- **Next:** Reassess start-time spread and optimize parallel-leaf tile shape only if phase budget supports improvement; preserve162 selected checkpoint.
- **Archive:** trajectory/20261015_034653_iter-188/

### Iter 189 — Reuse waiting TC warps for parallel-leaf input transformation

- **Hypothesis:** Five transformed-product CTA classes have idle TC warps before publication; using them for pack raises worker count and reduces front latency.
- **Changes:** Base188; raw-product CTAs96pack workers, transformed-product CTAs352including theirTCthreads; contiguous global worker partition62464 workers with full coverage, local pack barrier counts adjusted.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.019352 ms; mean speedup=0.5797x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012493 | 0.019166 | 0.6518 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009917 | 0.019537 | 0.5076 | 0.000754626 | 0 |

- **Analysis:** BF16 about19.17us FP16 about19.54us, improves188 but still much slower162. Original errors retained and nonfinite0. Trace187 allCTAstartspread0.224us, publication6.5us, so launch skew was not the main cause.
- **Next:** Test direct atomic recombination to remove serialized last-CTA global reads; explicitly report changed FP32 summation order and unchanged original precision gates.
- **Archive:** trajectory/20261015_034821_iter-189/

### Iter 190 — Direct atomic leaf recombination

- **Hypothesis:** Concurrent FP32 atomic contributions avoid serialized last-CTA reads of seven global product planes.
- **Changes:** Base189; pack workers also initialize four D quadrants before publication; each leaf directly atomicAdds its Winograd contributions, P4 negated. FP32 summation order is now nondeterministic, not bitwise97-preserving. Gates unchanged.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.030709 ms; mean speedup=0.3647x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012339 | 0.030011 | 0.4112 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009998 | 0.031407 | 0.3183 | 0.000754626 | 0 |

- **Analysis:** BF16 about30.01us FP16 about31.41us, worse189 and162. Expected original gate failures remain with eF near.0060152/.000754626 and nonfinite0. Atomic contribution route rejected; no numerical certification.
- **Next:** Restore189 and test bulk TMA tensor reduction if hardware-supported, otherwise return162 for independent direction.
- **Archive:** trajectory/20261015_034919_iter-190/

### Iter 191 — TMA tensor reduction of shared leaf tiles

- **Hypothesis:** Bulk tensor reduce-add replaces scalar atomics while avoiding last-CTA serialized recombination.
- **Changes:** Base190; shared two64x32FP32 reduction tiles, SW128 address permutation, fifth current-output tensor map, perWG TMAreduce contributions and full bulk completion waits. Nondeterministic sum order remains explicit.
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Runtime failed with misaligned address during explicit tuning. Reduction buffer follows48bytes of barriers and lacks128B alignment required by swizzled TMA output. No valid timings or correctness results; exclude candidate.
- **Next:** Align reduction array128bytes and correct static shared budget114944; rerun.
- **Archive:** trajectory/20261015_035040_iter-191/

### Iter 192 — Aligned TMA tensor reduction

- **Hypothesis:** Correct TMA reduction source alignment and measure aggregate hardware reduction versus scalar atomic190.
- **Changes:** Move128B-aligned reduction arrays immediately after operand storage so base is also1024B-aligned with driver reservation; retain total shared114816 instead of increasing padded allocation. Otherwise191 unchanged.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.015783 ms; mean speedup=0.7058x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012274 | 0.015676 | 0.7829 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009990 | 0.015891 | 0.6287 | 0.000754626 | 0 |

- **Analysis:** BF16 about15.68us FP16 about15.89us, much faster scalar190 but slower162. eF remains near.0060152/.000754626 with unchanged gates FAIL, nonfinite0. Sum order nondeterministic; no bitwise97 equivalence claimed. Valid performance experiment, not accepted winner.
- **Next:** Profile192 reduction waits and consider reusing retired operand memory for one full128x64 reduction tile to remove half-tile barrier round.
- **Archive:** trajectory/20261015_035129_iter-192/

### Iter 193 — Reuse retired operands for full TMA reduction tiles

- **Hypothesis:** Aliasing completed operand storage with full64x64FP32 output tiles removes half-tile TMA reduction round and shared allocation.
- **Changes:** Base192; TCgroups rendezvous after allWGMMA operands retire; reuse first32KiBoperand as two64x64tile buffers, rank3 output tensor map boxes32x64x2, oneTMAreduce per quadrant perWG; shared98432.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.015616 ms; mean speedup=0.7107x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.012277 | 0.015971 | 0.7687 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_parallel_128x64x128 v0 | launch | 0.009960 | 0.015262 | 0.6526 | 0.000754626 | 0 |

- **Analysis:** BF16 about15.97us FP16 about15.26us, mixed against192 and slower162. Expected approximation errors/nonfinite0 retained, FP32 sum order remains nondeterministic. Reclaimed16384sharedbytes but no occupancy class change.
- **Next:** Return162 and inspect prior vector consumer mappings before pursuing a distinct vector output epilogue.
- **Archive:** trajectory/20261015_035258_iter-193/

### Iter 194 — Vectorize P exchange and fixed-offset recombination

- **Hypothesis:** Paired shared exchange and fixed output offsets reduce epilogue address and memory instructions on the selected asynchronous architecture
- **Changes:** Restored iter162; pair-preserving row XOR layout; float2 TC stores, consumer loads, arithmetic and fixed chunk output stores
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011436 ms; mean speedup=0.9725x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012264 | 0.011328 | 1.0826 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009955 | 0.011544 | 0.8624 | 0.000754626 | 0 |

- **Analysis:** Valid finite approximate output with unchanged original precision failures. BF16 solution 11.328 us versus Dense 12.264 us; FP16 11.544 us versus Dense 9.955 us. No material improvement over iter162 and FP16 remains slower.
- **Next:** Restore selected iter162 and investigate a different source of producer or tensor execution latency
- **Archive:** trajectory/20261015_035421_iter-194/

### Iter 195 — Single CTA N64 with three operand slots

- **Hypothesis:** Wider N reduces repeated A loads and WGMMA issue, with three operand slots sustaining overlap
- **Changes:** N64 single-CTA tiles; grid64; three K256 operand slots; single P slot; 256 consumers
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Compiled but first invocation stalled: P-ready initialized with 256 arrivals while only128 TC producers arrive. Explicitly terminated the hung job step. No timing or numerical results valid.
- **Next:** Correct ready producer count to128, retaining consumer free count256, and fix descriptor tile extent toN64
- **Archive:** trajectory/20261015_035738_iter-195/

### Iter 196 — Correct and measure N64 single-P pipeline

- **Hypothesis:** Three K256 operand slots and wider WGMMA reduce issue and duplicated A traffic
- **Changes:** Correct P-ready producer count128 vs free consumer count256; B descriptor extentN64
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.014195 ms; mean speedup=0.7840x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012265 | 0.014095 | 0.8702 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009975 | 0.014296 | 0.6977 | 0.000754626 | 0 |

- **Analysis:** Finite expected approximate errors; original precision gates fail both. BF16 14.095 us and FP16 14.296 us versus Dense12.265/9.975 us. Wider tiles and64CTA grid lose to selected162. Reassessment used current162/187NCU and phase timing plus NVIDIA efficient GEMM/warp specialization documentation; no floor established.
- **Next:** Return to128CTA N32 grid and test two tensor warpgroups with balanced asynchronous input transform
- **Archive:** trajectory/20261015_035839_iter-196/

### Iter 197 — Two tensor warpgroups with asynchronous480-worker pack

- **Hypothesis:** Alternating seven products between independent tensor warpgroups overlaps issue and completion latency
- **Changes:** Restore162; second TC warpgroup takes odd product steps; 480 pack workers and128 streaming consumers; original slots, operation order and publication
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.012164 ms; mean speedup=0.9221x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012424 | 0.012081 | 1.0283 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.009992 | 0.012247 | 0.8158 | 0.000754626 | 0 |

- **Analysis:** Finite expected approximate output, gates fail. BF16 12.081 us and FP16 12.247 us versus Dense12.424/9.992 us; slower than selected162. Dual TC with480pack is not adopted.
- **Next:** Increase block to896 threads to restore608 transform workers while retaining dual tensor warpgroups
- **Archive:** trajectory/20261015_035937_iter-197/

### Iter 198 — Dual tensor warpgroups with608 transform workers

- **Hypothesis:** Restoring baseline pack parallelism removes transform slowdown of dual TC experiment197
- **Changes:** 896-thread block;608 transform workers; two tensor warpgroups retain alternating product ownership
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.011685 ms; mean speedup=0.9651x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.012332 | 0.011548 | 1.0678 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_cooperative_hilo_64x32x256 v0 | launch | 0.010195 | 0.011822 | 0.8624 | 0.000754626 | 0 |

- **Analysis:** BF16 11.548 us and FP16 11.822 us; Dense12.332/10.195 us. Recovers much of197 slowdown but not selected162 performance. Finite expected errors and both original gates fail. No adoption.
- **Next:** Profile dual-warpgroup instruction and resource behavior, then investigate cluster-local parallel leaf combination
- **Archive:** trajectory/20261015_040020_iter-198/

### Iter 199 — Cluster-local seven-leaf DSM combination

- **Hypothesis:** Seven parallel leaf CTAs and one DSM epilogue CTA eliminate global P traffic and last-CTA counters
- **Changes:** Base189; cluster8 grid256; retired operand memory reused for32KiB swizzled P; rank7 row-major DSM reads and exact original recombination;4MiB workspace; final cluster lifetime barrier
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Build succeeds but launch rejected: too many blocks in cooperative launch. Per-SM capacity check alone does not establish cluster-resident capacity. No timing or numerical evidence valid. Fresh198NCU64regs,zero local sectors,2.92Minstructions versus1622.52M; higher tensor activity did not reduce full latency.
- **Next:** Reduce cluster count by doubling output N extent per cluster, and explicitly query active-cluster capacity
- **Archive:** trajectory/20261015_040200_iter-199/

### Iter 200 — Four-WG leaf CTAs and explicit cluster capacity check

- **Hypothesis:** Doubling per-cluster M coverage halves the number of clusters to16 and retains one leaf per rank
- **Changes:** BM256 BN64 four tensor WGs,640threads,163968Bshared;128CTA grid;608 transformed pack workers; explicit cudaOccupancyMaxActiveClusters query
- **Bench:** Compiled=False/partial; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 0 rows; arithmetic mean of row median latency=0.000000 ms; mean speedup=0.0000x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|

- **Analysis:** Build succeeds; explicit occupancy query reports15 active clusters versus required16, so safely rejects before launch. No performance or numerical results. Cluster topology, not aggregate132SM capacity, restricts launch.
- **Next:** Use15 persistent clusters, with one cluster processing an extra output tile, and retain global input publication only once
- **Archive:** trajectory/20261015_040326_iter-200/

### Iter 201 — Measure fifteen persistent eight-CTA leaf clusters

- **Hypothesis:** Fifteen resident clusters with one processing a second tile fit the measured topology and expose DSM exchange cost
- **Changes:** 120CTA grid;15clusters; cluster0 processes extra tile15; one per-invocation global transform publication; same barrier phases reused after cluster lifetime synchronization
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 2 rows; arithmetic mean of row median latency=0.034099 ms; mean speedup=0.3261x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | sw_leaf_cluster8_128x64x128 v0 | launch | 0.012201 | 0.033944 | 0.3594 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | sw_leaf_cluster8_128x64x128 v0 | launch | 0.010030 | 0.034254 | 0.2928 | 0.000754626 | 0 |

- **Analysis:** Finite expected approximate output and both original gates fail; BF16 33.944 us and FP16 34.254 us versus Dense12.201/10.030 us. Slow cluster8 route rejected. Configuration string inherited stale grid/thread descriptions; actual source120CTA640threads4TCgroups163968Bshared is authoritative.
- **Next:** Inspect cluster resource and instruction counters, then return to162 for a new synchronization and supply direction
- **Archive:** trajectory/20261015_040441_iter-201/
