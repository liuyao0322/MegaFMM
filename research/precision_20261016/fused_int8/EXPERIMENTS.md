# 片上融合实验记录

全部为独立研究，生产`solution/`、库和bench未改。模型固定floor筛选和实际原始FP64 gate严格区分。精度失败候选的计时只用于诊断，不进入合格性能排名。

|候选|改动/结果|状态|
|---|---|---|
|F4|相对低部尺度，NVCC条件模板解析失败|未执行；方案另有F3保守数值筛选失败，不继续该表示|
|F5|回到自适应低部尺度；static constexpr CuTe布局对象无法用于device代码|未执行，失败源/日志保留|
|F6|布局以host/device constexpr函数构造，首次完整融合执行|普通72项中66项失败，BF16/FP16最大eF约129.94/129.93；诊断731.5/724.8μs|
|F7|slot和动态shared改为1024byte对齐，符合base_offset=0的SW128描述符|普通72项中60项失败，最大eF .050676/.006171；诊断732.6/724.3μs|

F6/F7两种dtype均0stack、0spill、124registers/thread。F7对齐修正大幅降低错误，但不足以通过gate；不能称为数值方案本身失败或精度改善winner。

单叶P5 probe以新生成的原始输入核对TMA原始数据、高部、量化尺度、INT8字节、native点积和两项整数补偿。首次probe两种dtype均出现约3.7万raw索引错误、约9千高部索引错误、约1.6万量化字节错误、4096项整数点积不符；native点积相对CPU高部点积eF仍仅约1.86e-7/5.37e-8。完整输出保存于`probe_first/`、`probe_layout_failed/`，非完整FP64验收或性能证据。

根因分析：本仓库CUTLASS的`GMMA::Layout_*_Atom<T>`是带`smem_ptr_flag_bits`的布局，swizzle作用于字节地址。`make_tensor(smem_ptr, layout)`会把该swizzle转移到pointer；直接对T指针使用`layout(r,k)`当作元素索引则缺少类型宽度转换。对FP16/BF16，probe中的B(0,1)索引为64，而正确字节swizzle对应元素索引为72。低精度高部的错误读写位置相互抵消，使native点积看似正确，随后从错误逻辑坐标量化到INT8布局时暴露问题。修复方向是先用`as_position_independent_swizzle_layout`转换为元素域索引，再进行generic共享内存读写；INT8的一字节元素没有相同宽度差异。

一手代码依据：`third_party/cutlass/include/cute/pointer_flagged.hpp`的`make_tensor`及`as_position_independent_swizzle_layout`，以及`mma_traits_sm90_gmma.hpp`中`base_offset=0`和对应的对齐assert。另见[NVIDIA异步拷贝文档](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/04-special-topics/async-copies.html)：128B swizzle以1024bytes重复，未按重复周期对齐时需要处理base offset。这不意味着TMA本身一概要求1024byte最低对齐；这里是所用CuTe零offset描述符与索引方式的具体约束。

后续候选必须重新核对上述probe，再运行完整72项精度、Graph/双流/sanitizer及完整计时。P7全局预处理原型的验证不能转移到融合版本。

## F8 — convert flagged layouts before direct generic indexing

- Direct FP16/BF16 shared array access now uses`as_position_independent_swizzle_layout`; TMA and GMMA descriptor paths unchanged. Single P5 probe, two dtypes:raw/high/exponent/quant/integer mismatches all0; native eF remains1.86e-7/5.37e-8. This isolates the implementation bug from the numerical method.
- Full72/72 ordinary original-input FP64 checks pass. MaxBF16/FP16 eF=5.09385609333e-4/6.31255707396e-5. Full-call BF16/FP16=731.992/720.103us versus currentSW10.195/10.003us. **Reject performance**, much worse than the separate-packing P9. No Graph/dual-stream/sanitizer acceptance is claimed for F8.
- The prototype demonstrates numerical correctness of this local adaptive representation in the tested cases, not a speed benefit from fusion. Profile the whole fused kernel before selecting the next change. The profiling driver returns after the first quick numerical check per dtype and has no timing arms; its profile is diagnostic only.

## F8 profile and next isolated change

- One first BF16 fused kernel in a dedicated profiling driver (no timing arms), same124registers and0stack/spill. NCU18passes reports933.47us, memory throughput9.81percent, DRAM0.14percent, compute16.97percent,130997494 executed warp instructions. Profiler timing is perturbed and not the731.992us ordinary result. Warp analysis attributes44.4percent of average instruction-issue spacing to L1TEX dependency waits; this alone does not identify a source instruction or prove a DRAM bottleneck.
- Static BF16 SASS13708instructions includes980MUFU.EX2,1369branches, noLDL/STL/LDG; fallback math creates a large instruction footprint. Static counts do not prove those fallback instructions execute for normal inputs; polling and instruction-cache behavior also require care. Do not apply NCU's generic FMA suggestion without numerical validation.
- First filter attempt profiled no kernels because default base names omitted namespace; log preserved. Corrected using`--kernel-name-base demangled --kernel-name regex:fused_int8::kernel`. Logs`f8_ncu*`, ignored binary report under`ncu/`.
- F9 targets a concrete uniform branch: exponent validation/factor construction outside16correction outputs, yielding contiguous normal-path converts/multiplies. General exponent fallback, separate RN multiply/add, exact quantized representation and all barriers retained. No conclusion before full72-case checks and full invocation timing.

## F9 — hoist consumer rescaling decisions

- Full72/72 ordinary checks pass, all printed eF identical toF8,124registers,0stack/spill.
- bf16: complete715.255us versus currentSW10.182us.
- fp16: complete701.239us versus currentSW10.004us.
- A modest reduction within a severely rejected prototype; the end-to-end gap remains. No independent three-process performance acceptance, Graph, dual-stream or sanitizer result is claimed. This change does not establish consumer scale construction as the main fused-kernel bottleneck.
- Next diagnosis must distinguish producer packing/reductions from consumer waits and polling, then hoist the producer's repeated inside/quantization scale decisions or reassess the fusion layout. M4's paired relative-scale model is supported numerically but has no implemented kernel yet.

## F9 producer/consumer trace — locate the actual long interval

- Clone frozenF9; timestamp producer0 and consumer128 around raw/leaf waits and major work. One fresh uniformBF16 FP64 check passes with eF5.52481148267e-5, identical toF9. Diagnostic10warmup/20kernel calls=716.614us,122regs versus124untraced,0stack/spill. Stores and scheduling changes perturb timing; no complete-call verdict.
- Median of same-CTA phase sums:producer pack687.072us,inside14.880us,raw issue+ready4.448us,slot-free wait2.112us,total710.976us. Consumer slot-ready wait685.344us,native main raw+leaf7.472us,correction0+1 15.968us,total711.664us. Raw and leaf main phase medians separately need not sum to the displayed combined median; inspect per-CTA rows for exact sums.
- Roles overlap:median useful work overlap17.456us. Do not add producer and consumer durations. The dominant measured interval is producer packing, not exposed MMA time or host tensor-map setup. The trace does not yet separate pack first-pass/quantization details.
- Next isolated candidateF10 hoists common inside exponents/factors out of pack element loops, retaining general exponent fallback. Frozen trace source/results/binary identity in`trace_f9_manifest.json`.

## F10 — producer inside-scale construction before element loops

- Cache the two inside exponents and four inverse/direct normal power-of-two factors once per leaf pack, selecting a uniform producer fast path before any named barriers. Per-element transforms use separate RN multiplies. Both exponents outside[-126,126] retain the prior general path. Quantization scale handling is unchanged.
- All72ordinary original-input FP64 checks pass, every printed eF identical toF9.
- bf16: complete395.751us versus currentSW10.120us.
- fp16: complete379.155us versus currentSW9.946us.
- This supports repeated inside scaling as a large packer cost, but still fails end-to-end performance badly. No Graph/dual-stream/sanitizer acceptance claimed. Next isolated change hoists quantization scale validation/factor construction from its element loops.

## F11 — quantization scale construction before element loops

- Cache four adaptive quantization exponents, validate their inverse scales in a uniform branch before the quantization loops, then use explicit RN multiplies in the common path. General exponents retain scale_pow2; no representation or barrier change.
- All72ordinary original-input FP64 checks pass, all printed eF identical toF10.
- bf16: complete245.429us versus currentSW10.325us.
- fp16: complete228.806us versus currentSW10.062us.
- Large reduction within the fused prototype, still much slower than independent-packingP9 and currentSW. Do not transfer P7 Graph/sanitizer acceptance. Next diagnostic subdivides the measured producer pack interval into native packing, reduction and quantization, instead of guessing from the overall time.

## F11 pack subphase trace

- FrozenF11 adds eight timestamps per pack, retaining the role trace. One uniformBF16 FP64 check passes with the same eF5.52481148267e-5; instrumented kernel255.563us versus ordinary complete245.429us,122regs,0stack/spill. Perturbed diagnostic, not performance acceptance.
- Median same-CTA sums: nativeA pack42.784us,nativeB51.936us,maximum reduction2.688us,exponent selection5.024us,A quantization48.192us,B quantization68.800us,publish/sync1.120us. Pack total220.608us; wrapper envelope225.760us. Individual phase medians need not sum to the median total.
- Second-pass quantization includes raw reloads and recomputation of transformed residuals. NextF12 retains first-pass residuals in per-thread registers; native high values still come from shared storage. Explicitly verify register/stack/spill effects and original precision before assessing complete-call speed.
