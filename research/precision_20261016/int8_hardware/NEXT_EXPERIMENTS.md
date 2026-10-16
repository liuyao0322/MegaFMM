# Remaining-cost experiments and outcomes


**Current checkpoint: P51 is the fully validated ordinary/Graph research reference; production performance target remains unmet.** Own same-roundP50→P51 ordinary reduction2.63/2.24%, Graph2.03/1.88% BF16/FP16, completecalls~27.4us versusSW~10us. Own standard78/79872exponents, eachmodeABBA312/319488exponents,54full-plane bit configurations plus18finite-edge32Kpacking configurations,12-pattern freshstress144/1152calls and scopedtools48/96calls each pass. SeeALL_NATIVE_PACK_GRID_HARDWARE.md/p51_validation.json. P50/P47 historical complete acceptance remains independent. P49 tuple reuse alone andN1–N3nativefusion rejected. New ownP51profile+history+freshCUDA12.4/ PTX8.7reassessment supports a separate consumer64x16/two-slot control, not a physicalfloor. CONSUMER_N16_DESIGN.md/generate_consumer_n16.py prepared and isolatedCPU-reproduced only; no P52application/build/measurement.

P27 supplied the fully validated research base, about41us ordinary full calls versus production SW about10us. Completed controls and remaining hypotheses are distinguished below; final retention and validation scope are recorded inEXPERIMENTS.md. Preserve seven native products, eight integer dots, exact per-column inside scaling, original rounded-input FP64 reference and all current preparation/descriptor costs. FrozenP27 profile reports joint pack56regs/65,576B shared, a shared-resource limit3CTAs/SM and issueactive53.16percent. Corrected196regs/196,736B is limited to1CTA/SM and issueactive35.71percent. All four measured kernels have zero local sectors on that input. Partial/finalize show high long-scoreboard/issue-active ratios,5.950/6.163. These are diagnostic metrics, not time fractions or a physical floor. Seeprofile_p27_stats.json and its manifest; never overlap GPU tasks.

## Two integer accumulator fragments, unchanged K128 scales — P28 rejected

NativeP24 scheduling fixed compiler-visible accumulator writes by separating the native and integer K loops. Integer correction still waits0 and rescales each K128. A controlled next consumer can use two distinct INT32 fragments, issueK0 thenK1, wait1 and consume only completedK0 whileK1 remains pending. IssueK2 into the oldK0 fragment with MMA ScaleOutZero, wait1 and consumeK1, then similarly issueK3; finalwait0 consumesK3. FP32 d updates remain in K0/K1/K2/K3 order and use the original four independent scale entries. Pairing within eachK128 and output recombination stay unchanged.

Use constant unrolled K indices or templated references, so indexing the two fragments cannot create dynamic local arrays. Initialize/fence both fragments before the pipeline. Keep the official CuTe accumulator-fence hints; do not remove them to silence a warning. PTX requires a hardware wgmma.fence between any thread register access, including a read during restoration, and a later MMA using those registers. CuTe fence_operand is only an empty-asm compiler hint; it does not replace this hardware fence. Fence each issue, including K2/K3 reuse after restoration. No CPU access to an in-flight fragment. Release each TMA slot only after its corresponding committed group is known complete. Inspect C7515 and actual stack/spills/regs before interpreting source-level overlap. P28's initial compile-only preflight lacked the read-to-reuse hardware fences and was archived without any device execution. This is an attempt to hide conversions and memory waits, not a claim of fewer integer MMAs or fewer source-level wait instructions.

Reference: [PTX8.7 WGMMA ordering/commit/wait](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations). Numerical and synchronization acceptance must be the new candidate's own, even if all printed errors equalP27.

## Cooperative preprocessing with exact phase boundaries — P29 measured

P27 still has separate partial reduction, inside finalize, joint pack and corrected-compute kernels. A distinct launch-count control can fuse only the first three with cooperative grid-wide barriers. Keep288logical pack blocks; the first256blocks perform the existing partial reduction using slab=block%8/chunk=block/8, synchronize the entire grid, the first128perform the existing8-column finalize, synchronize again, then all288execute the unchanged joint pack. Keep corrected as a separate kernel. Every thread in every block must participate in both grid barriers, including inactive phase blocks. Shared maxima/scratch must not be reused before the appropriate barrier.

Use cudaLaunchCooperativeKernel on the caller's stream. Query cooperative support and cudaOccupancyMaxActiveBlocksPerMultiprocessor during init outside timedrun/capture, and require grid<=activeBlocksPerSM*SMcount for each compiled variant.288blocks on132SMs need at least3resident CTAs perSM; do not infer that the fused kernel retains P27's standalone packer resources. No global spin barrier, host sync, hidden stream, caching of input-derived data or allocation inside run. Actual Graph capture/replay and two-stream behavior must be tested on localCUDA12.4, not inferred from general API documentation.

An optional later full fusion would use128resident CTAs, loop over the256reduction/288pack logical blocks, then after a third grid barrier reinitialize shared storage and run the original128corrected output tiles. Its dynamic shared requirement becomes196,736B and the pack phase loses the standalone packer's multi-CTA occupancy. It is a separate resource tradeoff, not a free removal of all launches. It needs independent numerical/timing evidence and new scoped sanitizers, not merely source equivalence.

Reference: [CUDA12.4 grid synchronization and occupancy](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#grid-synchronization), [CUDA12.4 execution API](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__EXECUTION.html). These support feasibility constraints, not a speed prediction.

P29 implements this first preprocessing-only fusion,52registers/65,576B dynamic shared/0stack-spill and actual3CTAs/SM on132SMs. Ordinary independentABBA versusP27 is39.405/39.425us BF16/FP16,3.967/3.580percent faster. Own synchronization validation is recorded separately, never inferred from source equivalence. SeeEXPERIMENTS.md andp29_abba.json.

## Single-CTA exact exponent reduction, then one grid barrier — P30 measured

A new reduction control could use the first128of288cooperative CTAs to compute8complete column exponents per CTA, eliminating the global partial/finalize boundary while retaining per-column semantics. With256threads, A usesk=block*8+tid%8 androw=tid/8+32*j forj0..31; reduce the four row groups inside each warp usingxor8/xor16, write8maxima per warp, then combine8warps percolumn. ForB, warp0..7 owns one of the8Krows, each lane loops over1024columns withstride32 and reduces its maximum across the warp. After a block barrier, the first8threads finalize the same log2/rn exponent from A/B maxima. All288CTAs, including inactive reduction blocks, execute one unconditionalgrid.sync before the unchanged joint pack. Keep the old workspace layout even if partial storage is unused, preserve the native consumer and all scale/cast math.

This trades fewer global phase barriers and partial traffic for longer per-thread A reductions and half-used A memory sectors (8FP16/BF16elements per contiguous segment). Only128CTAs perform the reduction. P30's completed ordinaryABBA versusP29 is37.721/37.854us BF16/FP16,4.512/3.742percent faster, and its own Graph/fresh stress/scoped tools pass. IndependentGraph versusP27 still regresses1.647/2.027percent. RetainP30 for ordinary research, P27 forGraph; seeEXPERIMENTS.md. A separate64CTA/K16 control would expose a different coalescing/parallelism tradeoff and remains unimplemented. No bandwidth causality or physical floor follows from these timings.

## Conditional identical-scale integer restoration — P31/P32/P33 measured

[UNIFORM_SCALE_DESIGN.md](UNIFORM_SCALE_DESIGN.md) develops an exact guarded alternative toP28's two live fragments: preserve all current quantized inputs/eight dots, but when the four currentK128restoration exponents agree for both output halves and lie in[-126,103], accumulate inoneINT32fragment over allK before restoring once. The worst paired integer bound is16,516,096<2²⁴; all intermediate restoration values are exactly representable under the guard. Different scales/exceptions retain the original perKpath. The CPU arithmetic control has zero bitwise differences on its generated tuples and catches a differing-scale negative control. This is not real-matrix or hardware acceptance. P31/P32/P33 now implement this guard with different predicate/retirement scheduling. All ordinary78checks pass; P33 independent ordinaryABBA improves2.48/2.20percent overP30 but Graph remains0.61/0.28percent slower thanP27. P31 alone has fresh9-pattern108/864stress, not sanitizer acceptance. No validation transfers. SeeEXPERIMENTS.md and their manifests.

## Wider quantization slabs are a different numerical algorithm

M9's passingK256/K512 screen is preserved separately inWIDE_QUANT_DESIGN.md andm9_manifest.json. Unlike the first two controls, it changes quantization scales and FP32 rescaling order, so equality to P27's printed errors is not required, but the original gate and fresh adverse cases are. A one-plane staged wider producer may reduce correction rescalings but increase raw reloads or reduce packing parallelism. Do not prioritize it merely because a numerical model passes, and do not restore rejected coarseinside.


## P34 single-dot-only guard after three-candidate reassessment

P31/P32 regress; P33 independent gain stays below3percent. OwnP33 consumer profile, prior failed schedules and primary PTX/Hopper/Nsight guidance have been reexamined before this new control. P33 has198registers,0measured local sectors, issueactive34.874percent on the profiled balancedBF16 input; this does not establish a physical floor. Actual timing-input metadata shows both single-dot groups eligible for every CTA, while paired groups have mixed eligibility. P34 enables the common-scale path only for groups0/1 and retains baselineperK for pairedgroups2–4. Keep all eight dots, same quantization/preprocessing, immediate slot retirement and hardware fences. Remove unreachable paired instructions only from the guarded branch and restrict host eligibility diagnostics to the enabled groups. P34 is now measured: ordinaryABBA versusP33 regresses0.956/0.874percent, all78ordinary checks pass; reject with no further Graph/stress/sanitizer claim. P33 now has its own Graph78, fresh9-pattern108/864stress and scoped tools36/72pertool, and is the ordinary research reference; P27 remains Graph reference. Seeuniform_ablation.json.


## Requantize existing planes to a common K512 scale — M10 numerical screen

REQUANTIZATION_DESIGN.md distinguishes doubleRN of existing INT8 values fromM9's directly quantized wider slabs. The negative control RN(RN(1.49)/2)=0 versusRN(1.49/2)=1 makes equality unsafe. A new60-configuration three-arm screen compares baselineK128, exactwideK512 and doubleRNK512 on shared new seeds5201–5203, unchanged per-column inside/seven main/eight corrections. Only after its matrix gate passes should a separate hardware control append a cooperative barrier and216logical requantization blocks; this adds about9.44MB read/write plus metadata/barrier costs, which must count in full calls. No hardware or speed conclusion follows from a numerical pass; preserve P33/P27 and require the new retained candidate's own checks.


M10 now passes all180rows, but initialP35 and vectorizedP36 hardware full calls68.270/68.421us and44.807/44.585us are slower thanP33. All78ordinary gates pass, no broader validation transfer. After three failed controlsP34/P35/P36, ownP36 two-kernel profile, review and primary-source re-search are complete (EXPERIMENTS.md). P37 tests only skipping a tile whose four stored exponents already agree, for which requantization is identity. No observed skip rate claimed, still pay second grid barrier. P37 now passes78ordinary checks but full43.084/42.980us still regresses versusP33; reject with no broader validation. P33 remains ordinary reference, P27 Graph.


## Exact reduction control P38, completed

The measured P35/P36/P37 post-pack common-scale approach remains rejected. First collect a separate ownP33 full two-kernel profile, retaining its existing consumer-only profile and separating profiler timings. Then consider four independent fmax accumulation chains in the exact per-column reductions before their final max combine. No extra grid barrier or changed quantizer; current single-chain loop is unroll4 but still has data dependencies. This is a compiler/ILP hypothesis, not an established bottleneck or speedup. Preserve global per-column semantics, work/cost accounting and all new candidate-specific checks. P38 is now implemented and measured: ordinary78 passes, independentABBA BF16 lowers0.285percent but FP16 rises0.244percent; rejected. SeeEXPERIMENTS.md andp38_abba.json.


## Paired packing P39 and reassessment

P33 full two-kernel profile and separate warp0 phase diagnostics are archived; all nonwarp0 post-grid timer boundaries are excluded from interval interpretation. Ten trace cases retain zero output bitwise differences but are diagnostics, not candidate acceptance. P39 computes both transformed high-plane maxima before quantizing either, adds36B scratch and preserves the existing quantizer/transposes and all dots. Independent ordinaryABBA againstP33 lowers1.234/2.735percent BF16/FP16; Graph againstP27 lowers1.762/2.777percent. Its own validation scope and final retention are recorded inEXPERIMENTS.md andp39_validation.json; never transfer certificates.

After P37/P38/P39 stay below3percent gains, ownP39 two-kernel NCU, failure-history review and new primary CUDA synchronization/warp-primitives search/read are complete before anyP40. Aggregate prep barrier ratio4.419823 is higher thanP33's4.108539 despite lower full latency; no source-level barrier count is a dynamic stall-time proof. Consumer remains198registers and zero measured local sectors, about19.8us under profiler; not a physical floor.

## P40: separate paired transpose lifetimes, measured

P39 still quantizes/stores its two planes sequentially through the same two B transpose arrays. A paired-plane helper also retains a retirement barrier after each plane even though its maxima and FP32 values are no longer overwritten. A new control could give paired B planes four nonoverlapping transpose arrays, quantize both, synchronize once, then publish both planes to their unchanged transposed global addresses. Raw group retains its existing helper/barriers. A paired path can omit both post-quantization retirement barriers because it writes distinct global bytes, never overwrites shared maxima/values and has no later shared consumers; the existing max-publication/finalization barriers remain unconditional.

B staging would grow from42,056B to51,272B, below A's65,612B already reserved by the joint kernel. Thus source layout suggests no larger joint dynamic shared request, but actual compiled resources/cooperative residency must still be checked. Inside paired storage source-level barriers would become2A/3B versusP39's4A/6B; not a speed prediction. In B, the shared publication barrier must be after both quantizers and before any cross-warp transpose read. Distinct arrays prevent the first plane's readers racing with second-plane writers. Preserve all casts, per-tile exponents, round-to-nearest/clamping/fallback, native and correction planes, proxy fences and full preparation/descriptor cost. Numerical inputs/reference/gates unchanged. New candidate requires its own ordinary, Graph, fresh stress and scoped tools before retention. NoP40 source, binary or measurement exists at this checkpoint.


P40 is now measured: ordinary gains0.848/0.104percent BF16/FP16, independentGraph1.319/0.684percent; no own fresh stress/tools, not retained overP39. IndependentP41 hardwareREDUX andP42 native-high exponent controls yield mixed/negligible full-call changes. Three sub3percent controls triggered ownP42 reprofiling, primary-source research and failure-history review beforeM11. SeeEXPERIMENTS.md andHARDWARE_REPORT.md. FrozenP39 sources/binary restored; no production change.

## M11 completed numerical screen; next independent hardware representation

BF16_CORRECTION_DESIGN.md andm11_summary.json retain72fresh configurations/216paired rows, all fixed-floor passes. Both original input dtypes and all eight standard correction terms remain. UnscaledBF16correction lowers eF on every paired case; scaled2^16 arm gives no material floor benefit here and has an explicit finite-high overflow counterexample. The model is FP32bmm/add with TF32disabled, not Hopper WGMMA or performance evidence.

Next implement an independent native-main/BF16-correction hardware control, retaining full-call accounting. Prefer unscaled residual for the first control, to isolate removal of adaptive quantization/restoration without fixed scaling overflow. Need explicitly test native BF16 underflow/subnormal behavior: the model cannot certify hardware flush/accumulation semantics, and a later protected scaling path must pay its own detection and restoration costs. Use four49,152B operand slots, prove phase/barrier reuse for all48K128steps, keep both paired terms and their fences before register reads/reuse. Independently validate exact1024inside exponents, original-inputFP64 gates for bothdtype, compiled resources, ordinary/Graph costs and fresh multistream/tools before retention. More16bit traffic and more MMA work may regress despite simpler preparation. No P43 implementation or acceptance at this checkpoint.


## P45 complete checkpoint and next unimplemented reuse control

P43/P44 improve hardwareaccuracy but regress againstP39. M12's72fresh configurations/216rows plusfinite-high scalar controls support nativep11FP16/p8BF16scaled lows, not arbitrary-range/hardware acceptance. P45 aliases nativehigh planes, uses11MiBplane data/two descriptors, preserves all seven main/eightcrossdots. Own independent ordinaryABBA lowers3.318/2.464percent BF16/FP16, Graph2.995/1.894percent, originalFP64 gates pass. Ownfresh9-pattern108configurations/864interleaved stream/Graphcalls, guards andthree scopedtools36/72each pass; racecheck0warnings/hazards. RetainP45researchreference, no production change or goal completion.

OwnP45two-kernelNCU, primary re-search/read and failedhistoryreview are complete before another hardwarecandidate. Fullprofile/timing exclusion and EmuGEMM/Twill/CUTLASS applicability are inEXPERIMENTS.md andLITERATURE_UPDATE_P45.md. P45's lower registers/plane bytes do not establish a physical floor or causal decomposition of runtime.

Next first model a corrected-native-leaf recombination: each transformed native main product is combined with its one/two scaled cross terms, then enters the existing seven-product Winograd recombination. Preserve all eight correctiondots and exact original gate; this changes FP32 addition order, so require its own paired numerical screen and same-inputP45-style control. Only then implement separate hardware reuse from frozenP45: use the same loaded high operands for its main and correction dots, keeping eleven-plane storage/descriptors for the first control. Model extra FP32accumulator pressure, all ready/free/fence lifetimes, bytes and28K128producersteps/88TMAcommands perCTA versus48/120 before drawing performance conclusions. Counts are source hypotheses, not dynamic or measured savings. Own precision/ordinary+Graph/freshstreams/scopedchecks remain mandatory ifretained. No M13/P46 implementation or numerical/hardware/performance claim at this checkpoint.


## P46完成后的下一消融

M13/P46已完成独立验证并保留P46，详见LEAF_REUSE_HARDWARE.md。普通降低13.69/13.80percent、Graph14.69/14.63percent，相对同轮P45；仍约30.3us、未进入生产。自身profile不证明下界。下一P47保持P46消费者及65,536B预处理共享请求，直接在每个元素变换后写原生高部/缩放低部，移除已失去跨线程用途的FP32暂存与barrier。DIRECT_NATIVE_PACK_DESIGN.md记录完整假设、原始资料和验收要求。共享请求量缩减单独后续消融；不能将P25/P26INT8量化时代的结果直接迁移。


## P47完成后的下一消融

P47自身完整精度/指数、独立ordinary+GraphABBA、新种子108/864双流保护区、三工具36/72各自通过，另54配置全11平面位比较零差异；已保留P47。普通相对同轮P46降低5.158/5.078percent BF16/FP16，Graph5.815/5.540percent，仍约28.8us未达生产性能。自身NCU/rawmetrics/源码生成身份完整归档，无物理下界结论。

下一P48只修改cooperative函数attribute、occupancy查询和launch的动态共享request65536→288B，足够64A/8Bfloat最大值。P47设备计算和消费者逐字保留，其他诊断分配不混作实际cooperative字节；实际容量/完整成本必须实测。现有288CTA已能驻留，高capacity不等于更高有效并行或更快。独立候选需要自己所有精度/Graph/新种子工具证据。gen已准备但未应用，见COMPACT_NATIVE_SCRATCH_DESIGN.md。新inside列几何或消费者调度另做后续，不与容量消融合并。


## P48已拒绝，转入M14数值筛选

P48普通BF16仅降低0.313%、FP16慢0.080%，Graph两种慢0.591/0.522percent。容量3→5但四函数设备指令一致，未产生值得保留的完整成本改善；无自己的压力/工具验收。已逐字恢复P47工作源码/二进制并另跑普通78与指数复核，全通过。

下一M14测试新种子9301–9303、12模式两类型，四臂：fullK128叶（对齐M13同输入控制）、fullK64叶、K64块输出累加、K64按产品直接更新四输出。保留所有7+8点积和原始门槛，先判断分K重组数值风险。只有数值通过后才考虑新的原生高低部片上生成N1；FUSED_NATIVE_DESIGN.md列出raw/TMA/shared/proxy/累加器资源与F13失败风险，不得跳过实际硬件验证或宣称融合必快。


## M14通过，下一项为原生融合N1硬件原型

M14全部72新配置/288条通过原始门槛及固定floor，两个同输入控制与M13匹配。streamK64最大FP16/BF16eF2.06438537745e-5/9.44446119237e-6；每类型33项误差下降、3项极小增加，不能宣称所有输入改善。数值模型不是WGMMA或性能证据。

下一原型放在独立fused_native研究目录，保留P47完整参考。128合作式CTA先精确完成1024inside指数并grid.sync，随后rawTMA+CUDA生成原生高低部+TC累加K64流式重组；全部7主项/8交叉不删。先验证两个原始输入描述符、四象限与MN/K swizzle映射、regular/async proxy可见性、raw/leaf槽位和双fragment读后复用fence。实际共享/寄存器/合作容量必须检查，名义144KiB不是编译资源结论。F13失败/重复变换与更频繁交接风险保留；不保证快于P47，更没有生产性能验收。模型结果m14_summary.json及m14_manifest.json，方案FUSED_NATIVE_DESIGN.md已冻结，不改写原文，新增硬件设计单独记录。


## After N1–N3: proposed P49 global all-plane reuse

Allthree raw native fusions fail full performance despite own layouts/original
gates/ordinary/Graph pairs. Ownprofiles and exact scopes are in../fused_native/REPORT.md;
no production change, P47 retained. N2 removes sevenfold withinCTA tuple scaling;
N3 caches exact inside after grid sync. Neither removes8xA/16xB spatial replication.
The12x emitted-native-store source count is not raw-load/physical-traffic scaling.

Next P49 is prepared but unapplied: same frozenP47 cooperative288grid,
65536Brequest,inside/barriers/tilegeometry,loadfastfallback/RN,all11planes and
byte-identicalconsumer. Group0 writes all7high/4low fromone tuple; Group1/2
packinginactive onlyafterexistinggridbarrier.96packingCTAs may reduce hiding;
querynewresources/layoutbits/ownoriginalgates/fullordinary+Graph/freshness/tools
before retention. SeeALL_NATIVE_PACK_DESIGN.md. No P49 implementation binary,
measurement or acceptance exists at this checkpoint.


P51 premeasurement proposal prepared after P50 acceptance/reassessment:
ALL_NATIVE_PACK_GRID_DESIGN.md +generate_all_native_grid.py. A16x128/B32x32,
384activepackingCTAs, cooperativegrid384/allthreadsunconditionalbarrier;
actualresidentcapacitymandatory, Bfactorlane%4 andcomponentdiagnosticgrids
128A/256Brequiredforbounds/coverage. Consumer/pow2byteexactP50. IsolatedCPU
generationonly atthischeckpoint; own54freshbit+18edge32Kpackingchecks mustpass
beforefirstP51MMA. NoP51speed/precision/toolclaim orcertificate transfer.


## P52 rejected; P53 premeasurement full-leaf parallelism

P51 is the fully validated retained research reference. P52's own nativeN16
layout/78originalgates pass, actual95registers/82048B/twoCTAperSM, yet independent
ordinary/Graph calls regress41–42percent to~39us. Own two-kernelNCU/profile and
all failure scope are inCONSUMER_N16_HARDWARE.md; no freshstress/tools. Restored
four P51 sources/binary exactly and own new-process78gates passed.

After P50/P51/P52 sub3percent results, reviewed ownP52profile, frozenhistory and
fresh primary CUDA12.4 occupancy/stream-ordering plus PTX8.7WGMMA requirements.
LEAF_PARALLEL_DESIGN.md prepares P53: preserve64x32/fullK512/nativep11p8/all15dots,
oneleafperCTA,grid896,two49152Bslots,Shared98432B; append7MiBleafworkspace and
same-streamFP32recombine, allthree kernels/costs in ordinary and Graph timing.
ScalarCPU RN dependency model1048576bitchecks0errors with changed-order negative
control34042differences. IsolatedCPUgenerator checks only; no P53 hardware claim.
Own six newly seeded pre-MMA layouts/all896CTAs/four-sequence reuse, seven-plane
coordinate coverage/guards and GPU recombination-vs-independentCPU oracle must
pass before any new MMA. Then originalgates/exactinside/independent ordinary+
GraphABBA; if retained, own new12pattern dualstream/guards/scopedthree tools.
More occupancy is not guaranteed speed; source payload counts are not physical
traffic, no floor established. Production/library unchanged, goalactive.


## P53 rejected; P54 prepared native N64 shape control

P53 is now measured and rejected. Own six fresh synthetic layouts415236096uint16
comparisons/11010048leafcoordinates and three freshGPU scalarrecombine3145728
outputbitchecks/guards pass before any MMA. Standard78/79872originalgates/exps
pass; consumer74regs/recombine18/zero spill, actual2CTA/SM. Independentordinary
P51->P53 BF16/FP16 27.676800/27.467400->41.052641/41.138800us (48.329/49.773%
slower); Graph27.535360/27.344280->37.369440/37.463359 (35.714/37.006%slower).
Eachmode312gates/319488inside, unlocked/noCI, noownfreshstress/tools. Ownthree
kernelNCU prep16.992us/52regs,leaf23.296us/74regs/98432B/sharedlimit2/issue25.172080%,
recombine6.240us/18regs,all0localRW. Profilertimingsexcluded/notABBAdecomposition.
P51sources/binrestoredbyteexact and newprocess78gates/insidepass.

AfterP51/P52/P53sub3percent, ownprofile/history and newlysearched/readprimary
CUTLASS EfficientGEMM and CUDA12.4Hopper guide informLEAF_N64_DESIGN.md. P54
preparedonlyfromfrozenP53: native64x64/fullK512/twoslots65536B/Shared131200B,
grid448,d/r32,all7+8dots,p11/8/inside/originaldtypes/RN unchanged. Three kernels,
7MiBleaf workspace and byte-identical scalar recombine preserved; no shortcut
costaccounting. NominalTMA2/3P53/instructions1/2,notphysicaltraffic/speedproof.
IsolatedCPUgenerationonly, noP54sourceapplication/build/deviceexecution.
Ownnewpre-MMAlayout+scalarGPUrecombine first, thenoriginalgates/exactinside,
independentP53/P54shapecontrol andP51/P54retentioncomparison ordinary+Graph;
ownfreshdualstream/guards/tools if retained. CurrentP51,productionunchanged.

## P54 closed; M15 compensation representation screen

P54 own layout/GPU scalar recombination/standard78 gates pass, but both independent
shape and retention comparisons lose ordinary and Graph latency. Own profile and
full timing scope are frozen in LEAF_N64_HARDWARE.md/p54_hardware.json. Restored
P51 sources/binary byte-exact and new-process78 original gates/exactinside pass.
No P54 fresh matrix stream/tools certificate; rejected. Earlier prepared-only
paragraphs are historical state at proposal creation, superseded by these results.

After own P54 profiling, fresh primary EmuGEMM/PTX reading and M8/M9/M12/M14 plus
P35–P39 review, FINE_QUANT_SCREEN_DESIGN.md fixes M15 before measurement:84 newly
seeded configurations, original12 patterns plus two row/column scale patterns,
five arms crossing coarse/fine fullK512 INT8 geometry with post/leaf recombination
and native leaf control. All7+8dots, original inside/input FP64 gates and stricter
floors retained. Direct fine-group packing is distinct from old requantization.
Numerical results cannot justify hardware retention or claim speed; complete prep,
metadata, reload, transpose, conversion and synchronization costs remain unresolved.

## M15/M16 measured numerical results

M15 complete420rows: all original gates pass; fine84 configs also pass fixed
floors; coarseFP16 has six stricter-floor negative rows (three inputs x orders).
M16 complete850rows/exit0:84 fresh plus1 old replay, all original gates and fine/
native floors pass.425 paired eF equalities,zero changed correction bits; M15's
zero product scales align with zero integer accumulations.5409scalarbits prove
combined-exponent model including a constructed underflow negative control, not
native CUDA execution or measured matrix accuracy gain. Reports/manifests frozen.

Next is a concrete direct fine-group packing/consumer proposal, not another native
leaf shape. Preserve all7+8dots,both inputs,exactinside,original gates/fulltiming.
Fine metadata must vary by output row/column; completeK512 INT32 accumulation has
own bound. Avoid double quantization; cost raw reloads/reductions/transpose/metadata.
Actual compile resource capacity and own pre-MMA packing/TMA probes remain gates.
P51 remains unchanged; no new GPU hardware performance claim from these models.

## P55 prepared direct fine-group hardware design

DIRECT_FINE_QUANT_DESIGN.md fixes the direct fine producer/consumer design after
M15/M16, before C++ generation/build/device execution.128 cooperative CTAs x768
threads, exactinside first256 then one all-thread grid barrier,64 B512x8 tasks+
64 A8x512 tasks, two original-tuple passes, seven native/nine INT8 planes per side,
fullK512 integer exponents9x64 eachside. Consumer retains P39 native64x32/grid128,
eight24576B slots plus432B onchip fine metadata,fullK512 all eight integer cross
dots in five groups,post correction order. Proposed shared55808Bprep/197248Bconsumer,
workspace12329472B,two kernels/four descriptors; actual compile capacities unknown.
CPU geometry/count checks cover both262144element packings and262144output fine
metadata coordinates.352vs480sourceMMA instructions/CTA, but source TMA bytes same
asP51 and commands15360vs11264; more batches and reload/reduction can lose. No
C++ source application/build/GPU result. Own packing/scalar/TMA probes are required
before first MMA; ordinary/Graph full costs and fresh stress/tools before retention.

## P55 rejected; reassess retained native schedule before another candidate

P55 own96 packing/scalar/exponent and six mixed TMA pre-MMA checks pass; standard78
and own84 fresh original-input single-stream gates/guards pass. Ordinary/GraphABBA
312 original gates/319488inside per mode pass, yet full latency regresses65–67/
62–63percent. Own two-kernel17pass NCU is diagnostic only; no own dual-stream/reuse
stress or scoped tools. Rejected, four working sources/bin byte-exact P51 restored
from retained commit plus frozen binary; new-process78/79872inside pass. See
DIRECT_FINE_QUANT_HARDWARE.md and own hardware audit/manifests. Production unchanged.

LITERATURE_UPDATE_P55.md records own profile/history and fresh primary CUTLASS plus
tensor-core model research after three no3percent results. P51 native representation
is the evidence-supported starting point; don't repeat fine INT8 solely for fewer
MMAs or treat scale-underflow as this batch's matrix error. Next investigate static
seven-leaf consumer dispatch against the existing dynamic step loop, first checking
prior controls for duplication. Preserve leaf order0,1,5,6,4,2,3, FP32 RN dependency
graph, fullK512/all7+8dots/exactinside and all hardware fences/slot release ordering.
Potential code/register/cache growth can regress. No P56 design/source/binary or
speed claim yet; own layout/lifetimes, actual compiled capacity and original gates/
independent ordinary+Graph full costs mandatory, then own streams/guards/tools if
retained. No floor/exhaustion established; goal active.

## P56 own full acceptance; retained research reference

Exactly one consumer unroll1→7 directive from P51, all other source bytes unchanged.
Own six pre-MMA full-grid TMA cases415236096values/guards pass; standard78 and
Graph78 original gates/79872inside per mode pass, printed eF equal P51. Independent
ordinary gains8.740/8.748percent BF/FP, Graph8.841/8.306percent, all312gates/
319488inside per mode pass. Own fresh14pattern168configs/1344two-stream independent
workspace/direct+Graph calls and output/workspace guards pass; scoped tools each
56configs/112calls zeroerrors, race zero warnings/hazards. P56 now replaces P51
only as research ordinary/Graph reference, production/Auto unchanged. Static prep
instructions identical; consumer128regs versus143, code1600→2048instructions;
source dots/payload/synchronization unchanged. No unique causal attribution/floor.
See STATIC_LEAF_SCHEDULE_HARDWARE.md and own validation/hardware manifests.

Next controlled direction: evaluate producer's dynamic seven-leaf dispatch from
frozen P56, separately from the accepted consumer expansion. Before code, freeze
source/CPU operand-plane+slot/phase equivalence and verify actual compiler changes,
then own pre-MMA checks and independent full-call gates. More code can regress;
no P57 source/design/binary or latency claim exists at this checkpoint. Do not
remove exactinside or any7+8dots to close remaining~25us versus~10us gap. Goalactive.
