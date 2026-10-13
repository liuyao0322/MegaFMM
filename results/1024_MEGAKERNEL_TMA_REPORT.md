# 1024 单launch TMA raw-alias 实测（iteration31）

本轮最佳新TMA配置是Mega31，普通启动BF16/FP16完整调用GPU event中位数为 **25.702400/26.413441µs**，同行独立调优cuBLASLt为12.256960/9.826560µs，Dense/SW为 **0.476880×/0.372029×**。同一冻结二进制中的Mega29为25.239041/25.427520µs，新TMA自身耗时分别增加 **1.836%/3.877%**。Graph下新31为24.929600/25.605760µs，也比同轮29增加0.620%/2.201%。这条TMA路线没有改善此前单launch实现；20行×3fresh=60个原始数值门限全部FAIL、nonfinite=0。Auto保持Dense，目标未完成。

普通启动的完整run在每次调用内从current A/B重新执行两次真实`cuTensorMapEncodeTiled`，随后启动一个kernel。两次encode均留在计时host loop中。事件区间覆盖全部GPU工作，也可能包含host提交造成的GPU空闲；它不单独测量每个CPU encode调用。另记录每组100调用的host wall含批末同步，除以100后再取五组median：新31普通BF16/FP16为 **25.77972/26.49597µs每调用**，同轮29为25.31608/25.50958µs。这是batch-per-call wall，不能解释为独立同步单调用CPU wall。Graph捕获时构造固定pointer的descriptor；replay不重做host encode，但每次从捕获地址读取fresh A/B内容，完整S/T变换、七P和U仍在kernel内。

源码checkpoint `20363e90a7a936e80f471fede0e16b96855ca11f`；bench SHA256 `9e3b6a7245217aa4f728622961e9b45640346fbf4945ed76019eef472c200785`。七源/两bin、初始build失败与修复证明、compiled资源、旧26CUDA字节及全部raw证据绑定在[small_tma_profile_iter31.json](/share/home/liuyao/workspace/MegaFMM/results/small_tma_profile_iter31.json)；20行和200个TIME/60个CHECK展开在[small_tma_measurements_iter31.json](/share/home/liuyao/workspace/MegaFMM/results/small_tma_measurements_iter31.json)。

## 测量协议

H10080GB HBM3，132SM，g08，allocation107488，CUDA12.4.99，CUTLASS3.5.1。BF16/FP16输入、FP32输出，同一源码/二进制；每行ordinary cuBLASLt独立调优，最多32候选、共同8GiB workspace cap。每行20warmup、100calls/group、5groups、3fresh输入；五个组均值的median、线性插值p10/p90、总体标准差/均值CV均完整保留，未删除波动组。Dense/SW=同行独立Dense_event_median/完整SW_event_median。ordinary与Graph分别比较，Graph setup/capture/tuning不计入稳态。未锁clock，环境快照不证明组内clock。Lt1为pack→七batch leaf GEMM→combine的多launch对照，不作为单launch目标完成。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega30 TMA64×16 | bf16 | 12.4394 [12.3379,12.7139] | 27.0643 [27.0385,27.0816] | 0.459622 | 1.4722/0.0764 | 12.52481/27.14900 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_tma30_launch_iter31_full1024.csv:2) |
| Mega30 TMA64×16 | fp16 | 9.9968 [9.9859,10.0099] | 27.4922 [27.4828,27.5302] | 0.363624 | 0.1064/0.0787 | 10.07930/27.57897 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_tma30_launch_iter31_full1024.csv:3) |
| Mega31 TMA64×32 | bf16 | 12.2570 [12.2492,12.2580] | 25.7024 [25.6888,25.7075] | 0.476880 | 0.0352/0.0404 | 12.33472/25.77972 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_launch_iter31_full1024.csv:2) |
| Mega31 TMA64×32 | fp16 | 9.8266 [9.8076,9.8386] | 26.4134 [26.4037,26.4387] | 0.372029 | 0.1412/0.0594 | 9.90316/26.49597 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_launch_iter31_full1024.csv:3) |
| Mega29 SIMT流水64×32 | bf16 | 12.2851 [12.2554,12.2996] | 25.2390 [25.2009,25.2583] | 0.486751 | 0.1636/0.1201 | 12.36321/25.31608 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter31_full1024.csv:2) |
| Mega29 SIMT流水64×32 | fp16 | 9.8944 [9.8783,9.9381] | 25.4275 [25.4249,25.4625] | 0.389122 | 0.2869/0.0706 | 9.97154/25.50958 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter31_full1024.csv:3) |
| Mega27 协作32×32 | bf16 | 12.2374 [12.1994,12.2533] | 30.9149 [30.8886,30.9888] | 0.395843 | 0.1994/0.1542 | 12.30950/30.98796 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_launch_iter31_full1024.csv:2) |
| Mega27 协作32×32 | fp16 | 9.8950 [9.8691,9.9128] | 29.9424 [29.9316,30.0017] | 0.330469 | 0.1995/0.1087 | 9.96555/30.01551 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_launch_iter31_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2746 [12.2536,12.2788] | 19.3731 [19.3625,19.4029] | 0.633587 | 0.0946/0.0944 | 12.34808/19.44336 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter31_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.8557 [9.8467,9.9112] | 19.5011 [19.4798,19.5119] | 0.505390 | 0.3110/0.0795 | 9.92870/19.57520 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter31_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega30 TMA64×16 | bf16 | 11.3050 [11.2794,11.3440] | 26.4531 [26.4201,26.5129] | 0.427358 | 0.2542/0.1590 | 11.41589/26.52554 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_tma30_graph_iter31_full1024.csv:2) |
| Mega30 TMA64×16 | fp16 | 9.5587 [9.4779,9.5748] | 26.8592 [26.8236,26.8619] | 0.355883 | 0.4688/0.0692 | 9.64463/26.93136 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_tma30_graph_iter31_full1024.csv:3) |
| Mega31 TMA64×32 | bf16 | 11.2877 [11.2818,11.3005] | 24.9296 [24.9254,24.9312] | 0.452782 | 0.0763/0.0114 | 11.36149/25.00263 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_graph_iter31_full1024.csv:2) |
| Mega31 TMA64×32 | fp16 | 9.3677 [9.3625,9.3820] | 25.6058 [25.5902,25.6186] | 0.365843 | 0.0935/0.0495 | 9.44167/25.67863 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_graph_iter31_full1024.csv:3) |
| Mega29 SIMT流水64×32 | bf16 | 11.3168 [11.3140,11.3199] | 24.7760 [24.7599,24.7877] | 0.456765 | 0.0261/0.0526 | 11.39366/24.85355 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter31_full1024.csv:2) |
| Mega29 SIMT流水64×32 | fp16 | 9.4368 [9.4275,9.4472] | 25.0544 [25.0527,25.0673] | 0.376652 | 0.0932/0.0271 | 9.51619/25.13400 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter31_full1024.csv:3) |
| Mega27 协作32×32 | bf16 | 11.2790 [11.2752,11.2871] | 30.5814 [30.5644,30.6109] | 0.368820 | 0.0533/0.0680 | 11.35753/30.68041 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_graph_iter31_full1024.csv:2) |
| Mega27 协作32×32 | fp16 | 9.3699 [9.3633,9.3782] | 29.7286 [29.7010,29.7460] | 0.315182 | 0.0705/0.0669 | 9.44759/29.81390 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_graph_iter31_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.2253 [11.2087,11.2318] | 17.0029 [17.0004,17.0217] | 0.660199 | 0.0934/0.0665 | 11.30251/17.08316 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter31_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.4022 [9.3954,9.4081] | 17.2253 [17.2209,17.2290] | 0.545840 | 0.0638/0.0234 | 9.47929/17.30567 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter31_full1024.csv:3) |

## 同binary自身耗时比较

| 模式 | dtype | retained29 µs | 最佳新31 µs | 29/31自身延迟比 | 新31耗时变化 | retained29/新31 batchwall µs/call | 同轮全部SW最小 |
|---|---|---:|---:|---:|---:|---:|---|
| launch | bf16 | 25.239041 | 25.702400 | 0.981972× | +1.836% | 25.31608/25.77972 | Lt1 19.373120µs |
| launch | fp16 | 25.427520 | 26.413441 | 0.962674× | +3.877% | 25.50958/26.49597 | Lt1 19.501120µs |
| graph | bf16 | 24.776001 | 24.929600 | 0.993839× | +0.620% | 24.85355/25.00263 | Lt1 17.002881µs |
| graph | fp16 | 25.054400 | 25.605760 | 0.978467× | +2.201% | 25.13400/25.67863 | Lt1 17.225280µs |

Mega31比Mega30更快，但同轮retained29在所有四种dtype/mode组合中都更快。1.836%/0.620%的小差异不宣称新的性能收益；FP16 ordinary差异3.877%也是回退方向。Lt1仍是所测全部SW控制中自身耗时最小，ordinary19.373120/19.501120µs，Graph17.002881/17.225280µs，但它有多个launch/global packed operands和P，不能替代真正Megakernel目标。比较依据是自身SW耗时，不从Dense分母漂移推代码收益。

本轮确实实现八条TMA原始块搬运及alias复用，保留producer128/consumer128、BM64 BN16/32 BK32、PipelineAsync2、两final A7/B7 banks。A21/S3和B12/T1复用槽位，所有原始四块都在局部寄存器snapshot后才覆盖；原始三槽无需SIMT重新store。原FP32-RN S/T依赖、最终operand一次RN cast、七完整K FP32 P、固定U和alphaRN乘法后active betaFMA保持。A使用K_SW64，B为MN_SW32或MN_SW64。native没有global S/T/P或矩阵workspace。静态8 TMA/14 HGMMA counts不是动态global traffic或性能因果证明；本轮没有ncu occupancy/traffic证据，不将普通与Graph差异全部归因于encode，也不宣称物理floor或优化方向耗尽。

## 编译资源与首launch门限

| 配置 | tile / BK | threads | dynamic shared bytes | ELF static shared bytes / align | ELF REG/thread | initial CTA pool | producer/consumer quota | role所需pool | 最高producer/consumer R（含隐式宽度） | stack/spill load/store |
|---|---|---:|---:|---|---:|---:|---|---:|---|---|
| Mega30 | 64×16 /32 | 256 | 72192 | 1024 /512 | 96 | 24576 | 64/96 | 20480 | 43/89 | 0/0/0 |
| Mega31 | 64×32 /32 | 256 | 86528 | 1024 /512 | 160 | 40960 | 64/160 | 28672 | 46/145 | 0/0/0 |

两dtype的上述资源相同。dynamic launch的72192/86528B含final operands、Pipeline/raw barriers及512B结构对齐；ELF另报1024B static shared，合计73216/87552B。初始per-CTA-owned pools分别24576≥20480、40960≥28672，方向64≤96≤96或64≤160≤160成立。独立审查全部CFG包括predicated BRA、raw/tail wait stubs、64/128bit寄存器隐式宽度、TMA与HGMMA隐式accumulator最高R79/R135；producer/consumer路径互不进入另一role，全部七段CS2R清零在首次HGMMA前。linked四函数逐instruction流与最终object完全相同，静态1224/1552条，STACK/LOCAL0。寄存器/shared约束只是可驻留条件，不能当作ncu实测occupancy。

ELF内compiler debug PTX证明descriptor aggregate为align64的256B参数，两张map偏移0/128，以grid_constant参数地址lowering，无.local descriptor副本；constantbank位置64B对齐。实际extern shared align512，四实例ELF shared align512，实际SASS基础0x400，bank/slot均512B倍数。原型静态布局断言最初误把保留byte swizzle的CuTe flagged placeholder直接作用到element index，导致首build8项assert FAIL。保留全部断言，改为layout_b的raw element offset→2×byte→layout_a swizzle→/2，执行host/device路径不变。实际编译的pinned CuTe CPU枚举A=14336、B16=3584、B32=7168元素和vector8、四个512B绝对起点，physical/vector mismatch均0，旧placeholder比较错10752/2688/5376项。初始失败、CPU实际模板证明、修复后build和独立compiled gate均保留，未绕过断言先launch。

## 正确性、接口与数值认证

默认GPU suite exit0，**2126CASE/933PASS**；selected30为22CASE/38PASS、31为24CASE/38PASS；CPU DAG3+stage6全部PASS。新增整数fixed-DAG覆盖多CTA、K64/128/192/256的prime/drain/phasewrap、padded strides/scalar outputs/alias/β0nullC0及fallback。36组同atom浮点原始U逐位对照分别用30↔28 N16、31↔29 N32，再以独立host volatile float RN alpha乘法→float std::fma检查指定epilogue；不声称直接导出七个P槽逐bit。40次current-map绑定实际覆盖Aonly/Bonly/两者更换、different allocations和different内容、AB+16新phase以及AB+2 Dense重绑定，每dtype/config含7native3Dense。Graph12fixtures共36fresh AB replay，8native4packed7，全部一kernel节点，固定capture binding的fresh内容确实重算。

公开Plan每run先检查span-end/overflow/overlap/D和activeC0 float4对齐/精确C0=D别名；native metadata为M%128、N%(2BN)、K%64、ldAB%8以及tensor-map stride界限和原共同certified range，实际ABnonnull16。metadata不符在共同SW域走packed7，共同certification或current AB16失败走Dense；raw helper native-only拒绝。empty/scale、levels0、AutoDense守合同，invalid32在degenerate前验证。algorithm_name()仍无binding参数，因此名字只证明metadata路由；本轮benchmark的compact cudaMalloc-aligned输入使native CSV标签有效。

20row×3fresh随机原始FP64 plain-GEMM门限全部FAIL/NF0；native BF16 eF约0.00601、FP16约0.000752，BF16/FP16 floor0.001/0.0001及3×同行Dense_eF保持。新strict专用60 gates为**36PASS/24FAIL/NF0**：uniform/range FAIL，positive/cancellation PASS，fp16_upper_range因共同certified range走Dense的12gates PASS，故48native/12Dense fallback。保留source29完整1290与source30新增60各自原身份；source30 full1350仅为源码域计数、未执行，本轮也未执行full1410。结构/bitwise一致性测试不能替代原始精度认证。

CUDA12.8 Compute Sanitizer2025.1.0对30/31分别memcheck/racecheck/synccheck，六项串行exit0、0errors/0racehazards/0warnings，两memcheck为0 leaked bytes；race/synccheck未测leak字段不当成实际0泄漏。旧26个CUDA TU和source23–30证据字节保存，没有改旧实现、扩大数值门限或启用Auto SW。
