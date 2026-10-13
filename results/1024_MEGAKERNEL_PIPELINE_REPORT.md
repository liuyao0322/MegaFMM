# 1024 单launch WGMMA流水实测（iteration30）

新Mega29在1024×1024×1024普通启动下为BF16 **25.1098µs**、FP16 **25.4029µs**，比同一冻结二进制中的旧Mega27分别减少 **19.219%/15.827%** 自身端到端耗时；同行独立调优cuBLASLt为12.1194/9.8550µs，Dense/SW为 **0.482655×/0.387950×**。Graph为24.8368/25.0938µs，Dense/SW为0.456742×/0.376342×。本轮没有快过cuBLAS，20行及60个fresh输入原始FP64数值门限全部FAIL，nonfinite=0。Auto仍使用Dense，持续优化目标未完成。

这次测量针对真正单launch路线：将每CTA分为128 producer和128 consumer，用两个final A7/B7 shared banks与PipelineAsync<2>重叠变换和七个完整P的WGMMA消费。原始FP32-RN S/T依赖、每个最终operand只cast一次、完整K累加、固定U、alpha RN乘法后active beta FMA均保留；原始A/B输入的变换在每个完整run中发生，无global S/T/P和矩阵workspace。native Mega28/29的complete Graph分别捕获一条kernel节点，未把外部预处理放到计时外。

源码checkpoint为`4a655fc7ed4c30497d262c7dd2a82137b2833989`；最终bench SHA256为`ccad7e48ef8b846103f569c54a4d193fa9cc806b188be03dd2413aa47f089175`。原型、正式源码、七源/双bin、编译资源、旧实现保留和全部raw证据由[small_pipeline_profile_iter30.json](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline_profile_iter30.json)绑定，20行及全部200组TIME/60fresh CHECK展开在[small_pipeline_measurements_iter30.json](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline_measurements_iter30.json)。

## 测量协议

H10080GB HBM3，132SM，g08，allocation107488，CUDA12.4.99，CUTLASS3.5.1。两dtype输入、FP32输出，同行ordinary cuBLASLt独立调优，最多32候选、共同8GiB workspace cap。每行20warmup、100calls/group、5groups、3fresh输入正确性检查；记录5个组均值的median、线性插值p10/p90、总体标准差/均值CV。每行的Dense和SW均使用完整调用事件计时；速度比=该行Dense_event_median/SW_event_median。Graph捕获/setup/调优不计入稳态，捕获包含完整同绑定run，普通启动和Graph分开比较。这里“端到端”event包含每次完整算子的全部GPU工作。另记batch提交后同步的host wall除以calls，再取五组中位数：新29普通BF16/FP16为25.18150/25.47569µs，Graph为24.91626/25.16739µs；它含批末同步与提交，不能当作独立单次CPU调用wall latency。clock未锁；环境快照为测量后记录，不证明每个计时组的硬件状态。

普通基线是直接A×B的cuBLASLt。Lt1是动态pack A7/B7→七batch leaf GEMM→combine的多launch SW控制，不能替代真正单launch目标。所有路线使用同一冻结源码/二进制，每行全部fresh inputs与计时组都保留；没有删除波动组。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Gate | WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---|---:|---|
| Mega28 流水64×16 | bf16 | 12.2960 [12.2666,12.3057] | 31.0915 [31.0640,31.1187] | 0.395478 | 0.1565/0.0781 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline28_launch_iter30_full1024.csv:2) |
| Mega28 流水64×16 | fp16 | 9.9162 [9.9058,9.9473] | 31.7434 [31.7202,31.7798] | 0.312385 | 0.1936/0.0958 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline28_launch_iter30_full1024.csv:3) |
| Mega29 流水64×32 | bf16 | 12.1194 [12.1173,12.1473] | 25.1098 [25.0966,25.1274] | 0.482655 | 0.1227/0.0552 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter30_full1024.csv:2) |
| Mega29 流水64×32 | fp16 | 9.8550 [9.8500,9.8656] | 25.4029 [25.3784,25.4298] | 0.387950 | 0.0709/0.0888 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter30_full1024.csv:3) |
| Mega0 原WGMMA64×32 | bf16 | 12.4160 [12.4056,12.4289] | 129.9754 [129.8550,130.0705] | 0.095526 | 0.0864/0.0793 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_mega0_launch_iter30_full1024.csv:2) |
| Mega0 原WGMMA64×32 | fp16 | 9.8662 [9.8530,9.8756] | 130.0397 [130.0283,130.3386] | 0.075871 | 0.1010/0.1504 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_mega0_launch_iter30_full1024.csv:3) |
| Mega27 协作32×32 | bf16 | 12.2848 [12.2822,12.3642] | 31.0838 [31.0685,31.0928] | 0.395215 | 0.3958/0.0353 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_launch_iter30_full1024.csv:2) |
| Mega27 协作32×32 | fp16 | 9.8963 [9.8847,9.9134] | 30.1795 [30.1469,30.1858] | 0.327915 | 0.1388/0.0640 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_launch_iter30_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2368 [12.2207,12.2404] | 19.2739 [19.2627,19.3126] | 0.634889 | 0.0743/0.1290 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter30_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.9018 [9.8488,9.9474] | 19.6486 [19.5613,19.6828] | 0.503941 | 0.4375/0.2779 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter30_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Gate | WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---|---:|---|
| Mega28 流水64×16 | bf16 | 11.2768 [11.2644,11.2850] | 30.6835 [30.6446,30.6904] | 0.367520 | 0.0816/0.0696 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline28_graph_iter30_full1024.csv:2) |
| Mega28 流水64×16 | fp16 | 9.4019 [9.3931,9.4117] | 31.4752 [31.4703,31.5181] | 0.298709 | 0.0882/0.0732 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline28_graph_iter30_full1024.csv:3) |
| Mega29 流水64×32 | bf16 | 11.3440 [11.3311,11.3657] | 24.8368 [24.8255,24.8475] | 0.456742 | 0.1550/0.0388 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter30_full1024.csv:2) |
| Mega29 流水64×32 | fp16 | 9.4438 [9.4273,9.4496] | 25.0938 [25.0854,25.0964] | 0.376342 | 0.1101/0.0210 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter30_full1024.csv:3) |
| Mega0 原WGMMA64×32 | bf16 | 11.2902 [11.2776,11.3516] | 129.9779 [129.8731,130.2259] | 0.086863 | 0.3001/0.1210 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_mega0_graph_iter30_full1024.csv:2) |
| Mega0 原WGMMA64×32 | fp16 | 9.3741 [9.3667,9.3796] | 129.7008 [129.6686,129.7142] | 0.072275 | 0.0608/0.0184 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_mega0_graph_iter30_full1024.csv:3) |
| Mega27 协作32×32 | bf16 | 11.3126 [11.2850,11.3219] | 30.6342 [30.6024,30.6820] | 0.369281 | 0.1489/0.1151 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_graph_iter30_full1024.csv:2) |
| Mega27 协作32×32 | fp16 | 9.5366 [9.5265,9.5378] | 30.2394 [30.1576,30.2666] | 0.315372 | 0.0596/0.1631 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_cooperative27_graph_iter30_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.2323 [11.2184,11.2370] | 16.8362 [16.8332,16.8442] | 0.667154 | 0.0817/0.0297 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter30_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.4781 [9.4646,9.5373] | 17.1894 [17.1733,17.1931] | 0.551390 | 0.3525/0.0528 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter30_full1024.csv:3) |

## 自身耗时与控制比较

| 模式 | dtype | 旧最佳Megakernel（Mega27）µs | 新最佳（Mega29）µs | 旧/新自身延迟 | 自身耗时减少 | 同轮全部SW中最小 |
|---|---|---:|---:|---:|---:|---|
| launch | bf16 | 31.0838 | 25.1098 | 1.237919× | 19.219% | Lt1 19.2739µs |
| launch | fp16 | 30.1795 | 25.4029 | 1.188035× | 15.827% | Lt1 19.6486µs |
| graph | bf16 | 30.6342 | 24.8368 | 1.233421× | 18.925% | Lt1 16.8362µs |
| graph | fp16 | 30.2394 | 25.0938 | 1.205055× | 17.016% | Lt1 17.1894µs |

Mega29在同轮所测0/27/28/29单launch候选中自身耗时最小。Mega28普通BF16与Mega27几乎相同（31.0915对31.0838µs），FP16更慢（31.7434对30.1795µs），没有据此宣称收益。Lt1仍是本轮全部SW对照中最小，为普通BF16/FP16 19.2739/19.6486µs、Graph16.8362/17.1894µs；它使用多launch和global S/T/P，不计为单launch目标完成。此前冻结source23全SW包络在1024为19.6432/19.3498µs Lt1（另一个测量源/协议）；新29没有改善这一历史全SW包络。自己的SW耗时变化是上述比较的依据，Dense分母变化不证明代码收益。

新29减少约五分之一单launch耗时是本次真实观测，不能分解成流水单一机制的因果收益：与旧27相比同时更换WGMMA、tile、producer/consumer分工、shared双bank与资源配额。两新配置仅BN/对应resource budgets不同，但仍有CTA数、shared量及初始寄存器保留的联动，未测ncu动态瓶颈或实际occupancy。本轮没有重试会话先前`InterprocessLockFailed`的ncu。

## 编译资源与合法性

| 配置 | tile / BK | threads | shared动态launch bytes | ELF REG/thread | 初始CTA register pool | producer/consumer quota | role所需pool | 最高producer/consumer R（含隐式宽度） | ptxas stack/spill load/store |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| Mega28 | 64×16 /32 | 256 | 71808 | 96 | 24576 | 64/96 | 20480 | 58/89 | 0/0/0 |
| Mega29 | 64×32 /32 | 256 | 86144 | 160 | 40960 | 64/160 | 28672 | 54/145 | 0/0/0 |

两个dtype编译资源一致。71808/86144B是正式类型`sizeof(SharedStorage)`及dynamic launch参数，含70/84KiB final operands、barrier和128B结构对齐；cuobjdump另报staticSHARED1024B，不与dynamic launch量混写。N16初始pool24576≥20480且64≤96≤96，N32pool40960≥28672且64≤160≤160；全CFG role路径、末尾wait stub与64/128bit隐式register宽度独立审PASS，HGMMA隐式accumulator最高R79/R135。linked四个kernel的逐(address,predicate,instruction)流与最终object一致，N16/N32静态1112/1448条，STACK/LOCAL/spill0。静态条数不等于运行指令量。N16的register/shared条件允许考虑2CTA/SM，N32寄存器保留限制为1CTA；没有把这个资源条件当成ncu实测occupancy。

最初的编译期穷举assert超出NVCCconstexpr预算，正式源码改为每slot四个边缘vector静态assert，独立CPU完整layout/absolute-byteaddress验证仍保留；device路径不因此改变。原型初始consumer160方案在N16改为96是明确授权的安全资源微调，所有配额方向/pool/角色SASS在实际launch前重新核验，未冒险启动不可证明的配额。

## 正确性与接口

默认GPU测试exit0，**2080 CASE/859 PASS**；CPU DAG3+stage6全部通过；Mega29选定范围24CASE/38PASS。新增整数fixed-DAG覆盖prime/drain、2stage phasewrap与多CTA、完整七P/U顺序及padding sentinel，new29随机18groups仅在同M64N32K32atom下比较原始U和最终D，不声称直接导出七P槽逐位比较。Mega28使用N16atom，未以N32atom作为浮点逐位参考。

oldMega0 activeα/β epilogue普通C++编译为先round(beta*C0)，再FMA(alpha,U,...)；新合同要求先round(alpha*U)，再FMA(beta,C0,...)。SASS分别证明两序列，因此旧最终D不能直接作为activeβ逐位oracle。保留最初failed diagnostic，同一原始A/B下alpha1/beta0的Mega0↔29原始U全部逐位PASS，再用该U+原始C0的host volatile float alpha multiply与float std::fma核对新指定epilogue全部PASS；alias/nonalias、K64/128/192、padded/shifted AB16和D4域均保留。没有为了参考contraction差异改变device或原始数值门限。

Graph12fixtures共36freshAB replays全部PASS，每个capture一kernel node；覆盖native、metadata tail回退、beta0/nullC0与activeC0=D。公共Plan每current run先验证checkedspan、overlap、D/C0float对齐与合法inplace；native要求M%128、N%(2BN)、K%64、ldAB%8、AB16，native metadata不适用按共同边界走packed7或Dense，actualAB16重绑定失败走Dense；empty/scale、invalid30预先验证和AutoDense均保留。`algorithm_name()`无current-binding参数，单独名字不能证明一次unaligned rebound实际route；本轮benchmark输入全部cudaMalloc aligned/fulltile，CSVactual对该范围有效。

## 原始数值门限与sanitizer

20row×3fresh=60随机原始输入FP64plain-GEMM门限全部FAIL，nonfinite=0，所有benchmark exit2。两新native的BF16 eF约0.00601、FP16约0.000752，原BF16/FP16 floor仍为0.001/0.0001，加3×同行Dense_eF。结构整数和matched rawU正确性不替代这一认证。

新配置strict专用范围60gates为**36PASS/24FAIL/NF0**：uniform与range各3trial全FAIL，positive与cancellation各3trialPASS；fp16_upper_range因共同certifiedrange边界走Dense的12gates PASS，故60中48native、12Densefallback。旧source29的1290gates与所有旧source23–29证据字节保留，未把旧1290重复测成当前1350全suite。CUDA12.8 Compute Sanitizer2025.1.0分别对28/29执行memcheck/racecheck/synccheck，六项均exit0，0errors/0racehazards/0warnings，两个memcheck零泄漏；checks串行完成。

本轮保留25个旧CUDA TU字节及前期Nsys原始报告/SQLite/stats哈希。前期Nsys的130911/30591ns仅是带instrumentation的单次kernel诊断，不用于评分、cold floor或本轮计时替代。
