# 1024 单launch 三bank vector wait1 实测（iteration33）

BF16最佳新Mega35为25.407679µs，同行cuBLASLt 12.287040µs，Dense/SW 0.483596×；相对同binary Mega29自身耗时变化+0.972%。FP16最佳新Mega35为25.605760µs，同行cuBLASLt 9.854720µs，Dense/SW 0.384863×；相对同binary Mega29自身耗时变化+0.266%。

本轮原始精度认证未通过，不能据性能数据宣布目标完成；Auto仍沿用原策略。

完整矩阵为20行、200条TIME、60个fresh输入CHECK：0个原始数值门限PASS、60个FAIL，nonfinite=0；记录行中0/20通过。结构一致性检查不能替代原始FP64数值认证。

测量源码checkpoint `96938eb558dd031d37ffc6f7f3ff74f57736de9c`；bench SHA256 `8d0d7ad7b728537962c78c880cbb67df8dcef59c283a895764cff4b383d0dd0b`。7源、双bin、编译资源与原始证据见[small_three_stage_profile_iter33.json](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage_profile_iter33.json)，展开测量见[small_three_stage_measurements_iter33.json](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage_measurements_iter33.json)。

## 测量协议

BF16/FP16输入、FP32输出；同一冻结源码/二进制，同行ordinary cuBLASLt独立调优，heuristic参数32、共同8GiB workspace cap。每行20warmup、100calls/group、5groups、3fresh输入检查。median与线性插值p10/p90来自五个组均值，CV为总体标准差/均值；保留全部组。Dense/SW使用同行Dense_event_median/完整SW_event_median。

Mega34/35、retained29及两bank Mega33在每次完整run中从current A/B重新构造变换、计算七个完整K的P和固定U。全部native Mega为单kernel、零矩阵workspace、无descriptor encode。Lt1为pack→七batch leaf GEMM→combine的多launch控制；完整调用包含这些阶段。

Graph捕获固定binding，每次replay仍读取fresh原始A/B内容并重做相应变换。setup/tuning/capture不计入稳态。event覆盖完整GPU工作并可能包含host提交空隙；batchwall是每组100调用含批末同步的host wall除以100再取五组median，不能解释为独立同步单调用CPU wall。未锁clock；环境快照不证明组内clock。

环境记录：H10080GB HBM3，132 SM，host=g08，allocation=107488，CUDA=12.4.99，CUTLASS=3.5.1。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega34 三bank wait1 64×16 | bf16 | 12.2224 [12.1999,12.2397] | 31.5424 [31.5184,31.5485] | 0.387491 | 0.1507/0.0434 | 12.29666/31.61720 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage34_launch_iter33_full1024.csv:2) |
| Mega34 三bank wait1 64×16 | fp16 | 9.8774 [9.8629,9.8931] | 32.1981 [32.1656,32.2536] | 0.306771 | 0.1503/0.1213 | 9.95016/32.26989 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage34_launch_iter33_full1024.csv:3) |
| Mega35 三bank wait1 64×32 | bf16 | 12.2870 [12.2598,12.2967] | 25.4077 [25.3715,25.4470] | 0.483596 | 0.1577/0.1388 | 12.35826/25.49694 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_launch_iter33_full1024.csv:2) |
| Mega35 三bank wait1 64×32 | fp16 | 9.8547 [9.8257,9.8596] | 25.6058 [25.6004,25.6584] | 0.384863 | 0.1727/0.1064 | 9.92823/25.67992 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_launch_iter33_full1024.csv:3) |
| Mega29 wait0 64×32 | bf16 | 12.2298 [12.2237,12.2339] | 25.1632 [25.1439,25.2000] | 0.486018 | 0.0418/0.1015 | 12.30244/25.23570 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter33_full1024.csv:2) |
| Mega29 wait0 64×32 | fp16 | 9.9488 [9.9184,10.0106] | 25.5379 [25.5300,25.5766] | 0.389570 | 0.4667/0.0841 | 10.02095/25.65375 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter33_full1024.csv:3) |
| Mega33 两bank wait1 64×32 | bf16 | 12.5891 [12.3315,12.8619] | 29.0970 [29.0698,29.1073] | 0.432661 | 1.8714/0.0582 | 12.67350/29.18295 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_launch_iter33_full1024.csv:2) |
| Mega33 两bank wait1 64×32 | fp16 | 9.8886 [9.8735,9.9116] | 29.2483 [29.2461,29.2635] | 0.338093 | 0.2030/0.0313 | 9.97306/29.33891 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_launch_iter33_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2416 [12.2263,12.3415] | 19.3162 [19.3099,19.3457] | 0.633749 | 0.4343/0.0893 | 12.32734/19.41566 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter33_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.9414 [9.9282,9.9537] | 19.5859 [19.5767,19.6186] | 0.507581 | 0.1153/0.1014 | 10.02681/19.67946 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter33_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega34 三bank wait1 64×16 | bf16 | 11.2912 [11.2826,11.3080] | 31.3635 [31.3195,31.3756] | 0.360011 | 0.1039/0.0818 | 11.37159/31.44357 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage34_graph_iter33_full1024.csv:2) |
| Mega34 三bank wait1 64×16 | fp16 | 9.4355 [9.4302,9.4423] | 32.0400 [32.0090,32.0616] | 0.294492 | 0.0569/0.0729 | 9.51351/32.13648 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage34_graph_iter33_full1024.csv:3) |
| Mega35 三bank wait1 64×32 | bf16 | 11.3520 [11.3460,11.3638] | 24.9517 [24.9421,24.9667] | 0.454959 | 0.0702/0.0457 | 11.42424/25.02618 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_graph_iter33_full1024.csv:2) |
| Mega35 三bank wait1 64×32 | fp16 | 9.4182 [9.4113,9.4263] | 25.3325 [25.3204,25.3408] | 0.371785 | 0.0704/0.0381 | 9.49173/25.41012 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_graph_iter33_full1024.csv:3) |
| Mega29 wait0 64×32 | bf16 | 11.3229 [11.3167,11.3390] | 24.8694 [24.8536,24.8730] | 0.455293 | 0.0883/0.0408 | 11.39579/24.94556 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter33_full1024.csv:2) |
| Mega29 wait0 64×32 | fp16 | 9.3802 [9.3716,9.3894] | 25.0336 [25.0216,25.0574] | 0.374703 | 0.0834/0.0681 | 9.45037/25.10559 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter33_full1024.csv:3) |
| Mega33 两bank wait1 64×32 | bf16 | 11.3642 [11.3547,11.3959] | 28.6400 [28.6246,28.6586] | 0.396793 | 0.1866/0.0547 | 11.45808/28.73447 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_graph_iter33_full1024.csv:2) |
| Mega33 两bank wait1 64×32 | fp16 | 9.5030 [9.4884,9.5633] | 28.8806 [28.8784,28.8831] | 0.329045 | 0.3672/0.0076 | 9.58855/28.97043 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_graph_iter33_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.3142 [11.3066,11.3196] | 17.1040 [17.0962,17.1146] | 0.661497 | 0.0510/0.0484 | 11.39996/17.19342 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter33_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.4717 [9.4317,9.4813] | 18.0531 [18.0248,18.0723] | 0.524656 | 0.2387/0.1325 | 9.55584/18.15193 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter33_full1024.csv:3) |

## 自身耗时比较

| 模式 | dtype | retained29 µs | 最佳新配置 / µs | 29/新自身延迟比 | 新自身耗时变化 | retained29/新 batchwall µs/call | 同轮全部SW最小 |
|---|---|---:|---:|---:|---:|---:|---|
| launch | bf16 | 25.163200 | Mega35 / 25.407679 | 0.990378× | +0.972% | 25.23570/25.49694 | Lt1 打包一次（多launch对照） 19.316159µs |
| launch | fp16 | 25.537920 | Mega35 / 25.605760 | 0.997351× | +0.266% | 25.65375/25.67992 | Lt1 打包一次（多launch对照） 19.585921µs |
| graph | bf16 | 24.869440 | Mega35 / 24.951680 | 0.996704× | +0.331% | 24.94556/25.02618 | Lt1 打包一次（多launch对照） 17.104000µs |
| graph | fp16 | 25.033600 | Mega35 / 25.332479 | 0.988202× | +1.194% | 25.10559/25.41012 | Lt1 打包一次（多launch对照） 18.053120µs |

## 相同BN32的阶段数比较：35对33

| 模式 | dtype | 两bank33 µs | 三bank35 µs | 33/35自身延迟比 | 三bank自身耗时变化 | 两bank/三bank batchwall µs/call |
|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | 29.096961 | 25.407679 | 1.145203× | -12.679% | 29.18295/25.49694 |
| launch | fp16 | 29.248321 | 25.605760 | 1.142256× | -12.454% | 29.33891/25.67992 |
| graph | bf16 | 28.640001 | 24.951680 | 1.147819× | -12.878% | 28.73447/25.02618 |
| graph | fp16 | 28.880639 | 25.332479 | 1.140064× | -12.286% | 28.97043/25.41012 |

比较依据是SW自身耗时，Dense分母变化不证明代码收益。Mega35与33都是64×32，比较三bank和两bankwait1阶段轴；与Mega29另行比较当前最佳保留路线。Mega34的64×16与33的64×32不同，不能从该比较单独提取阶段数贡献。同atom实现一致性测试分别比较34↔28与35↔29，没有把N16和N32作浮点逐位oracle。Lt1多launch控制不能替代单launch性能目标；性能差异不单独证明某项stall或traffic机制。

## 编译资源与实现检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | producer/consumer quota | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 35 | fp16 | 64×32 /32 | 256 | 129152 | 160 | 40960 | 64/160 | 28672 | 0/0/0 |
| 34 | fp16 | 64×16 /32 | 256 | 107648 | 96 | 24576 | 64/96 | 20480 | 0/0/0 |
| 35 | bf16 | 64×32 /32 | 256 | 129152 | 160 | 40960 | 64/160 | 28672 | 0/0/0 |
| 34 | bf16 | 64×16 /32 | 256 | 107648 | 96 | 24576 | 64/96 | 20480 | 0/0/0 |

dynamic shared与ELF static shared分别记录。静态寄存器/shared/SASS条数只说明编译资源；没有把它们解释为实测occupancy、动态traffic或物理性能floor。完整role-CFG/配额与隐式register-width审查及pipeline prime/retire/tail证明见profile中的独立审核原件。

CPU dag：archived exit=0，测试数=3，[three_stage_cpu_dag_iter33.txt](/share/home/liuyao/workspace/MegaFMM/results/three_stage_cpu_dag_iter33.txt)。

CPU stage：archived exit=0，测试数=6，[three_stage_cpu_stage_iter33.txt](/share/home/liuyao/workspace/MegaFMM/results/three_stage_cpu_stage_iter33.txt)。

CPU 状态模型：archived exit=0，status=CPU_THREE_BANK_ABSTRACT_PASS，states/transitions=14279/27569，randomized sequences/actions=448/115616，phase crosschecks=395523，[three_stage_cpu_state_model_iter33.txt](/share/home/liuyao/workspace/MegaFMM/results/three_stage_cpu_state_model_iter33.txt)。这是抽象状态模型检查，不是硬件memory-order/thread-convergence/liveness认证。

default：exit=0，2262 CASE /1269 PASS，[correctness_three_stage_iter33.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_three_stage_iter33.txt)。
selected34：exit=0，36 CASE /98 PASS，[correctness_three_stage34_targeted_iter33.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_three_stage34_targeted_iter33.txt)。
selected35：exit=0，38 CASE /98 PASS，[correctness_three_stage35_targeted_iter33.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_three_stage35_targeted_iter33.txt)。

新Graph configurations/replays为36/108；全部Graph为168/504。结构fixed-DAG/bitwise/Graph检查不替代原始数值认证。

## 原始数值门限与sanitizer

本轮strict专用范围60 gates：36 PASS /24 FAIL /nonfinite 0；48 native，12 Dense range fallback。BF16/FP16原floor保持0.001/0.0001，门限为3×同行Dense_eF+floor。

历史执行结果按各自源码身份保留：source29完整1290、source30新增60、source31新增60、source32新增60。source30 full1350、source31 full1410、source32 full1470和当前full1530仅是源码full域计数，未宣称本轮执行。

记录6项CUDA Compute Sanitizer运行，其中0项exit/errors/warnings/hazards非零；memcheck leaked-bytes记录为[0, 0]，race/synccheck未测泄漏字段保持未测。实际工具版本和各原始摘要见profile，不用结构测试PASS替代numerical PASS。

旧CUDA TU保存28个，历史raw/profile/CSV逐项绑定。本报告生成器从原始文件填数；独立数据审计仍需另外重算统计量和核对来源，最终审结与目标状态以冻结profile为准。
