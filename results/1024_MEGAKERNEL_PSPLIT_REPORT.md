# 1024 单launch P3+4 双consumer 实测（iteration34）

BF16实际可用最佳新Mega36 23.426239µs，同行cuBLASLt 12.356800µs，Dense/SW 0.527477×；相对Mega29自身耗时变化-7.394%。FP16实际可用最佳新Mega36 23.645761µs，同行cuBLASLt 9.916160µs，Dense/SW 0.419363×；相对Mega29自身耗时变化-7.039%。

预声明20行，实际16行、160条TIME、48个fresh CHECK；完整矩阵=否。原门限0 PASS /48 FAIL，nonfinite=0，实际记录行0/16通过。

完整原始数值认证未通过，不能据性能数据宣布目标完成；Auto沿用原策略。

## 实际缺失范围

下列行未运行，原因与已归档编译安全门限逐项绑定；没有把缺失行当作速度或正确性结果。

| route | dtype | mode | 分类 | 原因 | 门限证据 |
|---|---|---|---|---|---|
| mega37 | bf16 | launch | compiled_safety_blocked | Final37 REG80 STACK128 spill384store/256load+C7512; forcedPlan/nativehelpers disabled. Zero native37 launches; numerical/timing/Graph/SAN37 unexecuted. | [psplit_static_review_initial_iter34.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_static_review_initial_iter34.json) `f23363e85f642e957c605b3988d908c40fe2891c7400e5f8bc907227f37162a8` |
| mega37 | bf16 | graph | compiled_safety_blocked | Final37 REG80 STACK128 spill384store/256load+C7512; forcedPlan/nativehelpers disabled. Zero native37 launches; numerical/timing/Graph/SAN37 unexecuted. | [psplit_static_review_initial_iter34.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_static_review_initial_iter34.json) `f23363e85f642e957c605b3988d908c40fe2891c7400e5f8bc907227f37162a8` |
| mega37 | fp16 | launch | compiled_safety_blocked | Final37 REG80 STACK128 spill384store/256load+C7512; forcedPlan/nativehelpers disabled. Zero native37 launches; numerical/timing/Graph/SAN37 unexecuted. | [psplit_static_review_initial_iter34.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_static_review_initial_iter34.json) `f23363e85f642e957c605b3988d908c40fe2891c7400e5f8bc907227f37162a8` |
| mega37 | fp16 | graph | compiled_safety_blocked | Final37 REG80 STACK128 spill384store/256load+C7512; forcedPlan/nativehelpers disabled. Zero native37 launches; numerical/timing/Graph/SAN37 unexecuted. | [psplit_static_review_initial_iter34.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_static_review_initial_iter34.json) `f23363e85f642e957c605b3988d908c40fe2891c7400e5f8bc907227f37162a8` |

测量源码checkpoint `21f1135c1aa37036d50b5756f9001ff62a0a2f7e`；bench SHA256 `e5004b3fd0c9e12b80bc3ca4f681f0d37b483c015c010ad3b70df13ae2aebc3f`。8源、双bin、资源与原始证据见[small_psplit_profile_iter34.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_profile_iter34.json)，展开测量见[small_psplit_measurements_iter34.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_measurements_iter34.json)。

## 测量协议

完整目标为36/37/29/35/Lt1 × BF16/FP16 × ordinary/Graph。每行20warmup、100calls/group、5groups、3fresh检查，rotate_inputs=1。每行cuBLASLt独立调优，Dense与SW按同行相同ordinary/Graph模式计时，32 heuristic candidates、共同8GiB workspace cap。BF16/FP16输入、FP32输出。median和线性p10/p90来自五个组均值；CV为总体标准差/均值。Dense/SW为同行Dense_event_median/完整SW_event_median。

新P3+4路线采用BM64/BN32/BK32、384线程、两bank，将P3+4分配给两个独立consumer WG。两WG各自完成升序whole-K及每K32两个升序K16步骤，再wait0/release；producer tail和全CTA drain结束后复用typed shared union，发布全部七P后固定U和原alpha/beta epilogue。每CTA独占四象限输出tile；两consumer没有把K拆成浮点分段归约。

本轮实际运行的native Mega36/29/35每次读取当前原始A/B并重新构造相应输入变换，完整调用为一个kernel，零native矩阵workspace，无host descriptor encode。Lt1是pack→七batch leaf GEMM→combine多launch控制，其event和wall包含完整调用。它不能替代单launch目标。

Graph使用捕获时的固定地址，replay读取当前A/B内容并重做变换；三次fresh检查更换内容。setup/tuning/capture不计入稳态。event覆盖完整GPU工作并可能含提交空隙。batchwall是每组100调用含批末同步的host wall除以100，随后取五组median，不能解释为独立同步单调用CPU wall。clock未锁；环境快照不认证组内clock。

环境：NVIDIA H10080GB HBM3，132 SM，host=g08，allocation=107488，CUDA=12.4.99，CUTLASS=3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee。

## 同行Dense组均值的波动

下列行的Dense总体CV至少2%。全部五组均值保留，median、p10/p90和CV均使用这些组；没有剔除较慢组。该波动影响同行Dense/SW分母，不能据加速比变化单独认定代码收益。

Mega36 P3+4 64/96/96 min1 launch bf16：Dense CV=2.450637%，SW CV=0.136245%；Dense median/p90=12.356800/12.782784µs。stdout五个Dense event组均值µs=['12.3568', '12.2189', '13.0518', '12.2592', '12.3792']。
Mega36 P3+4 64/96/96 min1 launch fp16：Dense CV=26.632093%，SW CV=0.116721%；Dense median/p90=9.916160/14.486656µs。stdout五个Dense event组均值µs=['9.88992', '9.89856', '9.91616', '9.93472', '17.5213']。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega36 P3+4 64/96/96 min1 | bf16 | 12.3568 [12.2350,12.7828] | 23.4262 [23.4010,23.4681] | 0.527477 | 2.4506/0.1362 | 12.44348/23.51329 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter34_full1024.csv:2) |
| Mega36 P3+4 64/96/96 min1 | fp16 | 9.9162 [9.8934,14.4867] | 23.6458 [23.6218,23.6836] | 0.419363 | 26.6321/0.1167 | 9.99219/23.72476 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter34_full1024.csv:3) |
| Mega29 wait0 64×32 | bf16 | 12.3101 [12.2814,12.3178] | 25.2966 [25.2597,25.3228] | 0.486629 | 0.1361/0.1097 | 12.38843/25.37170 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter34_full1024.csv:2) |
| Mega29 wait0 64×32 | fp16 | 9.9571 [9.9449,10.0282] | 25.4362 [25.4103,25.4543] | 0.391455 | 0.4807/0.0775 | 10.03255/25.52674 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter34_full1024.csv:3) |
| Mega35 三bank wait1 64×32 | bf16 | 12.1485 [12.1455,12.1707] | 25.3283 [25.3188,25.3445] | 0.479640 | 0.0965/0.0450 | 12.22576/25.40318 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_launch_iter34_full1024.csv:2) |
| Mega35 三bank wait1 64×32 | fp16 | 9.8493 [9.8412,9.8594] | 25.5565 [25.5494,25.5851] | 0.385393 | 0.0817/0.0658 | 9.92248/25.62935 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_launch_iter34_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2534 [12.2441,12.2823] | 19.1619 [19.1355,19.1805] | 0.639468 | 0.1421/0.1089 | 12.33100/19.23848 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter34_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.8496 [9.8358,9.8554] | 19.5370 [19.5174,19.5513] | 0.504152 | 0.1050/0.0822 | 9.92749/19.61523 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter34_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega36 P3+4 64/96/96 min1 | bf16 | 11.3587 [11.3482,11.3632] | 23.0544 [23.0516,23.0593] | 0.492692 | 0.0682/0.0151 | 11.43558/23.13428 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter34_full1024.csv:2) |
| Mega36 P3+4 64/96/96 min1 | fp16 | 9.4355 [9.4236,9.4446] | 23.1571 [23.1338,23.1787] | 0.407457 | 0.0987/0.0857 | 9.50902/23.22741 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter34_full1024.csv:3) |
| Mega29 wait0 64×32 | bf16 | 11.3725 [11.3703,11.3780] | 24.8477 [24.8334,24.8537] | 0.457688 | 0.0338/0.0399 | 11.45118/24.93303 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter34_full1024.csv:2) |
| Mega29 wait0 64×32 | fp16 | 9.6086 [9.6056,9.6196] | 25.1219 [25.1092,25.1367] | 0.382480 | 0.0667/0.0522 | 9.68993/25.20051 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter34_full1024.csv:3) |
| Mega35 三bank wait1 64×32 | bf16 | 11.3114 [11.2999,11.3190] | 25.0096 [25.0051,25.0148] | 0.452281 | 0.0813/0.0191 | 11.38434/25.09540 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_graph_iter34_full1024.csv:2) |
| Mega35 三bank wait1 64×32 | fp16 | 9.5859 [9.5856,9.5886] | 25.3843 [25.3667,25.3880] | 0.377632 | 0.0164/0.0396 | 9.66323/25.46060 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_three_stage35_graph_iter34_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.3158 [11.3091,11.3202] | 17.1027 [17.0997,17.1050] | 0.661640 | 0.0446/0.0137 | 11.38850/17.17694 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter34_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.3808 [9.3742,9.3999] | 17.3526 [17.3406,17.3798] | 0.540598 | 0.1242/0.1058 | 9.45528/17.42453 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter34_full1024.csv:3) |

## 各新路线相对两条保留Mega的自身耗时

| 模式 | dtype | 对照 | 对照 µs | 新路线 | 新 µs | 对照/新自身延迟比 | 新自身耗时变化 | 对照/新 batchwall µs/call |
|---|---|---|---:|---|---:|---:|---:|---:|
| launch | bf16 | Mega29 | 25.296640 | Mega36 | 23.426239 | 1.079842× | -7.394% | 25.37170/23.51329 |
| launch | bf16 | Mega35 | 25.328319 | Mega36 | 23.426239 | 1.081194× | -7.510% | 25.40318/23.51329 |
| launch | fp16 | Mega29 | 25.436161 | Mega36 | 23.645761 | 1.075718× | -7.039% | 25.52674/23.72476 |
| launch | fp16 | Mega35 | 25.556481 | Mega36 | 23.645761 | 1.080806× | -7.476% | 25.62935/23.72476 |
| graph | bf16 | Mega29 | 24.847679 | Mega36 | 23.054399 | 1.077785× | -7.217% | 24.93303/23.13428 |
| graph | bf16 | Mega35 | 25.009601 | Mega36 | 23.054399 | 1.084808× | -7.818% | 25.09540/23.13428 |
| graph | fp16 | Mega29 | 25.121920 | Mega36 | 23.157120 | 1.084846× | -7.821% | 25.20051/23.22741 |
| graph | fp16 | Mega35 | 25.384319 | Mega36 | 23.157120 | 1.096178× | -8.774% | 25.46060/23.22741 |

## 相同BN32预算轴：37对36的实际配对范围

| 模式 | dtype | 64/96/96 36 µs | 64/80/96 37 µs | 36/37自身延迟比 | 37自身耗时变化 | 36/37 batchwall µs/call |
|---|---|---:|---:|---:|---:|---:|

36与37的算法均使用N32 atom。早期96配额候选及128配额修复尝试的编译失败身份保留在归档证据中；最终配置按冻结plan填写：Mega36 final=64/96/96, actualname=megakernel_wgmma_psplit_64x32; Mega37 final=64/80/96, actualname=megakernel_wgmma_psplit_asymmetric_64x32。只有实际CSV出现的配置才有性能结论；任一路线缺失时37↔36预算轴未实测，不能从其它配置或历史数据补出。作用域修复后重编译的最终身份不能被描述成早期失败原件已经通过，性能差异不能单独归因到动态occupancy或stall。收益判断使用SW自身耗时，Dense分母变大不证明优化。本轮仅实际enabled的36执行sameatom检查，以29的rawU及独立host epilogue为参考；37未执行这些检查，没有采用N16输出作逐位oracle。

有12个比较配对缺少实际行；准确键与缺失原因见measurements，没有从其它binary或历史数据补齐。

## 编译资源与作用域

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 源producer/WG0/WG1 quota | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 36 | fp16 | 64×32 /32 | 384 | 86144 | 96 | 36864 | 64/96/96 | 32768 | 0/0/0 |
| 36 | bf16 | 64×32 /32 | 384 | 86144 | 96 | 36864 | 64/96/96 | 32768 | 0/0/0 |

static/shared/SASS及初始寄存器池属于编译后门限和实现证据，不是实测动态occupancy、动态traffic或物理floor。完整隐式register-width/role-CFG、配额方向、pool安全、tail同步、actual object/linked binary对应关系见profile中的独立编译审查。

## 实际CPU与结构检查

CPU dag：实际归档exit=0，测试数=3，[psplit_cpu_dag_iter34.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_cpu_dag_iter34.txt)。

CPU stage：实际归档exit=0，测试数=6，[psplit_cpu_stage_iter34.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_cpu_stage_iter34.txt)。

CPU psplit：实际归档exit=0，status=PASS_CPU_ABSTRACT_PROTOCOL_AND_LITERAL_HEADER_OWNERSHIP_ONLY，states/transitions=126748/278868, random traces/actions=208/105568，[psplit_cpu_state_model_iter34.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_cpu_state_model_iter34.txt)。仅检查原prototype的抽象状态/源码模型范围，不认证最终修复配置的实际寄存器ownership、GPU memory-order或forward-progress；当前编译门限另见profile。

CPU psplit_asymmetric：实际归档exit=0，status=PASS_CPU_SOURCE_NORMALIZATION_AND_HYPOTHETICAL_REGISTER_POOL_ONLY，hypothetical initial80 states/transitions=75/160，[psplit_asymmetric_cpu_state_model_iter34.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_asymmetric_cpu_state_model_iter34.txt)。仅检查原prototype的抽象状态/源码模型范围，不认证最终修复配置的实际寄存器ownership、GPU memory-order或forward-progress；当前编译门限另见profile。

default：exit=0，2298 CASE /1361 PASS，[correctness_psplit_iter34.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_iter34.txt)。
selected36：exit=0，36 CASE /93 PASS，[correctness_psplit36_targeted_iter34.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit36_targeted_iter34.txt)。

新Graph configurations/replays=18/54；全部Graph=186/558。结构检查不替代原始FP64数值认证。

## 原始精度与sanitizer

本轮strict实际30 gates：18 PASS /12 FAIL /nonfinite 0；24 native、6 Dense range fallback。范围回退的PASS不认证native。BF16/FP16 floor为0.001/0.0001，门限保持3×Dense_eF+floor。

历史实际strict按各自源码身份保留：source29 full1290，source30/31/32/33各新增60。历史full1350/1410/1470/1530以及最终当前full1560是未重跑的源码范围计数。早期1590预测包含现已disabled的37，保留为初稿预测，不冒充最终执行范围；本轮只报告实际归档的new36范围。

实际归档3项CUDA Compute Sanitizer运行，其中0项exit/errors/warnings/hazards非零或未记录；memcheck leak bytes=[0]。race/synccheck的未测泄漏字段保持null；各实际工具摘要见profile。

保留旧CUDA 29个、历史证据530个绑定。其中onsite-only证据4个按现场SHA保留，不声称已归档为Git对象。统计、所有原件SHA、当前源码/二进制/Git绑定仍由独立审计另外核验。
