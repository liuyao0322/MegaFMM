# 1024 单launch vector wait1 实测（iteration32）

BF16最佳新Mega33为29.158080µs，同行cuBLASLt 12.266880µs，Dense/SW 0.420703×；相对同binary Mega29自身耗时变化+16.007%。FP16最佳新Mega33为29.407041µs，同行cuBLASLt 9.912640µs，Dense/SW 0.337084×；相对同binary Mega29自身耗时变化+15.767%。

本轮原始精度认证未通过，不能据性能数据宣布目标完成；Auto仍沿用原策略。

完整矩阵为20行、200条TIME、60个fresh输入CHECK：0个原始数值门限PASS、60个FAIL，nonfinite=0；记录行中0/20通过。结构一致性检查不能替代原始FP64数值认证。

测量源码checkpoint `f396662f498af7998b2401da49506184b96904ae`；bench SHA256 `8fe268ef4a42f51bcea88478d3451c69bc963b245a267db288ef24d5de1f3dab`。七源、双bin、编译资源与原始证据见[small_wait1_profile_iter32.json](/share/home/liuyao/workspace/MegaFMM/results/small_wait1_profile_iter32.json)，展开测量见[small_wait1_measurements_iter32.json](/share/home/liuyao/workspace/MegaFMM/results/small_wait1_measurements_iter32.json)。

## 测量协议

BF16/FP16输入、FP32输出；同一冻结源码/二进制，同行ordinary cuBLASLt独立调优，heuristic参数32、共同8GiB workspace cap。每行20warmup、100calls/group、5groups、3fresh输入检查。median与线性插值p10/p90来自五个组均值，CV为总体标准差/均值；保留全部组。Dense/SW使用同行Dense_event_median/完整SW_event_median。

Mega32/33及retained29在每次完整run中从current A/B重新构造变换、计算七个完整K的P和固定U，native为单kernel、零矩阵workspace、无descriptor encode。retained TMA31普通run每次重新执行两张current-pointer tensor-map encode后单kernel，encode留在计时host loop中。Lt1为pack→七batch leaf GEMM→combine的多launch控制。三类路线的完整调用边界均保留。

Graph捕获固定binding；TMA31仅在capture时encode，replay不做host encode，所有路线每次replay仍读取fresh原始A/B内容并重做相应变换。setup/tuning/capture不计入稳态。event覆盖完整GPU工作并可能包含host提交空隙；batchwall是每组100调用含批末同步的host wall除以100再取五组median，不能解释为独立同步单调用CPU wall。未锁clock；环境快照不证明组内clock。

环境记录：H10080GB HBM3，132 SM，host=g08，allocation=107488，CUDA=12.4.99，CUTLASS=3.5.1。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega32 wait1 64×16 | bf16 | 12.2602 [12.2143,12.2815] | 32.5133 [32.4309,32.5974] | 0.377082 | 0.2463/0.2451 | 12.33800/32.60708 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_wait132_launch_iter32_full1024.csv:2) |
| Mega32 wait1 64×16 | fp16 | 9.9190 [9.8888,10.0024] | 32.6182 [32.5146,32.6602] | 0.304095 | 0.6010/0.2011 | 9.99536/32.70793 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_wait132_launch_iter32_full1024.csv:3) |
| Mega33 wait1 64×32 | bf16 | 12.2669 [12.2010,12.3460] | 29.1581 [29.0991,29.2110] | 0.420703 | 0.5802/0.1694 | 12.34423/29.25172 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_launch_iter32_full1024.csv:2) |
| Mega33 wait1 64×32 | fp16 | 9.9126 [9.7627,9.9409] | 29.4070 [29.2097,29.4092] | 0.337084 | 0.8329/0.3283 | 9.98748/29.48424 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_launch_iter32_full1024.csv:3) |
| Mega29 wait0 64×32 | bf16 | 12.1939 [12.1856,12.2134] | 25.1347 [25.0833,25.1519] | 0.485142 | 0.1127/0.1462 | 12.27395/25.21137 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter32_full1024.csv:2) |
| Mega29 wait0 64×32 | fp16 | 9.8282 [9.7932,9.9011] | 25.4019 [25.3112,25.4838] | 0.386906 | 0.4869/0.2986 | 9.90388/25.47930 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter32_full1024.csv:3) |
| Mega31 TMA 64×32 | bf16 | 12.2534 [12.2442,12.2604] | 25.7574 [25.7474,25.7647] | 0.475724 | 0.0605/0.0307 | 12.32357/25.83029 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_launch_iter32_full1024.csv:2) |
| Mega31 TMA 64×32 | fp16 | 9.9030 [9.8894,9.9143] | 26.6182 [26.5992,26.6269] | 0.372040 | 0.1236/0.0504 | 9.97679/26.69285 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_launch_iter32_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2064 [12.1949,12.2125] | 19.3997 [19.3822,19.4055] | 0.629206 | 0.0664/0.0603 | 12.28268/19.48018 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter32_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.9910 [9.9776,10.0064] | 19.6733 [19.6637,19.7636] | 0.507848 | 0.1396/0.2796 | 10.07019/19.77599 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter32_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega32 wait1 64×16 | bf16 | 11.3696 [11.3660,11.3752] | 32.3008 [32.2301,32.3396] | 0.351991 | 0.0363/0.1694 | 11.44650/32.38223 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_wait132_graph_iter32_full1024.csv:2) |
| Mega32 wait1 64×16 | fp16 | 9.4282 [9.4205,9.4387] | 32.3379 [32.3321,32.4600] | 0.291551 | 0.0924/0.2405 | 9.50412/32.43273 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_wait132_graph_iter32_full1024.csv:3) |
| Mega33 wait1 64×32 | bf16 | 11.3456 [11.3284,11.3527] | 28.6048 [28.5974,28.6085] | 0.396633 | 0.1039/0.0199 | 11.42480/28.68783 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_graph_iter32_full1024.csv:2) |
| Mega33 wait1 64×32 | fp16 | 9.4464 [9.4284,9.4586] | 28.9830 [28.9698,28.9942] | 0.325929 | 0.1489/0.0375 | 9.52602/29.07212 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_wait133_graph_iter32_full1024.csv:3) |
| Mega29 wait0 64×32 | bf16 | 11.3187 [11.3146,11.3407] | 24.8544 [24.8457,24.8661] | 0.455401 | 0.1078/0.0403 | 11.39605/24.93586 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter32_full1024.csv:2) |
| Mega29 wait0 64×32 | fp16 | 9.4394 [9.4271,9.4719] | 25.1296 [25.1185,25.1372] | 0.375627 | 0.2136/0.0341 | 9.51675/25.20803 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter32_full1024.csv:3) |
| Mega31 TMA 64×32 | bf16 | 11.3248 [11.3014,11.3271] | 25.0317 [25.0301,25.0405] | 0.452419 | 0.1100/0.0201 | 11.40076/25.11026 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_graph_iter32_full1024.csv:2) |
| Mega31 TMA 64×32 | fp16 | 9.4157 [9.4117,9.4299] | 25.9066 [25.9023,25.9150] | 0.363448 | 0.0939/0.0234 | 9.49331/25.98919 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_tma31_graph_iter32_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.2906 [11.2723,11.3034] | 17.1274 [17.1206,17.1315] | 0.659212 | 0.1264/0.0306 | 11.36935/17.20440 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter32_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.4195 [9.4163,9.4255] | 17.2790 [17.2732,17.2911] | 0.545141 | 0.0453/0.0464 | 9.49989/17.36218 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter32_full1024.csv:3) |

## 自身耗时比较

| 模式 | dtype | retained29 µs | 最佳新配置 / µs | 29/新自身延迟比 | 新自身耗时变化 | retained29/新 batchwall µs/call | 同轮全部SW最小 |
|---|---|---:|---:|---:|---:|---:|---|
| launch | bf16 | 25.134721 | Mega33 / 29.158080 | 0.862016× | +16.007% | 25.21137/29.25172 | Lt1 打包一次（多launch对照） 19.399680µs |
| launch | fp16 | 25.401919 | Mega33 / 29.407041 | 0.863804× | +15.767% | 25.47930/29.48424 | Lt1 打包一次（多launch对照） 19.673280µs |
| graph | bf16 | 24.854400 | Mega33 / 28.604801 | 0.868889× | +15.089% | 24.93586/28.68783 | Lt1 打包一次（多launch对照） 17.127360µs |
| graph | fp16 | 25.129600 | Mega33 / 28.983040 | 0.867045× | +15.334% | 25.20803/29.07212 | Lt1 打包一次（多launch对照） 17.279040µs |

比较依据是SW自身耗时，Dense分母变化不证明代码收益。Mega33与29都是64×32，用于同宽wait1/retained wait0比较；Mega32的64×16与此控制宽度不同，不能从该比较单独提取wait1贡献。同atom的实现一致性测试分别比较32↔28与33↔29，没有把N16和N32作浮点逐位oracle。Lt1多launch控制不能替代单launch性能目标。

## 编译资源与实现检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | producer/consumer quota | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 33 | fp16 | 64×32 /32 | 256 | 86144 | 160 | 40960 | 64/160 | 28672 | 0/0/0 |
| 32 | fp16 | 64×16 /32 | 256 | 71808 | 96 | 24576 | 64/96 | 20480 | 0/0/0 |
| 33 | bf16 | 64×32 /32 | 256 | 86144 | 160 | 40960 | 64/160 | 28672 | 0/0/0 |
| 32 | bf16 | 64×16 /32 | 256 | 71808 | 96 | 24576 | 64/96 | 20480 | 0/0/0 |

dynamic shared与ELF static shared分别记录。静态寄存器/shared/SASS条数只说明编译资源；没有把它们解释为实测occupancy、动态traffic或物理性能floor。完整role-CFG/配额与隐式register-width审查及pipeline prime/retire/tail证明见profile中的独立审核原件。

CPU检查exit=0，DAG/stage测试数=3/6，[wait1_cpu_checks_iter32.txt](/share/home/liuyao/workspace/MegaFMM/results/wait1_cpu_checks_iter32.txt)。

default：exit=0，2188 CASE /1075 PASS，[correctness_wait1_iter32.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_wait1_iter32.txt)。
selected32：exit=0，30 CASE /72 PASS，[correctness_wait132_targeted_iter32.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_wait132_targeted_iter32.txt)。
selected33：exit=0，32 CASE /72 PASS，[correctness_wait133_targeted_iter32.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_wait133_targeted_iter32.txt)。

新Graph configurations/replays为20/60；全部Graph为132/396。结构fixed-DAG/bitwise/Graph检查不替代原始数值认证。

## 原始数值门限与sanitizer

本轮strict专用范围60 gates：36 PASS /24 FAIL /nonfinite 0；48 native，12 Dense range fallback。BF16/FP16原floor保持0.001/0.0001，门限为3×同行Dense_eF+floor。

历史执行结果按各自源码身份保留：source29完整1290、source30新增60、source31新增60。source30 full1350、source31 full1410和当前full1470仅是源码full域计数，未宣称本轮执行。

记录6项CUDA Compute Sanitizer运行，其中0项exit/errors/warnings/hazards非零；memcheck leaked-bytes记录为[0, 0]，race/synccheck未测泄漏字段保持未测。实际工具版本和各原始摘要见profile，不用结构测试PASS替代numerical PASS。

旧CUDA TU保存27个，历史raw/profile/CSV逐项绑定。本报告生成器从原始文件填数；独立数据审计仍需另外重算统计量和核对来源，最终审结与目标状态以冻结profile为准。
