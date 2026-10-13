# 1024单层 SW：融合输入与打包一次的端到端对照

在这一组固定的候选中，打包一次后调用 cuBLASLt batch7 的 **Lt1** 在两种 dtype、普通 launch 和 CUDA Graph 下都是观察到的最快 SW。它仍慢于各自行中独立调优的普通 cuBLASLt，且本次全部24行、72次新鲜输入检查都未通过原始数值门槛；Auto继续选择Dense。此报告是性能实验记录，四个尺寸都比cuBLAS快的目标仍未完成。

测量源码commit `bdbcdd9191072014aff7fa99dd9e19afd026d55e`，应用CUDA12.4/CUTLASS3.5.1，H10080GB（allocation107488/g08）。Benchmark二进制SHA256 `66bcc473b87bc5d51f46af9b9b6012ae7dace43c32d2f0c594bbf31e30206d5e`；测试二进制SHA256 `fae22ec4c77586bbbc9d34fa1820903e60ebaef08756b75dbf36eec442c31671`。证据checkpoint的后续HEAD不是测量源码身份。[完整原始行、源码/二进制/日志SHA与Lt配置](small_matched_measurements_iter29.json)，[测量前的24行协议声明](matched1024_plan_iter29.json)。

六条路线都计算同一原始低精度 A/B 对应的完整一层SW：每次调用包括当前输入的全部S/T变换或packing、七个完整K的FP32 P、P写入/读取、固定U合并和alpha/beta。H9/H10融合输入leaf加combine，共两次launch；H11/H12每次fresh uniform pack加匹配leaf再combine，共三次launch。Lt1/H7也每次重新打包，没有缓存输入预处理；Lt内部kernel数量由库实现决定。Graph捕获完整run，每次replay仍执行这些步骤。

所有行使用20次warmup、每组100次调用、5组、3份fresh原始输入检查，普通Dense与SW使用相同dtype/FP32输出和模式，分别调优，Lt heuristic_candidates=32预算（实际去重候选池见CSV config）及共享8GiB workspace预算。表中耗时为五个组均值的中位数；p10/p90、CV也来自五个组均值。GPU时钟未锁定，串行测量的组统计不是置信区间，不能把不同行的Dense分母当作代码收益。`Dense/SW>1`才表示SW在该行更快。

## 普通 launch：完整run，单位µs

| 路线 | dtype | Dense中位数 [p10,p90] | SW中位数 [p10,p90] | Dense/SW | CV Dense/SW (%) | 原数值门槛 | payload / reserved (MiB) | CSV行 |
|---|---|---:|---:|---:|---:|---|---:|---|
| H9 融合32×32 | bf16 | 12.3238 [12.2810,12.6317] | 43.0096 [42.8306,43.0686] | 0.286537 | 1.3396/0.2506 | FAIL（finite） | 7/7 | [small_fused9_launch_iter29_full1024,行2](small_fused9_launch_iter29_full1024.csv) |
| H9 融合32×32 | fp16 | 9.9984 [9.9613,10.0036] | 43.0349 [42.9597,43.0668] | 0.232332 | 0.2045/0.1188 | FAIL（finite） | 7/7 | [small_fused9_launch_iter29_full1024,行3](small_fused9_launch_iter29_full1024.csv) |
| H10 融合32×64 | bf16 | 12.2589 [12.2487,12.2866] | 38.2541 [38.1944,38.2932] | 0.320459 | 0.1444/0.1149 | FAIL（finite） | 7/7 | [small_fused10_launch_iter29_full1024,行2](small_fused10_launch_iter29_full1024.csv) |
| H10 融合32×64 | fp16 | 9.9590 [9.9291,10.0579] | 38.5088 [38.4773,38.5164] | 0.258617 | 0.7170/0.0513 | FAIL（finite） | 7/7 | [small_fused10_launch_iter29_full1024,行3](small_fused10_launch_iter29_full1024.csv) |
| H11 匹配打包32×32 | bf16 | 12.2022 [12.1740,12.2264] | 40.0851 [40.0532,40.1362] | 0.304408 | 0.1903/0.0984 | FAIL（finite） | 14/46 | [small_matched11_launch_iter29_full1024,行2](small_matched11_launch_iter29_full1024.csv) |
| H11 匹配打包32×32 | fp16 | 9.9734 [9.9348,10.0062] | 40.3974 [40.3775,40.4099] | 0.246883 | 0.3340/0.0355 | FAIL（finite） | 14/46 | [small_matched11_launch_iter29_full1024,行3](small_matched11_launch_iter29_full1024.csv) |
| H12 匹配打包32×64 | bf16 | 12.2992 [12.2617,12.3046] | 34.1165 [34.0529,34.1706] | 0.360506 | 0.1666/0.1531 | FAIL（finite） | 14/46 | [small_matched12_launch_iter29_full1024,行2](small_matched12_launch_iter29_full1024.csv) |
| H12 匹配打包32×64 | fp16 | 9.9430 [9.9119,9.9576] | 34.4358 [34.3911,34.5012] | 0.288741 | 0.2090/0.1442 | FAIL（finite） | 14/46 | [small_matched12_launch_iter29_full1024,行3](small_matched12_launch_iter29_full1024.csv) |
| Lt1 打包批量 | bf16 | 12.2298 [12.2143,12.2651] | 19.3750 [19.3604,19.3842] | 0.631212 | 0.1907/0.0608 | FAIL（finite） | 14/46 | [small_packed_lt1_launch_iter29_full1024,行2](small_packed_lt1_launch_iter29_full1024.csv) |
| Lt1 打包批量 | fp16 | 9.9302 [9.8090,9.9679] | 19.7350 [19.6773,19.7765] | 0.503178 | 0.7286/0.2337 | FAIL（finite） | 14/46 | [small_packed_lt1_launch_iter29_full1024,行3](small_packed_lt1_launch_iter29_full1024.csv) |
| H7 打包BK64 | bf16 | 12.2003 [12.1761,12.2829] | 21.8893 [21.8660,21.9126] | 0.557365 | 0.3956/0.0955 | FAIL（finite） | 14/46 | [small_packed_h7_launch_iter29_full1024,行2](small_packed_h7_launch_iter29_full1024.csv) |
| H7 打包BK64 | fp16 | 9.9338 [9.9265,9.9707] | 22.3165 [22.2775,22.3346] | 0.445131 | 0.2176/0.1141 | FAIL（finite） | 14/46 | [small_packed_h7_launch_iter29_full1024,行3](small_packed_h7_launch_iter29_full1024.csv) |

## CUDA Graph：完整run，单位µs

| 路线 | dtype | Dense中位数 [p10,p90] | SW中位数 [p10,p90] | Dense/SW | CV Dense/SW (%) | 原数值门槛 | payload / reserved (MiB) | CSV行 |
|---|---|---:|---:|---:|---:|---|---:|---|
| H9 融合32×32 | bf16 | 11.2589 [11.2172,11.3103] | 40.9030 [40.8904,40.9354] | 0.275258 | 0.3785/0.0511 | FAIL（finite） | 7/7 | [small_fused9_graph_iter29_full1024,行2](small_fused9_graph_iter29_full1024.csv) |
| H9 融合32×32 | fp16 | 9.3978 [9.3901,9.4166] | 41.2054 [41.1788,41.2374] | 0.228071 | 0.1265/0.0710 | FAIL（finite） | 7/7 | [small_fused9_graph_iter29_full1024,行3](small_fused9_graph_iter29_full1024.csv) |
| H10 融合32×64 | bf16 | 11.2672 [11.2628,11.3737] | 36.3965 [36.3331,36.4239] | 0.309568 | 0.4709/0.1177 | FAIL（finite） | 7/7 | [small_fused10_graph_iter29_full1024,行2](small_fused10_graph_iter29_full1024.csv) |
| H10 融合32×64 | fp16 | 9.5101 [9.4208,9.5192] | 36.7504 [36.7265,36.7574] | 0.258775 | 0.4873/0.0380 | FAIL（finite） | 7/7 | [small_fused10_graph_iter29_full1024,行3](small_fused10_graph_iter29_full1024.csv) |
| H11 匹配打包32×32 | bf16 | 11.2330 [11.2103,11.3414] | 37.1501 [37.0765,37.2354] | 0.302367 | 0.5330/0.2027 | FAIL（finite） | 14/46 | [small_matched11_graph_iter29_full1024,行2](small_matched11_graph_iter29_full1024.csv) |
| H11 匹配打包32×32 | fp16 | 9.4355 [9.4082,9.4515] | 38.0253 [37.9880,38.0787] | 0.248138 | 0.2041/0.1058 | FAIL（finite） | 14/46 | [small_matched11_graph_iter29_full1024,行3](small_matched11_graph_iter29_full1024.csv) |
| H12 匹配打包32×64 | bf16 | 11.3392 [11.3032,11.3750] | 31.3840 [31.3532,31.4372] | 0.361305 | 0.2784/0.1271 | FAIL（finite） | 14/46 | [small_matched12_graph_iter29_full1024,行2](small_matched12_graph_iter29_full1024.csv) |
| H12 匹配打包32×64 | fp16 | 9.5779 [9.3720,21.4297] | 31.9309 [31.8789,31.9688] | 0.299958 | 59.0005/0.1246 | FAIL（finite） | 14/46 | [small_matched12_graph_iter29_full1024,行3](small_matched12_graph_iter29_full1024.csv) |
| Lt1 打包批量 | bf16 | 11.3597 [11.3538,11.3652] | 17.0726 [17.0612,17.0787] | 0.665373 | 0.0448/0.0456 | FAIL（finite） | 14/46 | [small_packed_lt1_graph_iter29_full1024,行2](small_packed_lt1_graph_iter29_full1024.csv) |
| Lt1 打包批量 | fp16 | 9.4397 [9.4268,9.4794] | 17.8051 [17.7902,17.8128] | 0.530167 | 0.2935/0.0627 | FAIL（finite） | 14/46 | [small_packed_lt1_graph_iter29_full1024,行3](small_packed_lt1_graph_iter29_full1024.csv) |
| H7 打包BK64 | bf16 | 11.3123 [11.3027,11.3428] | 19.1370 [19.1096,19.1398] | 0.591124 | 0.1616/0.0739 | FAIL（finite） | 14/46 | [small_packed_h7_graph_iter29_full1024,行2](small_packed_h7_graph_iter29_full1024.csv) |
| H7 打包BK64 | fp16 | 9.4243 [9.3782,9.4728] | 19.8010 [19.7496,19.8191] | 0.475953 | 0.4438/0.1596 | FAIL（finite） | 14/46 | [small_packed_h7_graph_iter29_full1024,行3](small_packed_h7_graph_iter29_full1024.csv) |

H12/FP16/Graph这一行的Dense组均值有明显离群波动：Dense CV为59.0005%，p90为21.4297µs；同一行SW CV为0.1246%。保留全部原始组和本次中位数比值0.299958，它不用于声称代码提升幅度。配对改动的结论来自自己的完整SW延迟；ordinary launch表及其同行Dense比较另列。没有选择性删除或覆盖这个波动结果。

H9/H10仅七个FP32 P需要7MiB payload与7MiB caller-reserved workspace。其余路线的低精度A7/B7共7MiB，加P7MiB，payload14MiB；本次caller-reserved均为46MiB，其中32MiB是Dense回退候选池的最大scratch预留。此预留不等于native leaf运行的额外scratch；H11/H12/H7 custom leaf均为0，Lt1所选batch候选也为0。各自行的普通Dense独立workspace和实际tuned配置保留在51列原CSV中。

## 观察到的路线选择与配对结论

| 模式 | dtype | 当前六候选中自身SW最小 | 同行Dense | Dense/SW |
|---|---|---:|---:|---:|
| launch | bf16 | Lt1 19.3750µs | 12.2298µs | 0.631212 |
| launch | fp16 | Lt1 19.7350µs | 9.9302µs | 0.503178 |
| graph | bf16 | Lt1 17.0726µs | 11.3597µs | 0.665373 |
| graph | fp16 | Lt1 17.8051µs | 9.4397µs | 0.530167 |

这里按自己的SW延迟选择已测候选，没有启用一个固定dispatcher，也没有把数值FAIL候选认证为可自动使用。

- launch/bf16 同tile H9→H11：完整run 43.0096→40.0851µs，观察到SW耗时减少6.80%。
- launch/bf16 同tile H10→H12：完整run 38.2541→34.1165µs，观察到SW耗时减少10.82%。
- launch/fp16 同tile H9→H11：完整run 43.0349→40.3974µs，观察到SW耗时减少6.13%。
- launch/fp16 同tile H10→H12：完整run 38.5088→34.4358µs，观察到SW耗时减少10.58%。
- graph/bf16 同tile H9→H11：完整run 40.9030→37.1501µs，观察到SW耗时减少9.18%。
- graph/bf16 同tile H10→H12：完整run 36.3965→31.3840µs，观察到SW耗时减少13.77%。
- graph/fp16 同tile H9→H11：完整run 41.2054→38.0253µs，观察到SW耗时减少7.72%。
- graph/fp16 同tile H10→H12：完整run 36.7504→31.9309µs，观察到SW耗时减少13.11%。

配对控制保留32×32/32×64 BK32 tile、shared布局、barrier结构、MMA顺序及完整P累加。变化是S/T放置位置及其产生的packed输入读取路径，编译资源也随之变化；它不是共享内存移除或纯DRAM流量实验。默认GPU验证已通过：36份同原始随机rounded输入的H9↔H11/H10↔H12完整七P及最终D逐位一致，同时保留与Mega7最终D对照；四个新整数fixtures独立验证14个A7/B7 slots和全部七P。freshAB Graph每次在捕获外、同stream、固定指针上传原始输入并重算FP64参考，包含beta-zero/nullC0和active C0=D。

编译资源显示匹配BN32（H11）为40REG、16B stack frame、80B spill stores/loads；匹配BN64（H12）为48REG、0stack/spill。ELF显示两者LOCAL0，不能据此宣称H11没有spill。ptxas operand shared为4096/6144B，ELF SHARED为5120/7168B，两工具字段分别记录；没有查询runtime function attribute。静态指令/资源不等于动态瓶颈或时延解释，ncu本session先前lock失败后没有重试。

默认结构、严格原始FP64 gate与六项CUDA12.8 sanitizer的完整审计在本轮final profile中单列；结构精确fixtures或paired bitwise通过不会替代随机原始数值认证。本次原始Gate下24行全部FAIL/NF0，保留原BF16/FP16 dtype、阈值和AutoDense。旧source23–28证据与24个旧CUDA实现未改写，也没有硬件floor或优化方向耗尽的结论。
