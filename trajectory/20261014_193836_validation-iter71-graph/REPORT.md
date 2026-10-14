# 1024³ SW 优化阶段报告：iter71

2026-10-14。当前源码来自 iter71（8b1e8ca），普通调用结果见
`../20261014_193739_iter-71/results.csv`，同一源码的 Graph 结果见本目录 `results.csv`。
本报告接续最初首版报告，只测试 M=N=K=1024。

**结论：本次测量与压力输入均通过原精度门限，但 SW 尚未达到 cuBLASLt Dense 的性能。
普通调用 BF16/FP16 为28.33/28.22 μs，分别慢2.31/2.85倍。Auto继续使用Dense。
这是已测输入集合上的数值通过，不是任意输入的数值保证。后续优化目标仍在进行。**

## 当前架构

- 原生 CUDA cooperative kernel `sw_cooperative_hilo<BF16/FP16>`，实际路线名
  `sw_cooperative_hilo_64x32x128`。无转置、连续row-major、alpha=1、beta=0，FP32输出。
- 网格16×8，共128 CTAs，每个CTA负责四个输出象限中的对应64×32 tile，遍历4个BK128分片。
  每CTA512线程：warp0的leader发TMA，warp1–3和warp8–15共11个CUDA Core warp
  重组四块输出，warp4–7构成一个128线程WGMMA warpgroup。
- 每次调用开始，所有CTA的线程共同以FP32固定DAG生成S1–S4/T1–T4，各拆为输入dtype的
  hi/lo两部分，写入统一的8MiB caller workspace。所有输入相关结果都在本次调用生成。
  generic→async全局proxy fence与cooperative grid barrier后，进入分工流水。
- A使用K-major/SW128，B使用MN-major/SW64。BK128的每个A平面拆成两个64列TMA box，
  以满足SW128的inner-box字节限制；B直接装载32×128。两个operand槽循环复用。
- 统一动态shared buffer共188544字节：operand双缓冲98304字节，七块FP32 P共57344字节，
  四块FP32 Q共32768字节，其余为barriers和128字节对齐。
- FP32 S/T按固定顺序生成；叶乘法保留hi×hi、lo×hi、hi×lo，省略lo×lo。
  P1/P2各1项，P3/P4各2项，P5/P6/P7各3项，共15项，超过同样分块Dense的8项。
  各K分片的P按固定U DAG重组并累加Q，最终写入四象限。没有split-K或输出原子。
- WGMMA commit/wait完成后才释放operand，CUDA workers消费七块P后才允许复用product区。
  输入变换及grid barrier都在被计时的同一个kernel中。
- 每次普通run重新编码当前指针的4个TMA map，无跨调用输入值缓存。Graph固定指针绑定，
  每次重放仍重新生成workspace。run不分配、不进行host同步。workspace由调用方提供，
  要求256字节对齐、8MiB预算且不与输入或输出重叠。
- 需Hopper及cooperative launch，设备必须容纳同时驻留的128 CTA；本次在132 SM H100验证。
  编译器两种dtype均74 registers/thread、静态shared1024字节、stack/local均0。
  资源容量为最多1 CTA/SM；没有用该容量数字冒充实测occupancy。

## 完整调用性能

H100 80GB HBM3 / g08 / allocation107488；CUDA12.4，CUTLASS3.5.1头文件。
A/B为BF16或FP16，FP32累加和输出。Dense使用CUBLAS_COMPUTE_32F并独立显式调优。
每行20次warmup，5组×100次完整调用，顺序随机交替，报告组均值的中位数。
每行3轮新输入正确性检查。rotate_inputs=1，输入A/B共4MiB，属于warm-cache测量。
workspace预算同为8GiB；SW实际8MiB，Dense计划预留32MiB候选池，调优后实际配置见CSV。
没有减少trial数或排除SW变换、残差乘法、同步、P传递、重组、写出成本。

| dtype | 模式 | cuBLASLt GPU μs | SW GPU μs | Dense/SW | SW慢多少倍 | Dense wall μs | SW wall μs |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 | launch | 12.2666 | 28.3264 | 0.4330 | 2.309 | 12.3421 | 28.4114 |
| FP16 | launch | 9.9053 | 28.2166 | 0.3510 | 2.849 | 9.9836 | 28.2955 |
| BF16 | Graph | 11.5149 | 27.5709 | 0.4176 | 2.394 | 11.5943 | 27.6557 |
| FP16 | Graph | 9.4352 | 27.5062 | 0.3430 | 2.915 | 9.5207 | 27.5895 |

GPU是完整调用序列的CUDA event时间，wall包含主机调用和末尾同步的平均成本。
相对最初首版普通调用390.26/368.81 μs，当前约快13.78/13.07倍；该数字表示SW自身改善，
不能解释为相对cuBLASLt的加速。性能winner仍是Dense。

## 原精度验收

对相同原始已量化输入做FP64普通AB参考；不使用变换后的S/T作为参考。
保持eF_SW <= 3*eF_dense + floor，BF16 floor=1e-3、FP16 floor=1e-4，要求每轮通过且无nonfinite。
普通调用与Graph各两行共12次性能配套正确性检查全部通过：

| dtype | 标准随机集最大SW eF | gate上界（最差Dense eF对应） | 结果 |
|---|---:|---:|---|
| BF16 | 7.17661e-6 | 1.00255680e-3 | 所有trial通过 |
| FP16 | 8.25634e-7 | 1.03752761e-4 | 所有trial通过 |

另有六类输入×两dtype×三轮新输入×launch/Graph，共72次检查全部通过，无nonfinite。
同一dtype只捕获一次Graph，所有后续新输入复用同一Graph，每次先将输出用NaN位模式覆盖。
range16指数范围[-4,4]；cancellation为跨K半区配对抵消输入。

| 输入类型 | BF16最大SW eF | FP16最大SW eF | 结果 |
|---|---:|---:|---|
| uniform | 7.17661e-6 | 8.26069e-7 | 全部通过 |
| positive | 3.52073e-7 | 5.74113e-7 | 全部通过 |
| range16 | 6.58299e-6 | 8.96658e-7 | 全部通过 |
| cancellation | 1.16084e-5 | 1.72587e-5 | 全部通过 |
| zero | 0 | 0 | 全部通过 |
| identity | 2.24265e-7 | 1.29150e-9 | 全部通过 |

补充API检查通过：Auto保持Dense，强制SW拒绝transpose、输入未对齐、workspace预算不足、
workspace缺失、未对齐或与A重叠；精确workspace记账为8MiB。
CTest仅运行sw_1024_correctness与exact_winograd_dag，两项均通过；没有运行旧多尺寸GPU测试。

## 工具检查与证据边界

应用仍用CUDA12.4构建；工具为/share/app/cuda/cuda-12.8/bin/compute-sanitizer。
每个quick程序覆盖两dtype、一轮随机新输入、普通/Graph，共4次数值检查。
memcheck覆盖整个quick程序；racecheck和synccheck限制kns=sw_cooperative_hilo。
工具全部退出0，quick数值检查通过：

| 工具 | 结果 |
|---|---|
| memcheck | 0 errors |
| racecheck | 0 hazards、0 errors、0 warnings |
| synccheck | 0 errors |

原始输出保存在本目录。

此前ncu探测失败，遵守HINTS不重复；阶段瓶颈依据原生运行时间、编译器资源和独立clock64探针。
iter64插桩clone的CTA0结果约为：pack/grid barrier 10–11k cycles，TC operand wait约6.4k，
MMA+P store约40k，product-free wait约12.6k。插桩改变代码，不能将这些周期当作当前原生kernel
的精确耗时比例。没有宣称测得bank冲突或硬件stall比例。

保留失败和退化实验：iter63初次cooperative版本精度失败（eF约1.0），由P4的A22列偏移遗漏导致，
iter64修复后通过；iter65集中装载、iter66多P同时提交、iter67寄存器Q均未改善；iter68增大BK获益；
iter69因shared预算对齐断言编译失败，无有效性能；iter70 BK256单buffer慢于BK128；iter71增加CUDA
重组并行度改善性能。失败轮不纳入有效winner。

当前可继续优化的成本包括15项补偿乘法、逐P完成等待、shared P传递、每K分片重组边界、
以及全局S/T工作区生成。当前结果没有达到物理性能下界，也未宣告优化目标完成。
