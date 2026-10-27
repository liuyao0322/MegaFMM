# 统一 buffer / warp 分工 SW 首版：1024³ 实测

2026-10-14。范围：仅 M=N=K=1024，BF16/FP16，FP32 输出，连续 row-major、
无转置、alpha=1、beta=0。未展开其他矩阵尺寸或后续性能优化。

**结论：首版在本次 72 次压力集检查和 12 次性能评测配套的新输入检查中通过原数值
门限，性能约慢于调优 cuBLASLt 32–40 倍。不是加速 winner，Auto 保留 Dense。**
这是有限输入集合上的实测通过，不是对全部有限值输入的数值保证。

## 实现

- 原生 kernel：`sw_unified_v1<BF16/FP16>`；公开实际路线名
  `sw_unified_v1_hilo_64x32x32`，强制接口 `Algorithm::SWPipeline` / CLI `--algorithm sw`。
- CTA 为 256 线程：warp0 的 leader 发 TMA；warp1–3 共96线程做 FP32 变换、
  高低部分拆分、重组与输出；warp4–7 共128线程组成 WGMMA warpgroup。
- BM=64、BN=32、BK=32；每 CTA 拥有四象限中对应的四块输出 tile。网格16×8，
  共128 CTAs；每个 CTA 遍历 K/2 的16个分块，无 split-K、全局原子或跨 CTA 结果同步。
- 一块动态 shared buffer：raw双缓冲48KiB，FP32 S/T 48KiB，operand双缓冲24KiB，
  七个FP32 P 56KiB，四块FP32输出累加区32KiB，加 barriers 和对齐共213120字节。
- 编译器资源：BF16 62 registers/thread，FP16 61；静态shared1024字节，stack/local均0。
  动态+静态shared共214144字节，资源容量最多允许1 CTA/SM；没有声称测得实际occupancy。
- S/T按固定DAG在FP32生成，共用表达式保留；最终操作数用匹配输入dtype的 hi+lo 表达。
  叶乘法采用 hi×hi、lo×hi、hi×lo；省略lo×lo。P1/P2各1项，P3/P4各2项，
  P5/P6/P7各3项，共15项低精度乘法，而相同划分的普通Dense为8项。
- 每个K分块产生七个P后，由CUDA workers按U DAG重组，累加四个Q。此运算顺序与
  全K完成P后重组不同，已纳入原始输入FP64检查。P跨warp传递经shared，WGMMA累加器在寄存器。
- raw/op各自使用ready/free mbarrier及phase；SIMT写operand后执行generic→async fence；
  WGMMA commit/wait0完成后才释放operand。七个P重组完成后才复用product区。
- 每次普通run重新编码当前A/B的两个TMA descriptor，无输入值缓存、全局S/T/P或输出缓存。
  编码在完整普通调用计时内。Graph捕获固定绑定，但每次重放仍执行所有输入相关计算。
- `run()`不分配、不进行host同步；SW的caller global workspace为0。
  非法强制SW配置直接抛错，不静默fallback；Auto/Dense始终执行调优cuBLASLt。

## 性能

环境：NVIDIA H100 80GB HBM3，132 SM，CUDA12.4，driver API12090，runtime12040；
使用已有H100 allocation107488 / g08。CUTLASS头文件版本3.5.1。
每种dtype/mode：20次warmup、每组100次完整调用、5组，交替顺序随机化，报告组均值的中位数；
正确性每行3轮新输入；rotate_inputs=1，输入A/B共4MiB，属于warm-cache测量。
Dense每行独立显式调优，32个请求heuristics，同为8GiB workspace上限。无异常值剔除。

| dtype | mode | cuBLASLt GPU µs | SW GPU µs | Dense/SW | SW 慢多少倍 | Dense wall µs | SW wall µs |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 | launch | 12.133 | 390.256 | 0.03109 | 32.16 | 12.228 | 390.350 |
| fp16 | launch | 9.733 | 368.811 | 0.02639 | 37.89 | 9.810 | 368.914 |
| bf16 | graph | 11.032 | 387.515 | 0.02847 | 35.13 | 11.103 | 387.607 |
| fp16 | graph | 9.252 | 366.750 | 0.02523 | 39.64 | 9.326 | 366.833 |

GPU时间是完整调用序列的CUDA event计时；wall时间包含主机调用及末尾同步的平均成本。
两种模式都包含全部变换、残差乘法、P传递、重组及输出，Graph未预计算输入相关变换。

性能不佳的已知成本是：15项叶乘法、很大的shared占用、逐P的生产/消费握手、
FP32 P写入并重读、每K分块的七P重组边界。它们是实现事实；各项占耗时的比例、
bank冲突或具体stall原因未通过profiling测得。遵守HINTS，不重复此前失败的ncu探测。
本任务交付首版和比较，到此不自动继续优化。

## 原始精度验收

参考对相同原始、已量化的A/B进行FP64普通AB，不使用变换后输入。
门限保持 eF_SW <= 3*eF_dense + floor，BF16 floor=1e-3，FP16 floor=1e-4，
要求每轮通过且无nonfinite。

标准均匀随机性能集：BF16最大eF=7.15224591113e-6，gate约1.00256e-3；
FP16最大eF=3.09536268537e-7，gate约1.03753e-4。普通/Graph共4行、12次检查全部通过。

`test_sw1024`：六类输入（uniform、positive、range16、cancellation、zero、identity）
×两种dtype×三轮新输入×普通/Graph，共72次SW检查全部通过，nonfinite=0。
每种dtype只捕获一次Graph，然后对各类每轮新输入重放同一个graph；执行前用NaN位模式
覆盖输出，避免部分写出漏检。range16指数范围[-4,4]；cancellation使用跨K半区配对输入，
A第二半区=-A第一半区+1/256、B第二半区=B第一半区。测试同时检查Auto保持Dense、
SW不需要全局workspace，以及转置/输入未对齐的强制SW拒绝。

| 输入类型 | BF16最大SW eF | FP16最大SW eF | 结果 |
|---|---:|---:|---|
| uniform | 7.15225e-06 | 3.09536e-07 | 全部通过 |
| positive | 1.61913e-07 | 1.37579e-07 | 全部通过 |
| range16 | 6.53954e-06 | 3.55378e-07 | 全部通过 |
| cancellation | 5.3116e-06 | 7.33019e-06 | 全部通过 |
| zero | 0 | 0 | 全部通过 |
| identity | 2.24265e-07 | 1.2915e-09 | 全部通过 |

CTest仅执行 `sw_1024_correctness` 与 `exact_winograd_dag`，两项通过。
未执行旧的多尺寸GPU测试，符合本次只测1024的要求。

## 内存与同步工具

使用CUDA12.8路径下的Compute Sanitizer，应用仍由CUDA12.4构建。每个工具的quick
输入覆盖两种dtype、各一轮新输入、普通/Graph，共4次数值检查；全部通过。

| 工具 | 插桩范围 | 退出码 | 结果 |
|---|---|---:|---|
| memcheck | 整个quick程序 | 0 | 0 errors |
| racecheck | `kns=sw_unified_v1`，新SW内核 | 0 | 0 hazards、0 errors、0 warnings |
| synccheck | `kns=sw_unified_v1`，新SW内核 | 0 | 0 errors |

先前不加kernel过滤的racecheck约4分钟无诊断，被主动中止，该尝试不是通过，
单独保留interrupted日志。之后限定新SW内核的racecheck与synccheck均正常完成。
工具结果只覆盖以上quick输入，不替代72次数值检查或未测试域的保证。

## 修正与证据范围

- iter48初始swizzled operand版本数值失败，eF约1.32，保留失败记录。
- 独立内部WGMMA tile诊断在swizzled布局失败，在canonical interleave布局通过；
  它不是另一种矩阵尺寸的性能benchmark。
- iter49的共享文件系统时间戳缓存导致make未重新编译；该轮沿用旧二进制，不能作为
  interleaved源码的证据。已在ITERATIONS.md明确纠正。
- iter50强制clean build，compute-node源码SHA256与当前kernel一致；完整launch数值通过。
  本目录是同一源码的Graph验证。构建脚本保留强制重编译和源码哈希，避免再次沿用旧对象。
- 当前源码、性能CSV、扩展验证与资源日志、工具终态见本目录及validation.json。
  原始性能launch轨迹在 `../20261014_182746_iter-50/`；失败轨迹在iter48/49目录。
