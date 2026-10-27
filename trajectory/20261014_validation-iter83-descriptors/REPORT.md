# WGMMA descriptor 优化 iter83

仅测试 H100 g08（allocation 107488）、CUDA 12.4、1024³、BF16/FP16 输入及 FP32 输出。
源码提交 `9d678d8`，原版 iter82=`77e1489`。计时包含输入 S/T 变换、cooperative grid barrier、15 项叶乘法补偿及输出重组。

## 改动

iter83 使用 CuTe `make_tensor<GMMA::smem_desc<...>>`，每项矩阵乘法构造一次基础 descriptor，按 layout 给出的 K 偏移迭代，替代每条 WGMMA 前完整构造 descriptor。
iter85 在此基础上仅更新 descriptor 低32位，避免64位加法进位。所有 operand 地址位于 Hopper shared 的256KiB寻址范围内，K偏移不会溢出以16字节为单位的14位 start-address 字段；高位配置字段保持不变。CuTe 仍负责生成 swizzle 对应的偏移。

计算顺序、15项补偿、P顺序、barrier、warp职责、buffer布局、精度门限不变。iter84 首次尝试无法编译：CUTLASS 的 `raw_pointer_cast(DescriptorIterator)` 模板存在无法推导参数；失败已独立归档提交。iter85 以零偏移 descriptor 访问修正。

## 完整调用交替对照

每项20 warmup、5组×100次、3组 fresh inputs，A=iter82，B=当前版本。普通调用及Graph均按 A/B/B/A 顺序，各运行两轮。无锁频，短内核仍有时钟及调度噪声；表中为两次独立运行的行中位数均值，不能混同单次调用分位数。

| 模式 | dtype | iter82 μs | iter83 μs | 耗时下降 |
|---|---|---:|---:|---:|
| launch | bf16 | 21.41024 | 20.90512 | 2.36% |
| launch | fp16 | 21.44096 | 20.87312 | 2.65% |
| graph | bf16 | 20.57696 | 20.18816 | 1.89% |
| graph | fp16 | 20.85152 | 20.48816 | 1.74% |

## 分段计时与 NCU

对隔离副本插入 `%globaltimer`，每种dtype3份fresh输入、每份10次kernel×128CTA，共3840个CTA样本。计时区间仍包含异步WGMMA执行的重叠，不能视为全部可消除的软件开销。每个trace输出均与其未插桩副本bitwise一致。

| dtype | 原版 ready / issue / wait0 / TC循环 μs | 本版 ready / issue / wait0 / TC循环 μs | trace耗时扰动 |
|---|---|---|---:|
| bf16 | 2.432 / 8.608 / 0.352 / 13.408 | 2.432 / 8.064 / 0.384 / 12.896 | +0.46% |
| fp16 | 2.464 / 8.608 / 0.352 / 13.440 | 2.400 / 8.096 / 0.352 / 12.864 | -0.07% |

NCU2025.1.1，full set，39passes/dtype，cache-control none、clock-control none。NCU重放耗时只用于诊断，不代替上表完整调用计时。

| dtype | 总动态warp指令 | 寄存器/thread | Tensor elapsed activity | R2UR | 实际64x32 WGMMA |
|---|---:|---:|---:|---:|---:|
| bf16 | 5782811 | 106 | 19.38% | 995968 | 245760 |
| fp16 | 5806925 | 106 | 19.48% | 995968 | 245760 |

原版BF16/FP16动态指令为6119946/6144374，R2UR两者均995968，实际64x32 WGMMA均245760。R2UR没有减少。
iter83 BF16 `IMAD.X`=319488；iter85=3072，支持消除通用64位进位路径这一解释。动态指令也受barrier轮询次数影响，不能把全部指令变化直接等同descriptor指令变化。

每dtype全部16条P STS与42条P LDS仍1-way、0 excessive wavefront；见summary.json逐指令证据。动态shared=213120 bytes；资源检查无local、stack或spill。寄存器变化未提升每SM驻留CTA数，仍为1。

## 正确性与范围

正式bench全部fresh trial通过原始FP64 AB参考门限，BF16 eF=7.394587e-6，limit=1.002557e-3；FP16 eF=2.371355e-6，limit=1.037528e-4，nonfinite=0。
独立stress包含6种输入×2dtype×3fresh×launch/Graph，共72项，通过；Graph重放fresh输入、输出NaN污染和API拒绝检查通过。memcheck/racecheck/synccheck quick均0错误/0hazard。数学DAG的3个CPU测试通过。

源码和运行结果按每次iter分别提交。`standard_launch.csv`保留本次正式bench，paired目录保留交替运行CSV/日志及执行脚本，production_chain目录保留固定git提交生成器、插桩源码、原始CTA记录及bitwise结果，NCU原始报告/导出和资源、sanitizer日志一并保留。
生产链计时编译参数：`nvcc -O3 -std=c++17 -lineinfo -gencode=arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include -Xptxas=-v <source>/measure.cu -lcuda -o <measure>`，先创建输出目录，再执行 `<measure> <output> 5`。生成器固定源码提交；原版对照生成器见`../20261014_profile-production-chain/`。
