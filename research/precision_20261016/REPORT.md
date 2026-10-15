# 2026-10-16：精度来源、消融和针对性方案

本轮按用户要求先做相关研究，再分离误差来源，最后设计修复方案。**已完成这三个阶段，但“七乘法 FMM 精度达标且性能不退化”的总目标尚未完成。** 当前生产七乘法未修改，仍不满足原始精度门槛；本轮的直接输入稠密原型同时达标且更快，但它不是七乘法修复。

## 研究边界与可复核起点

- 分支：`research/precision-ablation-20261016`；起点 Git `4035bed`，实际 SW 源码最后改动为 `da8bf87`，不是 README 历史 iter282 标签。
- H100，节点 g08，既有 SLURM allocation 107488；CUDA 编译12.4、sanitizer/NCU12.8，Python kernelagent 环境的 PyTorch 2.10/cu128。GPU 时钟未锁定。
- 本轮硬件性能只覆盖连续、无转置、alpha=1、beta=0、1024³，FP16/BF16 输入、FP32 输出，不推广到其他 MNK、布局或完整 BF16 值域。
- 参考始终是原始、已经舍入的 A/B 的 FP64 AB。原始门槛不变：`eF <= 3*eF_dense + floor`，FP16 floor=1e-4、BF16 floor=1e-3，且无非有限值。
- `solution/sw_pipeline.cu` SHA256 为 `748e7ace9de0b6469d47922cb7cb531bbb56a9d55c9915b0194c0e0a72e916e7`；研究前后未改。完整身份见 [source_manifest.json](source_manifest.json)。原有 `results/current.csv`、`results/build.log` 的未提交改动不属于本研究。
- 原 SW 仍为两个 kernel、4MiB workspace、七个 Tensor Core 叶乘积，FP32变换后将 S/T 写成输入 dtype。Auto/Dense 仍为 cuBLASLt。

## 文献如何影响实验

[LITERATURE.md](LITERATURE.md)记录核验过的一手来源、版本、适用范围及原文位置，包含 Ballard 的稳定性/缩放分析、Dumas 等的准确七乘法公式、Ootomo/Yokota 的混合精度残差恢复、Tensor Core 舍入模型、FalconGEMM 和近期 FP8 分析。这是定向工程检索，不是系统综述。

据此分开检验三个问题：误差是否来自变换 cast；输入尺度是否放大误差；更完整的残差信息是否可能修复。不能将低增长因子、片上 FP32 重组或某篇论文的速度直接当作本项目达标证据。前轮已测的公式替换、低秩 oracle 和高低分量代价也纳入判断，没有恢复已经停止的 FP8 内核开发。

## 误差阶段消融：主因是 S/T 的低精度存储

[ablate.py](ablate.py)在1024³上使用两种 dtype、六类输入、三个 seed，共36组。反事实模型分别改变 FP32 变换、操作数 cast、FP32 叶乘法、FP32 重组，其余环节尽可能使用 FP64；TF32 关闭。它用于归因，不声称逐位模拟 Hopper WGMMA 内部舍入。摘要见 [ablation_summary.json](ablation_summary.json)，原始数据见 [ablation.json](ablation.json)。

下表为均匀输入三个 seed 的最大相对 Frobenius 误差：

| 反事实干预 | FP16 | BF16 |
|---|---:|---:|
| 仅变换用 FP32，随后不作低精度 cast | 1.30e-9 | 5.63e-10 |
| 仅将 S/T cast 为输入 dtype，叶与重组用 FP64 | 7.545e-4 | 6.020e-3 |
| 不 cast，叶乘法用 FP32 | 5.45e-7 | 3.38e-7 |
| 所有阶段用 FP32，并保留当前低精度 cast | 7.545e-4 | 6.020e-3 |
| 理想精确补齐八个一阶残差项 | 1.12e-7 | 7.16e-6 |
| 从八项中 oracle 选择最好的七项 | 1.525e-4 | 1.218e-3 |

因此，仅提高重组精度或微调 FP32 加法顺序无法消除当前约7.5e-4/6.0e-3的误差。完整保留残差信息有充分的数值动机。对全部256个残差子集枚举时保留误差交叉项，未假定各项独立或将范数简单相加。即使由 oracle 每次选择最好的七项，均匀输入仍超过相应实际 gate；不能凭少补一项宣称解决。

强消去输入进一步放大误差，cast 单独导致的误差可达 FP16 4.45e-3、BF16 3.63e-2。这里的分母是原始结果范数，结果接近抵消时相对误差更敏感。这不等价于各误差分量之间存在可加的方差分解。

原始消融还保存了少量历史表示的数值对照字段。其中极端范围的溢出保留为 NaN，不属于达标候选；不能将整个原始 JSON 当成全有限的硬件验收结果。发现 HINTS 中的 FP8 停止指令后，没有继续 FP8 kernel 路线。

## 缩放消融：修复失衡，不能补回尾数

[scaling.py](scaling.py)按 Ballard 的思路分别测试 inside、outside、两种组合顺序，使用精确二的幂缩放及 FP64 叶乘法，对照原始 AB。六类输入×三个 seed×两种 dtype，共36组，数据在 [scaling.json](scaling.json)。

- 均匀输入和本轮消去输入：幂二缩放没有降低变换 cast 的误差。
- 刻意构造的 K 方向失衡输入：FP16 从1.272–1.277降至7.573e-4–7.596e-4；BF16 从7.297–7.322降至6.007e-3–6.020e-3。
- 行/列块尺度不均衡时，某些组合缩放还会增加 Frobenius 误差。不能将一个缩放顺序视为普遍改善。

结论是尺度失衡与变换 cast 是两个相关但不同的来源。平衡可去除额外放大，之后仍需要恢复 S/T 的残差信息。实际实现必须计入每次输入的尺度 reduction、转换和同步成本。

## 保持七乘法的定向候选：INT8 残差补偿

[int8_model.py](int8_model.py)提出新的表示筛选：七个输入 dtype 主乘积，加八个一阶交叉补偿乘积；补偿两侧为分块幂二缩放的 signed INT8。左块64×128、右块128×32，K128部分整数点积绝对值上界为 `128*127²=2064512`，能放入 INT32。每个 K 块应独立缩放后累加，不能跨不同尺度直接累积整数。

模型使用舍入后整数的精确 FP64 点积和理想反缩放，主叶与重组也偏向理想精度，**不是实际 INT8 WGMMA 内核**。它还未包含 FP32 叶累加、转换/缩放舍入、寄存器和同步代价。

新 seeds901–903、两种 dtype、七类输入共42组：无 inside 平衡时36组低于固定 floor，六组 K 失衡输入失败。均匀输入最大 eF 为 FP16 6.95e-6、BF16 5.98e-5，消去输入最大为6.42e-5、5.19e-4。失败的失衡输入则高达1.76e-2、1.08e-1，不能忽略。

接着固定同一表示，只加入 inside 幂二平衡，在 uniform/inside_mismatch 两类共12组中全部低于固定 floor；失衡输入降至 FP16 6.76e-6、BF16 5.98e-5。随后将有平衡模型复核到全部七类42组，也全部低于固定 floor。数据见 [无平衡42组](int8_model.json)、[有平衡12组](int8_inside_model.json)及[有平衡完整42组](int8_inside_full_model.json)。这是比仅换重组顺序更有针对性的数值候选，但目前没有证据说明增加八个补偿乘积后还能保持完整调用性能。

[DESIGN.md](DESIGN.md)给出数学表示、二阶项消融、Hopper映射、inside尺度开销、实现顺序和拒绝条件。总计是七个主乘积加八个补偿乘积，不能称为只执行七次 Tensor Core 运算。

## 硬件对照：精度安全的直接输入稠密原型

为验证完整调用成本，并为补偿方案建立更强对照，本轮实现了独立的直接输入内核：[generate_direct.py](generate_direct.py)、[compare_direct.cu](compare_direct.cu)、[validate_direct.cu](validate_direct.cu)。它使用原始 FP16/BF16 输入、FP32 累加；无 S/T、无调用方 workspace。最终 tile 为64×128、BK128、四槽、128 CTA，采用 `wait_group 1` 流水线，90 registers/thread、196736B shared。它数学上是普通稠密乘法。

tile 消融表明，显式四象限64×32版本约10.3μs并未获得优势；扩大到64×64约7.5–7.6μs；64×128无重叠约6.88–6.95μs；最终流水线约6.5–6.6μs。主要收益不能全部归因于异步重叠，tile 调度改变更大。

在三个独立进程复测中，每次20 warmup，五组，每组四轮交错三个方法（正/反/反/正），每轮100完整调用，以 CUDA events 计时。Dense 独立调优，原 SW 通过现有静态库调用。普通调用包含本次 TMA 描述符生成和完整 GPU 工作；Graph 单列。每项为15组的中位数，加速比为组内配对比值的中位数：

| 执行方式 | dtype | 调优 cuBLASLt μs | 当前 SW μs | 稠密原型 μs | 相对 SW 加速 | 最差组相对 SW 加速 |
|---|---|---:|---:|---:|---:|---:|
| 普通调用 | BF16 | 12.063 | 10.120 | 6.519 | 1.550× | 1.535× |
| 普通调用 | FP16 | 9.605 | 10.016 | 6.630 | 1.508× | 1.500× |
| Graph | BF16 | 11.114 | 9.901 | 6.139 | 1.610× | 1.600× |
| Graph | FP16 | 9.334 | 9.955 | 6.419 | 1.549× | 1.543× |

这说明在本测量范围内，不做低精度变换的稠密实现有足够好的硬件表现；“八个块乘积必然比七个慢”不是可靠的性能判断。但它不能证明七乘法得到修复，也未接入生产分发。

测量是热缓存、相同数据指针重复调用，不是每次计时重新生成输入。三个进程的精度检查使用相同确定性 seed，不算新的独立数值样本。未控制 GPU 时钟、未声称跨机器置信区间，不能推断流式冷缓存或任意尺寸收益。全部组与原始记录保存在 [direct_results.json](direct_results.json)。`direct_graph.log` 曾与缩放 pilot 短暂重叠，已明确排除；正式复测均串行。

## 硬件验证及其边界

- 原始六类输入×三轮×两种 dtype：每个验收进程36项，六个进程共216条 CHECK 均通过原始 gate。这是36种确定性配置的重复验证，不是216个独立 seed。
- 扩展六类×三轮×两条 stream×两种 dtype：72项、每项八次交错普通/Graph调用，共576次新内核调用，含输出 guards，全部通过。覆盖较大动态范围、块尺度、强消去、K方向失衡及单位矩阵。
- [memcheck.log](memcheck.log)：0 errors；[racecheck_scoped.log](racecheck_scoped.log)：仅新 `direct_quadrants` 内核，0 hazards、0 errors、0 warnings；[synccheck_scoped.log](synccheck_scoped.log)：相同限定范围，0 errors。初次不限定内核的 racecheck 因大量 vendor instrumentation 耗时被终止，保留日志，不声称它通过。
- [dag_check.log](dag_check.log)：原有 CPU DAG 三项测试通过。未以新增实现镜像测试替代数值参考。
- NCU 仅为资源/执行诊断，选择性 profile 一个零输入 FP16 调用：Tensor active约30.17%、active warps约7.93%，详见 [ncu/summary.json](ncu/summary.json)。其7.2μs是 profiler 下的单次时间，不作为正式性能样本。
- FP16 强消去硬件 eF 最大约1.536e-4，依靠实际 Dense 误差项满足原始 gate，不能声称所有硬件用例均低于固定1e-4 floor。固定 floor 筛选与真实 gate 验收在本文中分开报告。

## 复现

所有 GPU 操作串行运行，先确认可用 allocation，按本仓库集群脚本执行。构建使用既有 baseline 静态库；身份应与 manifest 对照。生成源码已保存，以下重建最终三份 binary 并复测：

```bash
bash scripts/on_gpu.sh bash research/precision_20261016/build_variants.sh
bash scripts/on_gpu.sh bash research/precision_20261016/run_acceptance.sh
bash scripts/on_gpu.sh bash research/precision_20261016/check_tools.sh
python3 research/precision_20261016/summarize_direct.py
python3 tests/test_dag.py
```

数值模型使用 PyTorch 自带 CUDA runtime，避免被系统 CUDA12.4 的库覆盖：

```bash
bash scripts/on_gpu.sh env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261016/ablate.py
bash scripts/on_gpu.sh env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261016/scaling.py
bash scripts/on_gpu.sh env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261016/int8_model.py
bash scripts/on_gpu.sh env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261016/int8_model.py --inside-scale --output research/precision_20261016/int8_inside_full_model.json
```

`ablate.py` 的原始记录包含早期表示对照，最终脚本将这些对照置于默认关闭的`--legacy-representations`参数下；阶段消融和残差子集计算未改变，默认复现不运行 FP8 对照。原始日志保留以避免隐藏失败。下一步实施以 DESIGN 中的 INT8 方案和完整调用拒绝条件为准。
