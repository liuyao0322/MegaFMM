# MegaFMM FP32 科学计算 benchmark

这组套件回答一个具体问题：在稠密大方阵的科学计算中，将相同算法里的矩阵乘法替换为 MegaFMM Strassen–Winograd（SW）后，能否在预先声明的数值质量要求下缩短完整应用调用时间。主要应用是有限时间范围的协方差传播，辅以 SPD 矩阵平方根、逆平方根和密度矩阵 SP2 purification。配置、原始计时、独立质量检查和失败结果共同构成证据。

这里的“高精度”指 **FP32 输入、FP32 输出及严格误差门槛**。当前 MegaFMM 公共接口没有 FP64 计算路径；FP64 仅用于计时外的参考和诊断。不能将这些数据称为 FP64 HPC、标准 HPL 或双精度求解性能。严格质量门槛也不保证每个 case 或每个 backend 一定通过。

## 应用、输入和数值范围

主应用对应线性随机动力系统的协方差预测：

```text
x[t+1] = A x[t] + w[t],    Cov(w[t]) = Q
P[t+1] = A P[t] Aᵀ + Q
```

这一矩阵更新也见于 [FilterPy 的 Kalman 预测实现](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py)。本套件独立测量其中的稠密预测计算；没有观测更新、数据同化循环或控制任务的精度结论。每次完整调用都从原始 `P0` 开始，执行配置指定的步数；不把上一次计时的输出作为下一次输入。

`covariance` 的每步计算为 `(A @ P) @ A.T + Q`。两个乘法都经过所选 backend，状态保持 FP32。转置的连续化、必要的 cast、SW packing/reconstruction、输出分配和加法均在应用计时内。不执行输出对称化、FP64 修正或迭代精化。

输入由确定性 CPU 随机种子构造。`A` 是由共轭对称 Fourier 相位生成的实循环矩阵，再乘 `transition_radius`；精确算术下它是缩放的正交矩阵，生成时无需大规模 QR/EVD。默认半径 0.95，stress 为 0.995。元数据记录 FP32 转换的 Frobenius 扰动和由生成器推导的范数界；FFT 构造自身的舍入没有经过严格区间认证，不能把该诊断当作实测谱半径或严格浮点稳定性证明。

```text
P0 = R0 R0ᵀ / n + 0.25 I
Q  = 0.05 × (R1 R1ᵀ / n + 0.25 I)
```

`R0/R1` 使用 CPU FP64 随机数，Gram 乘法在所选设备上用 FP64 于计时外生成。最终实际输入 `A/P0/Q` 存为 FP32，并记录其身份。参考从这些已存储的 FP32 值转换到 FP64，独立执行相同递推。输入矩阵是稠密的；循环矩阵结构仅用来控制谱性质，被测计算不调用 FFT 或其他结构化快速算法。因此它是受控的稠密算法 benchmark，尚不代表任意真实动力系统的分布。当前 `covariance` 不支持导入外部输入，传入外部资产会明确报错。

补充应用复用既有 suite 算法：

| kind | 应用与计时工作 | 独立参考/检查 | 结论范围 |
| --- | --- | --- | --- |
| `matrix_function`, `sqrt` | SPD 矩阵平方根的耦合 Newton–Schulz 迭代，每步 3 次 GEMM | FP64 EVD 平方根、平方残差、对称性 | 受控矩阵函数算法 |
| `matrix_function`, `inverse_sqrt` | SPD 逆平方根，可用于重叠矩阵正交化，每步 3 次 GEMM | FP64 EVD 逆平方根、正交化残差、对称性 | 正交化计算；真实分子需要导入真实重叠矩阵 |
| `purification` | 固定正交基 Hamiltonian 的 trace-correcting SP2，每步 1 次 GEMM 和 trace 分支 | FP64 占据子空间投影、幂等性、占据数和参考误差 | 冻结 Hamiltonian 的 projector；完整 SCF 的收敛、能量和力仍需另测 |

这两个既有应用的谱参考使用生成器的原始 FP64 矩阵，所以相对参考误差中包含 FP32 输入存储误差；`covariance` 明确使用已存储 FP32 输入的 FP64 参考。报告保留各应用自己的参考定义，不能将不同含义的 residual 混合平均成一个“精度分数”。

## 预先声明的质量合同

所有上限在配置中固定，所有输出还必须有限。主要指标使用 Frobenius 范数，`r` 为 SP2 占据数：

| 应用 | 指标与上限 |
| --- | --- |
| 协方差传播 | `relative_error = ||P - P_ref||F / ||P_ref||F <= 1e-5`；`symmetry_error = ||Pᵀ-P||F / ||P||F <= 1e-5`；默认还要求 `cholesky_failure = 0` |
| 平方根 | `residual = ||X²-S||F / ||S||F <= 1e-5`；`relative_error <= 1e-4`；`symmetry_error <= 1e-5` |
| 逆平方根 | `residual = ||X S X-I||F / sqrt(n) <= 1e-5`；`relative_error <= 1e-4`；`symmetry_error <= 1e-5` |
| SP2 | `residual = ||D²-D||F / sqrt(r) <= 1e-5`；`trace_error = |trace(D)-r| <= 1e-3`；`relative_error <= 1e-4` |

协方差正定性检查在计时外对 FP64 `(P+P.T)/2` 做 Cholesky；这是质量门槛的一部分，不修改被测输出，原始输出的非对称性仍单独受限。还保存最大绝对误差、绝对 Frobenius 误差、trace 及其相对误差。SP2 的能量等附加量是诊断，不自动成为完整物理任务的正确性证据。

native 调优同时启用 `strict: true`，保留历史局部 GEMM 门槛。局部 GEMM 门槛、应用质量、执行成功和性能胜出是不同字段。没有合格 SW 候选时应留下失败；不能自动替换为 dense 并将其解释为 SW 成功。stress 中高 condition number、近单位半径长递推和窄 gap 可能导致任何 backend 质量失败；保留失败和相应运行时间，不放宽门槛、不删除困难输入。

## Baseline 的合同

四条路径执行相同的 FP32 应用算法、输入和迭代次数：

| backend | 比较用途 | 精度/路由约束 |
| --- | --- | --- |
| `cublaslt` | 主配对基线 | 独立调优的 native dense，FP32 PEDANTIC 计算，FP32 输出；与 MegaFMM 使用一致公共接口和输出合同 |
| `megafmm` | 被测实现 | native 七乘积 SW，完整调优候选池，strict 局部门槛；记录实际 plan/route/workspace |
| `torch` | 同精度框架基线 | `torch.mm`，禁用 TF32，保留框架 dispatch 和应用调用成本 |
| `practical_torch` | 允许框架选择较快精度策略的实际基线 | 允许 TF32，仍须满足同一应用质量门槛；不能将未通过质量的时间称为合格加速 |

`allow_tf32=true` 表示允许使用 TF32，**不保证硬件、库或该 shape 实际选用了 TF32**。该字段不是 tensor core 执行证据；更强的路由声明需要额外 profiler/库日志。配置不要设置全局 `allow_tf32: false`，否则会同时收窄 `practical_torch` 的默认策略。

这里的 PyTorch 基线也使用同一 recurrence/迭代算法。计时外的 FP64 EVD 是参考，不是已经计时的 cuSOLVER/EVD 算法竞争者。结果能够支持“在这套算法中替换 GEMM 的应用收益”；声称胜过最优科学求解器、结构化协方差算法或专门的矩阵函数库，还需要在相同问题和质量合同下加入那些算法的实际计时。

## 四级配置

| 配置 | cases | 协议和用途 |
| --- | --- | --- |
| `bench/configs/scientific_smoke.json` | 协方差 n=128/129；sqrt、inverse sqrt、SP2 各 n=64 | 1 独立进程、1 次 warmup、3 groups × 2 calls、2 fresh quality trials、2 套轮换输入；仅检查集成/尾部/产物 |
| `bench/configs/scientific_core.json` | 协方差 n=1024/2048/4096/8192，各 5 步；sqrt n=512、inverse sqrt n=1024，condition=100、26 步；SP2 n=1024、gap=0.1、25 步 | 3 独立进程、2 warmup、5 groups × 2 calls、2 fresh quality trials、2 套轮换输入；主性能和质量对比 |
| `bench/configs/scientific_large.json` | 协方差 n=8192/12288/16384，各 5 步 | 可选大规模扩展，3 独立进程，worker timeout 7200 秒；必须根据实际显存和队列时间选择 |
| `bench/configs/scientific_stress.json` | 协方差 n=1025/4097、半径 0.995、20 步；矩阵函数 condition=1e4、40 步；SP2 gap=0.01、45 步 | 3 独立进程，保留严格质量门槛，查找数值和奇数尺寸边界 |

全部配置只使用 `launch` 模式，将完整 Python 调度应用作为主结果。core、large、stress 使用 `heuristic_candidates=16`、`tuning_input_sets=2`；smoke 分别为 4 和 1。全部使用 `sw_tuning_candidates=0`，其含义是使用完整 SW 候选池，不是关闭调优。全部 native workspace 上限为 2 GiB；这是每个 plan 的上限，不是进程总显存上限。

`large` 只是可运行配置，不能把列出的最大维度当作已经完成的实验。单个 FP32 n=16384 方阵就是 1 GiB，FP64 方阵是 2 GiB；多组输入、多个 backend、参考、验证和调优临时量会明显放大峰值。FP64 Gram、递推和 Cholesky 也消耗时间，虽然它们不进入稳态应用计时。应先完成 smoke/core，再在资源许可时逐个选择 large case。OOM、timeout 或调优失败均是覆盖范围的一部分。

## 计时、统计和胜出定义

**完整应用同步后的 wall time 是主要性能指标**。它包括输入到输出的一次应用执行所需的 Python dispatch、分配、transpose/cast、SW packing/reconstruction、元素运算和必须的同步。参考计算、输入生成、质量检查、plan 创建/调优以及 warmup 在稳态计时外；setup/cold 成本单独记录。稳态加速不能直接说明一次性使用的总耗时，实际应用重用次数较少时还要考虑 setup 成本。

同时保存 CUDA event 时间，用来解释设备时间和端到端时间的差异。报告应使用 wall time 做应用配对 speedup，不能挑选更有利的 event 结果。协方差另可报告传统算法等效吞吐：

```text
nominal_flops = 4 × iterations × n³
equivalent_fp32_gflops = nominal_flops / (steady_wall_ms × 1e6)
speedup = cublaslt_steady_wall_ms / backend_steady_wall_ms
```

等效工作量统计两个传统 GEMM，未计 O(n²) 加法；它不是 SW 实际执行的 FLOP/指令数。主要结论仍使用完整应用时间。

同一独立进程中各 backend 接收相同输入，计时组轮换 backend 顺序并轮换输入。调优输入与 fresh quality 输入使用分开的种子；质量检查还覆盖实际计时路径。每个进程以所有有效计时 group 均值的中位数作为 `steady_wall_ms`，配对独立进程的 speedup 再做几何平均和 bootstrap 区间。三个独立进程是当前配置的统计单位，group/call 数不能冒充独立重复数。smoke 只有一进程，不提供有意义的性能置信区间。

所有采集到的 group 计时均保留在 `samples.jsonl`，包含 call 数、顺序、event 和 wall 的每调用 group 平均时间；它不是逐调用 latency 分布。不能只保存最快一组，或将 group 平均值当作单次 tail latency。`implementation_pass`、`gemm_gate_pass`、`application_quality_pass`、`performance_win`、`qualified_win` 分开保存；主 speedup 的合格结论还需要被测实现和比较基线都通过适用质量要求。快但不合格的记录必须在报告中清楚显示。

## 运行和复现

先使用已经配置好的 CUDA/PyTorch 环境以及有效 GPU allocation。构建和执行使用现有 cluster wrapper，完整依赖与构建说明见 [通用 suite 文档](experiment_suite.md)。以下是可复用命令示例，并不代表这些尺寸已经测量。

```bash
# 配置检查不导入 PyTorch，也不启动 GPU worker。
python scripts/run_suite.py --config bench/configs/scientific_core.json --list
python scripts/run_suite.py --config bench/configs/scientific_core.json --dry-run

# 隔离冻结源码、构建并运行 CTest，不覆盖当前内核实验。
# 默认使用工作区 kernel；--kernel-ref 可明确选择已提交版本。
bash scripts/on_gpu.sh python3 scripts/prepare_scientific.py \
  --snapshot .scratch/scientific-YYYYMMDD-run01 --kernel-ref 14d5781e

# 以下 python 应来自已安装兼容 CUDA 的 PyTorch 环境。
bash scripts/on_gpu.sh python .scratch/scientific-YYYYMMDD-run01/source/scripts/run_suite.py \
  --config .scratch/scientific-YYYYMMDD-run01/source/bench/configs/scientific_smoke.json \
  --library .scratch/scientific-YYYYMMDD-run01/build/libmegafmm_suite.so \
  --output results/scientific/smoke-YYYYMMDD-run01

bash scripts/on_gpu.sh python .scratch/scientific-YYYYMMDD-run01/source/scripts/run_suite.py \
  --config .scratch/scientific-YYYYMMDD-run01/source/bench/configs/scientific_core.json \
  --library .scratch/scientific-YYYYMMDD-run01/build/libmegafmm_suite.so \
  --output results/scientific/core-YYYYMMDD-run01

# 可选：先只选择一个大尺寸，显存及时间充足时再运行其他 case。
bash scripts/on_gpu.sh python .scratch/scientific-YYYYMMDD-run01/source/scripts/run_suite.py \
  --config .scratch/scientific-YYYYMMDD-run01/source/bench/configs/scientific_large.json \
  --case covariance_fp32_12288 \
  --library .scratch/scientific-YYYYMMDD-run01/build/libmegafmm_suite.so \
  --output results/scientific/large-12288-YYYYMMDD-run01

python scripts/report_suite.py results/scientific/core-YYYYMMDD-run01
python scripts/audit_scientific.py results/scientific/core-YYYYMMDD-run01 \
  --source-root .scratch/scientific-YYYYMMDD-run01/source

# 已有 matplotlib 的环境可导出科学图；不需要 GPU。
python scripts/plot_scientific.py results/scientific/core-YYYYMMDD-run01
```

如 shell 的 `python` 不是已有 PyTorch 环境，应替换为该环境解释器的绝对路径。`--case` 可以重复，筛选精确 case ID。stress 使用同样命令更换 config 和新结果目录。GPU benchmark 不自动下载模型或数据。

每轮产物至少包括：

```text
manifest.json, resolved_config.json   源码/库/设备/环境/协议/输入身份
runs.jsonl                           每个 case/backend/process 的时间、路由和门槛
samples.jsonl                        全部原始 group 计时观测
metrics.jsonl                        fresh trials 与实际执行路径质量值
summary.json, summary.csv            配对统计、覆盖和质量计数
report.md, report.html               便于审阅的性能与质量报告
workers/...                          payload、terminal result、stdout/stderr
```

发布数据应同时给出实际测量的维度、迭代数、backend、GPU/驱动/库版本、源码和库 hash、独立进程数、质量通过/失败计数、wall 时间以及配对 speedup 区间；不能只给出“最高加速比”。配置包含的 case、跳过的 case 和实际测量成功的 case 必须分开计数。

manifest 还记录实际 GPU UUID、PyTorch/CUDA runtime fingerprint 和存在时的原生 build manifest。显存组峰值包含共享输入/参考和共同驻留的所有后端，不是该后端独占运行的最低显存。

`audit_scientific.py` 会核对配置所要求的 worker/backend/计时组覆盖、独立进程身份、配对输入、wall 加速比、聚合与 worker 原始记录的一致性以及源码/二进制 hash。质量观测也逐字段比较并保留非有限值，成功应用运行必须具有配置规定的 fresh、timed-before、timed-after 观测；最终质量判定必须等于全部适用观测的逻辑 AND。完整性通过不意味着数值质量通过，两者分别报告。

运行中断后可用相同命令和输出目录加 `--resume` 补缺失任务；它保留已有终态失败，并核对源码、库和协议身份。修改代码、配置或二进制后要创建新目录。质量失败默认返回退出码 2；`--allow-quality-failures` 可用于诊断，只改变退出策略，失败指标和状态仍保留。执行失败、OOM 或 timeout 不因该选项变为成功。实际性能数字和已完成覆盖以对应结果目录为准。

## 已完成实验

[2026-10-26 H100 实测结果](../results/scientific_20261026/RESULTS.md)保存本轮的实际覆盖、端到端时间、配对区间、质量失败和调优成本，方法说明与实测结论分开维护。本轮冻结 kernel `14d5781e493b0b7e3ec2b8373e2ec5ea014306d7`，没有覆盖工作区中的后续内核实验。core 已覆盖至 8192 阶；12288/16384 仅提供可选配置，没有计入本轮覆盖。
