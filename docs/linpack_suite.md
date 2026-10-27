# MegaFMM LINPACK-style suite

该套件在单张 GPU 上求解稠密线性方程组 `Ax=b`，比较 cuBLASLt 和
MegaFMM Strassen–Winograd（SW）在完整分块 LU 求解中的表现。它复用应用
suite 的独立进程、输入轮换、fresh correctness trials、源码/输入/库 hash、
原始计时、失败保存及 HTML/CSV 报告。

**这是 FP32 LINPACK 风格应用测试，不是标准 FP64 HPL，也不是 HPL-AI。**
MegaFMM 的公共 GEMM 接口没有 FP64 路径，不能把这里的结果报告为 TOP500
HPL Rmax。输入生成器也不复现 HPL 的随机序列。它用于回答：在相同 LU
算法、面板求解和精度合同下，用 MegaFMM 替换矩阵更新是否影响整体速度和质量。

## 算法和精度

每个 block 执行以下步骤：

1. 用 PyTorch FP32 LU 分解整个矩形 panel，在所有剩余行中部分选主元。
2. 将 successive pivots 应用到之前的 L 列、右侧矩阵和右端项。
3. 用相同的 PyTorch FP32 三角求解计算 `U12`。
4. 用被测 backend 计算 `L21 @ U12`，执行 `A22 -= L21 @ U12`。

最后执行前代和回代得到单个右端项的解。两个 native backend 使用完全相同的
面板和三角求解代码；仅 Schur update 的 GEMM 路径不同。`megafmm` 使用
SW 调优器，`cublaslt` 使用独立调优的 PEDANTIC FP32 dense 基线。
默认配置启用 strict local GEMM 调优门槛；不合格候选不会变成 dense fallback。
`torch` backend 可用于 CPU 功能检查，也执行同一分块算法；它不是独立的最优
`torch.linalg.solve` 性能基线。

默认矩阵由固定 CPU 随机种子生成，元素均匀分布在 `[-1,1)`，随后存为 FP32。
先用这个已存储矩阵和全一向量构造 b，再将 b 存为 FP32。最终质量参考是
**实际存储的 FP32 A、b 的独立 FP64 solve**，不假定舍入后精确解仍为全一。
每次求解均从原始输入重新开始；不复用已分解矩阵。

## 质量与性能定义

在计时外用 FP64 计算残差，默认质量门槛为：

```text
backward_error = ||Ax-b||inf / (||A||inf * ||x||inf + ||b||inf)
scaled_residual = backward_error / (n * torch.finfo(torch.float32).eps)
quality_pass = all_outputs_finite && scaled_residual <= 16
```

矩阵无穷范数是最大绝对行和；向量无穷范数是最大绝对元素。
这里显式使用 `torch.finfo(fp32).eps = 2^-23`，不是 FP64 HPL 的 epsilon。
残差公式参考 [Netlib HPL 算法](https://www.netlib.org/benchmark/hpl/algorithm.html)，
16 的默认阈值参考 [HPL tuning](https://www.netlib.org/benchmark/hpl/tuning.html)。
将公式用于 FP32 是本套件的精度合同，不代表通过标准 HPL。

同时保存 `residual_inf`、`backward_error`、FP64 参考解的 `relative_error`，以及
用 FP64 epsilon 缩放的诊断残差 `scaled_residual_fp64_diagnostic`。后者不参与
默认 FP32 门槛。可以在 case 的 `quality` 中设置 `scaled_residual`、
`relative_error`、`backward_error` 或 `residual_inf` 的上限。
默认只约束后向误差；病态系统即使后向稳定，也可能有较大前向误差。
奇异矩阵或失败的 panel LU 作为执行失败保存，返回非零退出码。

性能使用同步后的 **wall time**，包括输入复制、分配、panel LU、pivot/status
的 host 同步、行交换、三角求解、每次 packing/cast、SW GEMM 和尾部更新。
参考求解、残差检查、输入生成及 native Plan 创建/调优在稳态计时外，setup
单独记录。由于有必须的 host 决策，仅允许 `launch` 模式。

```text
nominal_flops = (2/3) * n^3 + 2 * n^2
effective_gflops = nominal_flops / (steady_wall_ms * 1e6)
speedup = cublaslt_steady_wall_ms / megafmm_steady_wall_ms
```

这个 FLOP 数是传统 LU 加单 RHS 求解的等效工作量；不是 SW 实际执行的浮点
指令数，也不是标准 FP64 HPL Rmax。报告中的 GFLOPS 标注为 Equivalent FP32。
小矩阵可能主要测到 Python 调度、panel 和同步开销；不能把 GEMM 加速等同于
完整 solver 加速。`n <= block_size` 的功能边界不调用 GEMM，不能比较 SW 收益。

## 运行

先选择有效 H100 allocation 和已经安装 PyTorch/NumPy 的 Python，不自动下载依赖。
构建使用现有脚本和完整默认 target 集：

```bash
export MEGAFMM_JOB_ID=107488  # 示例；先用 squeue 核验有效 allocation
export SUITE_PYTHON=/share/home/liuyao/miniconda3/envs/lragent-train/bin/python
MEGAFMM_BUILD_DIR=build-linpack MEGAFMM_RESULT_DIR=results/linpack-build \
  bash scripts/build.sh

# 配置检查不导入 PyTorch，也不运行 GPU。
python scripts/run_suite.py --config bench/configs/linpack.json --list
python scripts/run_suite.py --config bench/configs/linpack.json --dry-run

# 快速检查：128 和 129 两个尺寸，包含不能整除 block 的尾部。
bash scripts/on_gpu.sh "$SUITE_PYTHON" -s scripts/run_suite.py \
  --config bench/configs/linpack_smoke.json \
  --library build-linpack/libmegafmm_suite.so \
  --output results/linpack-smoke

# 完整配置：512/1024/2048/4096，128/256 两种 block 和 1025 尾部。
# 每个 case 三个独立进程；每次实验使用新输出目录。
bash scripts/on_gpu.sh "$SUITE_PYTHON" -s scripts/run_suite.py \
  --config bench/configs/linpack.json \
  --library build-linpack/libmegafmm_suite.so \
  --output results/linpack-core

python scripts/report_suite.py results/linpack-core
```

可以重复使用 `--case ID` 筛选尺寸；`--resume` 仅补缺失任务，保留终态失败。
质量失败默认退出码为 2；`--allow-quality-failures` 仅用于诊断且保留失败记录。
一个独立进程的 smoke 不能建立性能置信区间。

结果包括 `runs.jsonl`（wall/event 时间、等效 GFLOPS、实际 native 路由）、
`samples.jsonl`（原始 group 均值）、`metrics.jsonl`（fresh 和计时路径质量检查）、
`summary.csv`、`report.html` 及 `report.md`。`manifest.json` 保存源码和二进制身份。

## 导入自己的系统

可以用现有安全 PT/NPZ loader 提供 `A` 和 `b`。A 必须是非空方阵，b 的形状为
`(n,)` 或 `(n,1)`；载入后存为 FP32。输入及文件 hash 随结果保存。

```json
{
  "id": "linpack_imported",
  "kind": "linpack",
  "dtype": "fp32",
  "block_size": 128,
  "input_path": "inputs/system.npz",
  "quality": {"scaled_residual": 16, "relative_error": 0.001}
}
```

将此 case 放入完整配置即可运行。相对输入路径从仓库根目录解析；缺失的外部
文件按现有 suite 行为记为 skipped，不生成替代矩阵。输入含奇异系统时会失败。

CPU 回归无需 native library：

```bash
OMP_NUM_THREADS=1 "$SUITE_PYTHON" -s -m unittest discover \
  -s tests -p 'test_suite_*.py'
```

## 本次验证记录

2026-10-23 在现有 H100 80GB allocation 上完成完整默认 CMake 构建、
128/129 阶 native smoke，以及最终完整配置的 6 cases × 3 processes × 2 backends。
最终 36 条后端记录全部执行成功并通过应用质量检查，保留了 180 条原始 group
计时和 216 次残差检查；最大 FP32-scaled residual 为 0.0249676122，门槛为 16。
最终 manifest 的源码与 shared library hash 均已核验匹配。
主机回归共 65 项，64 项通过，1 项需要 opt-in 的额外 GPU bridge 测试跳过；
两种 native backend 已通过上述实际 suite 运行验证。

本地最终产物：

- [HTML 报告](../results/linpack/core-final-20261023/report.html)
- [CSV 汇总](../results/linpack/core-final-20261023/summary.csv)
- [原始运行与 native 路由](../results/linpack/core-final-20261023/runs.jsonl)
- [质量数据](../results/linpack/core-final-20261023/metrics.jsonl)
- [源码/库身份](../results/linpack/core-final-20261023/manifest.json)

这轮配对几何平均加速比约为 0.995–1.011，说明完整求解的差异很小。
这些数字仅覆盖本配置、当前实现和本轮环境，不能据此宣称通用 solver 加速。
所有结果目录保存在本地 `results/linpack/`，未选取最有利的一轮替换完整记录。
