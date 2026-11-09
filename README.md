# MegaFMM

面向 NVIDIA H100 的单层 Strassen–Winograd 七乘法研究工程，提供 BF16/FP16/FP32 GEMM、完整调用调优与应用实验套件。输入保持原 dtype，累加/输出 FP32；禁止补偿乘积、BF16→FP16 或完整 dense AB 替代 SW。**当前仅支持普通 launch 验收**；部分旧代码/配置仍有 CUDA Graph 分支。

主线使用通用 fallback、padded Hopper 与参数化 FP32 TMA 内核；固定尺寸特化已移到 [归档](archive/sw_specializations/README.md)。最近验证基线为默认 21 项 FP32 池 + iteration242 正确性修复 `14d5781e`；工作树可能包含未验收实验。固定 30 形状目前仅 5/30 独立认证领先 cuBLASLt、8/30 单轮领先，未实现全胜。历史五形状/六模板成绩见 [ITERATIONS](ITERATIONS.md)，不作为当前配置池认证。

## 构建与快速测量

使用 CUDA12.4、C++17、CUTLASS3.5.1（默认 `third_party/cutlass/`，CMake 可设置 `MEGAFMM_CUTLASS_DIR`），目标 `sm_90a`。构建/运行在有效 H100 allocation 中；`scripts/on_gpu.sh` 使用 `MEGAFMM_JOB_ID`，旧 allocation ID 不可直接复用。构建脚本记录源码hash并强制重编译，避免共享文件系统时间戳缓存；使用独立目录保留现有结果。

```bash
# 先将 MEGAFMM_JOB_ID 设为当前有效 allocation。
bash scripts/on_gpu.sh nvidia-smi
MEGAFMM_BUILD_DIR=build-suite MEGAFMM_RESULT_DIR=results/suite_build \
  bash scripts/build.sh

# 普通调用；严格检查原始输入 FP64 误差。
bash scripts/on_gpu.sh build-suite/megafmm_bench \
  --algorithm sw-tuned-strict --dtype fp32 \
  --shapes '1,1,1;33,65,129;512,512,512;1024,1024,1024;1025,1537,769' \
  --warmup 20 --trials 20 --groups 3 --correct-trials 3 --rotate-inputs 2 \
  --output results/suite_build/fp32-launch.csv
```

改用 `--algorithm dense` 测 cuBLASLt；此时输出的重复测量加速比不是 SW 收益。`--dtype both` 仅表示 BF16/FP16，`all` 包含 FP32。`--workspace-limit BYTES` 限制调用方预算，`--rotate-inputs` 控制轮换工作集；实际路径、workspace、误差、逐候选排名/确认及原始组数据随结果保存。完整选项由 benchmark `--help` 给出，当前使用不启用图模式。

形状及测量默认参数统一放在 [bench/shapes.json](bench/shapes.json)。使用
`python3 scripts/run_benchmark.py --config bench/shapes.json --binary build-suite/megafmm_bench --dtype fp32 --algorithm sw-tuned-strict`
读取配置；可用 `--dry-run` 仅打印最终命令。脚本按 `squares → rectangles → boundaries`
读取 `[M,N,K]`，同时读取 dtype、workspace、warmup、trials、groups 和 correctness_trials；
显式命令行参数覆盖配置。C++ 可执行文件继续接受 `--shapes '...'`，JSON 由 Python 标准库解析。

## 应用实验套件

覆盖 GEMM、FP32 covariance/sqrt/inverse-sqrt/SP2/LU，以及 BF16/FP16 Muon、GELU/SwiGLU MLP。提供匹配 native cuBLASLt、MegaFMM、独立 PyTorch 基线，支持固定工作量/达到质量时间、输入导出、进程隔离、原始样本及 HTML 报告。执行状态、质量和速度分列，smoke 不证明应用加速。

使用兼容 CUDA 的 PyTorch，依赖见 [requirements-suite.txt](requirements-suite.txt)。完整 schema/命令见 [experiment_suite.md](docs/experiment_suite.md)。旧配置可能包含 Graph，先生成 launch-only 配置：

```bash
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('bench/configs/smoke.json').read_text())
cfg['modes'] = ['launch']
for case in cfg['cases']:
    case['modes'] = ['launch']
path = Path('results/suites/smoke-launch.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2) + '\n')
PY

# list/dry-run 不导入 PyTorch，也不启动 GPU worker。
python scripts/run_suite.py --config results/suites/smoke-launch.json --dry-run
bash scripts/on_gpu.sh python scripts/run_suite.py \
  --config results/suites/smoke-launch.json \
  --library build-suite/libmegafmm_suite.so \
  --gemm-binary build-suite/megafmm_bench \
  --output results/suites/smoke-run
python scripts/report_suite.py results/suites/smoke-run
```

| 配置/文档 | 用途 |
|---|---|
| [smoke](bench/configs/smoke.json) / [core](bench/configs/core.json) / [stress](bench/configs/stress.json) | 接入检查；代表尺寸与独立进程；条件数/动态范围/谱隙/长迭代/边界 |
| [applications](bench/configs/applications.json) | 科学矩阵、优化器状态、checkpoint MLP replay；缺文件跳过，不替换为 synthetic |
| [LINPACK 套件](docs/linpack_suite.md) | 部分选主元分块 LU，比较 trailing updates；wall time、等效 GFLOPS、FP64 残差；[smoke](bench/configs/linpack_smoke.json)/[扫描](bench/configs/linpack.json)，仅 launch，非标准 FP64 HPL |
| [科学 benchmark](docs/scientific_benchmark.md) | 协方差 `P ← A P Aᵀ + Q` 为主，1024–8192 方阵及矩阵函数/SP2；scientific_smoke/core/large/stress 配置，strict FP32 与 TF32 分列 |

其他旧配置同样修改顶层和各 case 的 `modes`。每次用新输出目录；`--case` 可重复，`--resume` 保留终态失败并继续缺失任务，报告可重建。输入导出见 [export_suite_inputs.py](scripts/export_suite_inputs.py)，模型评估见 [evaluate_suite_model.py](scripts/evaluate_suite_model.py)，不自动下载模型/数据。

科学 benchmark 可在 GPU allocation 内用 `scripts/prepare_scientific.py --snapshot <新目录> [--kernel-ref <提交>]` 冻结隔离源码、构建和 CTest，保护活跃内核实验。[2026-10-26 实测](results/scientific_20261026/RESULTS.md)：core covariance 1024/2048/4096 合格加速 1.060/1.031/1.102×；8192 的 1.155× 因对称性失败不合格。core sqrt/SP2 合格，inverse sqrt 不合格；stress 无合格 SW 加速。4096 setup 约 17.8s（Lt 0.83s），收益依赖计划复用；可选 12288/16384 未测。

native 基线匹配输入/FP32 输出、cast、workspace；practical PyTorch 可能有输出舍入或不同精度，单独报告。每次 packing/转置/cast/重构/停止检查计入应用时间，参考 eigendecomposition 不计时。synthetic、replay、完整任务不能混称：Muon 不证训练收敛、冻结 Fock 不证 SCF 能量/力、MLP 不证完整模型质量；pedantic FP32 收益不代表领先 TF32 或最优科学求解器。

## API 与算法

[gemm.h](include/megafmm/gemm.h)：`Problem` 描述 M/N/K、stride、transpose、dtype、alpha/beta；`Policy` 设置算法与预算；`create_plan → tune → run`。

| Algorithm / benchmark 参数 | 行为 |
|---|---|
| Auto、Dense / `auto`、`dense` | 独立调优 cuBLASLt；FP32 PEDANTIC，BF16/FP16 FP32 compute |
| Tuned / `tuned` | 兼容旧枚举，使用同一 dense 实现 |
| SWPipeline / `sw` | 显式七乘法，按布局/精度/预算/对齐选通用实现 |
| SWTuned / `sw-tuned` | 显式 `tune()`，仅测量七乘法候选，不以完整 dense AB 兜底 |
| SWTuned / `sw-tuned-strict` | 要求原始输入 FP64 gate；无合格 SW 则报错 |

`tune()` 在稳态计时外，可分配/同步；`run()` 不分配、不 host 同步。调用方按 `plan->workspace_bytes()` 分配限额内 workspace，并发使用独立 workspace；绑定变化重查/重建 descriptor，不缓存输入值。支持各路径声明的布局、系数和 C0=D；零尺寸/零 K、非法 stride/溢出/重叠处理见 [PLAN](PLAN.md)。

```cpp
megafmm::Problem p;
p.m = p.n = p.k = 1024;
p.dtype = megafmm::DType::FP32;
megafmm::Policy policy;
policy.algorithm = megafmm::Algorithm::SWPipeline;
auto plan = megafmm::create_plan(p, policy);
// A/B/D 为设备 float*；此固定 FP32 路径 workspace_bytes()==0。
megafmm::run(*plan, A, B, nullptr, D, nullptr, stream);
// SWTuned 先按 Plan 分配 workspace，再显式 tune()，随后 run()。
```

## 路径与精度边界

FP32 固定 SW 使用 FP32 变换、SIMT 叶乘法与重构，零 global workspace；兼容绑定走双槽 TMA，其他走同步 fallback。固定 TMA 条件：非转置、M/N/K≤32768、N/K 为 8 倍数、A/B leading dimension 为 4 倍数、指针 16B 对齐；M 可为奇数，实际路径由 `load=tma/synchronous` 报告。

FP32 SWTuned 还测量 materialized/direct-B/fused 等通用模板，包含 pack、七叶和重构；cooperative 路由需实际驻留及绑定检查，否则使用普通 launch fallback。候选池和 workspace 以 [注册表](solution/sw/fp32/sw_fp32_batch.h) 与 Plan/API 为准，不能沿用旧六模板或固定七平面公式。BF16/FP16 的 `SWTuned` 使用统一 Hopper 模板的 9 项参数配置，涵盖原生输入、padding、短 K 流水和双 consumer warp group；按完整调用实测选择。`SWPipeline` 仍使用通用 padded 配置，其余绑定走零 workspace 通用 SW。模板参数、调用链与验证见 [Tensor Core 模板说明](docs/tensorcore_template.md)。

参考为已舍入原始输入的普通 FP64 GEMM。gate 为 `eF_SW <= 3*eF_dense + floor`，BF16/FP16/FP32 floor=`1e-3/1e-4/1e-6`，逐 fresh trial 检查且无 nonfinite。BF16/FP16 默认 SW 调优将旧 gate 作报告项，strict 可显式要求；FP32 保留原 gate。原生 BF16/FP16 仍有广泛精度失败；逐位正确、数值合格、应用质量和速度分别报告，Auto 不自动改用 SW。

## 验证与导航

纯主机套件检查：`python -m unittest discover -s tests -p 'test_suite_*.py'`；数学恒等：`python tests/test_dag.py`。GPU 验证须匹配实际配置/源码，历史 sanitizer/CTest 只证明各自范围；示例命令不是已执行证据。

| 位置 | 内容 |
|---|---|
| [HINTS](HINTS.md) / [PLAN](PLAN.md) / [ITERATIONS](ITERATIONS.md) | 最新交接与约束；DAG/成本/验收；结论、负结果、原文恢复入口 |
| [solution](solution/README.md)、`include/megafmm/` | 实现与 API；solution 文档中的旧配置/性能须按轮次理解 |
| `bench/benchmark.cu`、`bench/suite/`、`bench/configs/` | GEMM 测量/FP64 参考、应用执行、配置 |
| `scripts/`、`tests/` | 构建、导出/评估/报告、复现与回归 |
| `results/`、`trajectory/`、`research/` | 原始数据/审计、逐轮源码、研究与负结果 |
