# MegaFMM

H100 上 Strassen–Winograd 矩阵乘法的研究工程。2026-10-14 按用户要求移除旧
SW 架构后，实现统一 buffer 与 TMA、CUDA Core、Tensor Core 分工的首版。

**当前保留 iter282（1bdfb29）的七乘法、无补偿实验内核。持续优化目标尚未完成。**
完整A/B/B/A普通调用：BF16/FP16 SW约 **10.273/10.238 μs**，同期调优Dense约 **12.228/9.807 μs**。
BF16加速比约1.190×；FP16仍慢约4.40%，还需减少约0.432 μs（SW时间的4.22%）。
Graph单独配对：SW约 **9.976/9.993 μs**，Dense约 **11.274/9.341 μs**；不替代普通调用目标。
279相对266普通BF16/FP16改善约2.43%/0.70%；282相对279仅约0.47%/0.35%，没有新的3%以上突破。
313逐字恢复后的新测BF16/FP16约10.293/10.246 μs，Dense约12.469/9.791 μs；FP16仍需减少约0.455 μs。
292–312的cluster分工、本地shared变换及同步改动均未改善保留版普通调用；错误或未编译候选明确排除。

**两种输入仍未通过原始精度门槛**：BF16 eF约0.0060152，FP16约0.000754626。
`Auto` 和 `Dense` 继续普通调优 cuBLASLt GEMM；当前 SW 必须显式指定。

[当前持续优化检查点（292–313，目标仍在进行）](trajectory/20261015_optimization-goal-checkpoint-313/REPORT.md)

[此前271–291检查点](trajectory/20261015_optimization-goal-checkpoint-291/REPORT.md)

[此前249–270检查点](trajectory/20261015_optimization-goal-checkpoint-270/REPORT.md)

[此前209–248检查点](trajectory/20261015_optimization-goal-checkpoint-248/REPORT.md)

[此前165–208检查点](trajectory/20261015_optimization-goal-checkpoint-208/REPORT.md)

[此前性能配对与完整独立验证（109–162）](trajectory/20261015_optimization-goal-checkpoint/REPORT.md)

当前实现使用4 MiB调用方workspace、196736B动态shared和八个BK128 operand slots。
输入变换是128 CTA×512线程的primary kernel；乘法是128 CTA×256线程的secondary，
TMA warp0、TC warps4–7，warp1 leader提前预取四个当前descriptor。
叶乘法按B转置乘A转置组织，TC私有寄存器直接重组输出；P1/P2可与变换重叠，及时释放slot。
使用官方PDL与首次变换块前的grid dependency wait，两个kernel统一设置100%shared carveout。
`run()`无分配、无host同步；每次为当前指针编码tensor maps，无跨调用计算值缓存。

279和当前282分别通过72项fresh-input逐位检查、三种quick sanitizer，以及256次双stream/Graph调用。
282验证源码与当前solution逐字一致。这些检查验证实现与同步，**不代表通过原始FP64精度验收**。

[历史四slot预取优化（iter100）](trajectory/20261015_validation-seven-prefetch/REPORT.md)

[历史七乘法 profiling（iter97）](trajectory/20261015_profile-iter97-seven-products/REPORT.md)

[历史补偿版输入变换优化（iter96，精度通过但更慢）](trajectory/20261014_validation-iter96-supply/REPORT.md)

## 当前目录

```text
solution/
  plan.cpp             Dense 规划、cuBLASLt 调优、指针检查和执行
  scale_output.cu      alpha=0 或 K=0 时的 FP32 输出处理
  sw_pipeline.cu       1024³ PDL 双kernel SW、统一 S/T workspace、TMA/WGMMA 流水
  sw_multisize.cu      多尺寸单层 Winograd、WMMA 七乘法、逻辑补零与融合输出
  internal.h           内部声明
include/megafmm/gemm.h  Problem、Policy、Plan 和 run/tune 接口
bench/benchmark.cu     普通 launch/Graph 计时和原始输入的 FP64 参考检查
bench/shapes.json      历史形状集合
scripts/               构建、检查、评测、profiling 和 AKO 记录辅助脚本
tests/                 Dense API/数值测试及独立数学模型
results/               当前验证结果与清理记录
trajectory/            历史逐轮源码快照和运行输出
```

## 基线合同

- A/B 使用 BF16 或 FP16；cuBLASLt 使用 `CUBLAS_COMPUTE_32F`，输出 FP32。
- 支持 row-major、transpose、leading dimension、alpha/beta 和兼容的 C0/D 原位视图。
- `tune()` 显式执行测量，可分配及同步；`run()` 不分配、不进行 host 同步。
- workspace 由调用方分配，受 `workspace_limit` 限制。
- 精度参考基于已经舍入到输入 dtype 的原始 A/B，以 FP64 计算普通 AB。
- 新 SW 实现仍须满足原门限 `eF_SW <= 3*eF_dense + floor`：BF16 floor 为
  `1e-3`，FP16 为 `1e-4`；所有 fresh trials 都须通过且无 nonfinite。
- 精度未通过的方案只能记录为诊断结果，不能称作合格 winner 或进入 Auto。

原 `PLAN.md` 的数值合同和数学公式继续作为参考；其中旧架构、路线与覆盖描述属于
历史设计。1024 专用路径保留原 buffer 布局与 warp 分工；多尺寸通用路径见下文。

## 构建与验证

在有效的 H100 allocation 中使用 CUDA 12.4。脚本沿用 `MEGAFMM_JOB_ID`，默认
为历史 allocation 107488；使用前确认分配仍有效。

```bash
bash scripts/build.sh
bash scripts/on_gpu.sh ctest --test-dir build -R 'sw_1024_correctness|exact_winograd_dag' --output-on-failure
```

新 SW 使用 CUTLASS 3.5.1 的 CuTe 布局、WGMMA/TMA 和 barrier 头文件。
默认位置为 `third_party/cutlass/`，也可在 CMake 配置时设置 `MEGAFMM_CUTLASS_DIR`。
构建脚本强制重新编译并记录源码 SHA256，避免共享文件系统时间戳缓存沿用旧对象。

保留的 Dense GPU 测试检查原始量化输入的 FP64 参考、两种 dtype、矩形、转置、
stride、指针元素对齐、alpha/beta、原位调优、fresh-input Graph 重放和 API 拒绝。
`tests/test_dag.py` 检查 Winograd 数学公式；`tests/mixed_precision_floor.py` 是
旧 one-cast 数值策略的独立限制示例，不代表新架构已认证。

直接测量 Dense 基线：

```bash
bash scripts/on_gpu.sh ./build/megafmm_bench \
  --algorithm dense --shapes '1024,1024,1024' \
  --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 \
  --output results/dense_baseline.csv
```

加 `--graph` 测量 CUDA Graph。当前比较双方都是独立调优的 Dense，输出的
`speedup` 仅反映重复测量差异，不代表 SW 加速。

首版 SW 只测 1024：

```bash
MEGAFMM_ALGORITHM=sw bash scripts/bench.sh <label>
MEGAFMM_ALGORITHM=sw MEGAFMM_GRAPH=1 bash scripts/bench.sh <label>
bash scripts/on_gpu.sh ./build/test_sw1024
```

`test_sw1024` 仅执行 1024³，检查两种 dtype、六类输入、每类三轮新输入以及同一
Graph 的新输入重放。它仍使用原始 FP64 精度门限，当前无补偿实现预计未通过。

## 多尺寸 SW

显式选择 `Algorithm::SWPipeline`（benchmark 中为 `--algorithm sw`）时：

- 连续、非转置的 1024³，且 alpha=1、beta=0，继续使用原 TMA/WGMMA PDL 专用路径，
  workspace 为 4 MiB，预算检查与实际分配一致。
- 其余非退化尺寸使用 `sw_multisize_wmma`：支持 BF16/FP16、矩形、奇数边长、转置、
  leading dimension、alpha/beta 和兼容 C0/D 原位视图。尺寸范围沿用公共 API 的 INT_MAX 上限。
- 通用路径将每个维度逻辑补到偶数，再切为四个子块；边界读取补零，不分配 padding buffer。
  每个 warp 计算一个 16×16 子块的七个 Winograd 乘法，输入变换在 shared memory 中完成，
  FP32 累加及输出重组，调用方 workspace 为 **0 字节**。
- M/N 为零或 K/alpha 为零时，分别走已有 empty/scale 路径。
- `run()` 保持无分配、无 host 同步，支持 CUDA Graph 和新输入重放。GPU 目标仍为 Hopper sm90。

通用路径用于尺寸与接口覆盖，尚未进行性能调优，不能沿用 1024 专用路径的加速比。
它沿用无补偿的变换舍入策略，**不保证通过原始输入 FP64 精度门限**，Auto 仍使用 Dense。
实现正确性测试将整数输入与原始 AB 精确比较，并将非整数输入与独立的标量 Winograd
舍入参考比较；后者不能代替原始 AB 精度验收。

```bash
bash scripts/on_gpu.sh ctest --test-dir build \
  -R 'sw_multisize_implementation|gpu_correctness|exact_winograd_dag' --output-on-failure
bash scripts/on_gpu.sh ./build/megafmm_bench \
  --algorithm sw --shapes '512,512,512;1025,1537,769;2048,2048,2048' \
  --dtype both --trials 10 --warmup 3 --groups 3 --correct-trials 2 \
  --output results/multisize.csv
```

benchmark 会报告实际路径、workspace、原始 FP64 精度和计时；未过精度门限时退出码为 2。
`test_multisize` 覆盖小尺寸、512³、1025/2048/4096 行矩形、四种转置组合、非连续布局、
元素对齐指针、alpha/beta、原位输出、padding 保护、fresh-input launch/Graph 和退化尺寸。

`scripts/bench.sh <label>` 保留 AKO 的源码快照逻辑；正式迭代记录 `ITERATIONS.md`
和对应 Git 版本。按用户最新指令持续 AKO4ALL 优化，目标仍在进行；每轮保存失败及有效结果，未通过精度或性能验收不宣布合格 winner。

## 历史与清理记录

- [架构清理记录](results/architecture_reset_20261014.json)：当前保留内容和验证结果。
- [Dense 冒烟测试](results/architecture_reset_dense_smoke_20261014.csv)：BF16/FP16 原门限检查。
- [ITERATIONS.md](ITERATIONS.md) 和 `trajectory/` 保留历史上下文；历史源码和已提交报告可从 Git 查看。
- 当前 `results/` 中的旧报告已被移除，旧迭代中的文件路径不代表当前文件仍存在。
