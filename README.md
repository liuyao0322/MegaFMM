# MegaFMM

H100 上 Strassen–Winograd 矩阵乘法的研究工程。2026-10-14 按用户要求移除旧
SW 架构后，实现统一 buffer 与 TMA、CUDA Core、Tensor Core 分工的首版。

**当前实现调优 cuBLASLt Dense 基线，以及仅支持 1024³ 的实验性 SW 首版。**
新 SW 在本次测试输入上通过原精度门限，但明显慢于 Dense，不能作为性能 winner。
`Auto` 和 `Dense` 都执行普通 GEMM；`SWPipeline` / `--algorithm sw` 显式执行新内核。
旧 Mega45 等路线已从当前源码和公共 API 删除。

[当前固定 slot 与分阶段供数版本报告](trajectory/20261014_validation-iter87-staged-ready/REPORT.md)

[基础 descriptor 优化报告（iter85）](trajectory/20261014_validation-iter85-descriptors/REPORT.md)

[流式重组版本报告（iter82）](trajectory/20261014_validation-iter82-streaming/REPORT.md)

[P shared bank conflict 修复报告](trajectory/20261014_validation-iter81-p-banks/REPORT.md)

[生产链条分段计时：TMA 等待、WGMMA 提交与完成等待](trajectory/20261014_profile-production-chain/REPORT.md)

[最初首版报告（历史）](trajectory/20261014_183039_validation-graph-1024/REPORT.md)

当前源码为 iter87，`sw_cooperative_hilo_64x32x256`。沿用 P1→P2→P6→P7→P5→P3→P4
流式重组；展开两个TC operand slot，并将TMA高、低部分分阶段发布，hi×hi发射后再等待低部分。
同期 A/B/C/C/B/A 对照较iter85，普通调用 BF16/FP16 耗时下降约 **6.32%/6.76%**，
Graph下降约 **7.43%/8.74%**。普通调用当前约 **19.29/19.30 μs**。
正式独立bench SW为 **19.32/19.28 μs**，同期调优cuBLASLt Dense为 **12.30/9.95 μs**，
SW仍慢 **1.57/1.94倍**。R2UR由995968降至178944/dtype，P shared读写仍1-way、0额外wavefront。
原精度门限、72项独立launch/Graph检查及3种quick sanitizer均通过。
仅验证1024³，Auto继续Dense，不能称为性能winner。

每次调用先在调用方提供的 **8 MiB 全局 workspace** 中重新生成 FP32 S/T 的高低部分，
通过 cooperative grid barrier 后，以 TMA 装入统一 shared buffer 并执行 WGMMA。
128 个 CTA，每个 512 线程：1 个 TMA warp、11 个 CUDA Core warp、4 个 WGMMA warp。
动态 shared 每 CTA 213120 字节，BM=64、BN=32、BK=256。需要支持至少128个同时驻留
CTA 的 Hopper cooperative launch；当前在132 SM H100上验证。`run()`不分配、不进行host同步。
高低部分补偿共有15项叶乘法，原始FP64参考和数值门限保持不变。详见当前报告。

## 当前目录

```text
solution/
  plan.cpp             Dense 规划、cuBLASLt 调优、指针检查和执行
  scale_output.cu      alpha=0 或 K=0 时的 FP32 输出处理
  sw_pipeline.cu       1024³ cooperative SW、统一 S/T workspace、TMA/WGMMA 流水
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
历史设计。新 buffer 布局、warp 分工和数值表示已实现，验证范围仅为本次 1024³ 测试集合。

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
Graph 的新输入重放。SW 仅支持非转置、连续布局、alpha=1、beta=0；不满足条件时
强制 SW 抛错，不静默退回 Dense。Auto 始终保留 Dense。

`scripts/bench.sh <label>` 保留 AKO 的源码快照逻辑；正式迭代记录 `ITERATIONS.md`
和对应 Git 版本。持续 AKO4ALL 优化已按用户要求暂停；本次按明确请求完成 P shared bank conflict 修复和下一版流式重组试验。

## 历史与清理记录

- [架构清理记录](results/architecture_reset_20261014.json)：当前保留内容和验证结果。
- [Dense 冒烟测试](results/architecture_reset_dense_smoke_20261014.csv)：BF16/FP16 原门限检查。
- [ITERATIONS.md](ITERATIONS.md) 和 `trajectory/` 保留历史上下文；历史源码和已提交报告可从 Git 查看。
- 当前 `results/` 中的旧报告已被移除，旧迭代中的文件路径不代表当前文件仍存在。
