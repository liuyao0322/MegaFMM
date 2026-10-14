# MegaFMM

H100 上 Strassen–Winograd 矩阵乘法的研究工程。2026-10-14 按用户要求移除旧
SW 架构，准备重新设计统一 buffer 与 TMA、CUDA Core、Tensor Core 分工。

**当前只实现调优 cuBLASLt Dense 基线，尚无新的 SW kernel，也没有已认证的 SW
winner。** `Auto` 和 `Dense` 都执行普通 GEMM。旧报告中的 Mega45 等路线已从
当前源码和公共 API 删除，其数据仅代表历史版本。

## 当前目录

```text
solution/
  plan.cpp             Dense 规划、cuBLASLt 调优、指针检查和执行
  scale_output.cu      alpha=0 或 K=0 时的 FP32 输出处理
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
历史设计。新 buffer 布局、warp 分工和数值表示尚未实现或验证。

## 构建与验证

在有效的 H100 allocation 中使用 CUDA 12.4。脚本沿用 `MEGAFMM_JOB_ID`，默认
为历史 allocation 107488；使用前确认分配仍有效。

```bash
bash scripts/build.sh
bash scripts/check.sh
```

当前构建不依赖 CUTLASS。已缓存的 `third_party/cutlass/` 不参与编译。

保留的 Dense GPU 测试检查原始量化输入的 FP64 参考、两种 dtype、矩形、转置、
stride、指针元素对齐、alpha/beta、原位调优、fresh-input Graph 重放和 API 拒绝。
`tests/test_dag.py` 检查 Winograd 数学公式；`tests/mixed_precision_floor.py` 是
旧 one-cast 数值策略的独立限制示例，不代表新架构已认证。

直接测量 Dense 基线：

```bash
bash scripts/on_gpu.sh ./build/megafmm_bench \
  --algorithm dense --shapes '128,128,128;1024,1024,1024' \
  --dtype both --trials 100 --warmup 20 --groups 5 --correct-trials 3 \
  --output results/dense_baseline.csv
```

加 `--graph` 测量 CUDA Graph。当前比较双方都是独立调优的 Dense，输出的
`speedup` 仅反映重复测量差异，不代表 SW 加速。

`scripts/bench.sh <label>` 继续保留 AKO 的源码快照逻辑；正式迭代仍须记录
`ITERATIONS.md` 和对应 Git 版本。本次架构清理不属于性能优化迭代。

## 历史与清理记录

- [架构清理记录](results/architecture_reset_20261014.json)：当前保留内容和验证结果。
- [Dense 冒烟测试](results/architecture_reset_dense_smoke_20261014.csv)：BF16/FP16 原门限检查。
- [ITERATIONS.md](ITERATIONS.md) 和 `trajectory/` 保留历史上下文；历史源码和已提交报告可从 Git 查看。
- 当前 `results/` 中的旧报告已被移除，旧迭代中的文件路径不代表当前文件仍存在。
