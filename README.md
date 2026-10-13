# MegaFMM

H100 上的单层 Strassen–Winograd 矩阵乘法实验工程。输入支持 BF16 和 FP16，
预加减在 FP32 中完成，最终叶操作数转回输入类型，Tensor Core 使用 FP32
累加，乘积与最终输出保持 FP32。数学、精度与性能验收依据见 [PLAN.md](PLAN.md)。

当前实现包含调优 cuBLASLt 普通 GEMM、完整七乘法 Winograd、CUTLASS Hopper
WGMMA 叶乘法，以及每个 CTA 独占四个输出 tile 的单次 launch megakernel。
自动调度在通过独立的数值和性能验收前使用普通 GEMM。强制实验路径仅表示选择
算法，不表示已通过应用精度认证。

## 构建与验证

依赖固定为 CUDA 12.4、CUTLASS 3.5.1、C++17，编译目标为 `sm_90a`。
在登录节点准备依赖，CUDA 构建与测量通过现有 Slurm allocation 执行：

```bash
bash scripts/setup_dependencies.sh
MEGAFMM_JOB_ID=107488 bash scripts/build.sh
bash scripts/check.sh
```

`scripts/on_gpu.sh` 通过 `srun --jobid` 建立设备绑定，只使用 allocation 分配的
一张 GPU 和四个 CPU。改变 allocation 时设置 `MEGAFMM_JOB_ID`；不会自动申请
或取消其他作业。若已经在 Slurm step 内，直接使用该 step。

独立整数/双线性 DAG 测试、GPU 功能测试和数值认证分开记录：

```bash
python3 tests/test_dag.py
bash scripts/on_gpu.sh ./build/test_correctness
bash scripts/on_gpu.sh ./build/test_correctness --strict-numerics
bash scripts/on_gpu.sh compute-sanitizer --tool memcheck ./build/test_correctness
```

默认 GPU 测试检查精确整数输入的算法、边界、布局、alpha/beta、原位输出及地址
合同。`--strict-numerics` 额外验证强制 SW 路径的固定误差门槛；未达门槛返回 2。
功能测试成功不等于 BF16/FP16 随机输入已经获得精度认证。

完整最终验证与完整测量入口：

```bash
bash scripts/verify_final.sh
bash scripts/final_suite.sh
python3 scripts/summarize_results.py --files \
  results/final_lt_launch.csv results/final_lt_graph.csv \
  results/final_lt_rotating.csv results/final_lt_rotating_graph.csv \
  results/final_hopper3_launch.csv results/final_hopper4_launch.csv \
  results/final_hopper3_graph.csv results/final_hopper4_graph.csv \
  results/final_mega3_launch.csv
```

两个最终脚本在固定数值门限失败时返回 2，并保留所有结果；运行时或功能错误
返回 1。不能把返回 2 当作通过。最终测量保留原测试集，增加预先固定的
12288/14336/16384 相邻大尺寸诊断窗口；launch、Graph、轮换输入分别保存。
`final_suite.sh` 也可接收测量标签只重跑选定组，例如
`bash scripts/final_suite.sh hopper3_graph hopper4_graph`；选定 CSV/log 会覆盖，
`final_runs.csv` 追加本次来源与退出码。

## API

公共入口在 [include/megafmm/gemm.h](include/megafmm/gemm.h)。

```cpp
megafmm::Problem p;
p.m = p.n = p.k = 8192;
p.dtype = megafmm::DType::BF16;
megafmm::Policy policy;              // Auto：使用已验收的路径
auto plan = megafmm::create_plan(p, policy);
void* workspace = nullptr;
if (plan->workspace_bytes()) cudaMalloc(&workspace, plan->workspace_bytes());
megafmm::tune(*plan, A, B, nullptr, D, workspace, stream);  // 计时之外
megafmm::run(*plan, A, B, nullptr, D, workspace, stream);   // 异步 steady-state
```

调用方持有 A/B/C0/D/workspace，并负责它们在 stream 完成前有效。`run` 不动态
分配、不进行输入内容扫描，也不隐式同步主机。`tune` 是显式预调优阶段，可以
同步及使用临时缓冲。C0 和 D 分开传入，beta=0 时允许 C0=nullptr；原位 C0=D
仅允许相同 leading dimension。A/B 与输出/workspace 重叠、C0/D 部分重叠会拒绝。
`tune` 会写 D；若 C0=D，调优内部会使用原 C0 快照，但调用方仍须在正式
计算前恢复自己希望使用的 C0 内容。

`algorithm_name()` 描述 Plan 首选家族。实际绑定的 A/B 地址不满足对齐时可
回退到 Dense；megakernel variant 3 的非完整 tile 或非零 beta 使用 variant 2。
本工程主性能 CSV 的输入由 cudaMalloc 分配，采用显式记录的尺寸和 beta=0，
其 `actual` 栏对应这些绑定条件；它不能作为任意重新绑定调用的运行时分支追踪。

`Policy::certified_range=true` 由调用方声明 PLAN 的有限值范围（维度不超过32768，
输入绝对值不超过16，alpha/beta 绝对值不超过1），不会自动扫描数据。强制实验
算法用 `WinogradLt`、`WinogradHopper` 或 `Megakernel`。未知范围、transpose、
不满足实验 fast path 对齐或 workspace 限制时使用普通 GEMM。Dense 支持
任意合法 leading dimension、transpose、奇数尺寸和退化问题。`levels=0` 为普通
GEMM，`levels=1` 为单层；双层须通过独立时间与误差关口后才能实现/启用，当前
显式拒绝 `levels=2`。

可用实验版本如下；它们均保持 FP32 预加减后的一次最终叶操作数 cast。

| 路线 | variant | 配置与用途 |
|---|---:|---|
| WinogradLt | 0 | 向量化 S/T packing、七个调优 Lt 叶乘法、向量化 FP32 U 重组 |
| WinogradHopper | 0/1 | 128×128×64 / 128×256×64 Cooperative 对照 |
| WinogradHopper | 2 | 64×128×64 常规 warp-specialized 对照 |
| WinogradHopper | 3/4 | 128×256×64、C=void；4 另使用 2×1 B multicast |
| Megakernel | 0/1 | M1：七个完整 P，64×32 / 64×64 tile |
| Megakernel | 2 | M3：三个 P/U 生命周期复用，64×128×32、12 KiB shared |
| Megakernel | 3 | M3 的完整 tile beta=0 双缓冲，24 KiB shared；其他条件用 2 |

variant 3 的完整 tile 条件是 M%128=N%256=K%64=0，A/B 地址16字节对齐，
lda/ldb为8的倍数。初始 Pingpong 对照遇到独立 synccheck barrier 告警，已在
第5轮替换为常规 warp-specialized；原始诊断与代码仍在 trajectory 中。

## 基准与优化记录

```bash
bash scripts/bench.sh baseline
MEGAFMM_ALGORITHM=hopper MEGAFMM_HOPPER_VARIANT=0 bash scripts/bench.sh
MEGAFMM_ALGORITHM=mega MEGAFMM_MEGA_VARIANT=0 bash scripts/bench.sh
MEGAFMM_GRAPH=1 bash scripts/bench.sh
```

包装脚本默认测4096、8192、16384方阵，两种 dtype，20次预热、100次计时、
5组随机交替比较、3轮新输入的完整 FP64 参考检查。可通过脚本中的
`MEGAFMM_SHAPES`、`MEGAFMM_DTYPE`、`MEGAFMM_TRIALS` 等变量选择诊断组。

```bash
bash scripts/on_gpu.sh ./build/megafmm_bench \
  --algorithm lt --dtype both \
  --shapes '512,512,512;8192,4096,8192;1025,1537,769' \
  --trials 100 --warmup 20 --groups 5 --correct-trials 3 \
  --graph --output results/example.csv
```

CSV 报告每种方法相同模式的完整 event 延迟、主机 wall 延迟、组间分位数与
离散度、实际 workspace、调优成本、相对普通 GEMM 的 speedup，以及对原始
已舍入输入的 FP64 参考误差。所有依赖输入的 S/T 变换每次调用重新计算，Graph
也包含全部变换。cuBLASLt 在256/16/2字节对齐候选池中分别请求最多32个 heuristic，合并去重后
实测合法候选择优；运行时按实际地址选兼容算法。双方总 workspace 预算相同。返回2表示至少一个数值门槛失败，仍保留所有测量结果；返回1表示
构建或执行错误。`--correct-trials 0` 的数据标为 unchecked，不用于认证。

AKO4ALL 的实验假设、结果与提交记录保存在 [ITERATIONS.md](ITERATIONS.md)，
每次包装脚本运行在 `trajectory/` 保存源码快照、输出和 CSV。工程遵循
profile → 修改 → 测量 → 记录 → 提交，最终保留经过复验的实现。


## 当前验收与适用边界

代码已通过独立 DAG 恒等式检查、完整结构/API 测试，以及最终 memcheck、
racecheck、synccheck（错误/告警均为0）。严格数值压力组共有300个候选检查，
两种 dtype 各150个；各有60个失败，全部输出有限。失败集中于 uniform 和
范围内幅度压力组。高 FP16 幅度组走 Dense 回退，其通过不能认证 SW。

固定门限为 `eF_SW <= 3*eF_Dense + floor`，BF16 floor=1e-3，FP16 floor=1e-4。
原始已舍入输入的 FP64 GEMM 是参考。初始、所有优化轮次和最终复测使用同一
门限，未因候选结果而放宽。`tests/mixed_precision_floor.py` 提供独立量化反例，
即使叶乘法精确计算也失败；tile/流水线调整无法恢复最终 cast 已丢掉的信息。
因此 Auto 在当前认证范围全部使用 Dense，双层和更激进融合没有获得启用条件。

最终证据在 [results/REPORT.md](results/REPORT.md)、[ITERATIONS.md](ITERATIONS.md)
和 `results/final_*.csv`。最终94行中，88行实际 SW 均未通过数值门限，
6行 Dense 回退通过，unchecked 与非有限输出均为0。源码、二进制与 CSV
校验和、设备及完整协议见 [results/final_manifest.json](results/final_manifest.json)。
性能图/误差图保存为 `results/plots/` 的 SVG/PNG，
不同模式、缓存条件和回退行分开。组间 p10/p90 是5个组平均时间的分位数，
不是100次调用的尾延迟，也不是置信区间。GPU频率未锁定；clock/power/temperature
快照随每个最终数据集保存，性能结论须结合组间离散度。

仍未实施或复现的计划候选包括 M2 的逐 K-stage 输出展开、更广的输入融合、
product-aware batch/scheduler、tail peeling、双层，以及同合同的 SubCuber 性能
对照。已核验作者源代码的比较边界，记录在
[results/subcuber_compatibility.json](results/subcuber_compatibility.json)：原 FP16
输出为 half、缺 BF16，部分 global-presum preset 会在预热后复用 B 变换，需
适配后才能与本工程动态 A/B、FP32 输出公平比较。这里没有引用其速度作为成绩。
