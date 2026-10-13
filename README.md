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

`Policy::certified_range=true` 由调用方声明 PLAN 的有限值范围（维度不超过32768，
输入绝对值不超过16，alpha/beta 绝对值不超过1），不会自动扫描数据。强制实验
算法用 `WinogradLt`、`WinogradHopper` 或 `Megakernel`。未知范围、transpose、
不满足实验 fast path 对齐或 workspace 限制时使用普通 GEMM。Dense 支持
任意合法 leading dimension、transpose、奇数尺寸和退化问题。`levels=0` 为普通
GEMM，`levels=1` 为单层；双层须通过独立时间与误差关口后才能实现/启用，当前
显式拒绝 `levels=2`。

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
也包含全部变换。cuBLASLt 请求32个合法 heuristic 并实测择优，双方总 workspace
预算相同。返回2表示至少一个数值门槛失败，仍保留所有测量结果；返回1表示
构建或执行错误。`--correct-trials 0` 的数据标为 unchecked，不用于认证。

AKO4ALL 的实验假设、结果与提交记录保存在 [ITERATIONS.md](ITERATIONS.md)，
每次包装脚本运行在 `trajectory/` 保存源码快照、输出和 CSV。工程遵循
profile → 修改 → 测量 → 记录 → 提交，最终保留经过复验的实现。
