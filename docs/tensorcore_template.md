# Hopper Tensor Core SW：统一参数模板

BF16/FP16 生产实现使用一个 `compute<T, Config>` 计算 kernel 模板和一个
`transform<T, Config>` 输入变换模板。每次完整调用包含输入变换、七乘法、FP32 重构；
运行时 M/N/K 仍由 `Problem` 提供。不同 tile 和调度策略是这两个模板的实例，
没有按 512/1024/4096 等矩阵尺寸注册独立 `.cu` 实现。

## 文件和调用链

- 公共接口：[gemm.h](../include/megafmm/gemm.h)。
- 参数、合法性约束、有限注册表：[sw_hopper.h](../solution/sw/hopper/sw_hopper.h)。
- 输入变换、TMA producer、统一 compute、资源查询和 launch：[sw_hopper.cu](../solution/sw/hopper/sw_hopper.cu)。
- WGMMA consumer 的算法定义：[generate_tensorcore_ptx.py](../scripts/generate_tensorcore_ptx.py)。
  CUDA inline PTX 需要字面寄存器列表，因此生成
  [sw_tensorcore_ptx.cuh](../solution/sw/hopper/sw_tensorcore_ptx.cuh)。
  生成器共享 barrier、K 循环、七产品顺序和重构定义；仅展开硬件指令操作数和 fragment。
  生成文件不定义独立 `__global__` kernel。修改生成器后执行
  `python3 scripts/generate_tensorcore_ptx.py`，用 `--check` 验证同步。

公共高速路径：

```text
create_plan(Problem, Policy{Algorithm::SWTuned, ...})
  → sw_configs(problem)：按布局、整除条件、workspace 和实际资源过滤
  → tune(...)：测量 transform + compute 完整调用，记录选择
  → run(...)
  → run_sw_config → sw_hopper::run → dispatch<T, Config>
  → transform<T, Config> + compute<T, Config>
```

`Algorithm::SWPipeline` 保持无需调优的通用 padded 配置及零 workspace 回退。
要获得恢复的原生高性能配置，请使用 `SWTuned` 并显式调用 `tune()`。
默认 `Auto`/`Dense` 仍使用 cuBLASLt。

```cpp
megafmm::Policy policy;
policy.algorithm = megafmm::Algorithm::SWTuned;
policy.workspace_limit = size_t(8) << 30;
auto plan = megafmm::create_plan(problem, policy);
// 分配至少 plan->workspace_bytes() 字节的 device workspace。
megafmm::tune(*plan, A, B, nullptr, D, workspace, stream);
megafmm::run(*plan, A, B, nullptr, D, workspace, stream);
```

`run()` 不分配内存、不 host 同步、不缓存输入变换。TMA map 按当前绑定重新编码。
调优成本不计入稳态调用，但需要应用复用 plan 才能摊销。

## 参数

```cpp
template<int N, int BK, int Stages,
         int Groups = 1, int U3Components = 0,
         bool Pending = false, int KTiles = 0,
         int PackThreads = 256, int PackCap = 512,
         int Raster = 0, bool Padded = false, bool SplitPack = !Padded, int PackVector = 8>
struct KernelConfig;
```

| 参数 | 含义 |
|---|---|
| `N` | A 的行跨度；每个 consumer 计算逻辑 `N × 64` tile |
| `BK`, `Stages` | K tile 和 TMA 环形缓冲深度 |
| `Groups` | 1 或 2 个 consumer warp group，逻辑输出列跨度为 `64 × Groups` |
| `U3Components` | 每个线程放入共享内存的 U3 分量数，范围 0 到 N/2，控制寄存器压力 |
| `Pending` | 允许一个待完成的 WGMMA group，并延迟对应 slot 的释放 |
| `KTiles` | 0 为运行时循环；1 用独立累加器跨产品流水；4 展开四个 K tile；均有 K 合法性约束 |
| `PackThreads`, `PackCap` | 输入变换 block 大小和 CTA 数上限 |
| `Raster` | CTA 遍历分组，0 表示默认顺序 |
| `PackVector` | 原生输入每线程每次处理 2 或 8 个元素；padded 使用 2 |
| `SplitPack` | A/B 分开分配 CTA；默认原生输入启用，padded 关闭 |
| `Padded` | 物化补零后的原始象限和变换；否则直接读取对齐的原始 A/B |

当前注册 9 项，ID 只是表内索引，不能作为稳定序列化接口：

| ID | 逻辑 tile | BK / stages | consumer 数 | 主要策略 |
|---:|---|---|---:|---|
| 0 | 64×64 | 128 / 4 | 1 | 通用 padded |
| 1 | 8×64 | 256 / 4 | 1 | 单 K tile、跨产品 pending |
| 2 | 32×64 | 128 / 8 | 1 | 四 K tile 展开、512 线程 paired 打包 |
| 3 | 64×128 | 64 / 4 | 2 | 16 个 U3 分量共享、合并打包 CTA |
| 4 | 128×128 | 64 / 4 | 2 | pending |
| 5 | 128×128 | 128 / 3 | 2 | 更大的 K tile |
| 6 | 192×128 | 64 / 3 | 2 | shared U3 |
| 7 | 192×128 | 64 / 3 | 2 | shared U3、pending、raster 4 |
| 8 | 16×64 | 128 / 8 | 1 | 窄 padded tile |

添加配置只需往 `KernelConfigurations` 元组添加合法参数；workspace、资源查询、
候选注册、运行时 dispatch 和名称自动派生。支持的 WGMMA 宽度及共享内存预算由
`static_assert` 检查。矩阵尺寸和绑定检查仍由 `eligible()` 和公共 Plan 完成。
BF16/FP16 调优池至多 10 项，其中 1 项为既有零 workspace 七乘法 fallback。

## 历史来源和数值合同

恢复依据是 `ee1c665f` 中的 combined full28：frontier short-K、冻结的 1024
调度、单/双 consumer 和宽 tile U3 策略。`9d051d0a` 简化后只保留的 padded
64×64/BK128/S4 不能代表此前的高性能候选池。

本次共享一个参数模板，并保留私有 PTX 累加器的生命周期；直接改写为 C++
fragment 操作曾引发编译器 WGMMA 串行化并显著变慢，失败数据记录在 ITERATIONS。
打包与计算都设置匹配的 shared-memory carveout。原生打包采用步长为 2 的幂时的位运算地址路径，并通过 `SplitPack` 分配 A/B CTA。

输入保持原生 BF16/FP16，累加和输出保持 FP32。七个叶子乘法、变换舍入和重构
次序保持既有算法。重构逐位匹配与原始 AB 的数值误差分开验证：历史原生 SW
在随机输入上的旧 FP64 精度门槛大多失败，本次性能恢复不意味着精度改善。
默认 SW 调优报告该误差；`sw_tuning_allow_inaccurate=false` 可要求严格门槛。

## 验证与复现

`tests/test_tensorcore_configs.cu` 直接覆盖所有注册配置和两种 dtype，使用三组
新输入检查原始 AB、输出和 workspace 哨兵；还包括非整齐尺寸、长 K 以及实际
资源可驻留检查。`tests/test_sw_paths.cu` 验证公开调用、预算和回退合同。

在获准的 H100 compute allocation 内使用 CUDA 12.4：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 9 build/test_tensorcore_configs -1 --short
compute-sanitizer --tool racecheck --error-exitcode 9 build/test_tensorcore_configs -1 --short
bash scripts/bench_tensorcore.sh iter-N
```

benchmark 默认读取 `bench/shapes.json` 全部 28 项，保留 20 warmup、5×100 完整
调用、3 fresh FP64 对照、2 rotating inputs 和 8GiB workspace。原始组、数值失败、
source/binary hash 均保留。可用既有 `MEGAFMM_SHAPES` 选择研究子集，子集结果不能
当作完整套件认证。独立进程重复用于检查小尺寸噪声；历史表与本次结果比较并非
同轮硬件锁频对照。

2026-10-27 最终完整 28 项测试及两轮小尺寸独立复测已经归档：
[结果报告](../results/tensorcore_template_20261027/REPORT.md)、
[逐项性能对照](../results/tensorcore_template_20261027/comparison.csv)。
16384 方阵 BF16/FP16 为 14.108/14.756 ms，相对修改前 padded 实现提升
4.78/4.56 倍。主要方阵仍比历史最低延迟慢约 3%–10%；只有两个标量 case
通过原精度门槛，不能将恢复吞吐解释为精度合格的 cuBLASLt 胜出。
8 项 CTest 及全部配置的 memcheck/racecheck/synccheck 通过。
