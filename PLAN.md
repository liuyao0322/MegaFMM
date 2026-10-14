> **2026-10-14 架构重置：** 旧 SW 实现和变体接口已删除，当前只保留调优
> cuBLASLt Dense 基线，并新增仅支持1024³的统一buffer实验首版，见README中的实测报告。
> 本文的数学公式与固定数值门限继续作为参考；旧实现、
> 调度和覆盖描述属于历史记录。新首版通过本次测试数值门限，但性能慢于Dense，Auto不变。
> 当前工程状态见 [README.md](README.md)，当前指令见 [HINTS.md](HINTS.md)。

# MegaFMM：面向 H100 的高性能 Strassen–Winograd 实现计划

日期：2026-10-13。下文保留实施前的设计与验收合同，新增实施状态与实验快照。

当前执行更新为 iteration23，测量源 commit 为
`ea4dc50c57b08b4dee98c23c024a01bb94f836db`。工程保留完整单层 V1、Hopper
V2、Lt batch7、Sm80 batched leaves，以及 M1/M2/M3、warp/product/cluster
megakernel。较快函数与对照 variant 使用独立入口并存，以同一冻结二进制复测。
小矩阵预声明九形状、两种dtype、20个请求配置各自 launch/Graph，共40 datasets、
720行，全部是已检查的实际SW、全部固定数值失败，nonfinite、unchecked和Dense
fallback均为零。更广 strict 集合31个请求配置（23个实际route名称）、930次检查为558通过/372失败/
nonfinite零；DAG、完整结构/API、memcheck、racecheck、synccheck五项均exit0。
完整协议为20次warmup、每组100次调用、5组随机交替、每行3轮新输入正确性。
结构通过不能替代随机数值门限，Auto继续使用调优Dense，强制SW用于实验。

这是一份当前性能快照，持续优化仍获授权，不表示已达到延迟下限、穷尽可行方向或
完成P0–P5。直接寄存器operand等scratch方向仍待接入与同合同验证。双层、较广
V3/V4、通用product-aware scheduler、专用tail peeling与同合同SubCuber性能
复现仍开放。详见 [README.md](README.md)、[ITERATIONS.md](ITERATIONS.md)、
[当前小矩阵报告](results/SMALL_REPORT.md) 与
[当前源/二进制/证据manifest](results/small_manifest.json)。
第5轮大矩阵94行/strict300次的 [旧报告](results/REPORT.md) 和
[旧manifest](results/final_manifest.json) 属于历史阶段，不替代新增路线的当前覆盖。

## 1. 实现方向与目标合同

采用 **单层 Strassen–Winograd、FP32 预加减、BF16/FP16 Tensor Core 叶乘法、FP32 累加及结果重组**作为主线。先实现可分解测量的完整参考版本；首轮性能实验同时开发 **单次 launch、每个 CTA 独立拥有四个输出 tile 的 megakernel**，直接检验消除全局 S/T/P 流量的收益。另保留 H100 分核与部分融合路线作为对照。递归层数、融合方式和 tile 由端到端测量选择。

用户已确认首要精度目标为 **BF16/FP16 输入、FP32 累加**。本计划将 BF16 与 FP16 分开调优，默认输出 FP32；低精度输出作为另一个显式模式，不能与 FP32 输出基线混用。

计算合同为：

\[
D = \alpha AB + \beta C_0,\quad
A\in\mathbb{R}^{M\times K},\ B\in\mathbb{R}^{K\times N},\ C_0,D\in\mathbb{R}^{M\times N}.
\]

初始 fast path 支持单卡、大尺寸、row-major、非转置稠密矩阵，`alpha=1, beta=0`；随后加入一般 alpha/beta、leading dimension、尾块与 transpose。API 从一开始显式区分输入 C0 和输出 D。原位 `C0=D` 必须单独验证；A/B 与输出或 workspace 重叠先拒绝。

首版 FP32 输出的认证范围设为 `0<M,N,K≤32768`，A/B 元素有限且绝对值不超过 16；扩展 alpha/beta 时要求两者有限且绝对值不超过 1，beta≠0 的 C0 也有限且绝对值不超过 16。这是保守的初始数值合同，后续以独立压力组扩展。低精度输出另立值域与误差合同，不沿用 FP32 输出的认证。调用方通过 policy 明确声明范围；未知或超出认证范围的输入选择普通 GEMM。dispatcher 不声称仅凭 shape 就能识别数据里的 NaN/Inf 或动态范围，run 也不隐藏内容扫描或 host 同步。

退化情形由公共入口处理：M=0 或 N=0 时成功返回且不发 kernel；K=0 或 alpha=0 时只计算 `D=beta*C0`，不读取 A/B；beta=0 时不读 C0。负维度、非法 leading dimension 和地址大小溢出在 plan 创建时拒绝。

交付目标是：正确的 Winograd 算法、同精度公平基线、可重复性能数据、按形状选择的 dispatcher，以及具有明确适用区间的加速实现。若某形状或精度模式不适合快速算法，dispatcher 使用调优后的普通 GEMM。

## 2. 仓库与现场环境

实施开始前的只读检查得到，以下表格描述起始现场：

| 项目 | 2026-10-13 检查结果 | 实施影响 |
|---|---|---|
| 仓库 | 只有 `.git/`，尚无提交、代码、构建或基准 | 从最小 C++/CUDA 工程起步 |
| 当前 shell | 登录节点 `mgt` | CUDA 构建、运行和 profiling 放在计算节点 |
| 现有 H100 allocation | `107488`，RUNNING，`g08`，1 GPU、4 CPU，TimeLimit=UNLIMITED | 实施时重新核验 allocation 状态和设备绑定，复用可用分配 |
| GPU 查询 | `NVIDIA H100 80GB HBM3`，81559 MiB，compute capability 9.0 | 按 HBM3 单卡 profile 设计；实际 SM 数仍用 CUDA device attributes 记录 |
| 驱动与状态 | `575.51.03`，700 W power limit，MIG Disabled | 频率、温度、利用率和共享干扰要随基准记录 |
| 可用工具链 | `module load cuda/12.4` 后 `nvcc 12.4.99`；有 `ncu`、CMake、g++ | 第一条构建路线采用 CUDA 12.4 + CUTLASS 3.5.1 + C++17 |
| 其他模块 | 有 CUDA 12.2、12.8、13.2 等模块 | 模块存在不等于 driver/runtime 兼容 |

采用 `-gencode arch=compute_90a,code=sm_90a` 编译 Hopper 专用核。H100 的 device capability 是 9.0；`90a` 是编译目标的架构专用能力标记。[NVIDIA Hopper Compatibility Guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/)

CUTLASS 3.5.1 官方列出 CUDA 12.2.2 兼容性，并推荐 12.4/12.5u1 的代码生成；现场已确认 12.4.99 可用，因此先锁 3.5.1 tag，记录解析出的 commit 和编译器版本，再做最小编译验证。[CUTLASS 3.5.1 README](https://github.com/NVIDIA/cutlass/blob/v3.5.1/README.md)

现有驱动满足 CUDA 12.4 路线。CUDA 13.x 的普通兼容路径要求驱动至少 580；本计划不把现有 13.2 模块作为默认运行环境。SubCuber 的新依赖在独立构建目录核验，可先考察 CUDA 12.8；不以安装模块或更换驱动代替兼容性验证。[CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[CUDA 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)

iteration23应用仍由CUDA12.4.99/CUTLASS3.5.1/C++17构建为`sm_90a`并运行于
H100。验证使用独立CUDA12.8 Compute Sanitizer 2025.1.0.0，实际路径为
`/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer`；它没有改变应用的
CUDA12.4构建/runtime。三个当前工具日志均记录路径/version与exit0。早先12.4
racecheck host SIGSEGV属于历史工具失败，不能算通过；设备与版本的实际证据见
[small_environment.txt](results/small_environment.txt) 和
[small_verify_runs.csv](results/small_verify_runs.csv)。

## 3. 文档和研究支持的关键决策

| 已核验的依据 | 对本项目的决策 |
|---|---|
| [Boyer 等：Memory efficient scheduling of Strassen–Winograd](https://arxiv.org/abs/0707.2347) 给出七乘法、十五加减和低 workspace 调度 | 使用唯一 S/T/P/U DAG；空间调度与 GPU 并行调度分别评估 |
| [Strassen's Algorithm Reloaded](https://www.cs.utexas.edu/~flame/pubs/FLAWN79.pdf) 将加减融入 packing 和输出路径 | 借鉴数据搬运融合；保留 Winograd 公共表达式的计数，不直接移植 CPU 微内核 |
| [Volta 上的 CUTLASS Strassen](https://arxiv.org/abs/1808.07984) 探索共享内存、寄存器及融合 | 支持减少流量的方向；V100 的 SGEMM 加速与交叉规模不作为 H100 BF16/FP16 指标 |
| [PLDI 2026 SubCuber](https://www.microsoft.com/en-us/research/publication/compiling-strassen-like-matrix-multiplication-algorithms-to-fast-cuda-kernels/) 按算法 DAG 与 GPU 生成融合调度，论文测 A100/H200 | 将作者实现纳入近期对照，单独复现 H100 与目标输出精度 |
| [Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.2.2/hopper-tuning-guide/index.html) 描述 TMA、cluster、共享内存和 L2 | 利用异步拷贝与计算重叠；按真实 scratch 算 stage 和 occupancy |
| [CUTLASS Example 49](https://github.com/NVIDIA/cutlass/blob/v3.5.1/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) 和 [Example 48](https://github.com/NVIDIA/cutlass/blob/v3.5.1/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu) 提供 Hopper collective 与 scheduler | 复用已验证的普通 GEMM，再替换需要线性组合的 producer/mainloop |
| [Ballard 等：Improving the Numerical Stability of Fast Matrix Multiplication](https://arxiv.org/abs/1507.00687) 分析递归、误差和缩放 | 默认先验证一层；独立建立混合精度误差标准，再决定双层与缩放 |

以下“候选方案”“成本模型”和阈值均为本项目设计或推导；没有完成 H100 性能实测，不承诺固定加速比。

## 4. 固定算法 DAG，先保证数学与精度语义

令三个维度都能二分，A/B 各分为四个象限，采用 [Boyer 等论文](https://arxiv.org/pdf/0707.2347) 的编号：

```text
S1 = A21 + A22        T1 = B12 - B11
S2 = S1  - A11        T2 = B22 - T1
S3 = A11 - A21        T3 = B22 - B12
S4 = A12 - S2         T4 = T2  - B21

P1 = A11 B11          P2 = A12 B21
P3 = S4  B22          P4 = A22 T4
P5 = S1  T1           P6 = S2  T2
P7 = S3  T3

U1 = P1 + P2         U2 = P1 + P6
U3 = U2 + P7         U4 = U2 + P5
U5 = U4 + P3         U6 = U3 - P4
U7 = U3 + P5

C11 = U1             C12 = U5
C21 = U6             C22 = U7
```

这是 **8 次输入块加减 + 7 次块乘法 + 7 次输出块加减**。`P4` 的输入与 `C21` 中的负号必须成套固定，避免混用其他编号版本。

基准实现对每个位置一次读取四个 A 象限，在 FP32 寄存器中构造全部 S；B 侧同理。S1/S2 等依赖在 FP32 中保持，到每个最终叶操作数写出时才转为 BF16/FP16。禁止把已经舍入的 BF16/FP16 S1 再读回构造 S2，却仍当作相同精度策略。

叶乘法的 P1…P7 保存为 FP32。重组核对同一位置加载七个 P，复用 U2/U3/U4，产生四个输出象限；在最终写出时一次应用 `alpha * Cij + beta * C0ij`。beta=0 时不读 C0；U 不建立全尺寸全局缓冲。

本次已用精确整数系数检查四个输出的全部 16 个双线性项，A/B 乘法顺序保持，DAG 恒等式通过。实施阶段将这一检查固化为独立算法测试，并加入小型整数矩阵验证。

**融合会改变操作计数。** 若各 P 的输入独立展开，A 侧加减为 `1+2+1+3=7` 次，B 同为 7 次；四个 C 独立展开输出需要 10 次加减。因此完全展开的版本是 24 次块加减，仍为七乘法，但不能再标为十五加减。每侧产品表达式共引用 14 个源块项。这些额外读取和算术要进入融合版本的成本记录。

## 5. 分层实现与候选版本

| 版本 | 具体数据路径 | 用途与进入条件 |
|---|---|---|
| V0：普通 GEMM | 同 dtype 的调优 cuBLASLt；另建 stock CUTLASS WGMMA GEMM | 公平基线与小矩阵 fallback |
| V1：完整 Winograd | 两个批量 S/T packing 核 → 七个同形 cuBLASLt GEMM → 一个 FP32 DAG 重组核 | 首个正确版本；分离变换、叶乘法与重组成本 |
| V2：Hopper 叶乘法 | 保留 V1 DAG，用 CUTLASS 3.5.1 的 TMA/WGMMA 叶核；比较七次 launch、Graph、统一槽位 batched | 建立可维护的 Hopper 主线；普通叶核吞吐先过关 |
| VM：megakernel | 一次 launch；独立 CTA 完成四象限原始输入载入、S/T、七乘法和四个输出，无全局 S/T/P 缓冲 | 与 V2 同在 P2 做原型；不要求先完成 V3/V4，细节见 §6.5 |
| V3：部分输入融合 | 自定义 producer，比较多源 vector load 与多源 TMA scratch；按产品消除选定 S/T buffer | 实测额外源加载、stage 与舍入影响；完整 P 和 FP32 重组仍保留 |
| V4：DAG 复用与输出融合 | 交错预加减/GEMM；或同一 CTA 拥有对应四个输出 tile，流式处理 P/U；另测 FP32 TMA reduce 输出 | V3 的 P 流量成为瓶颈后投入；寄存器、并行度与输出冲突必须可控 |
| V5：双层与 dispatcher | 固定两层展开为 49 叶乘法；按 dtype、形状、workspace 与误差模式选择 0/1/2 层 | 单层已有可信收益，并且双层的总时间与误差各自达标 |

V1 优先单 stream 顺序执行，构成依赖清晰的图；随后比较实际并发。七个大叶 GEMM 可能已各自占满 GPU，七条 stream 不自动带来七倍并行收益。

V2 的 strided batch 需要七组输入具有统一 layout/stride。原始 A/B 象限与四个 S/T 的混合 view 不能免费变成 `L=7` batch。可比较两种方案：保留 view 的七次调用；把所有七个左右操作数复制到统一槽位，计入额外 packing 与 workspace。[CUTLASS 3.5.1 batched pingpong kernel](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)

为混合 view 编写 product-aware persistent scheduler 是后续独立实验：工作项为 `(product_id, tile_m, tile_n)`，为每个 product 绑定正确 operand descriptor。先验证 tensor map 切换和布局约束，再判断它相对七次 Graph 节点是否值得。

尾块采用确定的两步路线：首版凡维度、stride 或象限 offset 不满足 fast-path 要求，整个问题交给普通 GEMM；P4 再引入 peeling。令 M0/N0/K0 为适合 fast path 的内部尺寸，先计算 `D[:M0,:N0]=alpha*A[:M0,:K0]*B[:K0,:N0]+beta*C0[:M0,:N0]`，再将 K 尾部乘积以 beta=1 累加到该内部输出。底部行用全 K 的普通 GEMM，右侧列仅覆盖前 M0 行并用全 K 的普通 GEMM，二者输出不重叠，分别应用原 beta 一次。与整体 fallback 比较完整时间后才启用。

两层默认采用明确的 `recursive_cast` 策略：第一层在 FP32 中形成 S/T，转回 BF16/FP16 作为第二层递归输入；第二层再在 FP32 中形成 S/T，并转为叶 WGMMA 输入。各级乘积及输出重组始终保留 FP32。另可实验 `deferred_cast`：第一层 S/T 保持 FP32，直到最终叶操作数才 cast；包括未变换的叶 view 也须按目标 dtype 转换。这两种策略分别记录误差、packing 和 workspace，不共享成绩；不得把 half 的跨层输出当作同一策略。

## 6. H100 核心实现细节

### 6.1 WGMMA 与 producer/consumer 流水线

BF16/FP16 统一采用 FP32 accumulator。WGMMA 的 warpgroup 为四个连续 warp，共 128 线程；shared operand 必须使用兼容的 CuTe layout/descriptor。A 可走 register 或 shared 路径，B 位于 shared；不能把任意普通二维 scratch 直接当作 WGMMA 描述符。[CUDA 12.2.2 PTX ISA：WGMMA](https://docs.nvidia.com/cuda/archive/12.2.2/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)

主循环建立 producer 获取空 stage、搬运/变换输入、consumer 等待就绪、发 WGMMA、完成后释放 stage 的环形流水线。同步必须分别处理：

1. TMA/mbarrier 的到达计数、传输字节数与 phase，确保 producer 输入已可读。
2. SIMT 写最终 shared operand 后的 generic→async proxy fence，以及参与线程之间的排序。
3. WGMMA 的 register/operand fence、commit 和 wait；读取累加器或覆盖 shared stage 前等待相应异步计算完成。

`__syncthreads()` 不替代这些协议。优先复用 CUTLASS/CuTe pipeline 封装，并验证 sanitizer 与实际 instruction sequence。[PTX 异步 proxy 与 WGMMA 同步定义](https://docs.nvidia.com/cuda/archive/12.2.2/parallel-thread-execution/index.html)

### 6.2 TMA 能力边界与融合方式

TMA 可以搬运、swizzle、multicast，以及对指定 shared→global 路径执行 reduction；它不提供“把两个 global 矩阵读出后相加/相减成为 WGMMA 输入”的通用操作。stock CUTLASS TMA mainloop 的 transform 类型参数也没有实现这类 Winograd 多源组合。[CUTLASS 3.5.1 TMA mainloop](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)

V3 应比较：

- vectorized/coalesced global loads → FP32 register 加减 → cast → GMMA-compatible shared layout；结合异步拷贝可用性设计流水线。
- 多个 TMA 载入原始 scratch → SIMT FP32 加减 → cast → 最终 shared operand。原始 scratch 和组合后的 tile 均计入 shared memory，不能照抄普通 GEMM stage 数。

tensor map 对 base、stride、box 和 swizzle 有限制；检查象限 offset 后的地址，而非只检查 A/B 分配地址。对 16-byte 输入对齐，BF16/FP16 为 8 个元素；实际 WGMMA tile/padding 要求另计。[CUDA 12.2.2 Tensor Map API](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

### 6.3 Tile、stage 和寄存器搜索

| 参数 | 初始候选 | 调优依据 |
|---|---|---|
| CTA tile `(BM,BN,BK)` | `(128,128,64)`、`(128,256,64)`、`(64,128,64)`；融合核另试 BK=32 | 独立 leaf 时间、寄存器、带宽与边界利用率 |
| schedule | 合法的 Cooperative、Pingpong 和常规 warp-specialized | stock Cooperative 要求 BM 至少 128；不强行套 BM=64 |
| stages | 从合法资源预算出发，试 2/3/4；stock 普通核允许 auto | raw scratch、最终 operand、barrier、epilogue 一并计数 |
| cluster | 先 `(1,1,1)`，后试 `(2,1,1)`、`(1,2,1)` 等 | multicast 源复用收益与 active clusters 的权衡 |
| scheduler | 默认、raster/swizzle、persistent、product-aware | L2 重用、负载均衡、调度开销与 tile wave |

H100 每 SM shared memory 为 228 KiB，每 CTA 上限 227 KiB，L2 为 50 MB；大 shared 分配需 dynamic opt-in。更高 occupancy 不直接等于更快 GEMM。[Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.2.2/hopper-tuning-guide/index.html)

例如 BF16/FP16 的 `128×128×64` 普通 A+B tile 为 32 KiB/stage；若每侧保留两个 raw tile 再写一个组合 tile，则为 96 KiB/stage，三 stage 已超过 CTA 上限。这只是两源例子，Winograd 展开的四源产品可能更贵。必须为自定义 storage 编写准确大小计算；stock `StageCountAuto` 不会替自定义 scratch 留空间。

输出融合分别评估七个 P 跨全 K 累加、小 tile 下按 P/U 生存期复用，以及分段贡献累加。七个 P 同驻留需要更小 tile 或不同 warpgroup 分工，不能直接照搬普通 GEMM 的大 tile；具体资源预算见 §6.5。将代码生成后的寄存器数、spill 和 SMEM 配额作为是否合法的实际证据。

### 6.4 输出路径与 cluster

默认输出使用无写冲突的 FP32 DAG 重组。CTA/cluster 内输出融合要明确唯一 owner 或有序归约。跨 CTA 直接更新同一输出，必须使用合法 reduction，不能用普通 read-modify-write。

TMA reduce-add 可作为明确允许归约顺序变化的性能模式：先完整初始化 `beta*C0`，beta=0 时直接清零而不读 C0；通过 stream/Graph 依赖保证初始化先于全部归约。每项贡献在 FP32 中乘 alpha，并按输出系数取正负，之后仅通过合法 reduction 写相应 FP32 输出，完成所有 bulk 操作再返回。该模式展开了输出贡献，舍入顺序、初始化和扇出成本单独测量；其结果不得假定与 U DAG 一致。CUDA 12.4 的 PTX 文档明确 bulk FP32 reduce-add 将 subnormal 输入和结果 flush 为带符号零，该行为需进入此模式的数值合同和测试。[CUDA 12.4 PTX bulk reduction](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk)

cluster/multicast 先用于同一产品不同输出 tile 的共同 A/B 源。跨产品复用原象限是后续实验；不能假定不同表达式和 tile 必然共享输入。DSM 的访问与退出须符合所有相关 CTA 的生命周期约束。

### 6.5 Megakernel：提前验证的主要性能候选

这里的 megakernel 是单次 kernel launch，多个 CTA 各自完成一个独立 Winograd 工作项，并非单个 CTA 计算整张矩阵。工作项 `(tile_m,tile_n)` 对应 C11/C12/C21/C22 中相同局部坐标的四个 `BM×BN` 输出 tile。该 CTA 拥有四个 tile 的完整 K 归约与最终写出，没有其他 CTA 更新这些输出，因此不需要全 grid barrier、cooperative launch 或全局输出 atomics。

一层 fast path 的数据流为：

```text
producer warpgroup                         consumer warpgroup
TMA/vector load A11/A12/A21/A22 ─┐
TMA/vector load B11/B12/B21/B22 ─┤
FP32 构造共享的 S1..S4/T1..T4   │
最终操作数 cast + swizzled SMEM ├─ stage ready → 七组 WGMMA
预取下一个 K-stage             │                P/U 或 C 留在寄存器
                               └─ stage free ← 完成对应异步读取
                                                四个输出应用 alpha/beta
                                                四个独占 tile 写回
```

warp specialization 采用合法的连续四 warp consumer 分组；一个起步布局是 producer warpgroup + consumer warpgroup，共 256 线程，producer 的 TMA 发起线程与其余 SIMT 变换线程分工。最终线程数、寄存器重分配与 stage 数以编译资源和实测选择。WGMMA 结果访问前的 wait 与 shared stage 释放仍遵守 §6.1 的协议。[Hopper 资源与 warp specialization](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html)、[WGMMA fence/commit/wait](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)

先比较三个明确不同的 megakernel 调度：

| 调度 | 累加器与 K 顺序 | 收益和代价 |
|---|---|---|
| M1：共享输入 DAG，七 P 跨全 K | 每个 K-stage 一次构造本 CTA 所需 S/T，向七个独立 FP32 P 累加；整个 K 完成后执行原 U DAG | 保留全 K 乘积后重组的语义，完全消除全局 S/T/P；七个 accumulator 限制 tile |
| M2：四 C + 一个/两个 P scratch | 每个 K-stage 的 P 计算完成即按展开系数加到四个长期 FP32 C；scratch 重用 | 降低部分寄存器占用；增加每 stage 输出更新与 wait，改变浮点累计顺序，独立验证 |
| M3：按完整产品流式处理 | 顺序 `P1,P2,P6,P7,P5,P3,P4`，每个产品扫描完整 K，复用最多约三个 P/U tile 的生命周期 | 保留原输出 U DAG并可增大 tile；需重复 K 扫描、输入载入和变换，作为资源对照 |

M1 不保证与 cuBLASLt 逐 bit 相同，但保留每个 P 的全 K 累加、同一次输入量化策略和最终 U 重组结构。M2 是另一种 FP32 累加策略；不能把逐 K-stage 的线性重组视为浮点下无成本地等价于先完整算 P。M3 若完全内联每个产品的输入表达式，输入加减为 14 次、输出为 7 次；不能仍标为十五加减。

当前已实现M1/M3大tile对照及小tile warp/product路线。小矩阵完整套件包括
Lt0/1、Hopper5–8、Mega4及Mega7–19；严格集合还包含Hopper0–4和Mega0–3/5/6。
Hopper5/6分别为64×64×32与32×64×32 Sm80 batch7，7/8为32×64×64的三stage
multistage与两stage pipelined家族。它们的实际schedule不同，不作为纯stage数对照。
当前路线和fallback条件见 [README路线表](README.md#api与实验路线) 与
[plan.cpp](solution/plan.cpp)，不能只凭请求的variant认定实际快路径执行。

Mega18是已接入的M2：四个长期C和一个P scratch，每K16 stage按14项展开贡献
更新C，FP32重组顺序与M1不同。它排除于M1逐位回归4、7–17、19之外，并使用
[独立stage/整数CPU模型](tests/test_stage_m2.py)及结构/API/alpha-beta/alias/Graph
验证。模型的标量RN dot不声称与SM80内部K16归约逐位相同；所有路线仍须接受
原始量化输入的FP64普通AB参考及相同冻结随机误差门限。

Mega19以8×8 owner使用native16×8 MMA，下半A置零；全部32 lanes和每产品四个
native C寄存器仍参与MMA，输出只使用上8×8槽。相对Mega16的16×8 owner，CTA、
逻辑B载入和HMMA发射约翻倍，全问题A读取近似不变；粒度变细伴随冗余工作。
RawInputs从每lane 96降至64 bytes，shared源码5376 bytes，七P accumulator仍为每lane
28个FP32。资源和指令证据见 [iteration23 profile](results/small_owner_profile_iter23.json)。
这组结果不证明直接寄存器operand等后续方向无收益。

**M1 的第一组资源预算**：单 consumer warpgroup、128 个线程，七个 FP32 accumulator 的平均寄存器数为 `7*BM*BN/128`。暂保留四个原始 A 加四个 S、四个原始 B 加四个 T，则 BF16/FP16 数据 scratch 为 `16*BK*(BM+BN)` bytes/stage，另计 barrier、对齐和 epilogue。

| `(BM,BN,BK)` | 七 P accumulator regs/consumer thread | 数据 scratch/stage | 双缓冲数据 | 首轮用途 |
|---|---:|---:|---:|---|
| `(64,32,32)` | 112 | 48 KiB | 96 KiB | 首个可行性原型，保留辅助寄存器空间 |
| `(64,64,32)` | 224 | 64 KiB | 128 KiB | 更大 tile；需严格控制 descriptor/指针寄存器及 spill |
| `(64,64,64)` | 224 | 128 KiB | 256 KiB | 当前保守双缓冲超 CTA 上限；需别名/生命周期优化后再试 |

H100 为每线程最多 255 个寄存器、每 SM 64K 个 32-bit 寄存器、每 CTA shared memory 上限 227 KiB。上表只是数据与 accumulator 预算，并非证明核可驻留或高效；producer 寄存器、寄存器分配粒度、barrier 和 store scratch 均须另计。BM/BN 更大的七 P 版本只有在多个 consumer warpgroup 重新分工且满足整个 SM 预算后才试。[NVIDIA Hopper occupancy 资源](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html#occupancy)

以 8192 方阵估算，`BM=64,BN=32` 有 8192 个独立 CTA 工作项，`BM=64,BN=64` 有 4096 个；单 launch 仍有大量并行工作。可在资源验证后改为 persistent CTA 连续领取独立 owner tile，而无需跨 CTA 的阶段同步。[CUTLASS persistent scheduler 示例](https://github.com/NVIDIA/cutlass/blob/v3.5.1/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu)

**省显存流量不代表变换免费。** A tile 的 S/T 会在不同 N 输出 tile 重算，B tile 的变换会在不同 M 输出 tile 重算；M1 只在 CTA 内复用 DAG，不能沿用全局只算一次的十五加减成本。忽略边界，令 `G=MN/(4*BM*BN)`、`Q=K/(2*BK)`，则：

\[
F_{add,M1}=G\left[Q(4BM\,BK+4BK\,BN)+7BM\,BN\right]
=\frac{MNK}{2}\left(\frac1{BM}+\frac1{BN}\right)+\frac{7MN}{4}.
\]

M2 使用展开贡献时，七个 P 对四个 C 共 14 个贡献项，约有 `14*BM*BN` 次 FP32 输出更新/K-stage，首 stage 可省部分零加法。producer 变换、SIMT 更新、WGMMA wait 与小 tile 效率一起决定是否超过 V2；只算七乘法减少不能判断。

VM 的验证顺序为：M1 `(64,32,32)` → 资源与误差检查 → M1 `(64,64,32)`/raw scratch 复用 → M2 与 M3 对照 → persistent/cluster。优先获取单次 launch 的完整延迟、实际 DRAM/L2 流量、Tensor Core 活跃度、register spill 与编译资源。global workspace 可消除矩阵级 S/T/P，但 descriptor、任务计数器等元数据仍正常统计。

## 7. 端到端成本、workspace 与进入门槛

以下是模型推导，不是性能测量。下式加减量针对每层 S/T 在全局复用的 DAG；megakernel 的 CTA 内变换重算采用 §6.5 的独立模型。对 L 层均匀二分：

\[
F_{\rm leaf}(L)\approx2MNK(7/8)^L,\qquad
F_{\rm add}(L)=\left(MK+KN+\frac{7MN}{4}\right)\sum_{i=0}^{L-1}(7/4)^i.
\]

| 层数 | 叶数 | 叶乘加量比例 | 方阵 DAG 加减量 | 同吞吐、零额外开销的算术比 |
|---|---:|---:|---:|---:|
| 1 | 7 | 7/8 | `3.75 n²` | 8/7 ≈ 1.143 |
| 2 | 49 | 49/64 | `10.3125 n²` | 64/49 ≈ 1.306 |

实测应判断：

\[
T_{SW}=T_{transform}+T_{leaves,actual\ schedule}+T_{combine}+T_{tail/padding}+T_{required\ overhead}<T_{dense}.
\]

融合核按整体时间测量，分项只用于诊断，避免重复计算重叠区间。令所有叶在真实调度下的有效算术吞吐为 parent 的 η，单层叶时间比约为 `0.875/η`。当 η≤0.875 时，即使其他开销为零也没有单层收益；先检查叶吞吐，防止把普通 GEMM 的损失误当 Winograd 问题。

V1 单层的额外缓冲为：

\[
W_{ST}=s_{in}(MK+KN),\qquad W_P=7s_P MN/4,\qquad s_{in}=2,\ s_P=4.
\]

方阵合计 `11 n² bytes`；8192 时约 0.688 GiB，16384 时约 2.750 GiB。另计 A/B/C0/D、库 workspace、padding、graph 和 allocator 对齐。若把全部七个左右操作数复制成统一 batch 槽位，输入 workspace 增为 `7 s_in (MK+KN)/4`，不得沿用 V1 数字。

保存再重读全部 FP32 P 的逻辑流量为 `14 MN bytes`。以 H100 SXM 产品规格的 dense BF16/FP16 理论吞吐约 989.5 TFLOP/s 和 HBM 带宽 3.35 TB/s 做数量级估计：8192 方阵的理想乘法节省约 0.139 ms，但 P 写读按全走 HBM 估计约 0.280 ms，尚未加入 packing。这解释了输出流量可能是重要瓶颈；不能据此宣布某个实测交叉规模。[H100 产品规格，标注的 Tensor Core 峰值含 sparsity，dense 需除二](https://www.nvidia.com/en-in/data-center/h100/)

workspace planner 先用完整 V1 获得稳定基线，再探索按 tile wave/产品生存期复用缓冲。限制 workspace 时必须重新核算并行度、kernel 数和总延迟。

## 8. 数值正确性与验收

参考输入为 **已经舍入到目标 BF16/FP16 dtype 的原始 A/B**。将其转成 FP64，计算普通 GEMM 参考；高精度参考不得先经过 S/T 变换。小中尺寸检查完整输出；大尺寸采用分块或 GPU FP64 参考，测试成本不进入性能计时。

FP32 accumulator 不会恢复 S/T 转换丢失的精度。FP16 要覆盖变换溢出；BF16 要覆盖较少尾数位带来的消去误差。FP32 累加和输出重组均保持到最终输出；允许输出 half 的模式也仅最后 cast 一次。

同时报告：最大绝对误差、Frobenius 相对误差、逐元素 `|error|/(atol+rtol*|reference|)`，以及以 `||A||F ||B||F` 归一化的误差。参考近零时单独报告绝对误差，避免只用相对误差。

测试集合包含：随机正负/正值输入、zero、identity、非方阵、相近象限差值、强消去、不同动态范围、接近 FP16 上限、奇数/非 tile 对齐维度、不同 leading dimension，以及一般 alpha/beta、原位 C0/D 和连续调用。

验收按BF16/FP16和递归层数分别记录误差分布；当前只实现0/1层，2层显式拒绝。
已从初始基线冻结每轮门限 `eF_SW ≤ 3*eF_dense + floor`，BF16 floor=`1e-3`、
FP16 floor=`1e-4`。每轮使用新输入，所有trial的pass取conjunction，双方均须无
nonfinite。CSV最大eF与最大gate不能替代逐trial检查，不能事后为通过某方案放宽。
CHECK日志默认六位有效数字与CSV十二位的审计使用舍入区间；接近边界时保留
uncertain状态，不虚构原始精确metric。当前720行SW均失败，Auto保持Dense。
高风险范围若无法满足合同，限制fast path或另行验证计入时间的缩放方案。

对有限、位于声明输入范围内的数据，不能接受算法产生 NaN/Inf。NaN/Inf 输入的语义单独定义；初期走普通 GEMM。sanitizer 检查越界、race 和同步；异步协议仍需代码审查，不能只凭 sanitizer 通过宣称正确。

## 9. 公平基准与 SubCuber 对照

### 9.1 基准配置

- 方阵：512、1024、2048、4096、8192、16384；32768 在显存与参考验证允许后加入。
- 矩形：`(8192,4096,8192)`、`(4096,8192,8192)`、`(8192,8192,4096)`、K 偏小以及细长形状；元组顺序统一 `(M,N,K)`。
- 功能边界：例如 `(1025,1537,769)`，以及象限基址不满足 fast-path 对齐的 stride/view。
- BF16、FP16 分别测 FP32 输出；低精度输出另建表。所有 baseline 使用相同 layout、alpha/beta 和输出 dtype。

上述大矩阵集合是原计划与历史阶段。当前小矩阵研究的预声明集合单独固定于
[small_shapes.json](bench/small_shapes.json)：128/256/512/1024/2048/3072/4096
方阵和`(1024,512,1024)`、`(512,1024,1024)`。每条路线分开launch/Graph、两种
dtype；完整协议20warm/100calls×5groups/3correctness trials，诊断协议20warm/
20calls×5groups/1trial。最终套件40labels/720rows，不混入诊断或历史数据。

cuBLASLt 根据形状、dtype、stride、workspace 上限查询多个合法 heuristic 候选，例如最多 32 个，预热并实测取优、缓存决策。baseline 与候选允许相同总 workspace 上限，实际各自使用量分别记录。普通 cuBLAS GEMM 和 stock CUTLASS 作为诊断对照。[cuBLASLt heuristics 与 API](https://docs.nvidia.com/cuda/archive/12.2.0/cublas/index.html#cublasltmatmulalgogetheuristic)

当前两侧总workspace上限均为8GiB，实际bytes分别记录。Lt保留32个heuristic、
实际绑定的alignment pool与dedup；小问题Lt调优为4次warmup、3组各10次调用的
median，大问题保留1warm/3calls。调优在正式计时与Graph capture之外。51列CSV
保存Dense/solution实际bound config与protocol，TUNE日志精确对应；custom核的
Lt config为`none`，未调优/重绑定/query不可得时明确NA，不能虚构算法或score。

### 9.2 计时合同

测量预分配、预调优后的完整 steady-state 调用：输入 ready → 每次所需 S/T 变换 → 全部 P → 重组/输出；另外报告首次规划/分配与主机调用成本。默认动态 A/B 场景每次都重算变换，即使 benchmark 使用同一地址。

分别提供 Graph 和普通 launch 的端到端表，双方采用相同计时模式。CUDA Event 起止之间包含所有相关 stream，通过明确 event join 包含全部 GPU 工作；另外用主机 wall clock 报告 CPU 调度影响。多 stream 时间不能只在一条 stream 上记录事件。

预热例如 20 次，测量例如 100 次，独立重复至少 5 组并随机交替测 baseline/候选；单组总计时应足够长。报告 median、分位数和离散度，不能仅报告最小值。常规 warm-cache 与轮换输入的大工作集模式分别报告。输入更新和正确性检查在计时外，但所有依赖输入的算法工作在计时内。

当前完整表固定20warm/100calls/5groups/3correctness；p10/median/p90/CV取五组
每调用均值的分布，不是100次单调用的尾延迟或置信区间。小矩阵最终集合是
warm_cache/rotate_inputs=1，各候选同一行Dense才是主要普通GEMM基线。GPU频率
未锁定，before/after telemetry不能证明组内时钟稳定。

跨variant经验包络按候选自身最低观测`solution_event_ms`选择，不按最大Dense/SW
选择；精确平局按source路径/CSV行号/actual route/variant确定。获选行的Dense
用于计算Dense/SW。它是selected-on-measurement的观测最小值，不是独立dispatcher
收益或数值认证。历史原始Lt0/SW另列跨运行比值，严格匹配形状、dtype、mode、
cache、rotate_inputs和工作集；旧packing/tuner/时钟/host状态可能不同，不能解释
为同行Dense收益。缺失/歧义显示N/A，actual-route历史匹配为家族层次。

相邻预声明尺寸的性能资格须保持同一源、actual route、variant、dtype、mode、cache
和rotate_inputs场景，各shape工作集符合声明；固定数值门限和完整采样须同时
通过。`Dense p10 / SW p90`只作保守组间离散度筛查，不是统计保证。当前BF16
128 launch经验包络为同行Dense3.938µs/SW3.915µs≈1.006×，是微小观测差异且
数值失败，没有达到≥1.05×并通过精度的认证要求。

固定 B 的权重复用场景单独报告 `T_precompute_B`、每次 amortized 时间与复用次数，不能混入动态 GEMM 结果。固定 beta≠0 时保证每次都使用同一个逻辑 C0；不能无意反复将上一次 D 当 C0。

主要指标为总延迟、相对调优 cuBLASLt 的 speedup、`2MNK/T` 等效 TFLOP/s、误差、峰值 workspace 和适用条件；等效 TFLOP/s 可能超过传统 GEMM 的硬件峰值，须与真实叶算术量的吞吐区分。

### 9.3 SubCuber 复现边界

检索时官方仓库版本为 [`c7b6fd8c72db4fc83982b26ea6e80e47567d674f`](https://github.com/microsoft/subcuber/tree/c7b6fd8c72db4fc83982b26ea6e80e47567d674f)，CUTLASS submodule 为 `a49a78ffefc86a87160dfe0ccc3a3a2d1622c918`，version 4.1.0，并依赖作者 patch。它与本项目 3.5.1 基线隔离，复现时固定 commit/patch/compiler。[SubCuber 固定版本 README](https://github.com/microsoft/subcuber/blob/c7b6fd8c72db4fc83982b26ea6e80e47567d674f/README.md)

当前 Hopper FP16 实现采用 FP32 accumulator，但 D 为 half。[FP16 类型与 schedule 定义](https://github.com/microsoft/subcuber/blob/c7b6fd8c72db4fc83982b26ea6e80e47567d674f/src/cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh#L19-L59)

最新双层 cooperative/pingpong TMA-reduce preset 要求 M 为 512 倍数、N 为 1024 倍数、K 为 256 倍数；不能默认适用于任意形状或 BF16/FP32 输出。它可借鉴 FP32 producer 公共表达式和交错预加/GEMM 调度，但仍使用 presum workspace。[双层 preset 与 can_implement](https://github.com/microsoft/subcuber/blob/c7b6fd8c72db4fc83982b26ea6e80e47567d674f/src/cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_level_2_cooperative_pingpong_max_fusion_tma_reduce.cuh#L131-L137)

必须单独检查正确性：README 说明 runner 不验证输出；已有 FP16 测试采用 Strassen–Winograd 参考，L2 容差为 `rel≤0.11 OR abs≤14`，不能替代本项目对原始输入的高精度 GEMM 检查。[作者测试标准](https://github.com/microsoft/subcuber/blob/c7b6fd8c72db4fc83982b26ea6e80e47567d674f/tests/base_test.cuh#L85-L107)

作者 cuBLASLt runner 只请求一个 heuristic algorithm，不能替代本项目的多候选 baseline 调优。[作者 baseline 选择](https://github.com/microsoft/subcuber/blob/c7b6fd8c72db4fc83982b26ea6e80e47567d674f/src/cuda/cublaslt_runner.cpp#L152-L188)

尤其要检查预计算是否落在计时内：adapter 中 `paramsM0_.run <= 1` 时才执行独立 global presum，之后复用；warmup=2 可能将 B 预变换移出 measured iterations。动态 B 对照必须重新计算并计时，或明确标成固定权重预计算场景。[SubCuber adapter 的 run/presum 条件](https://github.com/microsoft/subcuber/blob/c7b6fd8c72db4fc83982b26ea6e80e47567d674f/include/cutlass/gemm/device/strassen_gemm_universal_adapter.h#L1460-L1474)

## 10. Profiling 驱动的优化顺序

先测端到端，再定位时间占比最大的阶段。Nsight Systems 用于 kernel、stream 与依赖时间线；Nsight Compute 用于代表性核，避免把 profiler 的 replay 时间作为性能成绩。

记录 Tensor Core 活跃度、DRAM/L2 字节与命中率、shared 冲突、寄存器/spill、active warps/CTAs、等待与 barrier stall。指标名按本机 `ncu --query-metrics` 核验，不硬编码其他版本的名字。

| 观测结果 | 下一步 |
|---|---|
| 叶算术吞吐低于必要门槛 | 先修普通 Hopper GEMM tile/schedule/layout，再做算法融合 |
| S/T 流量占主导 | 比较 V3 多源加载和 V4 交错变换；检查额外源读取与公共表达式复用 |
| FP32 P 写读或重组占主导 | 测 tile wave、CTA 输出 owner、FP32 reduction；比较确定重组与归约模式 |
| launch/依赖占主导 | 同模式 CUDA Graph、batch 或 product-aware scheduler |
| scratch 导致驻留受限/spill | 减 BK/stages，缩短 P/U 生命周期，再测 warp specialization |
| 特定边界/padding 代价大 | tail peeling 或普通 GEMM；计入完整 tail 与 alpha/beta 语义 |

保存每次改动的假设、代码版本、形状、完整时间、误差、workspace 与 profile。连续无收益时重新归因；最终保留经全量验证的最佳版本，不按最后一次实验选择。

## 11. 工程结构与里程碑

原计划的模块划分已落为以下主要入口；其余优化以独立TU保留，完整source hash由
manifest记录：

```text
CMakeLists.txt
include/megafmm/gemm.h
solution/{plan.cpp,internal.h,pack_winograd.cu,combine_fp32.cu}
solution/{gemm_hopper.cu,megakernel_hopper.cu,gemm_batched*.cu}
solution/{small_mma*.cu,small_product*.cu}
tests/{test_dag.py,test_stage_m2.py,test_correctness.cu}
bench/{benchmark.cu,shapes.json,small_shapes.json}
scripts/{build.sh,bench.sh,profile.sh,small_suite.sh,verify_small.sh}
scripts/{summarize_small.py,write_small_manifest.py}
third_party/cutlass/          # 固定依赖版本
results/                     # 元数据、CSV/JSON、profile 索引
```

当前公共API为plan创建、`plan->workspace_bytes()`、显式`tune`和
`run(plan, A, B, C0, D, workspace, stream)`；run不动态分配或隐式host synchronize。
tensor map/Graph指针绑定成本单独记录；地址或leading dimension改变时正确更新
descriptor。0/1层可强制用于benchmark/回归，2层尚未实现并显式拒绝。

| 阶段 | 交付物 | 必须通过的关口 |
|---|---|---|
| P0：工具链与基线 | 固定依赖、最小 Hopper GEMM、cuBLASLt 调优、环境记录 | H100 上 build/run 正确；普通 WGMMA 叶性能可信 |
| P1：V1 正确实现 | 唯一 DAG、两侧 packing、七乘法、FP32 combine、workspace API | BF16/FP16 分别通过参考检查；alpha/beta 与地址语义明确 |
| P2：V2/VM 与成本分解 | stock Hopper leaves、Graph/batch、M1 megakernel 原型及完整对照数据 | 所有 transform 在计时内；验证单 CTA 输出 owner、资源、误差与总时间 |
| P3：主要瓶颈优化 | VM 的 M1/M2/M3、V3/V4 对照、自定义 pipeline、代表性 profiling | 资源与异步协议正确；精度通过；改善完整总时间 |
| P4：泛化与双层 | 尾块/矩形/stride、transpose 功能、可选两层、0/1/2 dispatcher | transpose 先支持普通 GEMM 路由，再测转换/专用 fast path；双层独立过时间和误差门槛 |
| P5：最终交付 | 可复现实验表、策略表、使用说明、最优实现 | 公平基线、误差、资源、输入假设和复现命令齐全 |

单层初始性能验收目标为：在预先固定的主测试集内，至少一段相邻大尺寸形状具有超过测量噪声的收益；候选目标 `median speedup ≥ 1.05`，并由重复数据支持。没有达到时报告结果并继续处理已识别瓶颈，不将纸面乘法减少当作完成性能目标。dispatcher 在未通过验收的形状继续选普通 GEMM。

双层只在 leaf、workspace、误差与总时间同时允许时进入。递归深度采用 host 侧固定展开，不引入 device dynamic parallelism。FP8、稀疏 Tensor Core、多 GPU/NVLink 属于后续扩展，不能当作本次稠密 BF16/FP16 合同的加速来源。

## 12. 计算节点执行与当前实验状态

实施开始先重新确认现有 allocation，使用其分配的设备和 CPU；需要 Slurm 建立设备环境时可在登录节点进入已有 allocation 的 step：

```bash
squeue -j 107488
scontrol show job 107488
srun --jobid=107488 --ntasks=1 --cpus-per-task=4 --gres=gpu:nvidia:1 --pty bash -l
module load cuda/12.4
cd /share/home/liuyao/workspace/MegaFMM
nvcc --version
nvidia-smi
```

若采用 SSH，g08 使用端口 2222，并核验 allocation 对应的 GPU；不能假定任意 SSH 会话继承 Slurm 设备绑定。现有分配失效时先查 `scontrol show partition h100`，按集群默认使用长驻 `salloc`，允许时请求 UNLIMITED，并保持 allocation shell。下载依赖在登录节点完成，计算节点承担构建、测试与 profiling。

实施前阶段已经完成：仓库/现场只读检查、官方文档与原始研究检索、最新 SubCuber 源码边界核验、算法双线性恒等检查，以及成本与 workspace 推导。该阶段尚未进行 CUDA 实现和 H100 测量；后续执行结果见开头的执行更新，验收合同仍由上述 P0–P5 定义。

iteration23当前快照的 [run metadata](results/small_final_runs.csv) 与
[verify metadata](results/small_verify_runs.csv) 均记录源commit
`ea4dc50c57b08b4dee98c23c024a01bb94f836db`。benchmark SHA256为
`a89d8cadf7f3dcbbecd41e6d7fac30de24ce55827bcbacf372126beaaf807d42`，
correctness SHA256为`51791901d93f465db30d41f419f1bc2f7334b02936c9e8c0365e21130679d392`；
两个角色各自全程同binary。当前manifest核对30个source hashes、720行、80份
telemetry、1,440条TUNE与2,160轮CHECK；不以文档后来的HEAD冒充测量版本。
五项验证exit0，strict exit2且930次中558通过/372失败/nonfinite零。历史大矩阵
94行中88实际SW失败/6Dense fallback通过，strict300中120失败/nonfinite零，
保留于旧报告/manifest，不混入本快照。

完整复现命令及独立工具选择见 [README](README.md#构建验证与复现)。当前结构、
资源和实验覆盖并不把未通过精度与性能验收的P1/P3/P5关口标为完成。Auto继续
调优Dense；持续优化将依据新假设、资源、完整延迟与同一冻结数值合同继续，直接
寄存器operand等候选仍可接入验证，不将本快照视为停止或穷尽声明。
