# MegaFMM

H100 上 Strassen–Winograd 矩阵乘法的研究工程。2026-10-14 按用户要求移除旧
SW 架构后，实现统一 buffer 与 TMA、CUDA Core、Tensor Core 分工的首版。

**已按用户要求恢复历史 iter313（79dad0b，逐字恢复 iter282）的 1024³ 专用 SW 实现。**
显式 `SWPipeline` 下，连续、非转置、alpha=1、beta=0 的 1024³ 且 workspace
预算至少 4 MiB 时，Plan 使用 `sw1024_pdl`；A/B 需 16 字节对齐。
其他符合条件的尺寸使用下述多尺寸 SW；不支持的布局、较小 workspace 预算以及未对齐绑定仍走通用 `sw_pipeline`。
专用路径仍是七乘法无补偿实现，**没有解决原始精度不达标问题**。
下文专用路径耗时均为历史数据；恢复后的验证结果另行记录。

恢复后已在 H100 上完成全量构建，相关 CTest **4/4 通过**，覆盖专用路径普通调用、
Graph 新输入重放、工作区重新填充和通用路径回退；quick memcheck 零错误。
原始 FP64 精度验收中，BF16/FP16 **各 36 项、各 24 项未达门槛**，均无 nonfinite。
本次未重新测量性能。详见[恢复与验证记录](trajectory/20261016_restore_sw1024/REPORT.md)。

**恢复后独立复测（2026-10-16）**：普通完整调用 BF16 SW **10.149 μs** /
同期 cuBLASLt **12.173 μs**，约 **1.199×**；FP16 SW **10.061 μs** /
cuBLASLt **9.587 μs**，约 **0.953×**。Graph 分别约 **1.121× / 0.929×**。
每模式 3 个独立进程、每进程 5 组×100 次、2 轮换输入集；上面耗时取 15 个组均值的中位数。
两种 dtype 原始精度门槛均失败；这不是数值验收通过。
[新测完整数据与所有样本](trajectory/20261016_sw1024_remeasure/REPORT.md)。

**多尺寸七乘法 SW**：`sw_hopper` 支持各维正数且不超过 32768 的连续、非转置、
alpha=1/beta=0 的 BF16/FP16。1024³ 保留原专用实现。
M 为 256 倍数、N 为 128 倍数、K 为 256 倍数时，使用动态 K 的 TMA/WGMMA
流水线：小形状采用 64×32×128，较大形状采用 64×128×64；调用方 workspace 为
`2*K*(M+N)` 字节。K>=2048 时使用分组行遍历，K>=16384 时对 A 使用
TMA evict-last、对 B 使用 evict-first 缓存提示。
其他尺寸按逻辑 ceil(M/2)、ceil(N/2)、ceil(K/2) 切分四象限，显式零填充并执行
相同七乘法；物理半尺寸分别向上对齐到 (16 或 64, 64, 128)，workspace 为
`16*KHpad*(MHpad+NHpad)` 字节，包含 S/T 和原始象限的打包存储。
1³ 使用单 warp 的七个 SIMT 叶乘积和 256 字节 workspace 标记，不使用 Tensor Core。
预算不足或 A/B 未按 16 字节对齐时继续执行通用 SW。所有变换、七乘法和重组均在完整调用计时内。
该版本的原始精度问题仍未解决，性能与验证记录将按正式实测补充；不替换 Auto/Dense。

**2026-10-16 精度研究：** [调研、误差消融与硬件对照](research/precision_20261016/REPORT.md)
和[针对性方案设计](research/precision_20261016/DESIGN.md)。以下研究以统一前的生产源码为起点，
原七乘法仍未通过原始精度门槛；达标且更快的直接输入稠密原型保留为研究对照。
[补偿路径硬件研究](research/precision_20261016/int8_hardware/HARDWARE_REPORT.md)的普通和Graph参考现统一为P56，P51/P50/P47/P46/P45/P39为历史完整验收回退。
P56仅把P51的原生consumer七叶外循环unroll1改为7，其余源码逐字保持；原始FP16/BF16输入、精确逐列平衡、七个主乘积和全部八个交叉修正点积、FP32 RN依赖图不变。
标准78项最大BF16/FP16 eF为1.61320e-5/2.16395e-5，打印误差逐项等于P51。
独立普通ABBA同轮P51→P56为27.465→25.065μs / 27.511→25.105μs，降低8.74%/8.75%；Graph27.496→25.065μs / 27.443→25.164μs，降低8.84%/8.31%。全部准备与描述符/计算成本计入，保留全部计时臂；未锁时钟，无置信区间。
consumer143→128寄存器、0stack/spill，共享内存仍196736B/实际1CTA每SM；准备核归一化SASS逐条相同。静态代码变大，不将收益唯一归因于寄存器或分支。
P56自身普通/Graph各78项门槛及79,872inside通过，新种子14模式双流168配置/1,344次普通/Graph交错调用、独立workspace及输出/workspace保护区通过；memcheck/racecheck/synccheck限定范围各56配置/112调用零错误，racecheck零警告/零hazard。
自身原生TMA探针在首次MMA前覆盖六组/约4.15亿次高低部值比较及槽位/guards，未把通用读取当作WGMMA等待证明。没有将P51的证书转移给P56。
完整调用约25.1μs，仍约为生产SW的2.5倍，尚未进入生产或改变Auto，原生产七乘法仍不满足精度目标。
[P56实现与受控消融](research/precision_20261016/int8_hardware/STATIC_LEAF_SCHEDULE_HARDWARE.md)、[自身验收](research/precision_20261016/int8_hardware/p56_validation.json)、[综合审计](research/precision_20261016/fused_int8/audit.json)及[后续消融](research/precision_20261016/int8_hardware/NEXT_EXPERIMENTS.md)保留确切范围与剩余性能差距。
[P57仅展开生产者循环](research/precision_20261016/int8_hardware/STATIC_PRODUCER_HARDWARE.md)未改善完整耗时：普通BF16/FP16慢0.47%/0.33%，Graph降低0.48%/慢0.08%；自身布局、78原始门槛和每模式312项ABBA通过，但没有自身双流/工具证书。已拒绝并从14d9720逐字恢复P56，独立新进程78项复核通过。
[P58把384个打包任务分配给128CTA](research/precision_20261016/int8_hardware/RESIDENT_PACK_HARDWARE.md)的独立普通/Graph对照点估计快0.37%–1.38%，低于工作流3%进展标准，未升级参考；自身72组打包/极端缩放逐位控制及78原始门槛通过，计算核源码/SASS相同P56。保留为后续融合对照，无自身双流/工具证书，当前工作版本已恢复P56并通过新进程78复核。
此前P51的384CTA打包路径、54组原生高低部逐位检查及18组极端缩放证据仍冻结在[P51硬件记录](research/precision_20261016/int8_hardware/ALL_NATIVE_PACK_GRID_HARDWARE.md)，作为历史回退；不同轮次的提速不相加当作直接测量。
[P59全局平面阶段单核融合](research/precision_20261016/int8_hardware/GLOBAL_PHASE_FUSION_HARDWARE.md)通过自身布局与78项原始门槛，但独立普通/Graph相对P56没有一致收益，相对P58慢0.23%–2.05%，已拒绝并逐字恢复P56后另跑78项复核；无自身双流/工具证书。连续三个候选未获3%进展后完成[再次调研和历史复核](research/precision_20261016/int8_hardware/LITERATURE_UPDATE_P59.md)，形成[P60成对原生打包方案](research/precision_20261016/int8_hardware/PAIR_NATIVE_PACK_DESIGN.md)。[P60成对原生打包实测](research/precision_20261016/int8_hardware/PAIR_NATIVE_PACK_HARDWARE.md)现已完成：自身转换oracle、84组全平面/两字节偏移及78原始门槛通过；独立普通BF16快1.82%而FP16慢0.17%，Graph快1.81%/0.91%，未升级P56，无自身双流/工具证书。已逐字恢复P56并另跑78复核。[P61精确inside归约K16方案](research/precision_20261016/int8_hardware/INSIDE_K16_DESIGN.md)已完成独立准备核与[84组首次MMA前验证](research/precision_20261016/int8_hardware/probe_p61_pack.json)：约4.84亿次平面位比较、与P56及独立标量参考的inside比较全部通过，尚无完整矩阵或性能验收，目标继续。

P48缩小共享申请量未获完整收益。[原生融合N1/N2/N3](research/precision_20261016/fused_native/REPORT.md)各自布局和原始精度通过，完整调用约615/118/105μs，均慢于同轮P47，全部拒绝；没有将其他候选的双流或工具验收转移给它们。[P52计算瓦片与TMA槽位实验](research/precision_20261016/int8_hardware/CONSUMER_N16_HARDWARE.md)通过自身布局及78项原始精度，但独立普通/Graph耗时约39μs，比P51慢41%–42%，已拒绝。[P53全K512叶乘积并行](research/precision_20261016/int8_hardware/LEAF_PARALLEL_HARDWARE.md)通过自身约4.15亿次布局比较、315万次GPU重组位比较及78项原始精度，74寄存器/实际两CTA每SM，但完整普通调用约41.1μs、Graph约37.4μs，分别比P51慢48%–50%/36%–37%，也拒绝；没有自身fresh双流/工具验收。已逐字恢复P51并另跑新进程78项复核通过。[P54原生N64叶瓦片](research/precision_20261016/int8_hardware/LEAF_N64_HARDWARE.md)已测量拒绝：自身约2.77亿次布局比较、315万次GPU重组位比较和78项原始门槛通过，106寄存器/实际一CTA每SM；独立普通约41.2μs、Graph约37.8–38.0μs，比P51慢50%–51%/38%，相对P53也未改善。再次逐字恢复P51并通过新进程78项复核。[M15细分组全K512整数补偿消融](research/precision_20261016/int8_hardware/FINE_QUANT_SCREEN_REPORT.md)和[M16指数恢复消融](research/precision_20261016/int8_hardware/EXPONENT_RESTORE_SCREEN_REPORT.md)合计168组新输入支持A8/B8细分组：全部原始与固定门槛通过，保留两种输入及7+8点积。粗分组有FP16额外固定上限阴性对照，原始门槛仍通过；指数恢复未改变这些矩阵的修正值，排除已观察的零因子作为该批误差来源。[P55直接细分组INT8硬件](research/precision_20261016/int8_hardware/DIRECT_FINE_QUANT_HARDWARE.md)现已实测拒绝：自身96组CPU参考打包、6组混合TMA布局、78项标准及84组新矩阵精度/保护区通过；普通完整调用约45.4–45.9μs、Graph约44.4–44.7μs，比同轮P51慢65%–67%/62%–63%。自身两核profile与全部计时臂冻结，无自身双流压力/工具证书。P55拒绝时已逐字恢复P51，新进程78项复核通过；[再次调研与历史复核](research/precision_20261016/int8_hardware/LITERATURE_UPDATE_P55.md)后完成上述P56原生调度消融并独立验收保留。目标继续进行。
[片上融合后续](research/precision_20261016/fused_int8/REPORT.md)修正布局后通过72项普通精度；继续移出逐元素缩放判断后，F13完整调用约224–240 μs，仍拒绝性能验收。
下方 iter282 等标签为历史检查点，实际起点和文件哈希以新报告为准。

**历史 1024 专用路径检查点：iter282（1bdfb29）的七乘法、无补偿实验内核。以下为历史计时，不作为恢复后的新测量。**
完整A/B/B/A普通调用：BF16/FP16 SW约 **10.273/10.238 μs**，同期调优Dense约 **12.228/9.807 μs**。
BF16加速比约1.190×；FP16仍慢约4.40%，还需减少约0.432 μs（SW时间的4.22%）。
Graph单独配对：SW约 **9.976/9.993 μs**，Dense约 **11.274/9.341 μs**；不替代普通调用目标。
279相对266普通BF16/FP16改善约2.43%/0.70%；282相对279仅约0.47%/0.35%，没有新的3%以上突破。
313逐字恢复后的新测BF16/FP16约10.293/10.246 μs，Dense约12.469/9.791 μs；FP16仍需减少约0.455 μs。
292–312的cluster分工、本地shared变换及同步改动均未改善保留版普通调用；错误或未编译候选明确排除。

**该历史专用实现的两种输入均未通过原始精度门槛**：BF16 eF约0.0060152，FP16约0.000754626。
`Auto` 和 `Dense` 继续普通调优 cuBLASLt GEMM；当前 SW 必须显式指定。

[当前持续优化检查点（292–313，目标仍在进行）](trajectory/20261015_optimization-goal-checkpoint-313/REPORT.md)

[此前271–291检查点](trajectory/20261015_optimization-goal-checkpoint-291/REPORT.md)

[此前249–270检查点](trajectory/20261015_optimization-goal-checkpoint-270/REPORT.md)

[此前209–248检查点](trajectory/20261015_optimization-goal-checkpoint-248/REPORT.md)

[此前165–208检查点](trajectory/20261015_optimization-goal-checkpoint-208/REPORT.md)

[此前性能配对与完整独立验证（109–162）](trajectory/20261015_optimization-goal-checkpoint/REPORT.md)

历史 1024 专用实现使用4 MiB调用方workspace、196736B动态shared和八个BK128 operand slots。
输入变换是128 CTA×512线程的primary kernel；乘法是128 CTA×256线程的secondary，
TMA warp0、TC warps4–7，warp1 leader提前预取四个当前descriptor。
叶乘法按B转置乘A转置组织，TC私有寄存器直接重组输出；P1/P2可与变换重叠，及时释放slot。
使用官方PDL与首次变换块前的grid dependency wait，两个kernel统一设置100%shared carveout。
`run()`无分配、无host同步；每次为当前指针编码tensor maps，无跨调用计算值缓存。

历史279和282分别通过72项fresh-input逐位检查、三种quick sanitizer，以及256次双stream/Graph调用。
282验证源码对应当时的solution。这些检查验证实现与同步，**不代表通过原始FP64精度验收**。

[历史四slot预取优化（iter100）](trajectory/20261015_validation-seven-prefetch/REPORT.md)

[历史七乘法 profiling（iter97）](trajectory/20261015_profile-iter97-seven-products/REPORT.md)

[历史补偿版输入变换优化（iter96，精度通过但更慢）](trajectory/20261014_validation-iter96-supply/REPORT.md)

## 当前目录

```text
solution/
  plan.cpp             Dense 规划、cuBLASLt 调优、指针检查和执行
  scale_output.cu      alpha=0 或 K=0 时的 FP32 输出处理
  dense_hopper.cu     TMA/WGMMA dense 候选与分发
  dense_ampere.cu     偶数边界尺寸的 SM80 Tensor Core dense 候选
  dense_wmma.cu       WMMA/显式 MMA 候选
  dense_padding.cu    打包、补零、dense 叶乘法与写回
  tuning_check.cu     调优阶段的数值筛选
  sw_pipeline.cu      通用 SW、WMMA 七乘法、逻辑补零与融合输出
  sw1024.cu           恢复的 1024³ 专用 PDL/TMA/WGMMA 七乘法（iter313 原文）
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
历史设计。当前所有尺寸共用单层 Winograd 实现，见下文。

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

单独检查统一 SW 的 1024 尺寸：

```bash
MEGAFMM_ALGORITHM=sw bash scripts/bench.sh <label>
MEGAFMM_ALGORITHM=sw MEGAFMM_GRAPH=1 bash scripts/bench.sh <label>
bash scripts/on_gpu.sh ./build/test_sw1024
```

`test_sw1024` 仅执行 1024³，检查两种 dtype、六类输入、每类三轮新输入以及同一
Graph 的新输入重放。它仍使用原始 FP64 精度门限，专用及通用无补偿实现均不保证通过。

## SW Pipeline

显式选择 `Algorithm::SWPipeline`（benchmark 中为 `--algorithm sw`）时：

- 连续、非转置、alpha=1、beta=0 的 1024³ 在 workspace 预算至少 4 MiB 时，
  Plan 选择 `sw1024_pdl`，运行时 A/B 16 字节对齐则执行专用两 kernel PDL 路径。
  指针未对齐时回到通用 SW；绑定元数据报告实际路径。专用 Plan 的调用方 workspace 为 4 MiB。
- 其他情况使用 `sw_pipeline`：支持 BF16/FP16、矩形、奇数边长、转置、
  leading dimension、alpha/beta 和兼容 C0/D 原位视图。尺寸范围沿用公共 API 的 INT_MAX 上限。
- 通用路径将每个维度逻辑补到偶数，再切为四个子块；边界读取补零，不分配 padding buffer。
  每个 warp 计算一个 16×16 子块的七个 Winograd 乘法，输入变换在 shared memory 中完成，
  FP32 累加及输出重组，调用方 workspace 为 **0 字节**。
- M/N 为零或 K/alpha 为零时，分别走已有 empty/scale 路径。
- `run()` 保持无分配、无 host 同步，支持 CUDA Graph 和新输入重放。GPU 目标仍为 Hopper sm90。

统一路径尚未进行性能调优，不能沿用历史 1024 专用路径的加速比。
它沿用无补偿的变换舍入策略，**不保证通过原始输入 FP64 精度门限**，Auto 仍使用 Dense。
实现正确性测试将整数输入与原始 AB 精确比较，并将非整数输入与独立的标量 Winograd
舍入参考比较；后者不能代替原始 AB 精度验收。

```bash
bash scripts/on_gpu.sh ctest --test-dir build \
  -R 'sw_pipeline_implementation|gpu_correctness|exact_winograd_dag' --output-on-failure
bash scripts/on_gpu.sh ./build/megafmm_bench \
  --algorithm sw --shapes '512,512,512;1025,1537,769;2048,2048,2048' \
  --dtype both --trials 10 --warmup 3 --groups 3 --correct-trials 2 \
  --output results/multisize.csv
```

benchmark 会报告实际路径、workspace、原始 FP64 精度和计时；未过精度门限时退出码为 2。
`test_sw_pipeline` 覆盖小尺寸、512³、1024³（连续及转置/stride/原位）、1025/2048/4096 行矩形、四种转置组合、非连续布局、
元素对齐指针、alpha/beta、原位输出、padding 保护、fresh-input launch/Graph 和退化尺寸。

`scripts/bench.sh <label>` 保留 AKO 的源码快照逻辑；正式迭代记录 `ITERATIONS.md`
和对应 Git 版本。按用户最新指令持续 AKO4ALL 优化，目标仍在进行；每轮保存失败及有效结果，未通过精度或性能验收不宣布合格 winner。

## 按 M/N/K 实测选优（进行中）

新增显式 `Algorithm::Tuned`，benchmark 参数为 `--algorithm tuned`。Auto 和 Dense 的
含义保持不变。Tuned 在 Plan 中保存与该问题尺寸、dtype、布局和当前设备对应的选择：

1. 独立调优原有 cuBLASLt 候选池，作为可执行 fallback。
2. 检查 Hopper dense 配置及其 cluster、swizzle、raster 和流水变体；输入使用原始
   BF16/FP16，FP32 累加和输出。配置覆盖窄 tile、大 tile 与不同 K 分块；实际可用候选
   由尺寸、布局、指针对齐和 shared-memory 限制决定。另有 SM80 指令集的 CUTLASS dense
   候选处理偶数但未按8元素对齐的尺寸，运行目标仍为H100。
3. 小输出（M×N≤4096、K≤256）加入直接 FP32 FMA 候选；非对齐或一般布局加入四种
   pack→Hopper GEMM→unpack 候选。后者将所有打包、补零和写回成本纳入调用计时，scratch
   受调用方 workspace_limit 限制。
4. 在 tune 阶段先检查候选输出与 Dense 的相对 Frobenius 差≤1e-5 且无 nonfinite。
   对未通过该检查的 beta=0 Stream-K/Split-K 候选，从当前已舍入的原始 A/B 计算完整
   FP64 参考，按原有 `eF_candidate <= 3*eF_dense + floor` 门槛复核，避免更准确的归约
   结果因求和顺序不同被误筛；保留 alpha、转置及 stride 语义。FP64 临时分配和同步只在
   显式 tune 中发生，metadata 报告复核次数与通过数，run 路径不变。正式 benchmark
   仍独立使用三组新输入执行原始 FP64 精度检查。
5. 将 Lt 池中最快的八个候选与自定义候选一起重新计时，匹配 `Policy::tuning_input_sets`
   指定的输入工作集（1–4套，默认1套；benchmark 使用 `--rotate-inputs` 设置）。按交替
   顺序进行五轮、每轮20次完整调用计时。中位数快至少3%的候选进入持续确认，短测只用于提名。
6. 按短测中位数顺序逐个持续确认，取消前两名上限。预热后执行五轮交替的100次完整调用，
   仍要求中位数严格低于Lt的97%且五轮全部获胜；首个失败配对即淘汰该候选，继续下一名，
   直到找到首个通过者或候选耗尽。否则保留 fallback。metadata记录每次尝试的候选ID、
   分组数和全部配对时间。正式 benchmark 独立执行100次×5组。

`lt_algorithm_config()` 报告选择、各候选中位耗时及 fallback 的 Lt 配置。未通过约束或数值
检查的候选耗时为 nan。调用前尚未 tune 时使用 Dense。run 只构造当前指针的 tensor maps
并执行，不分配或进行 host 同步；绑定变化时重新检查可用条件，不复用输入计算值。

稳定性验收必须独立重复，不能把 tune 内部胜出当作最终结论：

```bash
bash scripts/build.sh
python3 scripts/tune_mnk.py --repeats 3 --output-dir results/mnk_acceptance
```

默认覆盖 bench/shapes.json 全部14种尺寸和两种 dtype，每轮100次、5组、3轮新输入精度检查，
轮换两套输入。registry.json 记录每个尺寸/dtype 的实际路径、各组配对加速比、精度和选中配置。
只有每次都采用自定义候选、精度全通过、全部配对组均至少快3%时才记 stable_win；fallback
只算功能覆盖。该脚本仍有未解决尺寸时返回2，结果文件会保留。加 `--graph` 独立验收 Graph。
registry 是测量证据，不会自动加载为不经复验的运行时缓存。

当前目标仍是所有尺寸稳定超过独立调优的 cuBLASLt，尚未完成。iter353 的三次独立全套
验收中，28个尺寸/dtype组合全部通过原始输入 FP64 精度门槛，**16/28个组合**的全部15组
配对计时均快至少3%。下面给出这些组合的配对组中位加速比（括号内为最差组）：

| M×N×K | BF16 | FP16 |
|---|---:|---:|
| 512³ | 1.120×（1.112×） | 1.124×（1.119×） |
| 1024³ | 1.480×（1.469×） | 1.190×（1.186×） |
| 2048³ | 1.093×（1.063×） | 1.048×（1.042×） |
| 8192×1024×512 | 1.075×（1.061×） | 1.099×（1.077×） |
| 1024×8192×512 | 1.117×（1.109×） | 1.120×（1.093×） |
| 1025×1537×769 | 1.115×（1.111×） | 1.072×（1.070×） |
| 514×770×258 | 1.294×（1.289×） | 1.297×（1.290×） |
| 1³ | 1.085×（1.062×） | 1.091×（1.077×） |

完整证据见 [iter353验收记录](trajectory/20261015_mnk-validation-353/acceptance/registry.json)。
其他12个组合仍未解决，集中在4096³及更大的正方形和三种大K矩形；fallback与偶发计时
优势不计作自定义内核获胜。证据只认证对应版本，新候选和选择策略改动仍须重新验收。
此前iter351的13/28、iter344的11/28、iter339的9/28认证记录继续保留。

## 历史与清理记录

- [架构清理记录](results/architecture_reset_20261014.json)：当前保留内容和验证结果。
- [Dense 冒烟测试](results/architecture_reset_dense_smoke_20261014.csv)：BF16/FP16 原门限检查。
- [ITERATIONS.md](ITERATIONS.md) 和 `trajectory/` 保留历史上下文；历史源码和已提交报告可从 Git 查看。
- 当前 `results/` 中的旧报告已被移除，旧迭代中的文件路径不代表当前文件仍存在。


iter364/365 后续分别测量了 source-free TMA 的 K=128 与 K=32 分块，候选池扩至102个。
iter365 的10种尺寸、两种 dtype 共20个组合均通过三组新输入的原始 FP64 精度检查；
102个强制候选和6项相关测试通过，CUDA12.8内存与竞争检查均为0错误、0警告。
这些验证不等于新增稳定性能认证。[逐组性能记录](trajectory/20261015_mnk-validation-365/performance.json)
中，8192×4096×8192 BF16 的新候选虽有1.0317×的中位数比值，但五组配对加速比为
0.975、1.023、0.992、1.059、1.108，不能计为稳定胜出。最新独立三轮认证仍为上述16/28。
完整正确性与剖析证据见 [iter365验证](trajectory/20261015_mnk-validation-365/validation.json)。


iter366 加入交换访问方向的两个 source-free TMA 候选，按 `B^T A^T` 的列主序视图
计算并直接写回原有行主序 D，候选池为104个。10种尺寸、两种 dtype 共20个组合均通过
原始 FP64 门槛，104个强制候选和6项相关测试通过，内存/竞争检查均为0错误、0警告。
全部12个未解决的大K组合仍选择 Lt fallback，没有新增稳定胜出；
[逐组性能记录](trajectory/20261015_mnk-validation-366/performance.json) 与
[完整验证记录](trajectory/20261015_mnk-validation-366/validation.json) 分别保存性能和正确性证据。


iter367/368 分别测试消费者 WGMMA 窗口和每次展开8个 K 分块的主循环，候选池扩至110个。
iter368 的20个尺寸/dtype组合全部通过原始 FP64 门槛，110个强制候选、6项相关测试及
CUDA12.8内存/竞争检查全部通过，错误与竞争警告均为0。
12个尚未解决的大K组合仍选择 cuBLASLt 回退，没有新增稳定胜出；最新独立三轮认证仍为16/28。
循环展开降低了固定8192³ FP16诊断中的指令数，但未转化为正式性能收益，不能据此宣称加速。
证据见 [iter368逐组性能](trajectory/20261015_mnk-validation-368/performance.json)、
[完整验证](trajectory/20261015_mnk-validation-368/validation.json)与
[性能剖析](trajectory/20261015_mnk-validation-368/profiling.json)。


iter369 增加两个 source-free `128x128x64` ping-pong 候选（cluster M=1/2），
候选池为112个。两个计算 warp group 分别处理不同输出块，以重叠计算与 epilogue。
20个正式尺寸/dtype组合全部通过原始输入 FP64 门槛，112个强制候选和6项相关测试通过。
CUDA12.8 精度保护检查及候选内存/竞争检查均为0错误、0警告；候选 sanitizer 的确切
范围为 legacy `--wmma` 子集（24/25、33..38、43..111），包含新增110/111的全部用例。
新增 `12289x520, K=8` 用例保证两个计算组都覆盖持久化流水线复用。

本轮新增内核实际选中了四个短K长方形组合：

| M,N,K | dtype | 中位时间 ms | 相对独立Lt的中位加速比 | 五组中最小配对加速比 |
|---|---|---:|---:|---:|
| 8192,1024,512 | bf16 | 0.022183 | 1.1382 | 1.1362 |
| 8192,1024,512 | fp16 | 0.022357 | 1.1460 | 1.1398 |
| 1024,8192,512 | bf16 | 0.023285 | 1.1788 | 1.1735 |
| 1024,8192,512 | fp16 | 0.023281 | 1.1573 | 1.1361 |

这些形状此前已有认证胜出，当前是新内核的单轮正式信号，不能据此增加稳定胜出数量。
十二个未解决的大K组合中，十一个选择Lt回退；`8192x4096x8192` BF16选择已有K32候选，
中位加速比1.0314，但最小配对加速比约1.0153，仍不满足稳定3%门槛。
4096³ BF16 新候选在短批次排名中约0.2036/0.2038ms，但最终未被选中；现有日志缺少
所有短批次分组和确认尝试，因此不能仅凭排名断定top2确认上限是回退的唯一原因。

iter369 的112候选版本已完成三轮独立全套认证：**15/28 稳定胜出，13 个组合未解决**。
84/84条结果均通过原始输入 FP64 精度门槛，252次新输入检查和420组配对计时已独立核验，
源码及二进制指纹保持一致。2048³ FP16 虽然中位加速仍为正，但第二轮一组仅约1.001×，
因此未通过全部15组均至少1.03×的稳定门槛；该失败数据完整保留。
历史iter353的16/28仅认证其对应版本，不能沿用到当前候选池。
完整证据见 [iter369三轮认证与原始计时](trajectory/20261016_mnk-validation-369/acceptance/registry.json)
及 [独立核验](trajectory/20261016_mnk-validation-369/acceptance/independent_audit.json)。
证据见 [iter369逐组性能](trajectory/20261016_mnk-validation-369/performance.json) 与
[完整验证及确切检查范围](trajectory/20261016_mnk-validation-369/validation.json)。
下一轮持续确认策略的独立CPU原型已通过测试，尚未接入正式实现，详见
[策略原型和证据限制](trajectory/20261016_mnk-research-370-policy/proposal.json)。

### iter370：扩大持续确认范围（独立认证待完成）

取消短测五组全胜的提名前置条件和前两名确认上限，持续确认仍使用原有五组全胜及
严格97%中位耗时门槛。新增CPU回归检查覆盖第九名候选胜出、首次失败提前淘汰、
单个失败配对、3%边界、两种中位数定义及非有限计时。正式完整14种尺寸、两种dtype
共28条结果均通过三组新输入的原始FP64精度检查；单轮17个配置的全部5组达到1.03×。

| M×N×K | dtype | 持续确认尝试数 | 外部中位加速 | 外部最差配对 | 单轮判断 |
|---|---|---:|---:|---:|---|
| 4096³ | BF16 | 8 | 1.076× | 1.004× | 未达稳定门槛 |
| 8192×4096×8192 | BF16 | 3 | 1.095× | 1.002× | 未达稳定门槛 |
| 8192×4096×8192 | FP16 | 3 | 1.029× | 0.987× | 未达稳定门槛 |
| 4096×8192×8192 | FP16 | 4 | 1.101× | 1.066× | 潜在新增赢家，须独立复测 |
| 2048³ | FP16 | 1 | 1.087× | 1.069× | 本轮通过，先前369波动仍保留 |

扩大确认范围已让第八名、第四名等候选得到持续计时机会，但内部通过不保证外部稳定。
最新完成的独立认证仍是iter369的15/28；本版尚未完成三轮独立认证。
[本轮全部配对及候选确认记录](trajectory/20261016_mnk-validation-370/performance.json)。
