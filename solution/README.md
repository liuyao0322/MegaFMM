# 通用实现

主线按精度使用共享内核，不按固定矩阵尺寸注册源码路径。公共 API 见 [gemm.h](../include/megafmm/gemm.h)。

| 文件 | 用途 |
|---|---|
| [sw/sw_pipeline.cu](sw/sw_pipeline.cu) | 任意布局的零 workspace 七乘法 fallback |
| [sw/hopper/sw_hopper.cu](sw/hopper/sw_hopper.cu) | BF16/FP16 单一可配置 Hopper 模板（native/padded） |
| [sw/fp32/sw_fp32.cu](sw/fp32/sw_fp32.cu) | FP32 零 workspace 同步/TMA，BK16/32 |
| [sw/fp32/sw_fp32_batch.cu](sw/fp32/sw_fp32_batch.cu) | 同一份 packed 模板：物化或共享重构、可配置驻留任务循环 |

`SWPipeline` 保留通用 padded Hopper 和零 workspace 回退；`SWTuned` 在预算内按实际完整调用调优。BF16/FP16 至多 10 项（9 项统一模板配置 + 零 workspace 回退），当前FP32至多25项（4项零workspace + 21项packed模板）。默认Auto/Dense仍为cuBLASLt。旧特化代码在[归档](../archive/sw_specializations/README.md)，不参加生产构建或注册。

Tensor Core 的参数注册表在 [sw_hopper.h](sw/hopper/sw_hopper.h)，统一算法在 [sw_hopper.cu](sw/hopper/sw_hopper.cu)；详见 [模板说明](../docs/tensorcore_template.md)。

## 当前 FP32 默认候选池 — iteration209

默认池保留已接受的202全部19项，只增加两项完整验证后的通用配置。所有配置仍来自同一个模板主体；`SWTuned`先按任务数和预算筛选，再实际测量打包、七路叶子和重构的完整调用。不使用固定shape分支，实际协作容量、原始B绑定和安全回退检查保持有效。

| packed ID | Tile / BK | 策略 |
|---:|---|---|
|0|128×64 /32|DirectB物化，PDL3、交错FMA和产品L2保留|
|1|16×32 /32|共享重构，驻留CTA循环处理输出任务|
|2|64×32 /32|共享重构，较大线程输出tile|
|3|128×64 /32|DirectB物化，无PDL对照|
|4|64×32 /16|HybridRawB融合|
|5|128×32 /8|两warp/产品、四级HybridRawB流水|
|6–7|64×32 /16、32|四象限A、独立打包、producer展开|
|8|64×32 /16|七产品shared行XOR布局|
|9–10|64×32、32×32 /16|受保护的最后K块|
|11|32×8 /32|窄线程行片段、驻留任务循环，任务波数上限1|
|12|128×32 /32|DirectB物化、PDL2，任务波数上限2|
|13–14|64×32 /16、32|A七变换，B原始四象限TMA，任务波数上限8|
|15–16|64×64、64×32 /32|较小DirectB物化tile、PDL3，任务波数上限32|
|17|64×32 /16|四象限A与HybridB，独立打包PDL1，任务波数上限8|
|18|64×32 /16|A七变换与B原始四象限TMA，独立打包PDL1，任务波数上限8|
|19|64×32 /16|A七变换、B四象限打包，连续输入读取/向量输出打包、shared行XOR、PDL1，任务波数上限8|
|20|64×32 /16|同19，并以`OutputTailConfig<1>`裁剪只有一行有效输出的最后M块|

公开ID为`9500000+packed ID`，随编译注册表位置变化，不是稳定序列化接口。`TuningConfig<Kernel,Waves>`限制候选参与实测的输出任务数量：任务数不超过`Waves×SM数`才参加选优；这与执行时实际驻留容量检查分开，容量检查始终保留。

本轮验证：固定首批30个shape各三个独立进程，1890组配置/5670次fresh原输入FP64与三份冻结SW逐位检查；全78个JSON shape新旧池配对尝试，每池74项有效、相同4项严格精度拒绝；两项新策略各通过完整memcheck/synccheck/racecheck，7组匹配NSYS/NCU捕获。完整性能、选中ID、独立重复范围与退化检查见[固定30报告](../results/fp32_better209/focused_report.md)、[全量对比](../results/fp32_better209/report.md)与[接受决定](../results/fp32_better209/decision.json)。全量单轮结果不等同于全部shape独立认证，首批30项全部超过cuBLASLt的目标尚未完成。

`PackingConfig<Kernel,VectorB,Compact>`控制独立打包的输出向量化和紧凑任务网格；保留连续输入读取，支持奇数维度及非对齐原始输入。`OutputTailConfig<Kernel,Rows>`在CTA剩余有效M行数不超过`Rows`时，只保留可被有效输出消费的线程行累加器。真实K累加次序、零FMA舍入及所有producer/reader屏障到达数均保留。配置名记录启用的`vector_packing`、`vector_pack_b`、`compact_pack`和`output_tail_rows`，便于审计。

## 历史 FP32 候选池 — iteration176/177

默认注册表采用 iteration176 完整配对验收的11项配置：保留旧2/3/4/5/6/8，加入两项A4、RowXor和两项Guarded配置；删除旧0/1/7。原始14项和扩展64项均已尝试，新旧池各74项通过、4项严格精度拒绝。性能与范围见[全量对比](../results/fp32_pool176/report.md)和[审计](../results/fp32_pool176/audit.json)。

| 当前 packed ID | 配置 | Tile / BK | 作用 |
|---|---|---|---|
| 0 | 旧2 | 128×64 / 32 | 物化、PDL、保留产品与交错FMA |
| 1 | 旧3 | 16×32 / 32 | 驻留CTA循环处理输出任务 |
| 2 | 旧4 | 64×32 / 32 | 共享重构、高叶子复用 |
| 3 | 旧5 | 128×64 / 32 | 非PDL物化对照 |
| 4 | 旧6 | 64×32 / 16 | HybridRawB融合 |
| 5 | 旧8 | 128×32 / 8 | 每产品两warp、四阶段HybridRawB融合 |
| 6 | A4 | 64×32 / 16 | 四象限A、独立打包、producer展开 |
| 7 | A4 | 64×32 / 32 | 四象限A、较深K阶段 |
| 8 | RowXor | 64×32 / 16 | 七产品shared交接布局置换 |
| 9 | Guarded64 | 64×32 / 16 | 保护最后K块 |
| 10 | Guarded32 | 32×32 / 16 | 保护最后K块与较小输出tile |

公开ID为`9500000+packed ID`，数值随注册表位置变化，不是稳定序列化接口。`SWTuned`按实际完整调用测量选择，继续保留原始七产品/RN顺序、raw绑定回退、实际协作容量检查和caller-owned workspace。

iteration177已完成不带外部注册表覆盖的默认构建验收：完整构建/7CTest、29个叶子主体与176一致且无local、两次重点shape复测198组合/594次fresh检查及三冻结逐位比较通过，全78公开入口尝试为74通过/4严格精度拒绝。本轮22/74快于同轮cuBLASLt；完整表见[默认构建报告](../results/fp32_default177/report.md)与[审计](../results/fp32_default177/audit.json)。新旧池的配对改善证据仍来自176，不把跨轮波动当作注册表更新的收益。下面带历史轮次和旧配置数的结果保留各自原始范围。

## FP32 数据流模板

`KernelConfig` 的 `Flow` 参数为 `DataflowPolicy<ASource, BSource, Pack, Tail, Producer>`，其后可选 `ProductLayout`。当前默认注册21项通用配置，策略通过注册表参与 `SWTuned`，没有固定shape分支。

| 参数 | 可选值与行为 |
|---|---|
| A/B source | `PackedTransforms`：七路变换；`PackedQuadrants`：四象限，producer在shared展开；`RawQuadrants`：producer直接受保护读取原矩阵并展开。B另有`HybridRawB`，保留原三个raw产品、四个packed变换及全pack回退。 |
| Pack | `Automatic`：保留原执行选择；`SeparateKernel`：强制独立打包；`CooperativeResident`：尝试驻留协作打包，容量或绑定不满足时安全回退。融合计算与打包方式可以独立选择。 |
| Tail | `Padded`：保留原融合K循环；`GuardedFinalTile`：只计算有效半K，尾段受保护且保留规范零FMA舍入。存储仍对齐32，不改变M/N tile几何和真实K累加顺序。该显式选项用于融合路径，物化叶子已有有效K处理。 |
| Producer | `TmaIssuer`：原单线程发起TMA；`WarpTransform`：一个producer warp协作变换；`WarpTransformTmaB`：原始B四象限TMA和warp展开，绑定或对齐不满足时回退。分别使用搬运完成、变换完成和全部计算读者释放的事件。 |

四象限/原始矩阵数据源目前用于融合策略，必须配合 `WarpTransform`，非法组合在编译期拒绝。`RawQuadrants` 当前使用通用的受保护global读取，A侧尚未引入原始布局TMA转置；这是可用的实验策略，不代表已经取得性能收益。完整重构、alpha/beta、C=D及输出边界继续共用现有实现。

```cpp
using Flow = DataflowPolicy<
    OperandSource::PackedQuadrants,
    OperandSource::PackedTransforms,
    PackExecution::SeparateKernel,
    TailPolicy::GuardedFinalTile,
    ProducerPolicy::WarpTransform>;

using Candidate = KernelConfig<
    16,32,16,4,4,1,1,8,1,
    ExecutionStrategy::CooperativeFused,
    true,0,false,false,false,2,Flow>;
```

`shape(problem, id)`、workspace大小和TMA平面数量随配置变化。七路/四象限/原始读取分别需要7/4/0个输入workspace平面；两侧均原始读取时可用零workspace。`shape(problem)`继续表示原七路布局。旧`FusedRawB`参数自动映射到`HybridRawB`，显式策略与旧参数冲突时编译失败。配置名称记录全部策略字段，供调优结果审计。

功能验证注册表为[tests/fp32_policy_configs.inc](../tests/fp32_policy_configs.inc)，使用已有覆盖选项构建：

```bash
MEGAFMM_BUILD_DIR=.scratch/build-fp32-policies \
MEGAFMM_RESULT_DIR=results/fp32_policies \
MEGAFMM_FP32_CONFIG_HEADER="$PWD/tests/fp32_policy_configs.inc" \
bash scripts/build.sh
bash scripts/on_gpu.sh .scratch/build-fp32-policies/test_sw_paths --packed-only
```

实验注册表用于验证独立策略和组合，不是默认候选池或性能推荐。扩大默认池前仍需完整调用性能对照与严格数值验收。

本轮验收：[报告](../results/fp32_policies160/report.md)、[审计](../results/fp32_policies160/audit.json)。默认/实验构建各7项CTest通过；原JSON7个代表shape共168配置组合、504次fresh原输入FP64及三冻结SW逐位检查全部通过。默认19个计算入口SASS与159完全一致，实验32入口无local指令；3个代表配置通过memcheck/synccheck/racecheck。此为模板重构与正确性验收，不认证全shape性能提升。

参数 sweep 可以显式覆盖模板注册表，默认构建使用头文件中的21项。覆盖文件只列通用 `KernelConfig` 参数，不添加按shape分派的设备代码：

```bash
python3 scripts/sweep_fp32_configs.py --output results/my_sweep/configs.inc
export MEGAFMM_FP32_CONFIG_HEADER="$PWD/results/my_sweep/configs.inc"
export MEGAFMM_BUILD_DIR=.scratch/build-my-sweep
export MEGAFMM_RESULT_DIR=results/my_sweep/screen
export MEGAFMM_PACKED_COMPARE=1 MEGAFMM_SW_PRECHECK=1
export MEGAFMM_ALGORITHM=sw-tuned-strict MEGAFMM_DTYPE=fp32 MEGAFMM_ROTATE_INPUTS=2
bash scripts/bench.sh iter-N
```

`iter-N` 应替换为新的实验编号，并按项目流程记录结果、提交。默认读取完整 `bench/shapes.json`；用既有 `MEGAFMM_SHAPES` 可指定筛选子集，保留JSON原次数。生成器同时输出ID/参数映射JSON；`--ids 2,3,4,...` 可生成小型确认池，JSON中的 `sweep_id` 保留原sweep编号。新建Plan的公开ID仍按当前编译注册表位置计算，不能跨两种注册表直接比较ID。清除 `MEGAFMM_FP32_CONFIG_HEADER` 后构建会恢复默认注册表；CMake也可直接设置同名缓存变量。覆盖宏会一致传递给库、测试、公开benchmark和packed对比程序。

**2026-10-23 参数 sweep（157–158）：** 60项在原JSON的8个代表shape筛选，再将旧2/3/4/6和新16/19/47组成7项确认池，完成全部14shape及两次预先固定的6shape复测。662组合、1986次fresh原输入FP64和三冻结SW逐位检查均通过；全14确认池9/14快于同轮Lt，最高16384为1.161×。完整数据见 [best.csv](../results/fp32_sweep158/best.csv)、[audit.json](../results/fp32_sweep158/audit.json)。这不是60项×14shape穷举，也不是全14独立重复认证；默认9项注册表未变。

新47为64×32/BK16/CK16/3级融合流水，1024在完整轮与两次独立复测中较旧4快1.49%/1.79%/1.45%；包含筛选轮共20/20配对组快旧4和Lt。新19为128×128/BK32/CK16/M4/PDL3/zigzag/retain，4096仅11/20组快旧2，不能宣称稳定改善；8192/16384单轮较旧2快0.58%/1.08%。两个短K矩形仍选旧6并落后Lt。

6CTest、新19/47的memcheck/synccheck、两次公开严格SWTuned、16叶子无LDL/STL及9保留叶子指令等156均通过。公开实验构建（11候选）1024选择9500006（sweep47），54.044μs/Lt57.134μs=1.057×；4096选择9500005（sweep19），2476.832μs/Lt2682.144μs=1.083×。这些公开ID属于7项确认池，不能映射为默认9项中的同位置配置。

## 矩形调优检查点 — iteration129–134

用户将当前重点转向瘦高/矮胖矩阵。优先两个短K形状8192×1024×512与1024×8192×512；本轮没有接受新的性能配置，当前cu/header/tests仍逐字节等于已验收112（9ba57230），保持六packed/公开最多十项，无新增尺寸源码或阈值。

同轮原20warmup/5×100/3fresh原输入FP64/2rotate普通完整调用对照：129融合64×64/BK16与驻留循环、130逆序重构、132两/四输出带复用中间缓冲、133转置64×128及BK64均未建立收益。0.14%的单轮转置变化不作为可接受提升；失败配置/参数已全部移除。131是补丁检查在写入前失败后误触发的恢复验证，不是分带性能结果。

实际NSYS路由、NCU及SASS证据：融合省掉global产品面但FMA78.64%→65.56%（普通）/53.83%（驻留）；分带实际每call3/5/9launch，叶子FMA78.77%→约76.19%/73.03%；原十叶子指令始终保持112且无local。逆序没有完整调用收益。NCU kernel replay可能改变前驱缓存状态，58.7MB重构读计数只作诊断，不代表普通调用流量已被独立证明；PDL阶段walltime含等待/重叠，不能相加。

134新完整构建验证原JSON全部五个矩形和512/1024/2048/4096方阵：54组合/162freshFP64/450TIME、原三个冻结SW逐位、6CTest、十实际叶子SASS等112且无LDL/STL通过。以下为恢复后新同轮最佳直接配置，时间µs；不是新的全14独立性能认证，也不是公开SWTuned重新选型。完整14表继续使用下方112验收。

| M×N×K | id | SW µs | Lt µs | Lt/SW |
|---|---:|---:|---:|---:|
| 1024×8192×512 | 2 | 194.518 | 177.260 | 0.9113× |
| 4096×8192×8192 | 2 | 9599.792 | 10800.802 | 1.1251× |
| 8192×1024×512 | 2 | 196.805 | 181.628 | 0.9229× |
| 8192×4096×8192 | 2 | 9619.070 | 10783.294 | 1.1210× |

[本轮对照与限制](../results/fp32_rect134/round_report.json)、[恢复验证](../results/fp32_rect134/restore112/verification.json)、[完整恢复CSV](../results/fp32_rect134/restore112/current.csv)。

## 当前 FP32 配置与验证 — iteration112

代码 `9ba57230`。六项通用模板中仅 id2 启用 `RetainProducts=true`，在原128×64/BK32、每线程8×8、ComputeK16、GroupM4、PDL3和交错FMA基础上，用带L2保留提示的向量写回交接七路产品。其余五项保留，公开FP32池最多十项；没有尺寸阈值或专用源码。

```cpp
// ... Strategy,PersistentTasks,ProgrammaticMask,ZigzagFma,RetainProducts
KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized,false,3,true,true>
```

原七路数学、每个累加器递增K的RN FMA和重构关联不变。每次完整打包、七路计算和重构；rawB精确绑定守卫/全pack回退、协作真实容量检查、调用者stream/workspace、并发和run无分配/host同步继续保留。

iteration128 已从 `9ba57230` 逐字节恢复源码、头文件和原测试，移除121–125未获收益的 Direct-A 字段、消费者和诊断项，以及127的窄重构与共享内存偏好参数。新完整构建、四个原 JSON 中型形状的24组合/72fresh FP64检查、6CTest及10叶子指令指纹检查通过，且无叶子local读写。2048恢复复测为327.105μs/Lt341.757μs，本次5/5组领先，最慢组仍为340.188μs；这是恢复检查，不更新下方全14验收表或认证跨进程稳定领先。见[最新恢复审计](../results/fp32_leader/restore112128/verification.json)。

最近两轮2048诊断均通过原次数数值与逐位一致检查，但没有获得完整调用收益，已经从当前源码移除。下表单位μs，每行与同进程原方案比较；不可把不同进程的差值当作新收益。

| 诊断 | 同轮原方案 | CK4候选 | CK16候选 |
|---|---:|---:|---:|
| 124：循环外固定A布局，四K向量消费者 | 326.903 | 396.976 | 378.421 |
| 125：packedA原流水，rawA保留B片段并流式消费行对 | 326.935 | 358.699 | 382.970 |

两轮各有6CTest、16实际叶子中12个控制指纹等112、两个重构控制指纹等112、无叶子local读写、七种20调用路由及三份NCU捕获验证，见[124 profiling](../results/fp32_leader/profile124/diagnostics.json)和[125 profiling](../results/fp32_leader/profile125/diagnostics.json)。Direct-A少写三路A降低了打包时间，但叶子FMA效率下降，资源恢复至4CTA仍不足以获益。

127进一步测试了窄重构与较大共享内存配置的组合：2048同轮原方案326.966μs、窄重构自动配置330.433μs、较大共享偏好328.608μs。NCU确认pack132KiB/leaf228KiB均保持原配置，窄重构为32allocatedregs/128线程，配置从32KiB实际变为228KiB，14叶子指令均等112，两个窄重构主体等116，但FMA活跃比例仍约81.3%，没有净完整调用收益。该组合也已移除，详见[127 profiling](../results/fp32_leader/profile127/diagnostics.json)。

**最佳直接配置：**H100/CUDA12.4，原`bench/shapes.json`全14形状×6配置，20warmup、5×100计时、3fresh原输入FP64、2轮换输入，普通完整调用。同进程保留三个冻结SW和独立调优PEDANTIC cuBLASLt。下表为六项最低中位数，单轮9/14领先Lt，不能代替公开选择或全14独立性能认证。单位ms。

| M×N×K | 最佳id | SW ms | Lt ms | Lt/SW |
|---|---:|---:|---:|---:|
| 1×1×1 | 2 | 0.006665 | 0.002833 | 0.4250 |
| 512×512×512 | 3 | 0.013164 | 0.015066 | 1.1445 |
| 514×770×258 | 3 | 0.017342 | 0.011713 | 0.6754 |
| 1024×1024×1024 | 4 | 0.055383 | 0.056426 | 1.0188 |
| 1024×8192×512 | 2 | 0.194466 | 0.177280 | 0.9116 |
| 1025×1537×769 | 2 | 0.084183 | 0.073479 | 0.8728 |
| 2048×2048×2048 | 2 | 0.329493 | 0.343279 | 1.0418 |
| 4096×4096×4096 | 2 | 2.498673 | 2.668344 | 1.0679 |
| 4096×8192×8192 | 2 | 9.592382 | 10.792544 | 1.1251 |
| 8192×1024×512 | 2 | 0.196523 | 0.181345 | 0.9228 |
| 8192×4096×8192 | 2 | 9.625089 | 10.802840 | 1.1224 |
| 8192×8192×4096 | 2 | 9.592370 | 10.921303 | 1.1385 |
| 8192×8192×8192 | 2 | 18.904554 | 21.417356 | 1.1329 |
| 16384×16384×16384 | 2 | 147.909439 | 169.891296 | 1.1486 |

**最终六配置池重复测量：**筛查、全14及两个预先固定独立复测进程，均按原次数保留所有组；每个四中型进程24组合/72fresh检查。2048摘录如下，单位µs。

| 进程 | PDL id2 | 非PDL id5 | Lt | PDL领先Lt组数 |
|---|---:|---:|---:|---:|
| screen | 329.133 | 351.935 | 341.801 | 4/5 |
| full | 329.493 | 352.448 | 343.279 | 5/5 |
| repeat1 | 329.349 | 350.346 | 342.296 | 5/5 |
| repeat2 | 329.416 | 357.340 | 344.370 | 5/5 |

2048合计19/20组领先Lt。**仍有慢组，未证明稳定领先；继续优化目标保持进行中。** 八配置诊断池111的组数不作为本次最终池结论。

**公开SWTuned实际选择：**完整十候选池，5×20排名、最多四项5×100确认，按确认中位数选择。下表为公开API完整调用，时间µs；tune为单独调优成本秒。

| M=N=K | selected id | SW µs | Lt µs | tune s |
|---|---:|---:|---:|---:|
| 512 | 9500003 | 13.253 | 14.830 | 0.322 |
| 1024 | 9500004 | 55.571 | 57.050 | 0.360 |
| 2048 | 9500002 | 331.384 | 344.726 | 2.222 |
| 4096 | 9500002 | 2496.893 | 2720.739 | 16.588 |

**验证：**全14的84组合、252当前fresh原输入FP64及126冻结检查通过，所有当前结果与冻结SW逐位一致；6/6CTest。id2的memcheck/racecheck/synccheck零错误、零race警告，覆盖longK16384、tails/strides、非单位alpha/beta、C=D、raw绑定守卫和强制全pack回退、独立workspace并发。id2/5各24次轮换绑定调用通过，每四次完整调用间无复制或host同步。

最终benchmark十个实际叶子入口无LDL/STL，八个控制入口指令指纹等105，两个保留产品入口等111已测版本；id2的rawB叶子为113寄存器、全pack回退为115寄存器；49280B动态shared、4驻留CTA/SM。寄存器从126下降未增加CTA驻留。写回也改变编译器生成的指令与寄存器分配，不能把全部收益归因于L2提示。最终NSYS2024.6.2和NCU同时确认raw与fallback实际路由；PDL各kernel持续时间包含等待/重叠，不能相加，NCU replay不作性能认证。

完整证据：[全14CSV](../results/fp32_leader/validation112/full14.csv)、[最终审计](../results/fp32_leader/validation112/audit.json)、[六池复测审计](../results/fp32_leader/validation112/repeat_audit.json)、[公开选择](../results/fp32_leader/validation112/public_plan.csv)、[最终profiling](../results/fp32_leader/validation112/profile_diagnostics.json)。105保留为上一验收恢复点；失败DSM109、较大线程输出110和evict_first诊断均未留在本次六项池。

## 历史 FP32 检查点 — iteration105

代码 `b84f7db5`。在原六项通用模板内，仅将 id2 的 `ZigzagFma` 设为 true，使用相邻2×2输出的交错FMA次序；每个累加器仍按原K顺序执行RN FMA。id0/1/3/4/5保留，packed仍六项、公开FP32候选最多十项。512/1024的协作融合、真实容量检查、rawB绑定守卫与全pack回退保持；没有尺寸阈值或专用源码。

```cpp
// BM,BN,BK,TM,TN,WarpsM,WarpsN,ComputeK,GroupM,Strategy,PersistentTasks,ProgrammaticMask,ZigzagFma
KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized,false,3,true>
```

`ProgrammaticMask`默认0，bit0控制pack→leaf，bit1控制leaf→combine；`ZigzagFma`默认false。现有id2启用两个PDL边和交错FMA，其余五项保持原策略。id2每次调用仍完整执行pack、七路叶子和combine，使用调用者同一stream/workspace，不做run分配、host同步或输入缓存。

**最佳直接配置：**H100/CUDA12.4，原`bench/shapes.json`全14形状×6配置，20warmup、5×100计时、3fresh原始输入FP64、2轮换输入，普通完整调用；同进程保留三个冻结SW及独立调优PEDANTIC cuBLASLt。下表是六项中最低中位数，单轮9/14领先Lt，不能代替公开调优器结果或全14独立性能认证。单位ms。

| M×N×K | 最佳id | SW ms | Lt ms | Lt/SW |
|---|---:|---:|---:|---:|
| 1×1×1 | 2 | 0.006579 | 0.002786 | 0.4234 |
| 512×512×512 | 3 | 0.013240 | 0.014881 | 1.1239 |
| 514×770×258 | 3 | 0.017262 | 0.011652 | 0.6750 |
| 1024×1024×1024 | 4 | 0.055543 | 0.056802 | 1.0227 |
| 1024×8192×512 | 2 | 0.199437 | 0.176861 | 0.8868 |
| 1025×1537×769 | 2 | 0.085395 | 0.072976 | 0.8546 |
| 2048×2048×2048 | 2 | 0.334136 | 0.344159 | 1.0300 |
| 4096×4096×4096 | 2 | 2.495360 | 2.666738 | 1.0687 |
| 4096×8192×8192 | 2 | 9.643415 | 10.814574 | 1.1214 |
| 8192×1024×512 | 2 | 0.202053 | 0.182421 | 0.9028 |
| 8192×4096×8192 | 2 | 9.696656 | 10.808455 | 1.1147 |
| 8192×8192×4096 | 2 | 9.621558 | 10.903770 | 1.1333 |
| 8192×8192×8192 | 2 | 18.967066 | 21.397202 | 1.1281 |
| 16384×16384×16384 | 2 | 148.774094 | 169.846420 | 1.1416 |

**最终六配置池独立复测：**预先固定两个新进程，全部四中型×6配置，每个进程24组合/72fresh原输入FP64及冻结SW逐位检查均通过，保留每一组。2048下表单位µs。

| 进程 | 新PDL id2 | 原非PDL id5 | Lt | PDL领先Lt组数 |
|---|---:|---:|---:|---:|
| full | 334.136 | 349.717 | 344.159 | 4/5 |
| repeat1 | 333.996 | 355.752 | 342.758 | 3/5 |
| repeat2 | 336.145 | 357.132 | 344.434 | 3/5 |

2048合计10/15组领先Lt。**仍有慢组，未证明稳定领先；继续优化目标保持进行中。** iteration104八配置诊断池的15/15组不能代替本次最终池结论。

**公开SWTuned实际选择：**每形状搜索完整十候选池，5×20排名、最多四项5×100确认，选择确认中位数最低项。下表为公开API选择后的完整调用，时间µs；tune列为单独调优成本秒。

| M=N=K | selected id | SW µs | Lt µs | tune s |
|---|---:|---:|---:|---:|
| 512 | 9500003 | 13.248 | 14.836 | 0.314 |
| 1024 | 9500004 | 55.359 | 57.472 | 0.363 |
| 2048 | 9500002 | 335.885 | 344.196 | 2.229 |
| 4096 | 9500002 | 2504.018 | 2726.491 | 16.591 |

**验证：**6/6CTest；新id2的memcheck、racecheck、synccheck零错误，racecheck零警告，覆盖rawB/强制全pack回退、longK16384、tails/strides/guards、alpha/beta、C=D和独立workspace并发。id2/5各24调用的fresh轮换绑定探针通过，四次完整调用之间无复制、事件或host同步。稀疏精确绑定探针不替代上述随机原始输入FP64检查。

实际最终benchmark十个叶子入口无LDL/STL；八个保留指令指纹与99一致，两个交错PDL入口与104已测版本一致。新配置126寄存器、49280B动态shared、4驻留CTA/SM。PDL kernel持续时间包含依赖等待和重叠，不能相加；NCU replay不替代普通完整调用的性能证据。最终NSYS使用2024.6.2，NCU同时断言实际raw路由与leaf名称。

完整证据：[105CSV](../results/fp32_leader/iter105/current.csv)、[验收审计](../results/fp32_leader/validation105/audit.json)、[公开选择与成本](../results/fp32_leader/validation105/public_plan.csv)、[最终profiling](../results/fp32_leader/profile105/diagnostics.json)。失败的统一carveout100、延后pack触发101、跨片段预取102/103保留原始记录，相关失败参数和诊断候选均已移除。

后续106/107诊断均未保留：最后BK触发combine在2048为337.024µs，慢于同轮默认334.073µs；转置2×2FMA访问轴为350.832µs，慢于默认331.819µs。失败候选及参数已移除。108逐字节恢复105源码后，用全新完整构建复查四中型24组合/72fresh检查，6CTest通过；十个实际叶子指令指纹与105一致、无LDL/STL。[恢复验证](../results/fp32_leader/restore105/verification.json)保留原始组；2048此次中位数331.747µs/Lt342.112µs但仅3/5组领先，稳定性目标仍未解决。

## 历史 FP32 检查点 — iteration99

PDL（程序化依赖启动）已纳入同一份 `KernelConfig` 模板策略，packed仍为六项，公开FP32候选最多十项。新id2替换原14形状从未获胜的旧128×64/每线程8×4配置；512/1024的协作融合和非PDL大矩阵配置保留。调优器按完整调用实测选择，没有尺寸阈值或专用源码。代码提交为 `73570134`。

```cpp
using KernelConfigurations=std::tuple<
  KernelConfig<128,128,32,8,8,2,4,32,8>, // retained large-tile control, id 0
  KernelConfig<64,64,16,4,8,2,2,16,1>, // lower-register control
  KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized,false,3>, // both PDL edges
  KernelConfig<16,32,32,4,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>, // resident task loop
  KernelConfig<64,32,32,8,8,1,1,32,1,ExecutionStrategy::CooperativeFused>, // fused reuse
  KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized> // retained non-PDL directB
>;
```

`ProgrammaticMask` 默认0；bit0控制pack→leaf，bit1控制leaf→combine，新id2为3，其他项为0。pack/leaf触发后继，后继只先执行独立初始化，读取前驱结果前必须 `cudaGridDependencySynchronize()`；下一调用的普通pack继续等待上一调用结束。所有工作仍在调用者的同一stream内，计时包括每次打包、七路计算和重构。PDL只提供重叠机会，语义见[CUDA12.4文档](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)。

id2/id5均为128×64/BK32、每线程8×8、128线程、ComputeK16、GroupM4。DirectB的产品0/1/2直接TMA读取原始B三个子块；全部七路A及B3–6每次重新打包。raw描述符精确匹配N/K/ldb，保持原指针/stride对齐检查，变更绑定时在局部对齐描述符副本中替换B地址；不满足条件则全打包回退。原七乘法、递增K的FP32 FMA与RN重构关联不变。workspace仍按完整七路容量预算，run无分配、无host同步，保留alpha/beta、C=D、guards和独立并发。

协作融合id3/id4及真实occupancy容量检查保留；PersistentTasks限制驻留网格并循环处理任务，条件不满足则普通pack+fused回退。

H100/CUDA12.4，原 `bench/shapes.json` 全14形状×6配置，20warmup、5×100计时、3fresh原始输入FP64、2轮换输入，普通完整调用。同进程保留三个冻结SW对照及独立调优PEDANTIC cuBLASLt。下面是六项中的最低中位数，不能代替公开调优器选择：**单轮9/14领先Lt，尚无全14独立性能认证。** 单位ms。

| M×N×K | 最佳id | SW ms | Lt ms | Lt/SW | PDL id2 ms | 非PDL id5 ms |
|---|---:|---:|---:|---:|---:|---:|
| 1×1×1 | 2 | 0.006559 | 0.003325 | 0.507× | 0.006559 | 0.009218 |
| 512×512×512 | 3 | 0.013179 | 0.014883 | 1.129× | 0.050636 | 0.024263 |
| 514×770×258 | 3 | 0.017369 | 0.011723 | 0.675× | 0.034817 | 0.024764 |
| 1024×1024×1024 | 4 | 0.055414 | 0.056582 | 1.021× | 0.099784 | 0.066039 |
| 1024×8192×512 | 2 | 0.202486 | 0.177246 | 0.875× | 0.202486 | 0.207200 |
| 1025×1537×769 | 4 | 0.088146 | 0.073330 | 0.832× | 0.088181 | 0.089772 |
| 2048×2048×2048 | 2 | 0.338715 | 0.341941 | 1.010× | 0.338715 | 0.353118 |
| 4096×4096×4096 | 5 | 2.535849 | 2.667687 | 1.052× | 2.550267 | 2.535849 |
| 4096×8192×8192 | 5 | 9.888535 | 10.814548 | 1.094× | 9.893490 | 9.888535 |
| 8192×1024×512 | 2 | 0.205230 | 0.181675 | 0.885× | 0.205230 | 0.211205 |
| 8192×4096×8192 | 5 | 9.928963 | 10.834996 | 1.091× | 9.946841 | 9.928963 |
| 8192×8192×4096 | 2 | 9.845128 | 10.898109 | 1.107× | 9.845128 | 9.846807 |
| 8192×8192×8192 | 5 | 19.440771 | 21.421606 | 1.102× | 19.445093 | 19.440771 |
| 16384×16384×16384 | 2 | 152.424057 | 169.874985 | 1.114× | 152.424057 | 152.586655 |

预先固定两个额外独立进程，四个中等规模、全部六项、原次数，保留所有逐组样本。512/1024/4096的最佳直接测量配置在full及两个repeat共15组全部领先Lt；**2048仅9/15组领先，三个中位数领先不构成稳定胜出。** 2048摘录如下，单位µs。

| 进程 | PDL id2 | 非PDL id5 | Lt | PDL领先Lt组数 |
|---|---:|---:|---:|---:|
| full | 338.715 | 353.118 | 341.941 | 3/5 |
| repeat1 | 339.590 | 355.407 | 341.694 | 3/5 |
| repeat2 | 342.550 | 356.919 | 344.680 | 3/5 |

公开严格调优结果如下。setup/tune成本保留在原CSV中，不能以直接测量的最低值冒充实际选择。

| 规模 | 公开id | SW µs | Lt µs | Lt/SW | tune秒 |
|---|---:|---:|---:|---:|---:|
| 512³ | 9500003 | 13.252 | 14.829 | 1.119× | 0.316 |
| 1024³ | 9500004 | 55.490 | 57.579 | 1.038× | 0.363 |
| 2048³ | 9500002 | 341.692 | 344.161 | 1.007× | 2.242 |
| 4096³ | 9500002 | 2546.869 | 2717.964 | 1.067× | 16.768 |

完整84组合、252次当前fresh FP64和126次冻结控制检查通过，所有当前配置与三个冻结SW逐位一致；6/6 CTest及新id2的memcheck/racecheck/synccheck通过，零错误/零race警告，覆盖长K、尾部、stride/非对齐、精确raw元数据变更、强制全pack回退、非单位系数、C=D、guard和独立workspace并发。另对id2/5各做24次fresh3绑定测试，交替不同地址/内容的B，复用同一Maps/workspace，每四次完整调用间不插入复制或主机同步，并检查C=D累加。

最终benchmark十个当前叶子入口无LDL/STL；保留的八个指令指纹与已验收79完全一致，两个新增PDL入口的浮点运算指令序列与同几何非PDL入口一致。两种配置均为117寄存器、49280B动态shared、4驻留CTA/SM。

最终二进制NSYS2024.6.2诊断中，2048普通完整调用区间均值341.56µs，PDL337.48µs；1024则66.48→98.37µs。**PDL kernel持续时间包含依赖等待与重叠，不能相加或把combine持续时间当纯重构计算。** NCU补测确认rawB实际路由；最初两个误传可选参数0的NCU标签实际为fallback，原记录与初始缺少路由断言的审计均保留，最终审计改用明确raw_enabled补测。NCU replay也不替代普通调用性能或并发证据。

[完整CSV](../results/fp32_leader/iter99/current.csv)、[最终审计](../results/fp32_leader/validation99/audit.json)、[复测1](../results/fp32_leader/validation99/repeat1.csv)、[复测2](../results/fp32_leader/validation99/repeat2.csv)、[公开选择](../results/fp32_leader/validation99/public_plan.csv)、[协议](../results/fp32_leader/validation99/protocol.json)、[profiler诊断](../results/fp32_leader/profile99/diagnostics.json)。

复现完整JSON：

```bash
MEGAFMM_BUILD_DIR=.scratch/build-fp32-reproduce99 \
MEGAFMM_RESULT_DIR=results/fp32_leader/reproduce99 \
MEGAFMM_PACKED_COMPARE=1 MEGAFMM_PACKED_CONFIG=all \
MEGAFMM_DTYPE=fp32 MEGAFMM_ALGORITHM=sw-tuned-strict \
MEGAFMM_ROTATE_INPUTS=2 MEGAFMM_SW_PRECHECK=1 \
bash scripts/bench.sh
```

后续验证脚本为[validate99.sh](../results/fp32_leader/validate99.sh)，通过 `scripts/on_gpu.sh` 在已有分配内串行执行，已有证据目录不会被覆盖。公共API仍是create_plan→tune→run。上一验收点79的原始结果保留在[79审计](../results/fp32_cluster/validation/audit.json)，其2048未超过Lt；旧表只描述当时版本。

## 后续诊断 — iteration100

仅将PDL链的pack/leaf/combine共享内存偏好统一为100%，原六项对照保留，原JSON四中型原次数筛选全部28组合/84fresh检查通过。2048新配置341.856µs，慢于同轮默认PDL339.602µs，两者都只有3/5组领先Lt342.468µs。512的PDL路径50.279→20.829µs仍慢于原融合；1024几乎无改善。NCU确认实际划分132/228/32→全228KiB，十二个叶子SASS指纹均与99相同，未出现local溢出。设置生效没有转化为2048收益，因此临时项及SharedCarveout参数均移除，源码逐字节恢复已验收99，保留六模板。恢复后独立重新构建的四中型24组合/72fresh检查及6CTest通过，十个实际叶子指令指纹与99完全一致且无LDL/STL，见[恢复验证](../results/fp32_leader/restore99/verification.json)。完整记录见[100CSV](../results/fp32_leader/iter100/current.csv)、[100诊断](../results/fp32_leader/profile100/diagnostics.json)。

## 后续诊断 — iterations80–95

本轮继续尝试保留线程内复用的分组融合、部分物化、完成后重构、tile/BK与同步策略，均未得到更快的2048方案。80–95结束时生产代码恢复iteration79；这些失败原型没有留在当前候选池中。以下是历史诊断，下表是原JSON三形状子集筛选中的2048结果，单位µs；各行只比较同一进程的对照。

| 迭代 | 新实验 | 实验耗时 | 同轮id5对照 |
|---|---|---:|---:|
| 90 | 64×64/BK32/每线程8×8 | 352.341 | 347.547 |
| 90 | 128×128/ComputeK16 | 407.642 | 347.547 |
| 93 | 64×64/BK16，实际8驻留CTA | 365.609 | 343.603 |
| 93 | 128×64/BK16 | 353.570 | 343.603 |
| 93 | 64×128/BK16 | 353.457 | 343.603 |
| 94 | 单完成计数器，无轮询，最后CTA重构 | 355.207 | 347.116 |
| 95 | 各warp释放叶子双缓冲槽 | 350.870 | 347.089 |

上述有效实验保持20次warmup、5组×100次、3次fresh原始输入FP64校验、两组轮换输入；93的27组合、94/95各21组合均数值与旧SW逐位通过。91/92调用参数错误，比较程序在计时前拒绝，失败记录保留且没有性能结论。新同步原型未做完整验收，不作为已验证生产实现。

94 profile中融合计算与重构320.942µs，高于原叶子296.474加重构12.928µs；无local溢出，FMA活跃度74.03%低于原78.90%。95把barrier停顿采样3609降到199，但long scoreboard采样422增到3588，叶子294.579升到299.928µs，仍未改善吞吐。停顿采样仅作诊断，不能当作可直接相加或消除的时间。完整记录见[ITERATIONS.md](../ITERATIONS.md)及[后续实验索引](../results/fp32_group43/index.json)。

## 历史参数与性能记录

以下iteration60/61的配置数、候选数、时间和命令只描述当时版本，当前配置与结果以以上iteration99记录为准。

以下是加入融合策略前的 **iteration60历史参数与性能记录**，其三项/7候选数字只描述当时版本。

修改入口是 [sw_fp32_batch.h](sw/fp32/sw_fp32_batch.h) 的 `KernelConfigurations`：

```cpp
//                 BM  BN BK TM TN WM WN CK GM
using KernelConfigurations = std::tuple<
    KernelConfig<128,128,32, 8, 8, 2, 4,32, 8>, // id 0
    KernelConfig< 64, 64,16, 4, 8, 2, 2,16, 1>, // id 1
    KernelConfig<128, 64,32, 8, 4, 2, 4,16, 4>  // id 2
>;
```

| 参数 | 含义 | 约束或派生关系 |
|---|---|---|
| `BM, BN` | 每 CTA 的输出 tile | `BM=8*TM*WarpsM`，`BN=4*TN*WarpsN`；32 的倍数、各不超过256 |
| `BK` | 一次 TMA 搬运的 K 深度 | 支持16或32；与公共 workspace 的 K padding兼容 |
| `TM, TN` | 每线程累加矩阵 | 正的4倍数，保持 float4 shared load/store |
| `WarpsM, WarpsN` | CTA 中 warp 的二维布局 | 每warp的lane布局固定8×4；线程数自动为 `32*WarpsM*WarpsN`，不另行写死 |
| `ComputeK`（CK） | 完全展开的计算段长度 | 正整数且整除BK，可与搬运深度不同 |
| `GroupM`（GM） | CTA 的 M 方向分组 | 正整数；1等价于按行排列，末组处理实际tile数 |

模板静态检查上述约束。增加、删除或修改 tuple 中一项后重编译即可；metadata、运行分派、TMA box、资源查询、候选名称及调优器注册从同一张表生成。内部packed id为tuple位置，公开调优id为 `9500000+id`；id不是跨配置修改的稳定序列化接口。每个 prepared tensor map 绑定配置id及workspace，错配会被拒绝。新增配置会增加编译代码和调优开销，因此保持少量可解释的配置。

| 配置 | Tile | BK / ComputeK | 每线程 TM×TN | Warp布局 / 线程 | GroupM | shared / 寄存器 / 驻留CTA |
|---|---|---|---|---|---:|---|
| 0 | 128×128 | 32 / 32 | 8×8 | 2×4 / 256 | 8 | 65664 B / 127 / 2 |
| 1 | 64×64 | 16 / 16 | 4×8 | 2×2 / 128 | 1 | 16512 B / 77 / 6 |
| 2 | 128×64 | 32 / 16 | 8×4 | 2×4 / 256 | 4 | 49280 B / 79 / 3 |

上一轮51→58实际改动是BK16→32、ComputeK16→32、GroupM1→8，tile128×128、256线程及每线程8×8未变。本轮把这些常量和tile/线程布局一起参数化。双槽TMA、pack/combine的数学运算、七个原始乘积、FP32累加顺序和公共workspace布局保持；combine单独固定256线程，不随叶子线程数变化。

以下为iteration60同进程比较，全部14形状直接来自 `bench/shapes.json`，单位ms。保留20warmup、5×100、3fresh原始输入FP64检查、2组轮换输入、普通launch；每次pack/七路叶子/combine均计入时间。控制为冻结iteration58（33ec1bc8），另保留原简化版9d051d0a及独立调优PEDANTIC Lt。表中“最低耗时配置”只是直接测量的三个模板中位数比较，不冒充公开调优器选择或独立重复认证。

| M×N×K | 配置0 | 配置1 | 配置2 | 最低耗时配置 | 冻结58 | 原简化版 | cuBLASLt |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 0.035260 | 0.019561 | 0.022873 | 1 | 0.035511 | 0.029436 | 0.014697 |
| 1024×1024×1024 | 0.069627 | 0.071783 | 0.066924 | 2 | 0.070338 | 0.056894 | 0.057302 |
| 2048×2048×2048 | 0.406827 | 0.394844 | 0.377577 | 2 | 0.409222 | 0.389027 | 0.341838 |
| 4096×4096×4096 | 2.623014 | 2.835393 | 2.792449 | 0 | 2.638667 | 2.956619 | 2.658579 |
| 8192×8192×8192 | 19.922855 | 22.903004 | 21.313879 | 0 | 20.053019 | 23.857216 | 21.395422 |
| 16384×16384×16384 | 155.201080 | 185.372437 | 167.210251 | 0 | 156.305756 | 188.552795 | 169.877213 |
| 8192×4096×8192 | 10.246064 | 11.394665 | 10.869338 | 0 | 10.308294 | 11.918579 | 10.798341 |
| 4096×8192×8192 | 10.218163 | 11.571646 | 10.849181 | 0 | 10.288830 | 12.137254 | 10.677877 |
| 8192×8192×4096 | 10.141772 | 11.519550 | 10.803074 | 0 | 10.207675 | 11.969590 | 10.846549 |
| 8192×1024×512 | 0.226117 | 0.231326 | 0.224675 | 2 | 0.229036 | 0.217946 | 0.181597 |
| 1024×8192×512 | 0.224622 | 0.230836 | 0.224191 | 2 | 0.227542 | 0.218109 | 0.175871 |
| 1025×1537×769 | 0.093292 | 0.091327 | 0.095250 | 1 | 0.093531 | 0.087338 | 0.073576 |
| 514×770×258 | 0.025644 | 0.020385 | 0.023901 | 1 | 0.025831 | 0.021485 | 0.011666 |
| 1×1×1 | 0.009610 | 0.008789 | 0.009180 | 1 | 0.009578 | 0.006981 | 0.002860 |

所有42项配置/形状组合通过fresh FP64门限，并与冻结原简化版、iteration58逐位一致。小tile明显改善512³，1024³仍未追回原简化版。不能把参数可配置或上述最低耗时表解读为全尺寸最优或全部领先Lt。

当前复现全部模板的同进程比较（CSV每形状四行，含 `configuration_id` 及所有参数；`previous_ms` 现在指冻结iteration58，历史文件原样保留）：

```bash
MEGAFMM_BUILD_DIR=.scratch/build-fp32-strategies-reproduce \
MEGAFMM_RESULT_DIR=results/fp32_strategies/reproduce \
MEGAFMM_PACKED_COMPARE=1 MEGAFMM_DTYPE=fp32 \
MEGAFMM_ALGORITHM=sw-tuned-strict MEGAFMM_ROTATE_INPUTS=2 \
MEGAFMM_SW_PRECHECK=1 bash scripts/bench.sh
```

只测某配置，加 `MEGAFMM_PACKED_CONFIG=1`（默认all）；不指定 `MEGAFMM_SHAPES` 就读取完整JSON。公开使用仍是 `create_plan(..., SWTuned) → tune(...) → run(...)`，不指定packed id，由现有5×20排名及最多四项5×100确认选择。当前完整FP32候选池至多8项：原4项零workspace加4项模板；逐配置检查workspace预算。默认Auto/Dense仍为cuBLASLt。

完整CSV：[iter60/current.csv](../results/fp32_templates/iter60/current.csv)。旧性能/资源及profiling记录见下方历史章节。


公开严格SWTuned验证：重新创建Plan并搜索完整7候选池，原始输入FP64门限均通过。以下是公开API实际选择，20warmup/5×100/3fresh/2rotation，普通launch；tune时间另列，不计入稳态调用。

| 规模 | 选中packed配置 | SW ms | Lt ms | Lt/SW | tune秒 |
|---|---:|---:|---:|---:|---:|
| 512³ | 1 | 0.019553 | 0.014709 | 0.752× | 0.311 |
| 1024³ | 2 | 0.066919 | 0.057473 | 0.859× | 0.431 |
| 2048³ | 2 | 0.377711 | 0.341586 | 0.904× | 2.911 |
| 4096³ | 0 | 2.622073 | 2.685971 | 1.024× | 21.622 |

6/6CTest通过；三个模板全部通过memcheck、racecheck、synccheck（零错误，racecheck零警告），覆盖K尾部、末组CTA、stride/非对齐、系数、C=D、guard和独立并发workspace。检查最终benchmark的三个模板叶子SASS均无LDL/STL。`--packed-config 1`单配置测量另行通过。审计：[validation.json](../results/fp32_templates/validation.json)；公开API：[public_plan.csv](../results/fp32_templates/validation/public_plan.csv)。这轮验证证明可配置性与实际选择生效，不建立全形状最优或独立重复的性能认证。


## FP32 协作融合策略验证 — iteration61

H100 / CUDA12.4，完整原JSON14形状×4配置，共56组合，168次当前模板fresh FP64检查全部通过；每项与冻结9d051d0a、33ec1bc8及归档117逐位一致，三个冻结SW另有126次fresh FP64检查。保持20warmup/5×100/3fresh/2rotation和每次pack/七路计算/重构普通调用。下表是同进程直接测量的最低中位数，不代替公开调优器选择；大矩阵仍选择Materialized。

| M×N×K | 最快配置/策略 | SW ms | Lt ms | Lt/SW |
|---|---|---:|---:|---:|
| 512×512×512 | 3/cooperative_fused | 0.013082 | 0.014721 | 1.125× |
| 1024×1024×1024 | 3/cooperative_fused | 0.066527 | 0.056245 | 0.845× |
| 2048×2048×2048 | 2/materialized | 0.375560 | 0.341855 | 0.910× |
| 4096×4096×4096 | 0/materialized | 2.626019 | 2.667139 | 1.016× |
| 8192×8192×8192 | 0/materialized | 20.000658 | 21.426247 | 1.071× |
| 16384×16384×16384 | 0/materialized | 155.612396 | 169.989136 | 1.092× |
| 8192×4096×8192 | 0/materialized | 10.243974 | 10.817945 | 1.056× |
| 4096×8192×8192 | 0/materialized | 10.252057 | 10.830336 | 1.056× |
| 8192×8192×4096 | 0/materialized | 10.178196 | 10.944725 | 1.075× |
| 8192×1024×512 | 2/materialized | 0.225560 | 0.182471 | 0.809× |
| 1024×8192×512 | 2/materialized | 0.225573 | 0.177100 | 0.785× |
| 1025×1537×769 | 1/materialized | 0.091644 | 0.073636 | 0.804× |
| 514×770×258 | 3/cooperative_fused | 0.016752 | 0.011649 | 0.695× |
| 1×1×1 | 3/cooperative_fused | 0.006740 | 0.002840 | 0.421× |

公开严格SWTuned重新搜索完整8项候选，原始输入FP64合格后按确认中位数选择：

| 规模 | 公开选择 | SW µs | Lt µs | Lt/SW | tune秒 |
|---|---:|---:|---:|---:|---:|
| 512³ | 9500003 | 13.147 | 14.703 | 1.118× | 0.307 |
| 1024³ | 9500003 | 66.657 | 56.536 | 0.848× | 0.356 |
| 2048³ | 9500002 | 380.453 | 344.135 | 0.905× | 2.267 |
| 4096³ | 9500000 | 2638.499 | 2664.894 | 1.010× | 17.138 |

512³额外两个独立进程沿用同一二进制和完整次数，预先记录追加协议、保留所有结果，不进行有利重试：

| 进程 | 新融合 µs | 冻结117 µs | Lt µs | Lt/SW |
|---|---:|---:|---:|---:|
| 完整sweep | 13.082 | 13.115 | 14.721 | 1.125× |
| 追加复测1 | 13.152 | 13.180 | 14.622 | 1.112× |
| 追加复测2 | 13.180 | 13.230 | 14.739 | 1.118× |

6/6 CTest通过；所有4组模板经过memcheck/racecheck/synccheck，无错误、racecheck无警告；覆盖实际协作512、同形状强制容量回退、自然超容量1024、奇数K/shape、stride/非对齐、alpha/beta、C=D、输出/workspace guard、独立workspace并发及融合专属预算。NSYS确认512实际执行一次cooperative_fused_leaves，1024实际执行独立pack+fused_leaves；该零输入诊断只证明启动结构，不作为性能测量。检查实际benchmark内5个当前叶子入口SASS，均无LDL/STL。融合入口资源63寄存器、86144B shared、2驻留CTA/SM；本机132SM得到264个CTA的协作容量。

这轮证明策略接入、历史117等价的数值结果、512性能恢复及公开调优可选中；未证明全14形状独立重复领先Lt，也未重新加入历史1024向量epilogue等其他归档策略。原14形状的单轮直接最佳结果仍有未超过Lt的尺寸。所有原始组时间、追加协议、数值检查及identity在 [验证审计](../results/fp32_strategies/validation.json)；完整CSV：[iter61/current.csv](../results/fp32_strategies/iter61/current.csv)；公开选择：[public_plan.csv](../results/fp32_strategies/validation/public_plan.csv)。

## 精简阶段验证（优化前）

2026-10-22 精简版本9d051d0a在 H100 / CUDA 12.4 上完整构建，6/6 CTest 通过。覆盖三种精度、
奇数与矩形尺寸、零 workspace、转置/stride/系数、非对齐输入、边界 guard、
严格数值调优、失败后清除旧选择，以及共享 Plan 的双 stream 独立 workspace。

下面是精简后 FP32 的单进程普通调用测量：2 组轮换输入、每形状 3 次 fresh
数值检查、每组 20 次调用共 3 组。所有数值检查通过；该轮不能建立重复实验的性能认证。

| M×N×K | SW（µs） | cuBLASLt PEDANTIC（µs） | Lt/SW |
|---|---:|---:|---:|
| 1×1×1 | 4.034 | 4.435 | 1.100× |
| 33×65×129 | 9.952 | 7.904 | 0.794× |
| 512×512×512 | 29.936 | 15.336 | 0.512× |
| 1024×1024×1024 | 56.520 | 57.835 | 1.023× |
| 1025×1537×769 | 86.758 | 73.189 | 0.844× |

固定形状特化移除后，当前版本没有五形状全面领先的结论。原始数据及逐候选信息见
[fp32_launch.csv](../results/sw_general_simplification/fp32_launch.csv)，
[源码与二进制身份](../results/sw_general_simplification/validation.json)。


## 前一轮 FP32 packed 优化（iteration58）

2026-10-22 继续优化最终采用 **iteration58**，实测源码提交 `33ec1bc8`，恢复提交 `c99687d4`。
该历史轮次只有一项通用 packed 候选9500000。当前模板配置见上方章节。

- 统一128×128 tile、256线程、8计算warp，每CTA计算一个原始Winograd子乘积。
  TMA搬运BK32，两个shared缓冲槽；长K的阶段同步次数相对上一版BK16约减半。
- 固定M8 CTA分组尝试改善相邻块的数据复用，最后一组按实际tile数处理。
  仅分组的跨轮差异不足1%，本报告不单独认证分组的加速幅度。
- 分别测量重复BK16计算段和完整BK32展开，后者完整调用更快。
  因此本次保留BK32展开：叶子静态SASS从上一版1864条增加到3000条，
  重构仍656条，两者共3656条；不再声称本次优化缩短了静态指令主体。
  profiling与最终benchmark的实际指令逐条一致，叶子无LDL/STL。

| 资源 | 上版iteration51 | 本版iteration58 |
|---|---:|---:|
| shared / CTA | 32896 B | 65664 B |
| registers / thread | 119 | 127 |
| resident CTA / SM | 2 | 2 |
| 叶子静态SASS | 1864条 | 3000条 |

workspace公式及预算约束不变，8192³仍为1.3125GiB，16384³仍为5.25GiB。
每次普通调用均重新pack、执行七个原始FP32子乘积并combine；不缓存输入变换。
FP32计算及重构顺序、原始输入FP64门限、caller-owned workspace和独立并发保持。

本次对照方向均保留在git和ITERATIONS：

| 轮次 | 方向 | 8192³完整调用（ms） | 结论 |
|---|---|---:|---|
| 54 | BK32搬运、重复BK16计算段 | 20.355871 | 优于上版 |
| 55 | 128×64、8×4累加、79寄存器 | 21.472496 | 驻留3CTA，仍比54慢 |
| 56 | 64×128、4×8累加、77寄存器 | 22.005474 | 排除 |
| 57 | 在54上固定M8分组 | 20.268425 | 与54差异较小 |
| 58 | 完整BK32展开 | 20.003765 | 最终选用 |
| 59 | 128线程、8×16累加、219寄存器 | 20.384138 | 排除 |

55的NCU显示占用率升至37.23%，但叶子HBM读取8.19GB（上版4.54GB），
shared理想读取wavefronts增加约33%，FMA活跃率几乎不变。没有显著的额外
shared wavefront证据；更高占用率不保证完整调用更快。

最终全量同进程四方比较，单位ms：原简化版9d051d0a、上版iteration51、本版58，
以及独立调优的cuBLASLt PEDANTIC。形状直接来自bench/shapes.json；
20warmup、5组×100次、3fresh原始输入FP64检查、2组轮换输入、普通launch。
三份SW共126次数值检查全通过，本版与两份冻结SW逐位相同。setup/tune另列，
不计入稳态调用时间；每次pack/七路乘法/combine均计费。

| M×N×K | 原简化版 | 上版51 | 本版58 | cuBLASLt | 上版/本版 | Lt/本版 |
|---|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 0.029628 | 0.037662 | 0.035703 | 0.015126 | 1.055× | 0.424× |
| 1024×1024×1024 | 0.056872 | 0.073655 | 0.069956 | 0.056582 | 1.053× | 0.809× |
| 2048×2048×2048 | 0.389379 | 0.427510 | 0.409313 | 0.341708 | 1.044× | 0.835× |
| 4096×4096×4096 | 2.953601 | 2.764128 | 2.653229 | 2.669632 | 1.042× | 1.006× |
| 8192×8192×8192 | 23.801062 | 20.820896 | 19.998703 | 21.570061 | 1.041× | 1.079× |
| 16384×16384×16384 | 188.467773 | 163.730911 | 156.277283 | 169.705368 | 1.048× | 1.086× |
| 8192×4096×8192 | 11.893184 | 10.723030 | 10.292825 | 10.787151 | 1.042× | 1.048× |
| 4096×8192×8192 | 12.155161 | 10.720672 | 10.282470 | 10.673929 | 1.043× | 1.038× |
| 8192×8192×4096 | 11.959878 | 10.589472 | 10.206738 | 10.842324 | 1.037× | 1.062× |
| 8192×1024×512 | 0.218264 | 0.236840 | 0.228966 | 0.181567 | 1.034× | 0.793× |
| 1024×8192×512 | 0.217820 | 0.235010 | 0.227234 | 0.175829 | 1.034× | 0.774× |
| 1025×1537×769 | 0.087123 | 0.096821 | 0.093429 | 0.072905 | 1.036× | 0.780× |
| 514×770×258 | 0.021473 | 0.027485 | 0.025718 | 0.011629 | 1.069× | 0.452× |
| 1×1×1 | 0.006963 | 0.009304 | 0.009557 | 0.002766 | 0.974× | 0.289× |

本轮单次全量13/14比上版快（1³略慢），6/14比Lt快；4096³仅约0.6%的差异
不能建立稳定领先结论。小尺寸/短K仍慢于Lt，未增加按尺寸选择的专用路线。
这些是强制packed的完整调用测量，不代表公开调优器在所有形状上的最终选择。

独立新进程复测，未重新编译，次数及四方比较协议相同：

| M×N×K | 本版58（ms） | 上版51（ms） | cuBLASLt（ms） | 上版/本版 | Lt/本版 |
|---|---:|---:|---:|---:|---:|
| 8192×8192×8192 | 20.045311 | 20.793955 | 21.568876 | 1.037× | 1.076× |
| 16384×16384×16384 | 156.285004 | 163.772751 | 169.805161 | 1.048× | 1.087× |

新版叶子NCU指标（真实非零轮换输入，与最终benchmark同指令）：

| 指标 | 8192³ | 16384³ |
|---|---:|---:|
| FMA管线活跃率（%） | 76.699 | 76.925 |
| issue活跃率（%） | 82.775 | 82.979 |
| 实际warp占用率（%） | 24.847 | 24.958 |
| L2命中率（%） | 81.251 | 77.622 |
| HBM读取（GB，十进制） | 3.274 | 31.071 |
| PC barrier采样占比（%） | 10.836 | 10.973 |
| PC no-instructions采样占比（%） | 1.989 | 1.982 |

NSYS 8192³：leaves 96.1%，pack 2.7%，combine 1.2%。
NSYS 16384³：leaves 98.0%，pack 1.4%，combine 0.6%。
这些计数器用于解释变化，NCU replay耗时不作为benchmark成绩，PC stall采样占比
也不是可以直接扣除的墙钟时间。FMA活跃率包含整数管线活动。

验证：6/6 CTest；packed memcheck、racecheck、synccheck全部零错误，
racecheck零警告。包含改变输入、边界、stride、非零alpha/beta、C=D、
不匹配prepared maps/workspace拒绝及双stream独立workspace并发。

公开SWTuned严格模式在1024³/8192³重新完整调优，两项均仍选择9500000，
候选总数仍5项：

| 形状 | SW（ms） | Lt（ms） | Lt/SW | 调优耗时（s） |
|---|---:|---:|---:|---:|
| 1024³ | 0.070156 | 0.057526 | 0.820× | 0.802 |
| 8192³ | 20.052435 | 21.587971 | 1.077× | 254.671 |

完整复现：

```bash
MEGAFMM_BUILD_DIR=build-fp32-stages \
MEGAFMM_RESULT_DIR=results/fp32-stages-reproduce \
MEGAFMM_PACKED_COMPARE=1 MEGAFMM_DTYPE=fp32 \
MEGAFMM_ALGORITHM=sw-tuned-strict MEGAFMM_ROTATE_INPUTS=2 \
MEGAFMM_SW_PRECHECK=1 bash scripts/bench.sh final
```

不指定MEGAFMM_SHAPES时默认读取JSON全部14项。构建脚本从git分别读取冻结
9d051d0a和1e31c91b，仅用于四方bench，不增加生产内核。

原始结果：[全14表](../results/fp32_stages/final/current.csv)、
[独立复测](../results/fp32_stages/repeat/current.csv)、
[公开调优](../results/fp32_stages/validation/public_plan.csv)、
[完整身份与验证审计](../results/fp32_stages/validation.json)。
NCU/NSYS原始文件见[8192³](../results/fp32_stages/profile8192/)和
[16384³](../results/fp32_stages/profile16384/)；逐轮源码和失败记录见
[ITERATIONS](../ITERATIONS.md#iteration-54--bk32-transfer-with-repeated-bk16-compute)。
上述结论限于本机、这些形状及该普通调用协议，不是全部形状/应用的性能认证。

## 前一轮 FP32 packed 优化（iteration51）

2026-10-22，按计算分工、同步、复用和指令规模逐轮测量，保留 iteration51
（源码提交 `67bfd069`）。全部合法尺寸继续共用一项 packed 配置：

- 七个原始 Winograd 子乘积分别交给 grid.z 的 CTA；每 CTA 的 8 个 warp
  执行同一份 128×128/BK16 SIMT 指令流。thread0 提前发起下一槽 TMA，
  通过 ready transaction barrier 等待，CTA 阶段同步保证槽位安全复用。
- 128×128 叶子 tile 增加操作数复用；两个 shared 槽只保存当前子乘积。
  七个结果平面暂存到调用方 workspace，再由第三个 kernel 逐位保持原重构顺序。
- K 展开由原 BK32 缩到 BK16。U4/U8 的进一步压缩实测变慢，均已排除。
  独立 issuer 的七读者退役方案49、八 warp 分摊七条指令流方案50也保留了失败记录。

每次完整调用仍包含 pack、七叶计算和 combine，FP32 变换/FMA/重构不变。
workspace 为 `4×7×(mp×kp + kp×np + mp×np)` 字节，逻辑半尺寸向上对齐到32；
8192³ 为1.3125GiB，16384³ 为5.25GiB，比旧 packed 增加50%。预算不足时调优器
不注册此候选。`Auto/Dense` 仍使用独立调优的 cuBLASLt。

8192³ 的 NCU 对比（同一 H100，真实非零轮换输入，普通 launch）：

| 指标 | 冻结9d051d0a | iteration51 |
|---|---:|---:|
| shared / CTA | 172160 B | 32896 B |
| 寄存器 / thread | 113 | 119 |
| resident CTA / SM | 1 | 2 |
| 实际 warp 占用率 | 12.50% | 24.85% |
| FMA 管线活跃率 | 67.40% | 72.91% |
| issue 活跃率 | 73.28% | 79.83% |
| 叶子 HBM 读取 | 31.27 GB | 4.54 GB |
| 叶子 L2 命中率 | 39.56% | 74.97% |
| PC barrier 采样占比 | 17.65% | 15.25% |
| PC no-instructions 采样占比 | 5.43% | 0.94% |
| 叶子静态 SASS 指令数 | 3208（含重构） | 1864 |

新版重构另有656条静态 SASS，两者共2520条，比旧融合 kernel 的3208条减少21.4%；
打包 kernel 不变。NSYS 显示新版8192³叶子占96.3%、pack占2.6%、combine占1.1%。
这些计数器和阶段时间用于解释变化；NCU replay 时间不作为性能成绩，stall比例
也不是可以直接相加或扣除的墙钟时间。FMA管线活跃率包含整数管线活动。

以下是同一进程对冻结 packed、当前唯一 packed 和独立调优 cuBLASLt PEDANTIC
的完整调用比较，单位 ms。全部14个形状直接来自 `bench/shapes.json`；20次warmup、
5组×100次、2组轮换输入、每形状3次fresh原始输入FP64检查。双方84次检查全部通过，
新版与旧版逐位一致。该表强制测量 packed 配置，不等于自动调优池最终选择；
本轮6/14比旧 packed 快，4/14比Lt快，0.2%的微小差异不建立稳定领先结论。

| M×N×K | 旧 packed | 新 packed | cuBLASLt | 旧/新 | Lt/新 |
|---|---:|---:|---:|---:|---:|
| 512×512×512 | 0.029604 | 0.037638 | 0.014892 | 0.787× | 0.396× |
| 1024×1024×1024 | 0.057254 | 0.073683 | 0.056580 | 0.777× | 0.768× |
| 2048×2048×2048 | 0.389506 | 0.427906 | 0.341632 | 0.910× | 0.798× |
| 4096×4096×4096 | 2.953098 | 2.779966 | 2.651364 | 1.062× | 0.954× |
| 8192×8192×8192 | 23.753141 | 20.831083 | 21.519773 | 1.140× | 1.033× |
| 16384×16384×16384 | 188.696274 | 164.703903 | 169.607849 | 1.146× | 1.030× |
| 8192×4096×8192 | 11.874889 | 10.747088 | 10.768358 | 1.105× | 1.002× |
| 4096×8192×8192 | 12.107678 | 10.751998 | 10.653551 | 1.126× | 0.991× |
| 8192×8192×4096 | 11.940402 | 10.599579 | 10.823082 | 1.126× | 1.021× |
| 8192×1024×512 | 0.218329 | 0.236629 | 0.181494 | 0.923× | 0.767× |
| 1024×8192×512 | 0.217915 | 0.235034 | 0.175728 | 0.927× | 0.748× |
| 1025×1537×769 | 0.087420 | 0.097082 | 0.073098 | 0.900× | 0.753× |
| 514×770×258 | 0.021466 | 0.026387 | 0.011622 | 0.814× | 0.440× |
| 1×1×1 | 0.006956 | 0.009314 | 0.002769 | 0.747× | 0.297× |

更大的 tile 与第三次 launch 对小尺寸/短K存在代价，表中回退未剔除。当前没有全部14
形状领先的结论；独立重复实验及验证结果如下。

完整复现（冻结旧源码由构建脚本从git读取，不增加生产候选）：

```bash
MEGAFMM_BUILD_DIR=build-fp32-packed \
MEGAFMM_RESULT_DIR=results/fp32-packed \
MEGAFMM_PACKED_COMPARE=1 MEGAFMM_DTYPE=fp32 \
MEGAFMM_ALGORITHM=sw-tuned-strict MEGAFMM_ROTATE_INPUTS=2 \
MEGAFMM_SW_PRECHECK=1 bash scripts/bench.sh final
```

驱动为 [packed_fp32_compare.cu](../bench/packed_fp32_compare.cu)，不指定
`MEGAFMM_SHAPES` 时读取JSON全部14项。显式候选测量避免反复搜索无关零workspace
配置，完整pack/叶子/重构仍逐次计费；setup/tune不计入稳态耗时，另列setup_ms。
逐组TIME、fresh CHECK、源码/二进制身份、NSYS/NCU原始报告保存在
[本轮结果目录](../results/fp32_pipeline/)，完整单轮表在
[current.csv](../results/fp32_pipeline/final_retry/current.csv)，失败与排除方向在
[ITERATIONS.md](../ITERATIONS.md#fp32-pipeline-iteration49--2026-10-22)。

独立新进程重复（同样20warmup、5×100、3fresh、2轮换；双方12次数值检查全通过，
逐位一致；未重编译、未按结果重试）：

| M×N×K | 新 packed（ms） | 旧/新 | Lt/新 |
|---|---:|---:|---:|
| 8192×8192×8192 | 20.814075 | 1.142× | 1.034× |
| 16384×16384×16384 | 163.588074 | 1.152× | 1.037× |

公开 `SWTuned` 严格模式另行调优并确认1024³/8192³均选择同一候选9500000；
8192³完整调用 20.863855ms，Lt 21.598296ms，1.035×；
1024³为 0.073752ms / Lt 0.057494ms，仍有回退。

验证：6/6 CTest通过（32.73s），强制packed的memcheck/racecheck/synccheck均零错误，
racecheck零警告。新增prepared maps/workspace不匹配拒绝、双stream独立workspace
并发、3组改变输入、非零alpha/beta、非对齐输入、奇数stride、掩码尾块、C=D及guard检查。
SASS逐条hash确认profiling和最终benchmark的叶子指令完全相同，叶子无LDL/STL。
最大尺寸16384³补测NCU：FMA73.13%、issue80.03%、占用率24.96%、HBM读取61.46GB
（旧版274.98GB）；NSYS的combine阶段占0.6%。

复测在 [repeat/current.csv](../results/fp32_pipeline/repeat/current.csv)，公开调优记录在
[public_plan.csv](../results/fp32_pipeline/validation/public_plan.csv)，工具输出在
[validation](../results/fp32_pipeline/validation/)，身份审计在
[validation.json](../results/fp32_pipeline/validation.json)。这些实测支持8192³/16384³
在本机普通调用中的收益，不扩展为所有形状或应用的性能认证。
