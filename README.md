# MegaFMM

H100 上的单层 Strassen–Winograd 矩阵乘法实验工程。输入为 BF16/FP16，预加减
保持 FP32，最终叶操作数才转回输入类型；Tensor Core 累加、乘积与输出保持
FP32。算法、API、数值和性能合同见 [PLAN.md](PLAN.md)。

当前源树保留调优 cuBLASLt 普通 GEMM、完整七乘法与 Lt batch7、Hopper
WGMMA/Sm80 batched leaves，以及 M1/M2/M3、warp/product/cluster megakernel。
较快历史函数和对照 variant 以独立入口并存，在同一冻结二进制中复测；无需恢复
历史整棵源码树来选择已有路线。强制 SW 用于执行时间与数值实验，Auto 当前使用
调优 Dense。未认证路径的速度信号不会启用自动派发。

## 当前实验结果：iteration23 快照

用户已要求持续优化，并优先处理小矩阵。本快照记录 iteration23 的完整复测，
不表示优化停止、达到物理延迟下限或整份 PLAN 已完成。直接寄存器传递 operand
已在 iteration24 作为 Mega20/21 加入并完成诊断验证，未更新本快照的最优包络。
后续连续向量加载与寄存器重排方向仍在推进。

证据在 [小矩阵报告](results/SMALL_REPORT.md)、[small manifest](results/small_manifest.json)
和 [ITERATIONS.md](ITERATIONS.md)；历史 profile 中的 Graph 覆盖描述以
[验证证据勘误](results/small_evidence_errata.md)为准。预声明九形状为 128/256/512/1024/2048/3072/4096
方阵，以及 `(1024,512,1024)`、`(512,1024,1024)`；元组顺序为 M,N,K。
两种 dtype、20 个请求配置各自 launch/Graph，共 **40 datasets、720 行**，全部
是已检查的实际 SW，全部未通过固定数值门限；nonfinite、unchecked、Dense fallback
行均为零。严格数值集合更广：Lt0–1、Hopper0–8、Mega0–19 共31个请求配置
（23个实际route名称），两种
dtype、五种模式、三轮，共 **930 次检查，558通过、372失败、nonfinite零**。
DAG、完整结构/API、memcheck、racecheck、synccheck 五项均 exit0；strict 为 exit2。
结构和工具检查通过不替代随机数值认证。

实际测量 source commit 来自两个 run metadata，而非后来的文档 HEAD：

```text
source: ea4dc50c57b08b4dee98c23c024a01bb94f836db
build/megafmm_bench SHA256:
a89d8cadf7f3dcbbecd41e6d7fac30de24ce55827bcbacf372126beaaf807d42
build/test_correctness SHA256:
51791901d93f465db30d41f419f1bc2f7334b02936c9e8c0365e21130679d392
```

Manifest 核对30个 source hashes、80份设备 telemetry、全部 CSV/log/退出码、
1,440条精确 TUNE 配置和2,160轮新输入 CHECK。设备与构建记录见
[small_environment.txt](results/small_environment.txt)，原始来源见
[small_final_runs.csv](results/small_final_runs.csv) 和
[small_verify_runs.csv](results/small_verify_runs.csv)。

## 性能快照与比较定义

报告按候选自身最低观测耗时选择每个 shape/dtype/mode 的经验包络，明确标为
`observed-minimum-latency / selected-on-measurement`。精确平局按 source路径、
CSV行号、actual route、variant确定。Dense/SW 比值始终使用获选行的 Dense，
双方同形状、精度、布局、alpha/beta、workspace和执行模式；Dense独立预调优。
该包络会逐尺寸切换 variant，属于本次测量后选择，不是固定 dispatcher、独立
验证收益或数值认证。

下面均为方阵，每种方法的耗时是五组每调用平均值分布的 median，单位µs。
**所有获选行均数值失败，Auto保持Dense。** 完整p10/p90/CV、矩形及全尺寸数据
保留在报告；[launch主图](results/small_plots/primary_performance_sw_launch.png) 与
[Graph主图](results/small_plots/primary_performance_sw_graph.png) 分别导出PNG/SVG。

| mode | dtype | N | 最低观测SW路线 | 同行Dense µs | SW µs | Dense/SW |
|---|---|---:|---|---:|---:|---:|
| launch | BF16 | 128 | Mega19 owner8×8 prefetch | 3.938 | 3.915 | 1.006× |
| launch | FP16 | 128 | Mega19 owner8×8 prefetch | 3.833 | 4.008 | 0.956× |
| launch | BF16 | 256 | Mega10 product warps fast | 5.028 | 6.203 | 0.811× |
| launch | FP16 | 256 | Mega10 product warps fast | 4.752 | 6.259 | 0.759× |
| launch | BF16 | 512 | Mega16 prefetch | 5.935 | 10.829 | 0.548× |
| launch | FP16 | 512 | Mega16 prefetch | 5.902 | 11.061 | 0.534× |
| launch | BF16 | 1024 | Lt1 batch7 | 12.364 | 19.643 | 0.629× |
| launch | FP16 | 1024 | Lt1 batch7 | 9.813 | 19.350 | 0.507× |
| Graph | BF16 | 128 | Mega19 owner8×8 prefetch | 3.483 | 3.629 | 0.960× |
| Graph | FP16 | 128 | Mega19 owner8×8 prefetch | 3.404 | 3.819 | 0.891× |
| Graph | BF16 | 256 | Mega16 prefetch | 4.470 | 5.844 | 0.765× |
| Graph | FP16 | 256 | Mega10 product warps fast | 4.027 | 5.875 | 0.685× |
| Graph | BF16 | 512 | Hopper7 Sm80 batchK64 | 5.432 | 9.210 | 0.590× |
| Graph | FP16 | 512 | Hopper7 Sm80 batchK64 | 5.391 | 9.373 | 0.575× |
| Graph | BF16 | 1024 | Lt1 batch7 | 11.223 | 17.364 | 0.646× |
| Graph | FP16 | 1024 | Lt1 batch7 | 9.282 | 17.092 | 0.543× |

BF16 128 launch接近Dense，1.006×约为0.6%的观测差距；该行Dense CV为1.06%、
SW CV为0.23%，没有形成≥1.05×及固定精度同时成立的验收证据。相邻预声明尺寸
必须属于同一源/actual route/variant/dtype/mode/cache/输入地址对数场景，且各尺寸
工作集与声明一致；同时核对数值、完整采样与离散度。`Dense p10 / SW p90` 是
保守的组间离散度筛查，不是置信界。GPU频率未锁定，before/after快照不证明组内
时钟稳定；所有p10/p90/CV是组均值分布，不是100次单调用的尾延迟。

报告另列 `历史原始Lt0时间 / 当前SW时间`，严格匹配M/N/K、dtype、mode、cache、
输入地址对数及工作集。例如BF16 128 launch为9.029×，而其同行Dense/SW为1.006×。
两者基线不同：前者是旧完整七调用SW的跨运行历史锚点，后者是同行调优普通GEMM。
旧Lt0与当前Lt0虽同名，packing与Lt调优已改变；时钟、host submission和算法选择
也可能不同。原始actual-route比较只匹配家族，不要求variant相同，缺失或多重
匹配显示N/A。此前0.90×来自 [iteration12 BF16 128历史行](trajectory/20261013_135104_iter-12/results.csv)
的4.1776µs 调优cuBLASLt普通GEMM／4.6416µs SW，即Dense/SW≈0.900，属于当轮诊断结果。

Iteration24 的直接 native-register 对照保留完整七个 M1 乘积及固定 U 顺序。
同一二进制的 quick20warm/20calls/5groups/1fresh 记录中，BF16/FP16 128 launch：
Mega20 为4.4960/4.6656µs，Mega21 为4.6368/4.8560µs；均未改善先前最低观测SW耗时。
这些快筛结果和上述 full 快照使用不同采样规模，不能作为独立收益验收。
新 kernel 无 shared/LDSM 或 local spill，但寄存器增为128/120；更多标量输入加载与
失去 lookahead 共同影响执行，不能归因为单一机制。完整结构测试 exit0，两个新
variant 分别通过 standalone CUDA12.8 memcheck/racecheck/synccheck 定向检查。
严格数值33请求配置共990检查，594通过、396失败、nonfinite零，Auto仍为Dense。
原始24轮证据与覆盖限制另存，不改写源23的完整报告、manifest或profile。
详见 [第24轮独立审计](results/small_direct_profile_iter24.json) 和
[Mega21原始诊断数据](results/small_direct21_launch_iter24.csv)。

## 构建、验证与复现

应用构建依赖为 CUDA12.4.99、CUTLASS3.5.1、C++17，目标`sm_90a`，设备H100。
下载在登录节点完成，构建与GPU工作复用已有allocation：

```bash
bash scripts/setup_dependencies.sh
MEGAFMM_JOB_ID=107488 bash scripts/build.sh
```

`scripts/on_gpu.sh` 使用现有Slurm allocation分配的一张GPU和四个CPU；改变
allocation时设置`MEGAFMM_JOB_ID`，不会自动申请或取消其他作业。

当前工具验证使用独立CUDA12.8 Compute Sanitizer **2025.1.0.0**，实际路径保存于
三个工具日志。它与CUDA12.4构建/runtime分别记录，没有将应用重编译为CUDA12.8。
早先CUDA12.4 racecheck的host SIGSEGV不是通过证据，历史日志仍保留。

memcheck运行完整扩展suite；racecheck和synccheck各运行374个candidate smoke
CASE，未覆盖独立shortK、Graph、rebind及M1逐位扩展集合。后者的结构测试通过
不表示已在race/sync工具下独立复验。

```bash
# 完成并冻结build后运行；核对每条命令退出状态。
MEGAFMM_SANITIZER=/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer \
  bash scripts/verify_small.sh
bash scripts/small_suite.sh
```

`verify_small.sh` 将`test_dag.py`与独立`test_stage_m2.py`串行写入同一dag日志，
再执行完整结构/API、memcheck、全部候选racecheck/synccheck及strict。前五项失败
会中止；只有固定strict数值失败允许exit2。`small_suite.sh`全40labels使用
**20次warmup、每组100次调用、5组随机交替Dense/候选、每行3轮新输入FP64参考**，
warm_cache、rotate_inputs=1。数值exit2保留所有行并继续，runtime失败中止。
选定labels可重测同协议子集，最终manifest仍要求全覆盖且源与每个角色的binary
一致。`--warmup 20 --trials 20 --groups 5 --correct-trials 1`仅用于诊断。

CPU生成命令显式选择当前suite，避免混入历史或未完成CSV：

```bash
mapfile -t small_files < <(python3 -B - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from write_small_manifest import declared_cases
for label in declared_cases(Path('scripts/small_suite.sh')):
    print(f'results/small_final_{label}.csv')
PY
)
python3 -B scripts/summarize_small.py --files "${small_files[@]}" \
  --baseline-files results/small_baseline_lt_launch.csv \
    results/small_baseline_lt_graph.csv results/small_baseline_hopper4_launch.csv \
    results/small_baseline_hopper4_graph.csv results/small_baseline_mega3_launch.csv \
  --suite bench/small_shapes.json --output results/SMALL_REPORT.md \
  --plots-dir results/small_plots
python3 -B scripts/write_small_manifest.py --output results/small_manifest.json
```

## API与实验路线

公共入口在 [include/megafmm/gemm.h](include/megafmm/gemm.h)。

```cpp
megafmm::Problem p;
p.m = p.n = p.k = 8192;
p.dtype = megafmm::DType::BF16;
megafmm::Policy policy;              // 当前Auto使用Dense
auto plan = megafmm::create_plan(p, policy);
void* workspace = nullptr;
if (plan->workspace_bytes()) cudaMalloc(&workspace, plan->workspace_bytes());
megafmm::tune(*plan, A, B, nullptr, D, workspace, stream);  // 计时之外
megafmm::run(*plan, A, B, nullptr, D, workspace, stream);   // 异步调用
```

调用方持有A/B/C0/D/workspace，负责它们在stream完成前有效。`run`不动态分配、
不扫描输入内容、不隐式host同步。`tune`可同步及使用临时缓冲，会写D；C0=D时
内部保留原C0快照，调用方仍须恢复正式计算所需C0。beta0允许C0=nullptr，exact
C0=D要求相同leading dimension；A/B与输出/workspace重叠、C0/D部分重叠拒绝。

`algorithm_name()`描述Plan首选家族；实际A/B绑定不满足对齐可走公共Dense。
部分完整tile候选会使用较通用packed或其他既有路线回退，CSV实际route单独记录。
`Policy::certified_range=true`由调用方声明PLAN的有限范围，不自动内容扫描。
未知范围、transpose或无法满足合法实验路线/workspace条件时用Dense；Dense支持
合法leading dimension、transpose、奇数尺寸和退化问题。`levels=0`普通GEMM，
`levels=1`单层，当前显式拒绝`levels=2`。

下列variant保留于当前源树；均使用FP32 S/T依赖与一次最终叶操作数cast。

| 家族 | variant | 当前实现/对照 |
|---|---:|---|
| Lt | 0 | 完整七个调优leaf；小尺寸A/B packing融合 |
| Lt | 1 | 全部七组operand统一槽位，单Lt batch7与FP32重组 |
| Hopper | 0–4 | Hopper WGMMA tile/schedule对照；4含2×1 B multicast |
| Hopper | 5/6 | Sm80 `mma.sync` batch7，64×64×32 / 32×64×32 |
| Hopper | 7/8 | 32×64×64 batch7，三stage multistage / 两stage pipelined家族对照 |
| Mega | 0/1 | M1七个完整P，64×32 / 64×64 tile |
| Mega | 2/3 | M3完整产品生命周期复用；3完整tile beta0双缓冲，否则用2 |
| Mega | 4/5/6 | 每CTA 1/2/4个独立warp，16×16×16 MMA owner |
| Mega | 7/8 | paired最终转换与packed路径；8完整tile专门分支 |
| Mega | 9/10 | 七product warps并行M1；10完整tile快分支 |
| Mega | 11/12 | 16×8 owner；12另测B的vec4加载 |
| Mega | 13 | product cluster对照，保留七完整P与U |
| Mega | 14/15 | float2 epilogue/int32坐标；15另含alpha1beta0特化 |
| Mega | 16/17 | 寄存器lookahead预读 / raw `cp.async`双bank对照 |
| Mega | 18 | M2：四个长期C与一个P scratch，每K16按14项贡献更新 |
| Mega | 19 | 8×8 owner、native16×8 MMA下半A置零及lookahead预读 |

完整小套件测Lt0/1、Hopper5–8、Mega4及Mega7–19；严格/结构集合覆盖更广。
M1逐位回归为4、7–17、19，明确排除18。M2改变FP32贡献累计顺序，是PLAN允许
的另一策略；使用独立stage/整数/alpha-beta/alias/Graph参考。CPU模型的标量RN
dot不声称与SM80内部K16归约逐位相同。两者共享原始FP64参考和冻结随机误差门限。

Mega19全部32 lanes及每产品四个native C寄存器仍参加MMA，只用上8×8的C槽
v0/v1输出。相对Mega16的16×8 owner，CTA数、逻辑B加载与HMMA总发射约翻倍，
全问题A读取近似不变；更细粒度并行伴随冗余工作。RawInputs由每lane 96降至64 bytes，
源码shared仍5376 bytes，每lane七产品accumulator仍28个FP32值。实际编译资源
和指令证据见 [iteration23 profile](results/small_owner_profile_iter23.json)。

## 数值合同、历史阶段与开放方向

每轮正确性输入先舍入到目标dtype，再将原始A/B直接转FP64计算普通AB；参考不
经过S/T。逐试验要求`eF_SW <= 3*eF_Dense + floor`，BF16 floor=1e-3、FP16
floor=1e-4，双方不能产生nonfinite。门限从初始基线冻结，没有为通过候选而放宽。
CSV最大eF/最大gate不能替代全部新输入试验的conjunction，结构/sanitizer通过
也不能替代随机认证。[mixed_precision_floor.py](tests/mixed_precision_floor.py)
记录最终operand量化本身可产生超过门限的误差；改变tile仍须独立通过同一门限，不能替代数值认证。

第5轮大矩阵记录是历史阶段，保留在 [results/REPORT.md](results/REPORT.md)、
[final_manifest.json](results/final_manifest.json) 和`results/final_*.csv`/`results/plots/`：
当时94行中88个实际SW失败、6个Dense fallback通过；strict300次有120失败、
nonfinite零。它们不能替代新增小矩阵路线的当前覆盖，当前小报告/manifest独立保存。

较广V3/V4、多源producer/输出归约、通用product-aware scheduler、tail peeling、
双层及同合同SubCuber性能复现仍开放。合法偶数partial tiles可走通用packed路线，
其他情况依据guard回退；这不表示专用tail peeling/transpose快路径已实施。
SubCuber比较边界见 [subcuber_compatibility.json](results/subcuber_compatibility.json)：
作者half输出、缺BF16和部分预热后复用B变换的preset须先适配，不引用其速度为本工程成绩。
AKO4ALL持续保留假设、资源、正确性、完整时间与提交记录；直接寄存器operand等
可行新方向仍需独立接入与同合同复验，当前快照没有将P0–P5全部关口标为通过。
