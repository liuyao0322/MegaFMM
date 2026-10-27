# ITERATIONS — 结论与证据索引

2026-10-27 压缩。当前约束见 [HINTS](HINTS.md)，算法/验收见 [PLAN](PLAN.md)。本文件按研究阶段归并结论；不同阶段的编号会重用，须连同 dtype、目录和版本解读。压缩未重跑实验，历史记录不授权恢复任务。当前仅验收 launch，旧 Graph 数据仅供历史追溯。

## 原文与证据恢复

压缩前四份工作树原文（包括未提交文档更新）已写入**本地 Git blob**。用 `git show <blob>` 阅读，或 `git show <blob> | rg '关键词'` 搜索；不必覆盖当前文件。这些对象未新增提交，复制/清理仓库前应另行保存所需原文。

| 文件 | 2026-10-27 压缩前 blob |
|---|---|
| HINTS.md | `e718ed43c94a68073bf68bff9e9715c52680a15a` |
| ITERATIONS.md | `c2d963c467ae26cfb80a078c97f46f0742da1758` |
| PLAN.md | `3fa7ae9c81204bc7fc5f1e380a872febeadc81e8` |
| README.md | `4b9a928236a37127275045b91bea0f8c283d5481` |

更早完整迭代日志（8510 行/1,122,851B）：`c593560eaeb7eb6c965df81f7d422d66f2ccbd60`；旧 HINTS：`a770acea50cee1b9d242b3ffa040ea486dde9715`；旧 PLAN：`bcda0d87c32c0e8088f17172a6aa0d94179c7320`；旧 README：`18efa78669a07905fefb06a8526c88caa38030ef`。原始 CSV、源码快照、审计、失败日志和 hash 仍在 `results/`、`trajectory/`、`research/`。

## 最近有效基线与边界

**默认 21 项池：iteration209 (`a84e5bdf`) + iteration242 必需正确性修复 (`14d5781e`)；iteration243 复验。** 最近完成的 256 未增加候选，恢复源/头/type tests 到242，保留 benchmark 最大配置 workspace 修复。最新 HINTS 记录258准备开始，尚无完成或接受结论；这不是对当前进程/工作树的实时认证。

- fixed30：5/30 同候选独立认证领先 Lt，8/30 单轮公开领先，目标未完成。五形状：8191×513×1025(id16)、513×8191×1025(id20)、512×4096×1024(ids13/18)、512×4096×2048(ids13/14/18)、512×4096×512(id18)。
- 209 接受时 public78：74 valid、4 原 gate 拒绝，32/74 单轮 Lt wins；不是全范围独立证书。243 paired public78 同样各74有效/4拒绝，old/new 几何均值0.997484（修复约0.252%成本），无 >3% 回退，最大1.136%。
- 242 旧235在3×10000 fresh中3次失败，修复版0/30000；241此前旧235/239出现4/18000失败，240 sanitizer曾全部通过。五处按访问线程执行的 proxy fence 必须保留。
- 243 的180 paired packed processes共3780 rows/11340 fresh/22500 timing；修复242的1890 rows/5670 fresh全strict。旧235一次1024²×8192全21项精度拒绝：SW eF固定3.034e-6，Lt选择不同使 gate 2.422e-6或3.436e-6；保留失败，不归因于竞态修复、不放宽gate。
- 256 ordinary 750 strict rows通过不能覆盖 instrumented K8192 的原FP64拒绝；252因 caller workspace不足，**全部性能结果作废**。

证据：[243报告](results/fp32_better243/report.md)、[256诊断](results/fp32_better256/diagnosis_report.md)、`results/fp32_better242/`、`results/fp32_better252/`、`results/fp32_better256/decision.json`、`results/fp32_better258/`。

## FP32 近期筛查与正确性修复：210–258

共同协议：原fixed30、20 warmups、5×100完整调用、3 fresh FP64、2输入轮换、8GiB、3 frozen SW bitwise controls；预检/独立复测/最终池/公开路由不可互相替代。表中 rows 为各轮记录范围，不是新增认证数。完整命令、配置与审计在 `results/fp32_betterN/`；250顺序诊断另见 `results/fp32_order250/`。

| 轮次 | 方向与结论 |
|---|---|
| 210 | A4/PackedB4，870筛查rows；4096×512×256独立增益未每轮>2%，不推广。 |
| 211–214 | 更深PackedB4、双producer、列尾、materialized cache/raster，无接受增益；保留209。 |
| 215–218 | 融合fragment与BK8资源预筛；spill类型先剔除，实际4CTA也未带来独立收益。 |
| 219–224 | 64acc/缩短unroll/metadata仍spill；48acc 4×12/12×4达3CTA但无合格收益，BK16溢出先拒绝。 |
| 225–228 | ConsumerB四平面与balanced48；224-thread min3 spill，256-thread无local/2CTA仍败；未推广。 |
| 229–232 | producer subset ownership、compact piece pipeline无收益；min2控制有筛查信号，待最终缩减池复验。 |
| 233 | 最终22项池三轮fixed30复验失败，不能将48项诊断池信号当稳定收益；真实驻留未改善。 |
| 234–235 | 有界方向耗尽后恢复209，不声称物理极限。234误调用233的22项二进制而失败；235修复harness，630 strict rows/1890fresh/3750timing及30公开形状完成，公开7/30领先，无新认证。 |
| 236–237 | bounded barrier wait先compile-only再全30筛查，无合格收益；保留逐lane wait/arrival。 |
| 238–239 | whole-task列尾；239测到第21形状时旧config11精度失败，中止（20完成/21尝试、567rows/566strict），不推广。 |
| 240–241 | 对旧235/239复现persistent config11失败，证明早于TaskTail；sanitizer通过不排除竞态。241 18000 fresh出现4次失败。 |
| 242 | 恢复209，加入五处generic→async proxy fence，在原join/release前排序；数学、arrival、guard、default21不变。full-clean7CTest/53leaf无local/实际驻留及新旧各30000fresh对比，修复版零失败。 |
| 243 | original30×3 paired + public78复验正确性修复；7 routes/21 sanitizers/NSYS。范围和精度例外见上，不是新优化推广。 |
| 244–245 | task-tail/raster先资源预检，后全30筛查990strict/2970fresh/5550timing；0合格，恢复242。 |
| 246 | 单warp materialized96/128acc，840strict/2520fresh/4800timing；0合格。128acc冲突约33.35% odd/44.45% wide，96acc减少load仍慢；7组profiles后恢复242。 |
| 247 | 4计算warp配对原七产品+producer，930strict/2790fresh/5250timing，12sanitizers、7profiles；8新类型全拒绝，FMA降低/long-scoreboard增大，恢复242。 |
| 248 | A/B float4所有权，1080strict/3240fresh/6000timing；A冲突33.35→0.05%，B44.45→0.056%，叶375.376→345.216µs，但完整调用不胜旧池。12sanitizers/10profiles，恢复242。 |
| 249 | fused每产品128acc+ownership修复，870strict/2610fresh/4950timing；1CTA对旧2CTA、FMA下降，无合格收益；8profiles后恢复242。 |
| 250 | 物理B平面映射0,2,3,4,5,6,1，producer vector stores 6→4；810strict/2430fresh/4650timing，无全组合格。odd中位改善2.213%但两组失败，不能舍弃。 |
| 250顺序诊断 | 原/prewarm/prewarm/原12进程324strict + trace27strict；额外20 warmups未消除逆序惩罚。NVML/NSYS只诊断，锁频请求被拒，未改时钟。REPORT_v3纠正早期case-map错误，原主验收不变；恢复242。 |
| 251 | 每产品2warp/64acc、480线程、448readers、B0TMA物理B；840strict/2520fresh/4800timing。6独立进程168strict仍无original/Lt证书；15sanitizers/9profiles，恢复242。B0大tile+RowXor4与193重合，不盲目重复。 |
| 252 | 七产品交错lane/shuffle，25项57leaf，A8/B8。原跑23进程、22终态成功/1失败、7未测；新8平面下K4096少1MiB、K8192少9MiB workspace，所有252性能作废，不能归罪冻结参考。 |
| 253–255（CPU） | 紧凑输出、quadrant-specific gather、RegisterGather；CPU/UBSan/SASS/资源模型证明不等于GPU验收。255修正GatherConfig内嵌base误匹配的analyzer，保留初次失败。 |
| 256 | comparator改为所有执行ID与dense/frozen需求最大workspace，保持252 kernel binary、原8GiB guard/counts。实际benchmark memcheck：内存/bitwise通过，K8192全部25项strict精度拒绝；instrumented时间排除。fresh全30得750strict/2250fresh/4350timing、0候选，9实际profiles身份审计通过后拒绝，恢复242并保留分配修复。 |
| 257–258（交接） | physical-slot coalesced packing CPU提案 → 258 PREPARE；默认21不变，25types/57leaves。无完成/推广结论，检查原study状态后继续。 |

256实际wide诊断：old/new pack 5.376/278.592µs，leaf334.671/383.360µs；八槽packing流量与工作集增长主导，shuffle还有13–23% no-instruction stalls。零槽store已向量化，不能归因“标量零store”。下一coalesced physical-slot/A8-B4只是待验假设。

<a id="fp32-pipeline-iteration49--2026-10-22"></a>
<a id="iteration-54--bk32-transfer-with-repeated-bk16-compute"></a>

## FP32 通用模板与数据流演进：49–209

此阶段全部保留原七乘法、FP32 RN/升序K、完整调用、绑定/预算/并发约束。接受的是具体配置池与范围；恢复sanity、部分形状筛查不构成全目标证书。

| 轮次 | 方向、决定与证据入口 |
|---|---|
| 49–53 | 精简后通用packed pipeline探索，保留恢复、precheck失败及full14/独立验收；完整路径见本次原文blob。 |
| 54–59 | stage粒度/小寄存器tile/M8/完整BK32/8×16复用；最终恢复58 `33ec1bc8`（恢复`c99687d4`），唯一packed9500000：128×128/BK32/ComputeK32，127regs、65664B、2CTA。14形状126 FP64+frozen bitwise；13/14胜51、6/14胜Lt。8192两进程19.999/20.045ms，16384为156.277/156.285ms；6CTest/三sanitizers。`results/fp32_stages/validation.json`。 |
| 60 | 同一实现参数化为3通用模板；42 shape/config组合及各路径sanitizer通过，公开选择实测，无全Lt胜。`results/fp32_templates/validation.json`。 |
| 61–68 | materialized/cooperative/shared融合、persistent tasks；修复mbarrier重复初始化前invalidate。68保留5模板/公开max9，70组合、独立中型复测、安全/路由验证；1024改善，2048未稳胜。`results/fp32_persistent/validation/audit.json`。 |
| 69–79 | DSM/寄存器复用/direct B；79精确descriptor绑定guard与fullpack fallback，保留6模板/公开max10，84组合+独立/公开验收，2048稳定胜未证。`results/fp32_cluster/validation/audit.json`。 |
| 80–95 | 4+3部分积、last-CTA重构、interleave、release counters、BK16/warp slot释放等失败；81变换中止、87编译失败、91命令失败均保留，最终恢复79。 |
| 96–100 | leader wait、普通PDL及两个依赖边；99收敛6模板，100 shared carveout未改善后恢复99。 |
| 101–108 | late pack与跨ComputeK预取失败；105采用独立输出2×2 zigzag，full14单轮9/14胜Lt、2048仅10/15组领先。106晚触发/107转置zigzag拒绝，108恢复105（`b84f7db5`），sanity不替代独立验收。`results/fp32_leader/validation105/audit.json`。 |
| 109–115 | DSM/96acc失败；112 retained-product L2 hint接受（`9ba57230`），full14×6=84rows/252fresh/700timing，6CTest/三sanitizers/公开路由。单轮9/14，2048四进程19/20组领先仍不稳。113plain/normal store、114stream reconstruction无中位收益，115恢复112。hint同时改变代码生成，不能称纯缓存因果。`results/fp32_leader/validation112/audit.json`。 |
| 116–128 | float2/标量小CTA、重排FMA、warp A转置、Direct-A消费等无接受收益；窄重构32regs不证明共驻留。117最终6池未推广，120/126/128恢复112；128 sanity2048 327.105µs/Lt341.757，非新认证。 |
| 129–141 | 短K大融合、反向combine、输出band、矩形几何/PDL/stages/warp数/register limit等诊断。138去空CTA未证明收益；140换maxnreg连旧control SASS也改变；139/141无新矩形收益。恢复112。`results/fp32_rectN/`及原文路径。 |
| 142–145 | 独立产品TMA queues与4+3partial重构；142 subgroup的bar.sync aligned语义不合格，143改显式nonaligned并synccheck。新路径无接受收益，移除。 |
| 146–156 | fragment stripes/stream operands/metadata liveness/PDL partial/issuer位置/producer与更深slots/双warp B复用；约1%局部信号、spill及慢路径均保留，无全目标接受。 |
| 157–159 | opt-in参数扫描与扩展shape测量；固定池诊断低速路由。极窄pack网格约49.2%空CTA是结构证据，未单独量化收益。 |
| 160–165 | operand source/pack/tail/producer策略参数化；紧凑/vector/quadrant pack、shared RowXor、guarded final K，形成后续有界池候选，保持默认与诊断区分。165有独立验证。 |
| 166–175 | raw A producer、A multicast、八warp平衡7产品、分散A变换均未建立收益；167/170/173/175逐次恢复验证源。172消bank conflict仍慢，174 pack17.824→15.200µs却leaf169.184→220.015µs。 |
| 176–177 | 176 opt-in11 public78：各74有效/4拒绝，23形状>3%改善、无>3%回退，geomean1.0241；177同11类型/顺序升默认，7CTest/29leaf身份/两99row重复/78公开通过。单轮Lt wins23→22的波动不归因未变代码。`results/fp32_default177/report.md`。 |
| 178–180 | early release/peel末两K、AliasA无推广，恢复177。初期local spill修复与失败工件保留。 |
| 181–183 | original B TMA有筛查收益；182 bounded13 full78 geomean1.01793、Lt23→25，但wide独立不稳定，default11不变。183物理B前缀比较后恢复，后续另测小形状。 |
| 184–188 | 256/768 tile/窄行/DirectB调度。187两目标独立通过但513×8191×1025全域回退门槛失败，拒绝；188加入通用task-wave eligibility，接受default13：195rows/585fresh、public78各74/4、无>3%回退，6sanitizers。公开256为8.150µs/Lt比1.004、768为32.179µs/1.021；全公开仍单轮。 |
| 189–194 | fixed30 dataflow/raw-A transpose/双producer/guardedK/reuse筛查，保留失败并独立复测；194只保留4已验证通用实例+原13，接受default17，无exact-shape route；原all30 Lt目标未完成。 |
| 195–200 | resident pack、fragment/aspect/BK16/stages/register microtiles、warp stage retirement。196极端microtile数值失败未解决；200聚合reader释放产生96 races，禁止原样复用。 |
| 201–202 | SeparateKernelFused pack→leaf PDL；202保留旧17+2=19，三轮fixed30共1710strict、public78各74/4，无>3%回退，6稳定相对改善；公开Lt27→31，fixed30单轮7/30、独立4。 |
| 203–206 | raw A TMA/word-ldmatrix、128acc、guarded/coalesced vector pack均未过>2%门槛；减少指令/流量不等于完整调用收益，恢复202。 |
| 207 | PackedB4+guardedK，1290strict；1025×1537×769三进程改善2.13/1.90/2.26%，未每轮>2%，Lt仅0.946–0.953，拒绝。 |
| 208–209 | PackedB4 RowXor与row-tail在最终池验收：209旧19+2=21，三轮fixed30共1890strict/5670fresh/11250timing，public78各74/4，无>3%回退。公开Lt31→32，fixed30单轮8/30、独立5；仅进展，非全胜。 |

49–175更细的题目、计时/配置表、SASS/profile路径仍可从本次原文blob搜索；不要按表中合并范围臆造单轮接受结论。后续重点研究位置：`results/fp32_leader/`、`results/fp32_cluster/`、`results/fp32_persistent/`、`results/fp32_templates/`、`results/fp32_stages/`，以及原文中各轮专用目录。

## FP32 初始五形状研究：1–48（2026-10-20）

原生SIMT baseline固定/tuned×五形状×两历史模式20/20 gate通过，普通512/1024/irregular约36.02/230.14/284.95µs。iteration1真实双缓冲TMA、thread0八象限copy、transaction barrier/phase、fallback和BK16/32指令验证；3CTest、66FP64/48bitwise、安全/并发通过，零global workspace。1024 tuned215.934µs仍慢于Lt56.811，见 `results/fp32_tma/`、`trajectory/20261020_123729_fp32-tma_iter-1/`。

随后packed caller-workspace路径也逐调用重新pack/七叶/重构；固定/零预算仍零workspace。原始输出在 `results/fp32_speed/iter-N/`。

冻结46预声明46/47/48三独立进程：历史五形状×launch/Graph共60/60数值、30/30tuned性能通过；**普通launch子集15/15**，仅该版本/五形状/H100范围。20calls×3groups外部协议、2输入轮换、3fresh；tuner排名5×20/前四确认5×100另计。较早39–41的29/30失败仍保留（40 tiny Graph0.996986×）；不能挑有利重试。

| M×N×K | 轮48 SW/Lt µs | 三进程launch加速比范围 |
|---|---:|---:|
| 1×1×1 | 2.246/2.899 | 1.2906–1.5557× |
| 33×65×129 | 5.771/6.187 | 1.0721–1.7886× |
| 512³ | 13.262/15.315 | 1.1440–1.1548× |
| 1024³ | 55.558/56.672 | 1.0195–1.0237× |
| 1025×1537×769 | 71.362/73.213 | 1.0259–1.0283× |

45 float4重构是主要保留改进；46 tiny只读cache设备收益约0.1–0.2%，不能解释全部tiny速度。最终10CTest与tiny六候选×三sanitizer通过，向量写回/guard/concurrency验证保留；只认证各自范围。证据：`research/fp32_tiny_cache_20261020/repeat_protocol.json`、`repeat_audit.json`、`final_raw_audit.py`，失败协议 `research/fp32_static_warp_20261020/repeat_audit.json`；`results/fp32_speed/validation45.json`、`validation46.json`、`sanitizer33/`、`sanitizer45/`、`sanitizer46/`。

| FP32 iter | Direction | Recorded result / decision |
|---|---|---|
| 1 | TMA asynchronous loading | True TMA, fallback and forced-path proof;1024 tuned215.934µs, still slowerLt. |
| 2 | register tiling | Ten register-tile candidates;1024~110.542µs vs215.934; four CTests/20 numerical rows pass. |
| 3 | product-major register scheduling | Product-major scheduling; irregular~174.712µs, aligned retains earlier winners. |
| 4 | vector shared operands and small register kernels | Vector float2/4 shared operands;1024~99.533µs, irregular151.902; tiny register routes added; NCU lock failure retained. |
| 5 | asynchronous fallback and small large-K stages | cp.async arbitrary-pitch fallback/small stages; no material new winner; harness BK128 reference repaired. |
| 6 | warp specialization by product | Seven compute warps/thread224 TMA;512=24.525µs,1024=86.549µs, irregular unchanged; small33×65×129~5.922. |
| 7 | larger per-product register tiles and K unroll | Larger per-product tiles/K unroll;512=21.808µs,1024=73.446µs, irregular132.803. |
| 8 | interior address and mask specialization | Interior mask/address specialization;512=19.810µs,1024=69.854µs, irregular120.928; all original gates pass. |
| 9 | shared lifetime overlay and CTA register bounds | Shared lifetime overlay/min-block bounds;1024=70.278µs, irregular118.642; no material aligned gain. |
| 10 | padded A layout versus operand shuffles | Padded shared A beats shuffle candidates;512=18.973µs,1024=70.267µs, irregular115.302; still belowLt. |
| 11 | row-phase TMA for odd pitches | Row-phase TMA handles odd pitches with aligned views/extra columns; preserve general fallbacks and guards. |
| 12 | caller-workspace seven-batch TMA | Six budgeted native seven-batch TMA candidates; fresh pack, seven planes and combine charged; no cached transforms. |
| 13 | CUTLASS SIMT leaf pipeline comparison | Six CUTLASS FP32 SIMT leaves, unchanged pack/combine; comparison retained, no Tensor Core/precision substitution. |
| 14 | seven-plane 3D TMA with fused reconstruction | Seven-plane fused leaf/reconstruction, two3D TMA boxes/stage/thread224; ten additive schedules. |
| 15 | lane layout, BK64 and scalar barrier phases | Warp-layout/BK64/barrier phases;33 batch schedules retained, physical K work report corrected. |
| 16 | explicit operand pipeline and prepared maps | Register operand pipeline and immutable prepared maps;43 batch schedules; different workspace re-encodes maps. |
| 17 | vector packing and active K tail | Vector float4 pack and active-K tails, scalar fallback;55 batch schedules; all prior routes retained. |
| 18 | fold thin M tail and increase CTA output reuse | Thin-M folding/additional output reuse;74 batch schedules; unchanged ascending K/FP32 order. |
| 19 | instruction footprint and seven compute warps | Instruction-footprint/K-unroll and224-thread seven-warp CTAs;90 batch schedules. |
| 20 | balance irregular tiles across SMs | Irregular104×32/balanced extras/double-tail folds;104 batch schedules. |
| 21 | transpose four raw quadrants before transforms | Four-quadrant transpose scalar packing:shared29,568→16,896B, seven transforms unchanged. |
| 22 | unchanged-binary independent repeat | Unchanged21 independent full comparison; all20 gates pass, scalar pack/promoted104×32 separately validated. |
| 23 | distribute vector packing over eight warps | Eight-warp vector pack/quadrant transpose; incompatible bindings fall back; aligned writes/masks unchanged. |
| 24 | split each product over multiple compute warps | Two/four warps per product,480/928-thread CTAs;113 batch schedules; preserve old pool. |
| 25 | unchanged-solution independent full repeat | Unchanged24 independent five-shape full comparison, fresh independent tuning and original gates. |
| 26 | remove packing tile integer division | 2D packing grid removes division only when exact CTA count matches; otherwise original masks/grid. |
| 27 | unchanged-solution independent repeat of2D packing | Unchanged26 independent full repeat, all20 precision rows pass; retain raw timing variability. |
| 28 | cooperative packing and TMA leaves in one launch | One cooperative pack+grid.sync+fused leaves call within occupancy capacity, fallback retained. |
| 29 | accurate cooperative metadata and repeat | Correct descriptive joint-kernel resources/linear-pack metadata, independent repeat; GPU math unchanged. |
| 30 | independent TMA producer with per-slot reader release | Independent producer/thread224, per-slot seven-warp reader retirement;119 schedules, liveness/phase checks retained. |
| 31 | minimal seven-product degenerate schedules | Tiny zero-workspace1/8-thread schedules retain all seven products/14FFMA and signed-zero canonicalization. |
| 32 | eight-warp balanced leaf rows | Balanced eight-warp leaf ownership; four additive TMA schedules; unique14,336 output ownership verified. |
| 33 | cooperative packing helper CTAs | Cooperative helper packing CTAs retire before leaf barriers; two additive schedules. |
| 34 | frozen production independent repeat and sanitizer acceptance | Frozen33 independent complete benchmark; final selected/new paths30/30 sanitizer runs zero errors. |
| 35 | retain tiny products with one shared sink | Tiny volatile product digest reduces shared sink stores while retaining14FFMA/all products. |
| 36 | bounded K unrolling and compact unit epilogue | Bounded K unroll and unit epilogue; preserve tiny arithmetic; reassess host-fed tiny timing variance. |
| 37 | stable tiny Graph tuner sampling | Batched tiny Graph tuner sampling stabilizes ranking; diagnostic corrected to two rotating bindings; historical Graph implementation only. |
| 38 | elected per-SM cooperative compute CTAs | Elect one cooperative compute CTA per SM/work queue;128/129 new,130 batch schedules retained; no best gain. |
| 39 | compile-time warp product specialization | Compile-time product specialization; declared39/40/41 repeats before measurements. |
| 40 | first predeclared independent repeat of39 | First unchanged39 repeat; tiny Graph .996986×Lt fails that predeclared group despite launch wins. |
| 41 | second predeclared independent repeat of39 | Second unchanged39 repeat; group29/30 wins,60/60numerical, acceptance=False; failure retained. |
| 42 | warp-cooperative A register exchange | Warp A-register exchange; 136 total schedules retained, no new winner; cannot erase39–41 failure. |
| 43 | narrow balanced thread tiles and more resident CTAs | Narrow balanced tiles/higher CTA capacity,142 schedules; no new winner; residency alone not speed proof. |
| 44 | single raw tile with register-held tail overlap | Single raw shared tile/register tail overlap K48/64,150 schedules; prefix helps its own baseline but loses retained104. |
| 45 | four-element reconstruction/output | Float4 reconstruction/output,154 schedules; new150 improves1024;10CTest/12sanitizer combinations pass; no repeated acceptance yet. |
| 46 | tiny read-only A/B loads | Tiny __ldg candidates4/5;~.1–.2%device gain, none selected in canonical; all10CTest/18sanitizer combinations pass; predeclared46–48. |
| 47 | frozen46 implementation independent repeat | Frozen46 fresh process, identities match;20/20numerical,10/10tuned wins; no source changes. |
| 48 | frozen46 implementation independent repeat | Frozen46 third fresh process; group60/60numerical,30/30wins, accepted=True; preserve failed39–41 group. |

## 早期 BF16/FP16 与 Dense/MNK：iterations51–380

| Iterations | Measured change / conclusion | Evidence or selected reference |
|---|---|---|
| 51–71 | Unified shared-buffer TMA/TC/CUDA roles; typed CuTe tensor stores fixed swizzle-address mismatch (52 invalid,53 repair). Cooperative original-A22 addressing fixed64. BK128/producer scaling improved latency; complete milestone after71. | 完整原文中的逐轮 trajectory 索引 |
| 72–79 | Full-K leaves/reconstruction, streaming registers and deeper operand slots;76 selected before profiling. Earlier failed NCU probe superseded by successful CUDA12.8 NCU2025.1.1 capture, private TMPDIR. | `trajectory/20261014_ncu_iter76/REPORT.md` |
| 80–87 | XOR P layout removed bank conflicts; streaming reconstruction82; descriptor reuse83/85; fixed slots86 and staged high/low readiness87. Iter85→87 ordinary BF/FP20.586/20.701→19.286/19.301µs; still slower than Dense. These are historical compensated kernels. | `trajectory/20261014_validation-iter87-staged-ready/REPORT.md` |
| 88–96 | Paired input IO and combined A-TMA92 retained as96. Deferred retirement94 reduced explicit wait but increased operand-ready wait and worsened total time. Reduced waits alone are insufficient. | `trajectory/20261014_validation-iter96-supply/REPORT.md` |
| 97–108 | User retained exactly seven uncompensated products despite original gate failures.97 BF/FP eF~.0060152/.000754626; full-call~25% below compensated path, still slower Dense. Compact BK256/four slots100 retained;104 pointer-width bug fixed105, exclude104 timings. | `trajectory/20261015_validation-seven-prefetch/REPORT.md` |
| 109–162 | Asynchronous input publication140 plus4MiB workspace/independent role returns162 selected. BF11.337µs vsLt12.191, FP11.542 vs9.965; BF wins, FP loses, both old accuracy gates fail. Separate leaves/clusters/local transforms/dual accumulators/warp-MMA rejected. | `trajectory/20261015_optimization-goal-checkpoint/REPORT.md` |
| 163–208 | Third P slot, small tiles, independent producers, parallel leaves/atomics/TMA reduction, clusters and PTX recombination failed to beat validated162.208 restores162; no certified FP win.201 NCU LaunchFailed invalid. Inspect real stack/local stores even when resource summaries say LOCAL0. | `trajectory/20261015_optimization-goal-checkpoint-208/REPORT.md` |
| 209–248 | Transposed leaf operands/private PTX registers218, balanced packing224 retained247; launch BF/FP~10.978/11.090µs vsLt12.275/9.975. Split-K/last-CTA/atomic handoffs and most dual-WG work rejected. | `trajectory/20261015_optimization-goal-checkpoint-248/REPORT.md` |
| 249–270 | Matched shared carveouts made two-kernel PDL254 effective; descriptor prefetch266 retained. Scoped publication, multicast, K512, compact160-thread and alternative tile/worker routes rejected. Graph rows historical. | `trajectory/20261015_optimization-goal-checkpoint-270/REPORT.md` |
| 271–313 | K128/packed conversion282 selected, restored313. Additional counters, N40, K192, early loads, producer swaps, cluster4+3, local transforms failed. Original source later frozen as `solution/sw1024.cu`. | `79dad0b:solution/sw_pipeline.cu`; 冻结路径/hash 见 HINTS |
| 314–336 | Precision/layout/Falcon research retained original gates and full-call costs; no numerical solution.332 row-interleaving independent ordinary gain only~.15%BF/.11%FP;335 selective on-chip S3/T3 correct but~84%/75% slower;334 invalid scratch placement excluded.336 restores330/331. | `research/falcon_layout_fusion/REPORT.md` |
| 337–353 | Dense/MNK empirical tuner, small MMA, source-free raster/ping-pong and cluster candidates; independent full-suite351:13/28,353:16/28 stable custom wins with original gates. Remaining12 large-K combinations unresolved; these are Dense routes. | `trajectory/20261015_mnk-validation-353/acceptance/registry.json` |
| 354–362 | Register-budget/width limits, nine-warp variants, Stream-K, swizzle/raster, clusters and explicit Split-K did not add stable wins.355 compile bound corrected356;358 swizzle-one liveness failure removed359.362 cheap Dense-distance screen falsely rejected more accurate reductions. | 完整原文中的逐轮 trajectory 索引 |
| 363–370 | Original-input FP64 fallback recheck fixed false rejection without relaxing gate. More K/scheduling/epilogue trials; independent369:15/28;370:28/28 precision and17/28 single-sweep wins, not certified. User temporarily closed optimization2026-10-16:7CTests/guard sanitizers passed, candidate memcheck interrupted, racecheck not started, no three-sweep370 certificate. | `trajectory/20261016_mnk-validation-370/`; HINTS |
| 371–377 | Native variable-MNK seven-product SW, guarded padding and scheduling; improved coverage but numerical failures stayed explicit. Medium/compact/inline producer/delayed retirement generally failed to add full-target wins. | 完整原文与 trajectory |
| 378–380 |378/379 BF16→FP16 candidates violated user constraint and were withdrawn;380 removed mixed computation and restored native dtype. Their old precision/speed statuses are retained as rejected history. | 完整原文与 trajectory |

## 原生 BF16/FP16 后续研究：381–386 与 2026-10-17–18

原all28 SW全胜未完成；2026-10-17默认调优将旧gate设为报告项，finite接受不代表原精度通过。378/379 BF16→FP16路线撤销，补偿路线禁止恢复；Dense/MNK成绩与SW分开。

| 研究 | 结论与证据 |
|---|---|
| 381–384集成 | 七批与native narrow/short-K池；384 `abae2b5`，28rows/2328候选/84fresh/280groups，有限值接受但仅2旧精度gate通过、5单轮全组≥1.03。512 BF/FP6.561/6.431µs vsLt5.561/5.624；1024为10.345/10.327 vs12.003/9.585。`trajectory/20261017_191336_iter-384/`。 |
| 383/384独立ABBA | 512池增益仅约1.45%BF/2.42%FP，仍慢Lt；早期约8%研究收益不稳定。`archive/sw_specializations/docs/sw_small_native_results.md`、`sw_tuner.md`、`sw_cross_product_results.md`。 |
| 七批物化 | 8192较旧SW约1.29×但仅0.78–0.82×Lt；pack/七P/重构抵消leaf改善。`research/sw_seven_batch_20261017/`。 |
| transforms/producer融合 | on-chip/raw-TMA/cooperative一次pack/overlap/双WG未建立接受收益；早期dual-WG Slurm139/debugger SIGSEGV，根因未定、调试后计时无效。 |
| native512 BK256/pending/deep | PTX降寄存器仅约1–2%收益；pending2/3变慢，S5/S6未胜旧S4P1/Lt。deep212combo/1696calls/8rowABBA/24fresh/80groups、全部36ID确认通过，四次选旧S4P1；BF约6.22µs/0.894×Lt，FP约6.10µs/0.922×。`research/sw_small_deep_analysis_20261018/REPORT.md`。 |
| 重复与审计修正 | pending2/3曾与`sw_bk256_pending_20261017`重复；pending auditor ID上界修正保留初次失败、仅host重审无GPU重跑。禁止改名重复宣称新方向。 |
| static/raster/multicast | 固定512常量微小/噪声，raster/pack256更慢；C2/C4 multicast 480combo/3840calls+ABBA仍败，compact160/192无接受收益。`research/sw_small_fixed_shape_20261018/`、`sw_small_multicast*_20261018/`。 |
| descriptor frontier385/386 | low32减少指令，初次局部约1.39%BF/2.25%FP；旧A0异常制造的约23%聚合增益拒绝。四个新FP进程仅0.35–0.98%领先Lt，无广泛独立认证。 |
| cooperative大形状 | 两owner WG/U3/register budgets/pending/deep/BK128S2无Lt胜；NSYS反驳已测间隙的host提交延迟解释。`research/sw_cooperative_*`、`sw_wide_u3_*`。 |
| split-A128×64 | 576combo/4608calls、6row/2556候选/18fresh/60groups/432对照通过，六项全败；8192FP1998.052µs/Lt1741.224=0.87146。几何容量模型非因果证据。`research/sw_split_a_geometry_model_20261018/`。 |
| 宽tile与combined pools | N80/96初期138候选只确认66的声明作废，host修正仍无胜；combined28审计5980候选通过但仅8/28单轮胜。N160/192、normal orientation/K512虽局部改善，无全胜。 |
| normal BK layout | A-group寻址导致structured mismatch，full-A slice修复经CPU/240native与六行tuner审计；保留失败及resource key冲突修正，不据此称速度胜。 |
| 参数化冻结1024 | generated1024 pack/compute SASS等价；机械2048 BF/FP69.506/69.177µs，慢于旧49.479/49.292及Lt31.636/33.307。BN64/BK128/S4改善到53.867/54.285仍败；7tuple/public审计通过非多尺寸加速。BN128编译主动停止不是运行拒绝，Stage3可选但慢于S4。`archive/sw_specializations/docs/sw_parameterized.md`、`research/sw_parameterized_20261018/`。 |
| SubCuber | 固定MIT源`c7b6fd8c72db4fc83982b26ea6e80e47567d674f`仅审阅，FP16输出/参考/测量不同，本地BF16/FP32输出和性能未证；U3/operand ring直接alias生命周期失败。`research/subcuber_reference_20261018/REVIEW.md`。 |

原日志的330项逐轮标题/轨迹表、完整研究目录索引仍在本次与更早blob；搜索报告前核对所需dtype/阶段，目录名不证明完成或认证。

## 应用 benchmark（2026-10-26）

科学套件冻结内核242，完成smoke/core/stress共41workers/164backend records/780timing groups。core covariance1024/2048/4096合格加速1.060/1.031/1.102×；8192虽1.155×但对称性不合格。core sqrt/SP2合格，inverse sqrt不合格；stress无合格SW加速（1025质量通过但0.841×）。4096 setup约17.8s/Lt0.83s需复用，12288/16384未测。保留质量失败、全部进程和source-core/source身份；见 [RESULTS](results/scientific_20261026/RESULTS.md)。这些是固定算法/synthetic应用证据，不认证完整科学求解器。

## 追加记录约定

每轮只记录：编号/日期、假设与改动、精确范围/协议、数值及完整调用结果、安全/资源验证范围、接受/拒绝/未完成决定、冻结身份与原始审计路径。逐候选表、重复配置字符串、JSON原文、build tails、PID轮询放study artifacts。失败、作废数据和后续更正都保留；新结论不覆盖旧失败，当前约束维护到HINTS。
