# HINTS — 当前约束与交接

使用说明见 [README](README.md)，数学/API/验收合同见 [PLAN](PLAN.md)，历史结论与原文恢复入口见 [ITERATIONS](ITERATIONS.md)。本文件记录任务约束和最后交接，不证明旧进程仍存活，也不授权恢复历史任务。

## 当前范围与最后交接

- **2026-10-27 文档压缩**保留原交接与已有未提交内容，未执行或重新认证下述实验。
- **launch only**：2026-10-22 起不再支持/验收 CUDA Graph；代码和旧配置仍可能有 Graph 分支，不能将旧脚本自动视作 launch-only。
- **减少特化路径**：保留通用 BF16/FP16 fallback + padded Hopper、FP32 零 workspace + 参数化 packed TMA。后续允许有界通用模板池及实测选择；禁止恢复固定尺寸分支或数百候选。旧特化在 `archive/sw_specializations/`，不参加主线构建。
- **FP32 最近已验证恢复点**：iteration209 的默认 21 项池，加 iteration242 必需的 proxy-ordering 修复，提交 `14d5781e`；iteration243 完成复验。固定 30 形状仅 **5/30 独立认证领先 Lt、8/30 单轮公开领先**，全胜目标未完成。当前工作树可能包含未验收实验，不等于恢复点。
- **最新原文交接是 iteration258 PREPARE starting**：使用 CPU physical-slot packing257 提案，25 types/57 leaves，默认旧 21 项不变。尚无该轮完成/性能认证结论。继续前检查 `results/fp32_better258/` 的终态、冻结身份和实际进程；旧 PID/allocation 不作为现态。原要求为 full-clean 7 CTest、精确 CMake CPU 对照、实际 2 CTA（计 1024B reserve）、12 maps/36 bindings、12 sanitizer 后才可跑原 fixed30/fullcounts；benchmark 终态紧接 `close_iteration.py` → `commit_iteration.py`，失败也记录。
- **iteration256 已拒绝**：750 strict rows/2250 fresh/4350 timing records，无合格新候选；9 组实际 NCU/NSYS 审计通过后源码/头/type tests 恢复 `14d5781e`。保留 benchmark 按所有执行 ID 的 `workspace_size(p,id)` 取最大值的修复。instrumented K8192 精度拒绝不能被普通运行通过覆盖。CPU RegisterGather255 仅有 CPU 证明；下一物理槽 packing/A8-B4 假设没有性能认证。

## 必须保留的合同

- 仅原始七个 Winograd 乘积；禁止残差、高低位分解、补偿/交叉乘积、BF16→FP16、FP8 残差或完整 dense AB 冒充 SW。P56/P61–P64、iter378/379 均为已否决历史。
- 输入保持 BF16/FP16/FP32，累加/输出 FP32；FP32 变换、升序 K 的 RN FMA 和重构不降精度。参考从同一原始已舍入输入计算普通 FP64 AB，保留 alpha/beta、stride、transpose 语义。
- FP32 保留原始 FP64 gate；BF16/FP16 默认 SW 调优的旧 gate 为报告项，strict 可要求通过；nonfinite、实现错误和内存失败始终拒绝。逐位匹配、原始 AB 精度、应用质量、性能分别报告。
- `tune()` 显式执行；`run()` 不分配、不 host 同步、不缓存输入变换。调用方提供限额内独立 workspace；绑定变化重新检查/编码，保留布局、别名、范围、并发和 caller-stream 合同。
- 固定 SW/零预算 FP32 保持零 global workspace；双槽 TMA、transaction barrier、明确 issuer、BK16/32 和同步 fallback 保留。实测必须证明实际走 TMA，不能以静态模板替代路由证据。
- **保留 iteration242 全部五处按访问线程执行的 generic→async proxy fence**，位于既有 join/release 之前；不改 reader/publisher arrivals、尾块和复用协议。禁止 fence-off 策略及原 iteration200 reader aggregation（96 races）。sanitizer 通过不能证明旧竞态不存在。
- **workspace 按实际配置计算**：多配置比较需覆盖全部 ID 以及 dense/frozen reference，保留 8GiB study guard。旧固定七平面公式不足以覆盖 A8/B8；iteration252 全部性能数据因此作废。

## 测量与工作纪律

- 保留 dirty work/results；使用隔离源码/build/result，冻结中的源码、二进制、脚本不修改。只暂存明确属于任务的文件。禁止擅自 push/merge、启动 agents、重启安静进程或并行 GPU timing/profiling。
- 已授权优化按 AKO4ALL：benchmark `iter-N` 终态 → 立即追加 ITERATIONS → 立即 scoped commit（包括失败）→ 审计/profile。已关闭实验用归档身份验证，恢复源码后不要重跑依赖 live source 的旧审计。
- fixed30 既定协议：20 warmups、5×100 完整调用、3 fresh FP64、2 rotating inputs、8GiB workspace、3 frozen SW bitwise controls；所有形状/组/失败保留。新候选须同一配置在 3 独立进程各 >2% 优于最佳 eligible old21、15/15 paired wins；Lt 认证另需三个中位数和 15 组都 >1。最终缩减池、public78、安全/路由、无 >3% 回退另验。
- 同进程独立调优 cuBLASLt，匹配输入/output dtype、shape、预算和模式；每次 pack/cast/transpose/reset/七叶/重构计入完整调用。setup/tuning、CPU wall time另报。旧五形状协议的外部 20×3 与 tuner 5×20/前四 5×100 不混用。
- 新方向先查 ITERATIONS/research 负结果；计时前核验实际 spill/stack/local、寄存器/shared、驻留（含 driver reserve）、布局/绑定、安全和新输入。编译资源、CPU 模型、占用率、静态指令减少、NCU replay 或重叠 stage 区间都不能单独证明收益/物理极限。
- CUDA/SLURM 工作使用 `cluster-llm-ops` 并核验 allocation。历史 H100107488/g08、CPU108873/g04 等不保证有效；NCU 可使用私有 TMPDIR，不删除他人锁。SSH/timeout/静默不等于终止。
- 长期优化每轮完成及至少每 5 分钟报告延迟/加速比、独立认证数、未解决项与当前实验；区分已认证、筛查候选和 dense fallback。

## 科学计算 benchmark（2026-10-26）

本任务建设/实测 FP32 科学应用，不授权修改或推广正在进行的内核实验。隔离快照固定验证内核 `14d5781e`，保护工作树；主应用为有限步协方差传播，保留 sqrt/inverse-sqrt/SP2。比较 PEDANTIC cuBLASLt、strict torch、独立标识 TF32-allowed torch；launch-only，预声明 FP64 质量门槛、完整应用 wall time、独立进程、全部原始组/失败、设备及源码/二进制身份。

已完成 smoke/core/stress：41 workers、164 backend records、780 timing groups，执行完成与数值合格分列。core covariance n=1024/2048/4096 合格加速 1.060/1.031/1.102×；8192 的 1.155× 因对称性失败不合格。core sqrt/SP2 合格，inverse sqrt 不合格。stress 无合格 SW 加速：1025 质量通过但仅 0.841×，其他失败。4096 setup 约 17.8s（Lt 0.83s），需计划复用；12288/16384 未测。synthetic/fixed-algorithm 结果不证明完整科学求解器收益。

设计：[scientific_benchmark.md](docs/scientific_benchmark.md)；配置 `bench/configs/scientific_*.json`；[结果](results/scientific_20261026/RESULTS.md)。core/stress 对照 `.scratch/scientific-20261026/source-core`，smoke 对照 `source`，归档/hash 已保存。应用套件允许 BF16/FP16 覆盖，但不重启其内核优化。

## 历史边界

- 原生 BF16/FP16 all28 未完成；Dense/MNK 成绩不能转为 SW 认证。旧 1024 实现数值失败仍保留。
- 冻结 `archive/sw_specializations/solution/sw/sw1024.cu`（来自 `79dad0b:solution/sw_pipeline.cu`），SHA256 `ebc37e9dd171dd5855cdaed91fbb030d250c052a4c459746dd33b82cc634264b`，禁止修改。历史参数化须先证明 generated1024 等价，再比较 2048；不能从归档推断恢复授权。
- SubCuber 仅源码审阅；其 FP16 输出和测量合同不同，无本地同合同 H100 证书，见 `research/subcuber_reference_20261018/REVIEW.md`。
- 旧进展逐次轮询、完整候选表和研究负结果在原始 Git blob/研究目录；以后只向本文件维护最新交接与长期约束，逐轮数据放 ITERATIONS 和 study artifacts。
