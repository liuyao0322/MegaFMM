# 本轮实验记录

本轮仅新增研究脚本、模型和独立 CUDA 原型，未编辑生产 solution，因此不占用生产迭代371编号。起点为4035bed；所有原始记录保留，失败和排除项不覆盖。

| 顺序 | 问题/单项改变 | 记录 | 结果/决策 |
|---|---|---|---|
| 1 | 文献与既有公式、低秩、补偿结果核验 | LITERATURE.md | 将cast、累加、尺度失衡分开检验 |
| 2 | 36组阶段消融；枚举全部256个残差子集 | ablation.json | cast主导；均匀输入最优七项补偿仍失败 |
| 3 | inside/outside幂二缩放，36组 | scaling.json | 消除K方向失衡的额外放大；不能补回均匀输入cast尾数 |
| 4 | 原始输入直接点积，64×32，wait0 | direct_launch.log | 精度通过但10.3μs，FP16/BF16均慢于同期SW，拒绝为性能候选 |
| 5 | 初始32列Graph pilot | direct_graph.log | 曾与缩放分析重叠，排除性能样本 |
| 6 | 仅扩大tile到64×64 | direct_b64_w0_launch.log、direct_b64_w0_graph.log | 约7.5–7.6μs普通调用，保留研究 |
| 7 | 64×128、四槽、wait1 | direct_b128_w1_launch.log | 约6.6μs，进入复测 |
| 8 | 64×128保持wait0的消融 | direct_b128_w0_launch.log | 约6.88–6.95μs，重叠有额外收益但tile贡献更大 |
| 9 | 独立普通/Graph各三进程，所有正式计时串行 | acceptance_launch_*.log、acceptance_graph_*.log | 全部精度检查通过；15组均快于SW |
| 10 | 双流、Graph、fresh输入、guards、范围/失衡 | extended_validation.log | 72项/576次原型调用通过 |
| 11 | NCU、memcheck | ncu/、memcheck.log | 资源诊断完成，memcheck 0错误 |
| 12 | 不限定kernel的racecheck | racecheck.log、tools_driver.log | vendor instrumentation耗时，被终止；不声称通过 |
| 13 | 只限定新kernel的racecheck/synccheck | racecheck_scoped.log、synccheck_scoped.log | 完整72项通过，0 hazards/0 warnings/0 errors |
| 14 | 新INT8残差表示，理想数值模型42组 | int8_model.json | 36组低于floor，六组K方向失衡失败 |
| 15 | 固定INT8表示，仅加入inside，先复核12组 | int8_inside_model.json | 全部低于floor |
| 16 | inside+INT8推广到全部七类42组 | int8_inside_full_model.json | 全部低于floor；仍需真实GPU完整成本和精度验证 |

研究结论、完整计时合同及边界见REPORT；下一步方案见DESIGN。未把稠密原型重命名为FMM或接入Auto，未恢复停止的FP8内核路线。默认消融复现已关闭早期遗留表示对照。
