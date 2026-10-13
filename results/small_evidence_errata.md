# 小矩阵历史验证证据勘误

2026-10-13，独立源码复核发现 iteration23 profile JSON 的一处 Graph 覆盖描述过宽。原始文件与 manifest 继续保留其已归档的字节和 SHA；本文件给出更准确的解释，不改变性能或数值测量。

适用证据为 [small_owner_profile_iter23.json](small_owner_profile_iter23.json) 的 `validation_coverage.Graph`，测量源 `ea4dc50c57b08b4dee98c23c024a01bb94f836db`。该项曾将 identity19 Graph 描述为 C0=D 并恢复 fresh inputs。实际 `cluster_graph_suite` 在 beta=-0 时捕获的调用是 `run(..., nullptr, D, ...)`；D 与测试的 C buffer 使用同一 allocation，但 C0 实参为空，因此不是活跃 C0=D。三个 replay 前重新上传的是 D 初值，A/B 内容保持不变。

这项测试证明 identity/null-C0 的 Graph 捕获与重播、实际所选 native/fallback 路径的输出、以及输出 padding；它不能单独证明重播时修改 A/B 内容，也不能称为活跃 beta 别名验证。一般系数、非零 beta 的 C0=D 仍由独立结构及回退测试覆盖。完整 memcheck 包含这一实际 Graph 套件；racecheck/synccheck 只运行 candidate smoke，没有运行该扩展套件。

iteration24 源 `bb15c03` 中新20/21复用相同 Graph helper，因此该限制同样适用。它们新增的随机 native bitwise 比较另行重新上传原始 A/B，不是 Graph 捕获内的测试。性能基准的3轮新输入数值检查与这项 Graph helper 分开记录。

后续验证将单独增加修改 A/B 内容后的 captured-Graph replay，并逐次重新计算原始输入的 FP64参考；其结果必须绑定后续源码与二进制，不能追溯扩大本快照的覆盖。
