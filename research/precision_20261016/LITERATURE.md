# 精度问题的定向文献调研（2026-10-16）

研究问题：对于已经量化为 FP16/BF16 的输入，以原始 AB 的 FP64 结果为参考，能否使当前单层七乘法同时满足原始精度门槛和完整调用延迟？

这是面向工程决策的定向检索，不是系统性综述。查询覆盖 Strassen/Winograd stability、inside/outside scaling、mixed-precision Tensor Core correction、accurate rank-seven formulas、FalconGEMM precision。只采用论文原文和官方指令文档。检索命中、存在性核验与对本项目的适用性分别判断，不将论文中的速度或模型结论转移为本仓库结果。

## 已核验的主要来源与适用边界

| 来源 | 原文位置与相关内容 | 对本项目的意义与限制 |
|---|---|---|
| Ballard, Benson, Druinsky, Lipshitz, Schwartz, *Improving the numerical stability of fast matrix multiplication*, SIAM J. Matrix Anal. Appl. 37(4), 2016；[作者论文原文](https://arxiv.org/html/1507.00687) | §4–6；将算法增长因子与输入范数分开，并讨论 inside/outside 对角缩放。§6 明确考虑将缩放系数舍入到二的幂。 | 支持独立测试缩放和算法公式。缩放可修复输入尺度不均衡，但不自动增加变换结果的尾数位。本轮用原始输入参考单独消融，而非据文献推断本项目会达标。 |
| Dumas, Pernet, Sedoglavic, *Towards automated generation of fast and accurate algorithms for recursive matrix multiplication*, 2025；[v1 原文](https://arxiv.org/html/2506.19405v1) | §4–5，特别是公式98、101、102；在等价分解中降低增长因子，并结合直线程序优化。 | 支持公式替换这一候选方向。仓库前一轮已独立核验张量恒等式并测得 FP16 eF 约4.9–5.3e-4，不能因公式“更准确”就推断满足1e-4 floor。 |
| Ootomo, Yokota, *Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance*, [2023 v3 原文](https://arxiv.org/html/2203.03341v3) | §2.3、§3.2–3.4；保存转换残差、缩放以降低下溢，并区分 Tensor Core 内部累加误差。 | 支持把转换误差与累加误差分开，以及显式保留残差信息。该工作针对特定 SGEMM 仿真和硬件；它的精度/速度不直接证明本项目的补偿流水线可行。 |
| *An SMT Formalization of Mixed-Precision Matrix Multiplication: Modeling Three Generations of Tensor Cores*, [2025 v1 原文](https://arxiv.org/html/2502.15999v1) | 形式化建模涉及 Volta/Turing/Ampere；结论中说明 Hopper 仅有初步观察。 | 不将老架构的内部舍入规则直接当作 Hopper WGMMA 的精确模型。本轮误差归因采用反事实数值模型，最终精度仍以真实 Hopper 输出对原始 FP64 参考检查。 |
| Zhu et al., *FalconGEMM: Surpassing Hardware Peaks with Lower-Complexity Matrix Multiplication*, [2026 v4 原文](https://arxiv.org/html/2605.06057v4) | §III-B、§IV-E/F；融合 GEMM 与 Combine H，片上保留高精度中间乘积；也讨论标准 GEMM 与 LCMA 的选择。 | 我们已使用 FP32 乘积和重组，因此其避免低精度 H 写回的机制不处理本项目的 S/T 操作数量化。其 H20/A100、大尺寸或离线静态权重的性能也不能转移到当前动态1024³ H100调用。 |
| Xie et al., *Fast Matrix Multiplication in fp8: Certified Coefficient Optimization and Measured Error*, [2026 v1 原文](https://arxiv.org/html/2609.26077v1) | §2–3 的 Φ 模型；附录E.4披露 Winograd 实际 kernel 与所提供系数不一致的问题。 | 论文报告的5/3系数界适用于其明确噪声模型，不能当作本项目所有 FP16 七乘法方案的精度下界。它进一步说明必须独立检查数学分解与实际编码路径。本轮不恢复被停止的 FP8 内核路线。 |
| Huang, Yu, van de Geijn, *Implementing Strassen's Algorithm with CUTLASS on NVIDIA Volta GPUs*, [2018 论文](https://arxiv.org/abs/1808.07984) | 将预处理、计算和重组的内存成本纳入实现与性能模型。 | 支持比较完整调用，而不是只数七次还是八次乘法。其 FP32/Volta 交叉点不作为 H100 阈值。 |
| NVIDIA, [PTX ISA 官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/) | Hopper WGMMA、TMA、异步组、屏障与操作数格式。 | 新原型使用原始 FP16/BF16 输入与 FP32 累加。未来 INT8 研究必须使用真实整数 Tensor Core 指令并核验缩放/同步；数值模型不能代替指令级实现。 |

## 与已有仓库证据的衔接

- [前一轮公式/补偿研究](../precision_20261015/REPORT.md)已经排除了“直接换一种已测七乘法公式就能通过FP16门槛”的简单判断。
- [七乘法数值替代研究](../../trajectory/20261015_numerical-alternatives-seven-products/REPORT.md)已经说明 BF16 改用 FP16 操作数不足以解决双 dtype 目标；本轮不把它当作完成方案。
- 前一轮的低秩 oracle 筛选位于 `../precision_20261015/low_rank_screen.json`。它使用完整精确误差做 SVD，本来就偏向低秩修正，却仍需数百秩；不能将极少秩的廉价后修正视为已支持的方案。

## 本轮决策

1. 首先用阶段消融确认是否仍是变换 cast 主导，避免在错误阶段加精度。
2. 用幂二缩放分别测试一般输入和刻意构造的尺度不均衡输入，确认改善范围。
3. 以零变换的直接输入内核建立精度安全、性能更强的硬件对照；明确其为稠密算法。
4. 保留七乘法时，需要带着残差信息进入计算，而非仅换重组顺序。新提出的 INT8 残差表示仅在独立模型中筛选；在全部转换、缩放和同步成本未计入前不宣布性能可行。

本研究使用 AI 辅助检索、代码编写与分析。以上来源存在性及相应原文位置已核对；本地实验结论由脚本与保存的原始日志支撑。未进行外部模型复核，也未上传仓库内容供第三方模型评审。
