# MegaFMM 算法与验收计划

当前任务/交接以 [HINTS](HINTS.md) 为准；运行见 [README](README.md)，证据见 [ITERATIONS](ITERATIONS.md)。仅支持普通 launch；历史 Graph、all28、特化/双层/补偿设计不构成继续执行授权。主线采用少量通用实现及有界模板池。

## 数学与 API 合同

`D = alpha AB + beta C0`，A 为 M×K、B 为 K×N、C0/D 为 M×N。输入 BF16/FP16/FP32，累加和输出 FP32；原生 dtype、七乘积和 FP32 运算关联固定。禁止 BF16→FP16、补偿乘积或完整 dense AB 替代 SW。

- 保留各路径声明的 row-major、transpose、leading dimension、系数及 C0=D；拒绝负尺寸、非法 stride、地址溢出及非法 A/B/output/workspace 重叠。
- `create_plan()` 校验；`tune()` 显式测量，可分配/同步；`run()` 不分配、不 host 同步、不缓存输入变换。调用方按 Plan 分配限额内 workspace，并发使用独立 workspace；绑定变化重查并重建 descriptor。
- M/N=0 不发计算 kernel；K=0 或 alpha=0 仅缩放 C0、不读 A/B；beta=0 不读 C0。
- Auto/Dense/Tuned 走独立调优 cuBLASLt；SWTuned 只在七乘法中选择。固定/零预算 FP32 SW 保留零 workspace fallback；预算内 packed/fused 策略按完整调用调优。

## 固定单层 Winograd DAG

象限按逻辑半尺寸划分，奇数边界补零；P4 与 C21 的负号成套固定。

```text
S1 = A21 + A22        T1 = B12 - B11
S2 = S1  - A11        T2 = B22 - T1
S3 = A11 - A21        T3 = B22 - B12
S4 = A12 - S2         T4 = T2  - B21

P1 = A11 B11          P2 = A12 B21
P3 = S4  B22          P4 = A22 T4
P5 = S1  T1           P6 = S2  T2
P7 = S3  T3

U1 = P1 + P2          U2 = P1 + P6
U3 = U2 + P7          U4 = U2 + P5
U5 = U4 + P3          U6 = U3 - P4
U7 = U3 + P5

C11 = U1             C12 = U5
C21 = U6             C22 = U7
```

共享表达式时为 8 次输入加减 + 7 乘法 + 7 次输出加减；独立展开可达 24 次加减，不再称“十五加减”。[test_dag.py](tests/test_dag.py) 独立验证恒等式。低精度 S/T 依赖在 FP32 中保留，仅最终叶操作数舍入回原 dtype；P/U 为 FP32，最后应用 alpha/beta。FP32 叶乘法不降精度，逐位匹配不能代替原始 AB 精度。

## 工程与资源约束

- 每次重新构造 pack/变换，七叶与重构、必要 reset 全部计时；融合需明确输出所有权，CTA 内重算和额外读取计入成本。固定 FP32 双槽真实 TMA/BK16/32，非法绑定同步 fallback；cooperative launch 按实际驻留容量检查。
- 分别证明 TMA/mbarrier 字节、arrival、phase、generic→async proxy fence、WGMMA fence/commit/wait 和 reader 完成后槽释放；`__syncthreads()` 不能替代全部协议。保留 iteration242 五处 proxy fence。带 swizzle 的 CuTe layout 不当裸数组；cluster/DSM 须证明跨 CTA 生命周期。
- workspace 以实现/API 的配置尺寸为准，比较器覆盖所有执行 ID、dense 和 frozen reference；禁止沿用固定七平面估算分配 A8/B8。检查实际 stack/spill/local、寄存器/shared、driver reserve 与 occupancy 后再计时。
- 历史冻结 1024 源码/参数化约束在 HINTS 与 [归档](archive/sw_specializations/docs/sw_parameterized.md)。递归、低精度输出、多 GPU、FP8 或改变重构顺序的原子/TMA-reduce 需另立合同。

## 成本模型（非实测承诺）

L 层均匀二分、每层全局复用 S/T：

```text
F_leaf(L) ≈ 2MNK (7/8)^L
F_add(L)  = (MK + KN + 7MN/4) Σ[i=0..L-1] (7/4)^i
T_SW = T_transform + T_leaves(actual schedule) + T_combine
     + T_tail/padding + T_required_overhead < T_dense
```

单层 7 叶、方阵加减 3.75n²，理想同吞吐比 8/7≈1.143；历史双层模型为 49 叶、10.3125n²、64/49≈1.306。单层叶吞吐若仅为 parent 的 η，叶时间比约 0.875/η；η≤0.875 时零额外开销也无收益。融合整体计时，重叠分项不相加；等效 `2MNK/T` 与真实叶 FLOP/s 分列。

历史 V1（输入元素 s_in=2B、P=4B）：`W_ST=s_in(MK+KN)`、`W_P=7s_P MN/4`；方阵 11n²B，8192≈0.688GiB、16384≈2.750GiB。复制七组左右操作数时输入存储为 `7s_in(MK+KN)/4`；另计 A/B/C0/D、库 scratch、padding/对齐。七组 P 写读逻辑流量 14MN B，不等于 HBM 流量。

CTA 重算模型：`G=MN/(4BM BN)`、`Q=K/(2BK)`，
`F_add,M1 = G[Q(4BM BK+4BK BN)+7BM BN] = MNK/2(1/BM+1/BN)+7MN/4`。
128-thread consumer 的七 P 累加器平均需 `7BM BN/128` 寄存器；保守低精度 scratch 为 `16BK(BM+BN)` B/stage，另计 barrier/对齐/输出。64×32×32 为 112 regs、48KiB/stage；64×64×32 为 224 regs、64KiB/stage。模型不证明实际编译资源或性能；历史 M2 每 K-stage 的 14 输出贡献改变顺序，另验。

## 数值与性能验收

- 参考：原始输入先舍入到目标 dtype，再用 FP64 普通 `alpha AB+beta C0`；报告 eF、最大绝对误差、近零误差、nonfinite。逐 fresh trial 判定，不能用各自最大误差/最大 gate 的比值替代。
- gate：`eF_SW <= 3*eF_dense + floor`，BF16/FP16/FP32 floor 分别 `1e-3/1e-4/1e-6`。FP32 必须保留；BF16/FP16 默认调优按后续指令将旧 gate 作报告项，strict 可要求。nonfinite、实现/内存错误始终拒绝；任务质量另验。
- 覆盖随机正负/正值、零/identity、强消去/相近象限、动态范围/溢出、signed zero/subnormal、奇数/矩形/stride/transpose、系数、C0=D、新输入及双流。
- 同进程独立调优双方，匹配 dtype/output/layout/workspace/mode/工作集，包含每次转换/重构；多流 event join。setup/tuning/CPU wall time 单列；固定 B 另报预计算及摊销。
- fixed30 原协议与独立接受条件见 HINTS。保留全部 raw groups、独立进程、p10/median/p90/CV；组均值分位数不等于单调用尾延迟/置信区间。筛查、跨配置包络、fallback、单轮领先不等于独立认证。

## 应用交付

| 关口 | 必需内容 |
|---|---|
| 可复现 | 声明式配置、隔离执行/输出、输入来源/seed/hash、环境与源码/二进制身份、失败/缺失状态 |
| 覆盖 | GEMM；FP32 covariance、sqrt/inverse-sqrt、SP2、LINPACK 风格 LU；BF16/FP16 Muon、GELU/SwiGLU MLP |
| 公平性 | 匹配 native 与独立 practical PyTorch 基线；strict FP32 与 TF32 分列；FP64/应用质量和速度分列 |
| 时间与报告 | 固定工作量/达到质量时间分列，包含停止检查；setup/tuning/摊销、原始样本、全部独立进程、路由/workspace、可再生报告 |

smoke、synthetic、真实 replay、完整任务各属不同证据；Muon replay 不证训练收敛，冻结 Fock 不证 SCF 能量/力，MLP replay 不证模型质量。参考 eigendecomposition 不计时，不能声称超过最优科学求解器；FP32 LU 不是标准 FP64 HPL。科学 benchmark 固定已验证内核快照，应用任务不授权修改活跃内核实验。

## 工具与研究入口

H100/CUDA12.4/CUTLASS3.5.1/C++17/sm_90a；实际环境另记录，allocation 使用前核验。保留 dirty work，独立 build/result；NCU 私有 TMPDIR 不删除他人锁。

设计依据：[Winograd/低空间调度](https://arxiv.org/abs/0707.2347)、[packing/output 融合](https://www.cs.utexas.edu/~flame/pubs/FLAWN79.pdf)、[CUTLASS Strassen](https://arxiv.org/abs/1808.07984)、[稳定性](https://arxiv.org/abs/1507.00687)、[Hopper](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html)、[CUTLASS3.5.1](https://github.com/NVIDIA/cutlass/blob/v3.5.1/README.md)。仅为背景，不替代项目测量。

SubCuber 固定 `c7b6fd8c72db4fc83982b26ea6e80e47567d674f`，patched CUTLASS4.1.0 与本项目隔离；FP16 输出、单 heuristic、warmup 后 presum 复用不符合动态输入同合同基线，见 `research/subcuber_reference_20261018/REVIEW.md`。早期 SMALL_REPORT/small_manifest 不在当前工作树，不作为现存证据。
