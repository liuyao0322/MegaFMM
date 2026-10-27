三 operand bank 的未来 scratch 原型及独立 CPU 模型已完成，状态为 `CPU_SOURCE_THREE_BANK_PASS_NOT_COMPILED`。没有分配正式 policy ID，也没有 CUDA/host 编译、GPU/SAN/Graph/数值/性能数据；没有修改当前 iteration32 的正式源文件或接管其执行 lease。

源码 `.scratch/megakernel_wgmma_wait1_three_stage.cu`，SHA256 `f78b1fbf1541583315ce084cac3e05afd7b5d5887f64b928c0f504c9902afd23`，来自冻结 wait1 原型 `80e4ee8f6df3ba738ef10101b9224c701abbb517de993a7ea7d1d710bbd8ae91`。还原唯一符号名、两条新增注释、bank 数注释与 Stages3→2 后逐字节等于基底。唯一执行配置变化是 Stages=3；没有合入 release1。full/empty count 仍各为128、两个 consumer release 仍由全部128消费者执行。向量输入、S/T最终一次转换、七P累加次序、固定 U、alpha/beta、当前 A/B binding guards、consumer wait1 和最终 wait0 全部保留。

三个 bank 使 producer 最多提前构造三个尚未退休的 stage；consumer 的 `pending==2` 调度不变，最多两个 committed unretired groups。每次 wait1 后只释放最旧 bank，最新组仍保留另一 bank；尾部 wait0 完成最新组后排空。多一个可填充 bank 的调度余量只是优化假设，尚无动态 profiling 证据把两 bank 的回归归因于这个因素。

实际核读 pinned CUTLASS3.5.1 `PipelineState`：默认 index/phase/count 为0，每次递增 index/count，到 index==Stages 时归0并翻转 phase，支持非2的幂的 Stages。consumer read/retire 对应 index=stage%3、phase=(stage//3)&1；producer 初相位相反。`PipelineAsync<3>::producer_tail` 独立 acquire 三个 bank，每次推进状态，并不缩减为两个 tail acquire。[CUTLASS3.5.1 Pipeline 源码](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/pipeline/sm90_pipeline.hpp)

CPU 模型 `.scratch/wgmma_wait1_three_stage_model.py` SHA256 `46d094203f57794e33f9d3b33a5e663d1b7145fe374ee2bb18fdd16d256b72ff` 实际运行：

- BFS stage0..17 加32/33/64/512，共14279 states、27569 transitions；每个规模有一个 fully drained terminal，无 deadlock。
- stage16 有287 states/549 transitions；stage512 有9215 states/17909 transitions。对所有stage≥3，最大 owned banks 实际达到3；最大 committed unretired groups 始终为2。
- 14个规模各32随机轨迹，共448 sequences/115616 actions；包含32/33/64/512的多次 phase wrap，全部通过。
- 独立逐 bank 的 arrival 计数，对 stage0..512 核对 full/empty 复用 phase 和三个 producer_tail acquire，共395523 checks 通过。
- 允许最新组先于旧组完成。保留四类具体不安全状态反例：wait1 后释放最新 bank、仅一组时用 wait1 当完成证明、尾部 wait1 后 generic U 读取、下一 producer bank 仍被占用时覆盖。

模型实际达到“第三 bank 已填充、两个组仍未退休”的状态，也验证了旧组可退休而最新组仍在途。模型把128 producer 的 publication 和128 consumer 的 release 分别抽象为整体事件，不证明每个线程的实际 PC、硬件内存序、编译器收敛、WGMMA rounding 或 spill。上述数字来自新三 bank 模型的执行结果，没有继承两 bank 模型的 PASS。

共享资源只作 ABI/布局预测，未实际编译：

| dtype | BM/BN/BK | 单bank operands bytes | 三bank operands bytes | 6个barrier预测bytes | align128后的SharedStorage预测bytes |
|---|---|---:|---:|---:|---:|
| BF16/FP16 | 64/16/32 | 35840 | 107520 | 48 | 107648 |
| BF16/FP16 | 64/32/32 | 43008 | 129024 | 48 | 129152 |

两个布局的第三 bank offset 分别为71680/86016 bytes，和前两 bank 一样是128与512 bytes 的整数倍；单bank布局未改。增加shared容量可能改变实际驻留CTA数量，不能因为源码quota仍是producer64、consumer96/160就沿用旧的owned pool、register maxima或occupancy结论。

后续正式实验在首次 launch 前必须重新编译并审核四个 BF16/FP16 × BN16/BN32 实例：实际 dynamic/static shared、初始 CTA-owned register pool、quota 方向、完整 role CFG/implicit register widths、accumulator 初始化、spill/local/stack、object/linked SASS 身份、mod3 index/phase/三个tail acquire、128线程 aligned 集合参与及最新组在途时无 generic P 访问。

GPU 验证还需同atom raw-U/指定epilogue、短/奇偶/phase-wrap尾部与fresh bindings、beta0/null、exact C0=D、one-kernel Graph、顺序 mem/race/sync、原数值gate。完整1024 ordinary/Graph分别对比同binary两bankwait1、retained29与同行独立调优cuBLAS。任何这些结果仍为pending，CPU状态PASS不能认证三bank更快或能超过cuBLAS。
