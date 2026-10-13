# S2 FP32 even/odd planar：CPU 坐标协议

本轮只写新的 `psplit_rsA_hybridS_planar_future37_*` 前缀。已冻结的 Hybrid S 四件逐 SHA 核对、执行前后无漂移。模型没有 import 任何旧模型、调用旧 main、生成候选 CUDA、编译或运行 GPU。实际 CPU stdout 为 `review.json`，真实进程退出码单独存于 `cpu.exit`；JSON PASS 只属于 literal CPU geometry。

本地 CuTe `Layout_K_SW32_Atom_Bits` 是 `ComposedLayout<Swizzle<1,4,3>,smem_ptr_flag,Layout<Shape<8,256>,Stride<256,1>>>`。`upcast<32>` 将 underlying atom 变为 float 的 Shape8×8/Stride8,1，flagged pointer 保留 SW32 函数并对实际 byte address 使用 `addr XOR ((addr AND 128)>>3)`。`tile_to_shape` 默认 GenColMajor：先扩 row block，再扩 K block，最后 plane。源码、六份相关 CuTe header 的 SHA 和关键定义行已记录于实际 stdout。

Planar 布局采用 Shape64×(BK/2)×2：plane0 存 even-k 的 S2 FP32，plane1 存 odd-k。对 `j=k/2`，每平面先用上述 float atom 扩为 64×BK/2。plane0 的 byte address 正好重现原 half/BF16 raw A 的 k-even packed-u32 地址；plane1 地址等于 plane0 加 `128*BK` bytes。两平面仍占完整 S2 的 FP32 payload，BK32 为 8192B，BK64 为 16384B，没有减少 S2 的逻辑存储或读字节量。

Producer 使用现有源码的 task 划分：`task=tid+i*128`、`row=task/(BK/8)`、`k=(task%(BK/8))*8`。一个逻辑 S2 vector8 按 even/odd 切成两组 vector4，CPU 枚举证明各组地址连续、16B 对齐、未越界，且所有 producer 任务覆盖每个逻辑值恰好一次。实际 CUDA 的 even/odd shuffle、register packing、临时寄存器开销与两次 `st.shared.v4` 实现尚未完成。

Consumer 直接采用本地 `ALayout_64x16=CLayout_64x16` nested shape/stride 解出的原 RS 坐标：每个 k-even pair 分别从 `(row,k/2,plane0)`、`(row,k/2,plane1)` 读取两个 scalar-u32。CPU 枚举证明 pair 坐标、4B 对齐、访问边界和 producer 写入的逻辑标签一一对应，没有遗漏、重复或平面混用。

每 BK 枚举六个 128B 对齐 origin，包括 SW32 byte 周期的不同 phase 和名义 Hybrid 两 bank 中 S2 的起点。以下 bank 数字按一次假设 warp scalar-u32 坐标集合统计 `bank=(byte_addr/4)%32`，计数该 bank 上的 distinct words；同地址重复只算一个 word。

| BK | 每种 comparator 的 warp batches | 原 packed-u32 最大 distinct words/bank | flat S2 even/odd | planar S2 even/odd |
|---:|---:|---:|---:|---:|
| 32 | 192 | 1 | 2/2 | 1/1 |
| 64 | 384 | 1 | 2/2 | 1/1 |

Flat 两路各有 32 个 distinct words 分布于 16 个 bank；planar 各路有 32 个 distinct words 分布于 32 个 bank。该差异仅是 local CuTe 定义下的 literal scalar address comparator，不是实际 compiled instruction grouping、transaction、冲突周期、WGMMA bank traffic、吞吐或速度收益证明。两路 scalar load、Producer 两次 store、S2 FP32 payload 和 Hybrid 的十等价量逻辑 distinct-slot 消费依然保留。

BK32 完整枚举为 1536 个 producer vector8 tasks、3072 次 planar vector4 store 检查、6144 个 consumer pair、12288 次 scalar load 检查；BK64 为 3072/6144/12288/24576。所有平面均为完整 byte-word 双射，无重叠、越界或错配。构造没有证明候选实际 C++/CUDA 类型实例化、布局 ABI、shared/register fit、physical A lifetime、GPU 同步、原 FP64 GEMM 数值门限或相对 cuBLAS 的性能。
