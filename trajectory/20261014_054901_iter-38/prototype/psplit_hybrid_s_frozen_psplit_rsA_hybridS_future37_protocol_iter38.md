# Hybrid S：后续 CPU 模型协议

本协议与 `psplit_rsA_hybridS_future37_model.py` 只检查输入变换的 FP32 RN DAG、最终 dtype cast 和逻辑槽计数。没有生成候选 CUDA、调用旧模型 main、编译或运行 GPU。`review.json` 为实际 CPU stdout；真实进程退出码单独记录于 `cpu.exit`。JSON 的 PASS 不用于推断进程退出码。

Producer 保存四个 raw A dtype 槽、一个 S1 dtype 槽和一个 S2 FP32 槽：共六个逻辑槽，占七个 16-bit 槽等价量。先计算 `s1=RN_FP32(A21+A22)`，再用这个尚未 cast 的 FP32 值计算 `S2=RN_FP32(s1-A11)`。S1 只在最终一次 dtype cast 后供 P5 使用；不能从已 rounded S1 槽重建 S2。WG0 构造 P3 的 `finalcast(RN_FP32(A12-S2))`，WG1 构造 P6 的 `finalcast(S2)` 与 P7 的 `finalcast(RN_FP32(A11-A21))`。P1/P2/P4 保留原始输入 bits。

每个逻辑 A cell 的算术与逻辑槽消费如下。16-bit 等价量只是 bytes 的计数单位，同时适用于 BF16 和 FP16。

| 路线 | A 存储槽等价量 | Producer/WG0/WG1 FP32 加减 | 总加减 | Producer/WG0/WG1 final cast | WG0/WG1 distinct-slot 消费等价量 |
|---|---:|---|---:|---|---|
| SS | 7 | 4/0/0 | 4 | 4/0/0 | 3/4 |
| raw RS-A | 4 | 0/3/3 | 6 | 0/1/3 | 4/3 |
| Hybrid RS-A | 7 | 2/1/1 | 4 | 1/1/2 | 4/6 |

Hybrid WG0 的四等价量来自 A11/A12/S2(FP32)；WG1 的六等价量来自 A22/S1/S2(FP32)/A11/A21。合计十等价量是按 WG 列出的逻辑 distinct-slot consumption，不能视为实际物理 unique LD 请求、共享内存 transaction 或 measured traffic。WG0 可复用 A11/A12，WG1 可复用 A11/A21/A22；真实指令分组和 whole-stage A 寄存器生存期需要新 CUDA 实现及 fresh compiled gate。

以 BM64、BN32、两 bank、七个 B dtype 槽计算的 nominal payload 如下；加 32B barrier 后 128B 对齐的数值仅为解析存储估计，尚未证明任何候选实际布局、dynamic shared、fit 或 occupancy。

| BK | raw RS-A 每 bank bytes | Hybrid/SS 每 bank bytes | raw RS-A 两 bank + barrier 对齐 bytes | Hybrid/SS 两 bank + barrier 对齐 bytes |
|---:|---:|---:|---:|---:|
| 32 | 30720 | 43008 | 61568 | 86144 |
| 64 | 61440 | 86016 | 123008 | 172160 |

在 1024 方阵一层递归、leaf K=512 的抽象域中，BK32/BK64 分别对应 16/8 个 stage、128 个 CTA。各 CTA 的完整 K 算术与逻辑 payload 总数保持一致。Hybrid 相比 raw RS-A 将 A 加减从六次降到四次，同时 A 存储恢复七等价量，consumer distinct-slot 消费由七增到十等价量；相较 SS，producer A 加减从四次减到两次。这些权衡不能直接推出端到端速度或 cuBLAS 加速比。

Bit-model 对 BF16 与 FP16 的有限输入 bits 做确定性 quadruple 采样，所有输入可由各自 dtype 表示且 `abs<=16`。比较原 SS、原 raw RS-A 和 Hybrid 七个最终 A 操作数的 16-bit bits，报告精确测试域与 mismatch 计数，不声称覆盖所有 quadruple，也不认证 CUDA converter、FTZ、NaN payload 或原 FP64 整体 GEMM 门限。

反例使用 A11=A21=1、A12=0，BF16 A22=2^-8 或 FP16 A22=2^-11。S1 恰在 midpoint，RN-even 后为 1；正确 FP32 S2 仍为 A22，错误的 rounded-S1→S2 链得 0，P3/P6 的最终 bits 与正确链不同。模型明确拒绝这一替代。

执行前后核对原 RS-A manifest 的八个 bindings 加 manifest 本身共九件，以及 BK32 future36 manifest 的十三个 bindings 加 manifest 本身共十四件。两个 manifest SHA 预先固定，所有受保护文件前后无漂移；执行期间所有 file write open 均被拒绝。这里只证明上述 CPU 模型与受保护输入，不认证原始精度政策、编译资源、硬件同步、性能或目标完成。
