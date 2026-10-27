BN64 的候选 CUDA/API 已由 frozen45 具体生成，属于未来独立试验的私有源码；当前没有 CUDA 编译、GPU 运行、Plan 注册或性能成绩。源码保留原 rawA4 TMA、SW128/1024 对齐、每 ordinary 调用的 current-A map encode、32768 transaction bytes、SIMT B onecast DAG、128 writer publication、wait0、wholeA3+4/whole7P、原三次 CTA epoch、两 bank drain、fixed U 和 RN alpha/beta FMA。

N tile 从32扩为64，CuTE RS atom 相应采用 m64n64k16，每 product 每线程32个 FP32 P。B 使用原 Layout_MN_SW32_Atom 模板的64×64×7 tiling。Tail 的 product stride 必须4096、row stride 必须64，完整尾部为28672个 FP32 words；原 XOR row permutation 在64列内仍是双射。

B 绝对 halfword 原始布局为 `q*4096+(k//8)*512+(n//16)*128+n%16+16*(k%8)`。令 `O=origin+bank*90112`，实际 byte 地址为 `sw32(O+32768+2*raw)`；`sw32(x)=x^((x&128)>>3)`。每 slot/K16 atom descriptor start 为 `O+32768+q*8192+atom*2048`，leading256B/encoded16、stride1024B/encoded64，layout type3/base offset0。这些是 pinned CuTE 源码推导，尚非实际 kernel descriptor trace。

预期每 bank90112B，两 bank union payload180224B；Pipeline barrier32B 和 raw_done16B 在 payload 后，raw_done 预期 offset180256；align1024 后 logical SharedStorage181248B。114688B typed tail 在两 bank payload 内，在全部 consumer retire 后激活。未来需 fresh `sizeof`、`offsetof`、alignment 和 Driver static/dynamic shared 记录；这些预测不能填入 actual 字段。

Producer/WG0/WG1 源码 quotas40/224/240 分别用于三个128线程组。两个 consumer 的 INC 分别置于各自 branch、第一条 P clear 之前。请求总量64512 words等于384×168；若实际 owned startup I=168，则 producer释放16384 words，两consumer需要7168+9216，算术刚好成立。WG0 P96+A48、WG1 P128+A64 给其他 live values 的源码下界余量分别80/48 words，不能证明实际 fit。Producer40尤其需要 fresh highest GPR<40；parent45 的37/165/164和I168不能继承。

1024 grid从128 CTA降为64 CTA，同一rawA供更宽N范围，A transform重复次数减少；每 CTA 的B/P/tail工作和P寄存器需求增加，设备上的并行工作单元减少。端到端收益未知，CPU模型不预测加速、bank conflict、occupancy或residency。

实际 CPU 验证涵盖五个绝对 origin、两个 bank、opaque uint4/halfword transport、独立 descriptor reconstruction、C/A fragment partitions、tail ownership、三种D stride的唯一输出范围、padded/极限A lda、完整七 product 整数DAG和wholeK，以及128-writer/TMA phase/两bank release/wholeA+P pending interleaving。最终40个负例均拒绝，包括SW32错误A、旧BN32 B stride、旧tail stride、真实127 arrivals的早读、错误raw epoch、单WG释放后的bank reuse、pending A覆盖、pending P读取、提前release/tail/U、少owned pool和错误quota。

第一次 model 实际 exit1，保留原 source/stdout/stderr/exit：负例的 P5=0，使错误 U7 sign 不可区分。只修新prefix fixture为显式 P5≠0，第二次实际exit0。随后加强127-arrival和完整metadata反例，第三次最终实际exit0；初次独立源码验证exit0，增加 pinned barrier 类型/结构断言后的第二次验证exit0。CUDA candidate/API/delta在这些 fixture 改动中未改变。

独立只读 subagent 对 pinned CuTE 同时复核了 B byte layout、descriptor fields、C64×64 coords、A fragment与quota算术，报告未发现应在CPU阶段否决的几何矛盾。该临时复核结果仅辅助评审，本prefix的冻结实际CPU日志为可重放证据。

旧100 future文件和41个旧library/source文件逐项保留hash。formal42 owner获准修改的六个CMake/API/Plan/bench/test文件仅记录前后观察，不当作drift；新formal46源码不属于本候选写租。本prefix无public47符号注册，也不会自动开始43。

未来准入须完整 fresh compiled/resource/SASS/initialization/pending ABI、零STACK/LOCAL/spill、SAN和原始FP64门限。CPU整数恒等式只是模型，不代表实际FP16/BF16精度；所有actual CUDA、CAN、Driver、GPR、shared ABI、SAN、precision、latency和cuBLAS speedup字段均为null。
