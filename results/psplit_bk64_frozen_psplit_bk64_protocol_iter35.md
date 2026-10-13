BK64 的未来独立轴已准备，状态为 `PASS_CPU_SOURCE_GEOMETRY_AND_ABSTRACT_PROTOCOL_ONLY`。没有正式policy ID、CUDA/host编译、GPU/Graph/SAN/数值/性能结果；本任务只写全新 `.scratch/psplit_bk64_*` 文件，没有接管 iteration34 的正式执行 lease。

基底是冻结保守 P3+4 原型 `.scratch/cooperative_st_once_psplit_candidate.cu` SHA256 `84e56c44deb1585f6be457461dcc9073cad6631316e159249e77dbbb77c78664`，384 threads、producer64/consumer96/96、BM64/BN32/BK32、两个 operand banks。候选 `.scratch/psplit_bk64_candidate.cu` SHA256 `55f119259536c1c788f0d8508c8005f820463ef8fa6b316018ea2733caf0d8e1`。

冻结基底的 iteration34 初次编译资源 gate 已知 **FAIL**，不能写成未知或可用于GPU。nvcc build exit=0，但保守 FP16/BF16 都是 REG96/STACK128、388B spill stores/384B spill loads，并出现 C7512（寄存器资源不足导致WGMMA串行化）；cap80版本为 REG80/STACK192、516B/512B spill，情况更差。实际读取的原始证据是 `results/psplit_initial_build_iter34.txt` SHA `4b69eaf27e25c888105d3080234f06e9cf5154a9288a1b57bf0f894a27a14011`、object资源 SHA `47d82a6253b07637ddbe46fc5f5f91907ed221f0f4ac963e388cd3b525a52a82`，以及 CPU snapshot SHA `17b2e7776520e028ce7f92a72f76bdc3ab63cd1204db5f63071368d85824bc66`。ROOT确认初次正式源码绑定84e56且没有GPU launch；本agent只核读上述raw logs/resource/snapshot，没有独立复审旧object与源码完整对应。BK64候选仍只表达冻结84e56上的几何轴，未导入core34正在处理的register-scope修复，也未编译。

纯 BK 常量替换存在两个具体 blocker，因此候选包含两项必要的几何修正：

1. 原 A helper 固定 `row=task/4,k=(task&3)*8`。BK64 的512个task会产生256个越界row task（第一个task256→row64/K0），并留下256个有效K32..63 vector未写。候选改为 `/8` 与 `&7`，每个row覆盖八个vector8。原vector8加载、逐元素FP32 S/T DAG、最终一次RN转换、原值直接复制和七个slot存储保持。
2. 原父矩阵 `p.k%64==0` 会接受leafK32/96等partial BK64 stage。候选改为 `p.k%128==0`，使halfK完整覆盖64；parentK64/192在该native helper中拒绝，128/256/384/512/1024/2176接受。未来正式Plan须保持当前guards和真实fallback选择，不得把partial leaf当native成功。

还原独立符号名和注释，以及 BK64→32、A task索引、Kguard三项几何变化后，源码逐字节等于冻结基底。精度、S/T、P分组、U、epilogue、quota、union复用和同步均没有变化。完整diff与只有三个执行变化的semantic diff分别保存在本prefix内。

实际核读 pinned CUTLASS3.5.1 的 SW32 bit layout、upcast16、tile_to_shape、flagged pointer及selector/gemm代码。A atom8×16 elements可整除target64×64×7，repeat8×4×7；B atom16×8可整除target32×64×7，repeat2×8×7。默认LayoutLeft先重复第一mode再K再P。F32 BF16/FP16 selector在N32仍指向m64n32k16 atom，gemm K循环按ascending k遍历。这里确认的是源形状的divisibility条件和文字API，不是实际CuTe/CUDA实例化成功。[CUTLASS布局与selector源码](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/atom/mma_traits_sm90_gmma.hpp)、[GEMM遍历](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/algorithm/gemm.hpp)

BK64每P每stage有四个K16子步，WG0的P1–3共12条、WG1的P4–7共16条，合计28条。parentK128/256/384/512/1024/2176/4224的每P全K16 offset序列和旧BK32逐项相同；不同P之间的交错会随stage变长而改变，各P独立。真实GPU P/U bitwise是否相同仍需核实，异步issue credit、编译器scalarization及硬件backpressure也未测试。

Literal-header Python物理地址推导按byte处理SW32 pointer swizzle，避免把byte swizzle施加到element index。每个bank含A28672/B14336个16-bit elements，A57344B、B28672B，总86016B；B origin57344、第二bank origin86016均为512B整数倍。对absolute shared base phases0/128/256/384、两个bank和所有七slot，实际核对43008个连续16B对齐vector及344064个element address，覆盖唯一、无bank越界。A512/B256 producer task每stage覆盖完整。这个推导仍需以实际CuTe layout/descriptor实例化和GPU访问验证。

两WG协议没有改动，冻结CPU模型经过fresh execution并绑定原model SHA `d5a159f926b9608a467d39ca806027f14772ce44fc035f078fc337f5ffa6755f`。full arrival128、empty arrival256，两组独立wait0后各release128；producer_tail独立acquire两个bank。包含initial、全部bank退休、P发布的三个uniform384-thread CTA barriers保留。两consumer可以独立完成/释放，第一组的128 arrivals不能允许另一组仍在读时重用bank。

CPU同步BFS覆盖stage0..17、31/32/33、63/64/65、511/512，共126748 states/278868 transitions；每规模只有一个安全终态，无deadlock。14个规模各16随机轨迹，共224 traces/106400 actions。BK64的1024对应8 stages，实际962 states/2562 transitions；stage16/17/33也分别实跑。三种unsafe witness继续存在：empty误设128导致单WG早释放、只有一个consumer结束就做tail覆盖、缺第二tail CTA barrier导致U读未发布P。模型是三组各128线程的集体事件抽象，不是实际线程PC、内存序或资源进展证明。

P tail总57344B不随BK变化，现在完全落在bank0的86016B内，bank1 overlap=0。七个slot的byte spans从[0,8192)到[49152,57344)，P6不再跨bank boundary。候选仍要求全部两个bank和两个consumer排空后才切换typed union，没有修改同步轴。实际枚举14336个FP32 P写者，全部唯一；U的七slot读取源码和输出坐标拓扑未改，由其推导读取同一完整集合；三个1024输出stride1024/1025/1041各覆盖1048576个输出且不写padding。32 active-beta exact C0=D轨迹核实262144次原C0读取和262144次D写入；32 beta0/null轨迹没有C0读取并完成262144次D写入。任意partial overlap不在该证明范围内。

共享内存仅作独立C++ ABI预测：

| 项目 | 预测bytes |
|---|---:|
| 两bank operands/union payload | 172032 |
| 两full+两empty uint64 barrier | 32 |
| 追加barrier后 | 172064 |
| align128后的SharedStorage | **172160** |
| 额外P tail allocation | 0 |

payload翻倍而barrier仍为32B，因此按payload加barrier后重新对齐。实际dynamic、额外ELF static shared及总驻留资源均pending。源码quota仍为64/96/96，role合计32768 registers；数学上uniform initial quota在8粒度下须88..96并满足各direction，实际initial allocation、完整role CFG/implicit R widths、spill/local/stack及最新WGMMA在途P访问都需新编译gate，不继承 iteration34。

1024 grid仍128 CTAs，halfK从16个BK32 stage变为8个BK64 stage。source logical总input仍50331648B、最终operand shared stores仍88080384B、transform add/sub仍25165824、WGMMA K16 instructions仍28672、P store+U load仍14680064B。这些是逻辑请求量，不能解释为实测HBM/L2/shared traffic、occupancy或时间收益。

后续应把仅三个执行变化的几何diff移植到通过fresh compile/resource gate的Psplit基底，或为该独立BK64候选修正寄存器作用域后重新通过自己的fresh gate。当前CPU PASS不能继承旧84e56的GPU可用性。

下一次正式执行的所有硬件gate仍需完成：实际BF16/FP16 CuTe模板及canonical descriptors、current object/linked SASS一致性、初始CTA-owned pool/role CFG/implicit R/spill/inflight P访问、dynamic/static shared、matching-atom全P/rawU/指定epilogue、原FP64误差gate、short/phase-wrapped尾、fresh currentAB及真实fallback/Graph、serial mem/race/sync，以及完整1024 ordinary/Graph调用对比retained bestMega和同行独立调优cuBLAS。BK64是否更快没有承诺。
