未来 BK64 候选已基于实际 consumer-only rev6 源码准备，状态 `PASS_CPU_SOURCE_GEOMETRY_AND_ABSTRACT_PROTOCOL_ONLY`。没有正式 policy ID、host/CUDA build、GPU launch、Graph/SAN、数值或性能结果。全部新写只在 `.scratch/psplit_bk64_scope34_*`；旧七个 `psplit_bk64_*` 文件仍冻结。

冻结基底从 `solution/megakernel_wgmma_psplit.cu` 按bytes复制到本prefix baseline，full SHA `fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91`。候选 full SHA `d119f6cf9e922f870a2acca90803231feb330d8cba134ea91942e101b4b4baeb`。实际diff只三处，无符号或注释变化：BK32→64；A helper `task/4,(task&3)*8`→`task/8,(task&7)*8`；父矩阵 Kguard `%64`→`%128`。逆向还原三处后与新baseline逐字节一致。继承的“All256 tasks”和“BN32:128 tasks”注释仍是BK32历史计数，本CPU报告中的实际BK64任务数是A512/B256。

这两个辅助几何修正必需：只改BK会生成256个越界A task（第一个task256→row64/K0），同时漏掉256个有效K32..63 vectors。新A映射对64rows×8vectors唯一覆盖；B索引本来由Cols/8推导，扩大到256个task后覆盖64K×4vectors。父K64/192不能提供完整半K64 stage，native helper拒绝；128/256/384/512/1024/2176等可完整覆盖。Plan集成仍须保留common current span/end、overlap、scalar alignment、exact C0==D、certification/range、degenerate、invalid-before-empty/scale以及真实fallback/Auto guards。

consumer-only q0..q3未初始化数组及rmem views仍限定在consumer分支，producer分支不存在这些q的访问；producer dealloc64、consumer alloc96及post-allocation clear原样保留。两个独立whole-K consumer loops仍分别拥有P1–3/P4–7。vector8原值读写、逐元素FP32 S/T DAG和最终一次RN转换、P/U/epilogue顺序均逐字节保留。改变BK可能改变实际寄存器与descriptor存活、unroll和issue行为，不能从byte scope一致推出编译资源一致。

初始PipelineAsync发布由全384线程的 `__syncthreads()`完成；producer完成两bank的producer_tail后、两consumer各自wait0/release完整K后，沿两个不同role-local callsites执行同一非aligned `barrier.sync 6,384`，两个asm均带memory clobber。这是一个全CTA退休event，不是两个连续barrier。consumer在该event后重新取得volatile tail_tid并发布全部P；最后全384线程的 `__syncthreads()`在任何U读取前发布完整P。共三个逻辑CTA events、四个词法call sites（两次syncthreads及两处role-local retire）。source-level uniform role participation只作为CPU检查；实际线程CFG、非aligned named-barrier内存序和资源进展仍须新硬件gate。

full arrival128/empty arrival256，两consumer独立wait0后各release128；一组的128arrival不能提前允许另一组仍在读的bank被reuse。producer_tail仍独立acquire两个bank。fresh抽象模型BFS覆盖stages0..17、31/32/33、63/64/65、511/512，共126748 states/278868 transitions，每规模一个安全终态、无deadlock；224随机轨迹/106400 actions。1024的半K512对应8个BK64 stages，实际962 states/2562 transitions。三个negative witness保留：empty128导致单WG早reuse、仅一consumer结束就覆盖tail、缺P发布barrier导致U早读。这是三组各128线程的集体事件模型，不代表物理per-thread PC、compiler CFG、硬件内存序或资源充分性。

新model以importlib导入原frozen geometry/protocol模块的函数，两个旧main均不调用；还安装Python audit hook拒绝任何文件写入，避免旧protocol main的Path.write_text覆盖原review。旧七文件hash在执行前后复核；没有用仅stdout重定向替代输出安全。实际执行 `python -B .scratch/psplit_bk64_scope34_model.py`，session47431 exit0，stderr0B，stdout完整归档到新prefix；句柄已关闭。

Pinned CUTLASS3.5.1 literal-header推导：A SW32 atom upcast16为8×16elements，target64×64×7的repeat8×4×7；B为16×8elements，target32×64×7的repeat2×8×7；defaultLayoutLeft先重复首mode再K再P。目标形状可整除atom，N32 selector仍为m64n32k16，K64有四个ascending K16子步。代码从worded headers推导byte-pointer SW32，核对base phase0/128/256/384、两bank、全部七slots，实际43008个连续16B对齐vectors/344064个element addresses，覆盖唯一且无bank越界。这里没有实际实例化CuTe/template、canonical descriptor或C++ sizeof。

每P每stage四K16，WG0共12条/WG1共16条，合计28条；父K128/256/384/512/1024/2176/4224的每P全程K16 offsets与BK32逐项相同。P之间交错随stage长度改变，各P全程独立。GPU accumulator位级一致性和异步issue credit/backpressure须fresh验证。

一bank含A28672/B14336个16-bit elements，即A57344B+B28672B=86016B；B origin57344、bank1 origin86016均为512B倍数。两bank union payload172032B，barriers仍为四uint64=32B，合计172064B，按align128后的独立C++ABI预测为 **172160B**。P tail仍57344B，全部位于bank0内、bank1 overlap0、P6不跨bank boundary；保持两bank和两consumer完整退休，没有删减同步。实际dynamic/static shared、total residency与occupancy未测。

实际枚举14336个P唯一写者，并另按所有384线程 `linear=tid;linear<2048;linear+=384` 明确枚举14336次U读取：2048个U owner无重复，P写与U读覆盖完全相同的七slot word集合。1024的ldD1024/1025/1041分别唯一覆盖1048576输出，不写padding。32 active-beta exact C0=D轨迹验证262144原C0读取/262144D写；32beta0/null轨迹0读取/262144D写。partial overlap不在允许或证明范围内。

1024 grid仍128CTAs，K stage数量减半；source logical总input50331648B、operand shared stores88080384B、transform add/sub25165824、WGMMA K16 instructions28672、P store+U load14680064B都与原BK32相同。这不是实测HBM/L2/shared traffic、occupancy或延迟，尚无BK64更快的结论。

实际rev6 BK32背景证据：本agent只读build/object resource日志，FP16/BF16均REG96、STACK0、spill0、psplit无C7512，ELF static shared1024B。ROOT进一步报告final36 fresh独审 `CAN_LAUNCH_36_ONLY` PASS：source仍fdaae75a…，roleR58/93/93，initial96/pool36864≥32768，无earlyclear/inflightgenericP，nonalignedbar6/threecollectives/tailSTS/alias通过，binary SHA prefix e5004b3f、test prefix33f31ae9。该final36 full gate由ROOT拥有，本agent没有独立读取其完整报告；ROOT消息当时GPUcorrectness/latency仍pending。所有这些只属于BK32基底，绝不能继承为BK64 gate。

BK64下一轮仍须自己的fresh compile/object/linked SASS和actualCuTe canonical descriptors、actualinitial ownedpool/roleCFG/implicitR widths、spill/local/stack/C7512/earlyclear/inflightgenericP、dynamic/static shared；随后matching-atom所有P/rawU/指定epilogue位级与原FP64误差gate、short/wrapped tail、currentAB/Graph及真实fallback、serial mem/race/sync，再测完整1024ordinary/Graph调用并对照retained Mega控制与独立调优cuBLAS。没有正式policy ID，也不混入iteration34执行或归档。
