# Future35 RS A independent CPU review

**PASS仅限修正后source/API推导与分析模型，actual CUDA compile/layout/register gates全部pending。** Final candidate SHA `4e32c7cf4bd7b916ae5fad3cab8aba601fb629aaca41ffd310ce40014113fca1`，model SHA `aab2e1940470681b51d83596440ae49c5f2369e6abf2a1ff33d7f3c5ab7644ee`，parent report SHA `2f58b36daa5f70b9a0a693af19c2cc3036c5a4519b55a0bb5178900c3608d26a`，base pinned `fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91`。未运行parent完整模型（它会写parent report）、未编译/GPU/git/formal edit，未改旧冻结文件，仅写本两个独立review。

本次发现并由parent修复了一个真实source API前提：半精度 `Element` 仅align2，外置plain arrays最初失去CuTe owning `array_aligned`默认align16保证；`recast<uint32_t>`不自动检查alignment，builder、MMA unpack和A tensor fences都会使用packed32 view。两行arrays现有 `alignas(16)`，仍无initializer。不是实际GPUmisalignment或编译失败记录；是源码必须给足的API对齐条件。原c4 draft已被替代。BK32旧task计数/no raw shared bank两行注释也已修正。

pinned RS traits中 `ALayout_64x16=CLayout_64x16`，FrgTypeA默认Element；真实API链为partition_A(identity64x64)→logical rank3(V8,M1,K4)，make_fragment_A→make_fragment_like<Element>。make_fragment_like把mode0强制LayoutLeft，所以packed uint32 view logical(V4,M1,K4)，u(pair,0,atom)对应packed index pair+4*atom、Element indices2*pair+8*atom。8这个V mode内部可能保留层次shape；此推导不声称编译后的literaltype已经打印验证。独立CPU仿射检查4096 unique A cells与32768 pair alignment/contiguity条件，未实例化C++模板。

pinned half/bfloat默认constructor显式defaulted，uint16storage无in-class initializer；aligned external arrays无initializer，decltype不求值，不构造runtime owningfragment或隐式zero-fill。rmem view仅是API类别，不保证硬件REG。Tensor A operand fence会recast到uint32并对16words逐个+r；gemm rank3 dispatcher升序0..3，mma_unpack每次A需4u32，q需16float，与此布局源码推导一致。

G0预构造[A11,A12,S4]×4K16 atoms，G1预构造[A22,S1,S2,S3]×4atoms，全部stage输入存在a0..a2/a3完整fragment中；源码没有4-word atom loop重用。各required A有issue前和wait0后的tensor fences，后者位于empty release前，下stage重写晚于上stagewait0。G0不读/fence未写a3。source virtual A48/64packedwords与P48/64floats形成96/128寄存器下界；这不是实际physical allocation，LLVM隐藏copy、寄存器合并、local/spill和所有implicit widths仍要PTX/SASS全审。

A派生onecast顺序保留：G0以3RN算S1/S2/S4但只cast S4；G1以3RN算S1/S2/S3并各cast一次。A共6RN/cell对比SS4RN，B仍4RN；BK64 A4096/B2048 cells，ST总RN32768对比24576，为+1/3。总final A/B casts仍16384+8192=24576/CTA/stage；producer只做B8192RN/casts，A移动到consumers。raw slots bitcopy。B helper、paired converter及fixed U/alphaRN/betaFMA经normalized names逐字匹配base。perP wholeK仍升序K16，BK64 stage内0/16/32/48；跨P interleave改变，独立P accumulator未改。

producer rawA4 store32768B、finalB7 store28672B/CTA/stage；消费者rawA4/3两WG read合57344B，计数是source逻辑而非实测指令/事务。双bank operands122880B，float tail57344B只覆盖bank0；barriers预期offset122880/size32在union外，SharedStorage align128预期123008B。source仍先producer_tail与两WGwait0/release→drain6，再Pstores→publish0→fixedU。即使tail只覆bank0，仍排空两个bank。native source仅一个普通kernel launch，BK64 guard p.k%128确保halfK整tile；future Plan集成未完成。

parent模型的literal布局、virtual labels及NumPy语义限定合理，没有把它们称实际CUDA/API/SASS/GPU通过。finite quadruple样本不是4象限Cartesian exhaustive；65536 raw packing验证只是打包代数，未验证CUDA FTZ/NaN payload/转换器。负控可拒绝虚拟pending label复用，但不能证physical A源寄存器已保持到wait0。所有parent evidence bindings已独立hash核对，无mismatch。

source quota64/192/192与初始I152/160/168仅pool假设，final pool57344。expected shared123008B也只支持payload模型至多1CTA；不能由此预测性能。builder暂存、描述符/索引、隐藏copies及producer64下commonU仍需actual完整门。后续还需BF16/FP16真实template/layout、精确PTX/SASS roleCFG与quota/无spill/serialization/A异步源生命期、fences/barriers与object/link身份，然后才能做GPU FP64/alias/SAN/progress及E2E/相对cuBLAS测量。本文不给fit、launch、正确性或加速比结论。

完整细项、counts、resolved findings和19个independent bindings见[JSON](/share/home/liuyao/workspace/MegaFMM/.scratch/psplit_rsA_future35_independent_review.json)。Parent模型额外15个evidence bindings也核对零mismatch。
