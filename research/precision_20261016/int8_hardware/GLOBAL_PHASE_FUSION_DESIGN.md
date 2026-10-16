# P59：保留全局高低部平面的单核阶段融合

2026-10-16测量前设计，P56为研究参考，P58为未升级的128CTA准备对照。
P58四个完整调用点估计改善0.37%–1.38%，没有3%进展；自身全部平面逐位控制通过。
这只能支持继续测试128CTA，不构成P59的精度、同步或性能证书。

平衡路径以128CTA/256线程的合作式fused_global内核替代P58的两核：

1. 逐字P58的prepare_resident阶段：所有CTA各算八列精确inside，在一次无条件
   grid.sync后，每CTA依次打包B tile b/b+128、A tile b。保留全部11平面、每个
   输入四元组仅变换一次、每个打包线程已有的三个fence.proxy.async.global。
2. 所有线程第二次无条件grid.sync，随后复用同一shared空间，初始化P56的
   ready/free屏障并__syncthreads。任何TMA不得在本次grid同步前发起。
3. 将线性CTA b映射至原P56的m0=(b/16)*64、n0=(b%16)*32，逐字P56 producer
   与consume，全部七主项/八交叉点积、完整K512/四K128、RN图、wait/free相位、
   warpgroup硬件及编译fence保持。

共享申请为max(65536,196736)=196736B，不增加workspace11804032B或隐藏中间结果。
每次run先为当前工作指针编码两tensor maps，再合作启动一次；描述符只依赖地址，
输入数据的generic写与async读取按上述逐写线程proxy fence和grid barrier排序。
映射保持原布局，CPU检查全部262144输出坐标、阶段顺序和槽界限。init查询实际
编译fused_global的合作支持/SM数/active容量，要求128<=active*SM；generic读取
探针额外guards的197888B申请也单独查容量。不能从P58的三CTA容量推断P59容量。

把P58预处理body抽取为forceinline设备helper，表达式及同步逐字保持，原独立
preprocess包装仍存在于比较器的diagnostics-only组件。普通平衡完整计时仅实际
fused_global一次启动，所有实际阶段和输出均计入。Graph仅capture编码描述符，
不缓存输入派生数据；run无分配/host同步/额外stream。非平衡历史诊断仍用原两核，
不得把该分支的证书或耗时当作新融合结果。

首次MMA前，自己的六组新种子原始输入先用冻结P56生成独立expected全局高低部。
合作探针按逐字候选helper、第二grid.sync及producer执行，经TMA在四槽读取全部
28批次/128CTA，并与expected比较原生位。包含普通高缩放、抵消、极端32列缩放
三模式，每类型三组；逐列inside、shared/workspace/coverage guards检查。通用
读取没有WGMMA，不证明真实累加器等待、矩阵门槛或工具同步安全。另检查实际
fused_global编译资源/stack-spill/警告。所有自身探针与源身份冻结后才允许MMA。

完整标准78原始FP64AB门槛及inside；独立普通/Graph P56→P59保留参考比较，以及
P58→P59隔离融合的比较，全部准备/描述符/计算/输出成本计入。若有值得保留的
收益，自己的新种子双流/独立workspace/普通Graph交错/guards与限定三工具必需。
否则冻结失败臂、逐字恢复P56并新进程78复核。自身NCU仅诊断，计时排除且不加和。

N1–N3的片上raw变换按输出瓦片重复，改变K粒度与数值重组；P59保留全局平面、
一次输入变换和原K512数值图，不重试同一实现。不过更多共享申请、prep并行度、
合作屏障和编译代码资源可能抵消启动减少，不预断收益或物理下界。P57/P58是
P56之后连续两轮无3%进展；若P59也无3%，再下一候选前须自身profile/历史/新调研。

依据本轮查读的[CUDA12.4网格同步与实际容量](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#grid-synchronization)
和[PTX8.4 generic/async代理fence](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
固定同步约束。这里是方案推导，不是NVIDIA对本内核的验证或性能结论。
