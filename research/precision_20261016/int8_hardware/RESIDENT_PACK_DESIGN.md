# P58：128个实际合作CTA承接全部384个打包任务

2026-10-16测量前方案。P56是完整验收研究参考；P57 producer展开已拒绝并从
14d9720逐字恢复。先隔离打包并行度，再考虑全局平面准备/计算的单核融合。

P58只改变合作式预处理的任务分配：grid384→128，256线程与65536B动态共享申请
保持。128CTA全部执行原有每CTA八列的精确inside计算和无条件grid.sync；随后
每CTA依次打包B的tile=blockIdx.x、blockIdx.x+128及A的tile=blockIdx.x。显式
传入逻辑tile给pack_pair，数学/原始读坐标/高低部写坐标/类型转换/快速缩放与
ldexpfallback规则逐字保持。全部B256/A128逻辑任务各执行一次，不重复变换。
原pack_joint及独立pack诊断继续用blockIdx.x作为逻辑tile，不改变其启动范围。
无需新增grid barrier；每个打包线程保留原fence.proxy.async.global。

consumer/pow2/compare逐字P56，全部7主/8交叉点积、完整K512/四K128、七叶顺序、
RN图和等待/释放fence保持。workspace11804032B、两描述符、两内核及普通/Graph
完整成本不变；run无分配、主机同步或输入缓存，Graph仅capture编码描述符。
BF16/FP16原始舍入输入FP64AB和3*dense+floor门槛不变。无平衡诊断路径保持384。

CPU独立映射验证B256/A128任务集合、每边262144坐标各恰一次、inside1024列覆盖及
快速缩放索引。先隔离生成，不修改当前P56。自己的新种子54组九模式全部11平面
逐位比较及inside，以及18组有限极端值/下溢/32K值边界打包对照必须先于首次MMA。
实际编译寄存器、栈/spill及合作容量必须检查；不能沿用P56证书。消费者逐字相同
不重复TMA布局研究，但完整标准78原始门槛和独立普通/GraphABBA仍必需。
若改进则自身双流/独立workspace/guards/三工具；否则冻结全部臂、逐字恢复P56并
新进程78复核。自身NCU仅诊断，计时排除且不相加。无物理下界或收益预断。

后续P59才考虑128CTA在发布全部全局平面后执行第二次无条件grid.sync，再进入
P56消费端。这将Shared提高到196736B并改变编译资源，是独立实验；P58较低容量
代价不等于融合最终代价。N1–N3按输出瓦片重复原始变换，与此一次生成全局平面
方案不同；其精度/工具证书不能转移。

本次重新查读[NVIDIA CUDA12.4 grid同步与容量](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#grid-synchronization)
说明合作式启动需要实际容量容纳整个网格，跨阶段可用网格同步；
[PTX8.4 proxy fence](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
要求generic/async代理间显式有序。这些支持同步设计约束，不预测速度；P58本身
仍由内核完成边界连接计算，未实现P59的网格内发布阶段。
