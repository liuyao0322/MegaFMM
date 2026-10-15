# 片上自适应 INT8 补偿原型

本设计从已测P9的约65μs完整调用出发。其约41μs全局预处理本身超过当前SW约10μs完整调用，因此需要改变物化和调度方式。目标是保留七个native主乘积、八个INT8一阶交叉项，将动态输入的变换、量化、尺度选择和重组放进同一个CTA。此文描述研究原型，不是已通过性能验收的方案。

## 先做的数值消融

- F1：将全局per-column inside缩放改为每个输出CTA、每个K128块、两份K半区独立的局部幂二缩放；INT8块改为A32×128、B128×64。FP32叶模型42/42低于固定floor。
- F2：进一步按K128、P0/P1/P5/P6/P4/P2/P3及其即时补偿顺序直接累加四个输出块，避免保留全部中间叶。42/42低于固定floor，最大FP16/BF16 eF约7.28e-5/5.30e-4。该模型没有逐位模拟Hopper累加顺序。
- F3：再令低部尺度固定为高部尺度除以2^10(FP16)或2^7(BF16)，以便成对整数项共用反缩放。三组FP16强消去输入超过固定floor，最大1.20e-4；不采用这条简化。该失败是保守固定floor筛选结果，未声称是实际Dense依赖gate的硬件失败。
- F4：首次成对实现未编译，原始源码和编译错误保存于`f4_compile_failed/`，无计时或精度结果。
- F5及后续：回到F2支持的自适应高/低尺度和分别反缩放，实现下述新管线。F5编译问题、F6/F7硬件精度失败和布局归因见[实验记录](EXPERIMENTS.md)。数据来源在`../int8_hardware/tile_inside_*`，没有恢复旧SW架构或FP8路线。

## 一个CTA中的数据和角色

每个CTA负责原始四个512²输出象限各自32行×64列的tile。grid16×8，256线程：0–127是CUDA producer，128–255是Tensor Core consumer；producer内部由thread0发TMA。每次调用只编码原始A/B的两个tensor maps，启动一个kernel，无全局中间操作数、无host分配/同步。比较器的128byte workspace只是接口兼容占位，kernel不使用。

共享内存布局：

|区域|大小|用途|
|---|---:|---|
|四份A原始tile|4×8192bytes|四象限各32×128输入dtype|
|四份B原始tile|4×16384bytes|四象限各128×64输入dtype|
|两个叶slot|各49152bytes数据及16bytes尺度，128byte对齐|native A/B高部和INT8 A/B高/低部|
|barrier、reduction scratch和inside指数|其余对齐空间|raw ready/free，slot ready/free，producer归约|

slot和动态shared按1024bytes对齐，保持所用CuTe描述符的零swizzle base offset假设。generic元素访问需将带smem pointer flag的布局转成元素域索引，不能直接将GMMA布局算出的offset加到T指针。共享总量由`sizeof(Shared)`定义，并在编译时要求小于H100上限。A native为K-major SW128，B转置视图为MN-major SW128；两份INT8操作数均K-major SW128。原始A的TMA使用split-K维编码，B直接加载连续N轴。

## 每个K128块的协议

1. producer thread0在前一raw-free阶段完成后，发8次TMA加载四象限A/B，总98304bytes。raw-ready为transaction barrier。
2. producer128线程等待raw-ready，从本CTA局部四象限选择两份inside指数。每个幂二D同时作用于对应A列组和B行组，保持该tile的精确AB恒等式；并非修改FP64参考。
3. consumer同时从原始raw共享数据计算P0/P1，等待WGMMA全部完成后，由四个warp leader到达raw-free。producer在完成当前五个叶打包之前也不会重写raw。
4. producer顺序打包P5/P6/P4/P2/P3，复用两个slot。先算变换、输入dtype cast和高/低部amax，再选独立幂二INT8尺度，第二遍重算变换并量化。重算避免同时保留两侧所有元素造成过大寄存器压力。所有producer执行async-proxy fence和128线程named barrier，再由thread0发布slot-ready。
5. consumer等待各slot-ready，执行native主乘积并立即重组，再分别执行对应1或2个INT8点积、恢复各自尺度并重组。每个K128点积绝对值≤2064512，INT32可精确保存，FP32可精确转换。所有读取完成后四个consumer warp leader释放slot。
6. 四个K块完成后，consumer直接写四个FP32输出象限。原始FP64验收、fresh输入、Graph/双流guards与完整计时继续决定候选是否可接受。

producer使用named barrier1、计数128；全CTA barrier0只用于初始化。WGMMA组等待、slot free和raw free均保留，不能以单个warp完成替代整个consumer组读取完成。

## 预期收益与待验证项

预期去除约11.5MiB全局高部/INT8中间数据物化，以及独立inside/A-pack/B-pack三次kernel边界。这不意味着延迟能降至10μs：每CTA重复局部缩放/量化、两次变换计算、shared带宽、寄存器压力和每K的同步都可能抵消收益。先通过完整数值检查，再按完整调用计时；不能把片上计算量或理论Tensor Core吞吐当作速度证据。

此原型的累加顺序和量化粒度与P7/P9不同，因此P7的sanitizer、Graph、双流和精度结果均不能转移。需要独立验证，也不能只运行uniform quick检查就宣传精度通过。生产`solution/`与库保持不变。
