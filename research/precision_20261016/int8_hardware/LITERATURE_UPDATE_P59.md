# P57–P59三轮无3%进展后的重新调研

2026-10-16。已先读自身P59 profile及P56/P57/P58/P59全部硬件记录、P35/P36整数
重定量化、P46高部复用、P47直接原生写、P49四元组复用、P50/P51打包几何和
N1–N3重复raw变换的历史，再新检索并查读以下原始来源。P56仍保留，生产不改。

P59单核融合对P56普通收益约0.10%，对独立P58准备对照普通/Graph皆退步，不能
把剩余耗时解释为一个可免费消除的内核启动。自身17pass fused profile33.280μs/
114regs/issue14.633669%、零local读写，实际Shared仍196736B/一CTA容量。
P56两核profile17.376/15.936μs不是与它同期配对的可相加耗时；DRAM6.273280/
12.130816MB与P59的7.500032MB也不能直接换算ABBA或物理下界。P59源码480MMA/
名义138412032B TMA不变，没有达到物理下界证据。

1. [NVIDIA向量化内存访问文章](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
   （2025-08-04更新，本轮重新查读正文）说明宽访问可减少访存指令，要求实际地址
   对齐，且寄存器压力/并行度可能抵消收益。文章用copy内核，不能据此预报本原生
   FMM准备提速。P36也读过此文，但它是已有INT8平面重定量化，非本次原生平面写。
2. [CUDA12.4大小与对齐规则](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#device-memory-accesses)
   区分每线程访问宽度和warp内存事务；合法单指令访问要求自然对齐。新方案须
   证明每个plane/row/quad offset对齐，不能让原始2B对齐输入出现未对齐32bit访问。
3. [NVIDIA CUTLASS v3.5.1 numeric_conversion源码](https://raw.githubusercontent.com/NVIDIA/cutlass/v3.5.1/include/cutlass/numeric_conversion.h)
   包含FP32两元素到FP16的RN half2专用转换，以及sm80以上到BF16的RN bf16x2转换。
   本地version.h同为3.5.1，已读本地对应实现；本地文件另有插入，线上行号不作为
   本地位置。这证明有明确RN配对转换工具，不能替代极端/次正规值逐位比较。

代码特定推导：P56实际准备SASS已使用部分配对转换，但仍有176条静态STG.E.U16，
两类型相同，另有74条LDG.E.U16.CONSTANT。这是静态指令数量，不是动态次数、
瓶颈占比或“完全未向量化”的结论。P59独立准备静态264条U16 stores，体现其
展开的三个逻辑任务；与实际fused SASS不能混算。减少原生平面的逐元素16bit写
是P47/P49/P51之后尚未独立测量的轴，需在P56原384CTA几何上隔离测试。

下一P60候选仅把相邻两列作为一组读写，所有变换与低部残差仍按各元素FP32 RN，
转换与平面存储配对；保留7高/4低、两种输入、完整consumer、inside、原始门槛和
完整成本。不能用half2算变换或省略八项修正。输入4B不对齐时用原16bit标量读取
fallback，不加强既有输入对齐条件。输出平面仍满足原TMA要求的对齐。
更多live values/指数读取、不同warp索引和代码体积可能回归。先CPU覆盖/对齐/
factor映射、自己的标量转换与全平面位对照/guards，再完整硬件/性能/streams/tools。
这是新待测推导，无P60源码/编译/设备结果，不作性能或普遍精度承诺。
