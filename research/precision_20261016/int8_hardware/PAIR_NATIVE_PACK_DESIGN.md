# P60测量前方案：相邻两列的原生平面配对读写

P56仍是完整验收普通/Graph研究参考。P57–P59三轮无3%后自身profile/历史和新原始
来源复核见LITERATURE_UPDATE_P59.md。P60隔离原生准备的访存指令粒度，保留P56
384合作CTA、256线程、65536B申请、A16×128/B32×32瓦片、精确inside/grid barrier。
consumer/compare/pow2逐字P56；workspace11804032B、两核两描述符与全部计时不变。

一个线程处理相邻两列：i=2*(tid+256*j)，r=r0+i/Cols，c=c0+i%Cols；A每线程四
pair，B两pair。每pair读取四个象限的原始两值。输入基址4B对齐时用32bit字；
若只有2B对齐，逐元素16bit读取fallback，保持原输入契约。所有input象限偏移、
row stride、tile c0、pair c及全部plane输出字偏移须独立证明4B对齐。输出TMA
工作平面的原对齐约束保持，不增加API限制或主机同步/分配/隐藏缓存。

每个分量独立执行P56的FP32 __fmul_rn/__fadd_rn/__fsub_rn与general pow2 fallback，
不可改成half2/bfloat162算术。高部按T(v)的RN成对转换；原四个低部依然为
T(RN(RN(v-float(high))*2^p))，p11 FP16/p8 BF16。可用本地CUTLASS3.5.1的两值RN
转换器，必须先自己的GPU scalar oracle逐位比较高/低，覆盖两分量不同inside、
有限极端/次正规/零及边界。只把两原生值写为一个32bit字，不改变平面、数值图。

A的两个列分量使用各自inside[c]/inside[c+1]及各自+512指数。B同一pair共用行
指数；为保留warp广播，预取pos=r0+2*warp+(lane%2)+16*((lane/2)%2)，在j0/1时
从srcLane=(lane/16)+2*j广播。此lane的pos必须恰为r0+tid/16+16*j。全部warp
参与normal guard和shuffle，fallback仍逐分量原函数，不能读未初始化值。

源码层原生平面写5767168个16bit值→2883584个32bit字，字节数11534336不变。
对齐的raw打包读2097152个16bit值→1048576个32bit字，读字节不变；inside reduction
仍遍历原始A/B，不在此减半计数内。不是物理事务、动态机器指令或速度证明。

先冻结CPU每边262144元素各恰一次、4B字边界/quad offsets和B广播factor映射，
隔离生成；自己的两分量转换/残差标量oracle、54组新种子全部11平面/inside及
18组极端缩放对照，另有原始A/B偏移2B的打包fallback检查，必须在首次MMA前通过。
独立实际编译资源/合作容量与SASS核查，不能只看源码请求或沿用P56证书。

然后原78项FP64门槛、inside、独立普通/Graph P56/P60 ABBA，全成本全臂；若保留
则自己的新种子双流独立workspace/普通Graph交错/guards及限定三工具。否则冻结
失败臂、逐字恢复P56并新进程78复核。自身profile仅诊断，计时不作性能样本。
此处仅方案/CPU检查，无C++、二进制、GPU结果或速度结论；目标仍在进行。
