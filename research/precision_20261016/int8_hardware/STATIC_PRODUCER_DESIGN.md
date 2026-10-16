# P57：仅展开TMA producer的七叶外循环

测量前设计，2026-10-16，基于已独立验收的P56。P56只展开consumer并获得约8%完整
收益；producer仍动态枚举七叶。这轮只在其外循环前增加`#pragma unroll 7`，消费端
及所有其他源码逐字保持。P25/P26是准备核unroll对照，非本producer控制。

展开请求固定producer plane/低部条件及交易字节；是否实际展开须看机器码。
原循环是两个相邻for，不增加内部K pragma或改变循环边界。保持七叶顺序
0,1,5,6,4,2,3、每叶完整四K128、全部7主/8交叉点积、wait/free相位、ready事务
字节、四49152B槽/Shared196736B及消费端全部编译/硬件fence。七原生高部和四缩放
低部、p11/p8、原始精确逐列inside及显式FP32 RN数值图不变。

每CTA名义载荷1081344B、88TMA命令、28供数批次、源码480MMA不变。grid128/256线程，
准备384CTA/256线程/65536B、workspace11804032B、两核/两描述符及完整调用成本不变。
run无分配、host同步或输入缓存，Graph仅capture编码描述符。原FP64AB与两种输入
及原始门槛保持。更多代码/寄存器/指令缓存压力可能回归，不从源码提前推断收益。

先冻结CPU operand-plane/事务字节/槽相位序列及源身份，然后隔离生成p57；工作仍P56。
自己的新种子全网格TMA读取探针131179+503trial+281FP16，六配置，全部28批次/四槽
及高低部/shared-workspace-output guards通过后才运行MMA。通用读取不证明真实MMA
等待顺序，原P56的工具证书不转移。实际编译资源与合作式容量必须检查。

原始78项FP64门槛/inside与独立普通/Graph P56/P57 ABBA含全部准备、描述符、计算。
若改善则自身新种子双流独立workspace/普通Graph交错/guards和限定范围三工具；若
退步则冻结全部臂，逐字从保留提交恢复P56，并新进程78复核。自身NCU仅诊断，
所有profiler计时排除于ABBA。生产/Auto/正式iter计数不变，目标继续。
