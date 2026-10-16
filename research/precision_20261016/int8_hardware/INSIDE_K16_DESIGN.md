# P61测量前方案：精确inside归约从128CTA×8列改为64CTA×16列

P60成对原生打包已经自身逐位验证与普通/Graph独立测量，未达3%且FP16普通轻微
回退，不升级P56。此次独立从保留P56出发，只改变inside最大值归约的线程映射；
不组合P60打包改动。早期NEXT_EXPERIMENTS.md在P30后提出64CTA/K16但未实现；
现在结合P60自身准备核profile继续检验，而非把剩余成本已归因于该归约。

384合作CTA/256线程、65536B申请、一个无条件grid.sync、全部原P56原生打包和
consumer/compare/pow2保持。仅前64CTA各计算16个完整K列的A最大值/B行最大值。

A：k0=block*16，k=k0+tid%16；row=tid/16+16*j，j=0..63。warp里xor16合并
两行，lane<16把每warp的16个最大值写到ma[warp*16+lane]。跨8warp按原顺序
fmaxf组合，得到每K完整1024行的最大值。fabs和有限值最大值保证归约分组不改变
结果；仍须自己的GPU逐列inside逐位对照，不能只凭结合律转移验收。

B：每warp处理kr=warp与warp+8两行；各lane仍遍历col=lane+32*j，j=0..31；
warp内保持原xor16/8/4/2/1最大值归约，lane0写mb[kr]。block barrier后tid<16
写原log2f/float2int_rn表达式，零值分支不变。scratch为ma128float+mb16float，
在原65536B内，grid barrier完成后才被原打包复用。所有384CTA参与同一barrier。

源码层读取元素数不变。按自然32B地址集合的简化模型，A每warp每次从4个半满
sector地址变为2个满sector地址，总sector地址访问131072→65536，独特sector仍
65536；这不是实际cache/DRAM事务。B模型sector访问不变。代价是归约活跃CTA
128→64、每线程A读取32→64、每warpB行1→2；降低并行度可能完全抵消访问粒度收益。
没有声称当前访存是瓶颈，也没有速度或物理下界结论。

先CPU覆盖原始A/B各1048576位置恰一次、warp/block归约集合、scratch边界与全K
指数覆盖；再隔离实现，自己的新种子原始输入→inside/全部11平面逐位对照与guards，
包括BF16极端fallback和2B输入。实际资源和合作容量必须重新查询。首次MMA前
通过后，原78门槛、全成本独立普通/Graph ABBA；保留需自己的新种子双流/工具，
失败则冻结并逐字恢复P56另跑78。此文只方案，没有C++、GPU或性能验收。
