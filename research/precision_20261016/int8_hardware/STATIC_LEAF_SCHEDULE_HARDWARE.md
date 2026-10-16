# P56：原生 consumer 七叶外循环展开

P56只把P51的consumer七叶外循环`#pragma unroll 1`改为`unroll 7`。全部其余源码
逐字保持，准备核、producer、内部K循环、数值表示、FP32 RN依赖图和同步规则未改。
[测量前设计](STATIC_LEAF_SCHEDULE_DESIGN.md)、[CPU检查](p56_proposal_cpu.json)、
[隔离源码](p56/)与[源身份](p56_source_manifest.json)在首次候选MMA前冻结。
本文记录本候选自身证据；研究保留不等于生产精度/性能目标已经达成。

## 改动与实际编译

CPU抽象调度672事件逐项相同，含480次MMA、28次供数/commit/等待与已完成槽位释放，
以及七次硬件warpgroup_arrive；输出262144坐标完整覆盖。这是源调度模型，不是实际
异步执行或数值硬件认证。原始FP16/BF16输入、p11/p8原生低部、全部7主/8交叉点积、
完整K512、0,1,5,6,4,2,3叶顺序和显式RN重组保持。

完整编译的consumer两类型均128寄存器（P51为143），Shared196736B，零stack/spill、
无C7515/C7510警告；256线程、128CTA网格，实际仍1CTA/SM。准备核仍BF52/FP48寄存器、
65536B共享/256线程/384CTA，合作式容量实际3CTA/SM×132SM≥384。四49152B操作数槽、
11平面/两描述符、workspace11804032B、两核完整调用不变。run无分配、host同步或
输入缓存，Graph只capture时编码描述符。

[精确二进制静态检查](p51_p56_static.json)显示两种准备核的归一化SASS指令文本与P51
逐条一致。consumer静态指令1600→2048，HGMMA.64静态25→120，BRA静态69→83，
支持编译器确实展开了consumer。静态分支总数反而增加，不能声称静态分支更少，
也不能把静态计数当作执行数量。源码动态点积/480MMA、28供数批次、88TMA/CTA和全网格
138412032B名义载荷保持；编译调度、寄存器分配与代码大小一起变化，不能唯一归因。

## 自身pre-MMA检查与实际矩阵

[布局探针](probe_p56_layout.json)使用128171+491trial+269FP16的新种子，六个合成
配置/全128CTA/28批次/四槽复用，415236096次uint16高低部值比较、1572864输出坐标、
shared/workspace/output guards均零错误。producer与候选逐空白相同。通用读取代替MMA，
因此不是实际WGMMA等待顺序证明。探针读核自身32B stack/0spill，不与candidate
consumer的0stack/0spill混淆；所有布局通过后才首次执行候选MMA。

[标准78项](p56_balanced.json)原始FP64AB门槛和79872inside指数比较通过，无NF，
打印eF逐项与P51相同。最大BF16/FP16 eF1.61320431617e-5/2.16394614392e-5。
未进行全部矩阵输出逐位比较，不由相同打印eF声称逐位等价或普遍精度定理。

## 独立完整调用性能

| 模式 | dtype | P51（μs） | P56（μs） | P56降低 |
|---|---|---:|---:|---:|
| 普通ABBA | BF16 |27.465440274|25.065080166|8.740%|
| 普通ABBA | FP16 |27.511400223|25.104760647|8.748%|
| Graph ABBA | BF16 |27.496039867|25.065160275|8.841%|
| Graph ABBA | FP16 |27.443079472|25.163640022|8.306%|

[普通](p56_abba.json)和[Graph](p56_graph_abba.json)各四个新进程、P51/P56/P56/P51，
每候选每dtype十组完整计时臂，全部保存。每模式312原始门槛/319488inside检查通过，
是78个确定性配置重复，不能算312独立数值输入。准备、描述符与计算完整计入，
未锁时钟/无置信区间；不和P50→P51另一轮收益相加当作直接测量。

## 自身压力与工具验收

自己的stress源和种子在编译前冻结，129173+499trial+271stream+131pattern+277FP16；
14模式包含极端/次正规互补缩放、反向缩放、32K交替及新增8行/列缩放和抵消。
原始舍入输入及FP64 AB不变。普通/Graph比较的Dense为生产tuned Dense，stress沿用
历史cuBLASGemmEx控制，同时强制原始门槛和额外BF16 1e-3/FP16 1e-4固定floor。
双流、独立workspace、输出前后128float及workspace前后256B保护区；每配置交错提交
8次普通/Graph，配置末尾检查输出/guards，非每次中间提交单独检查。

[自身验收](p56_validation.json)完成168个新配置/1344次交错调用，全部原始门槛和额外
固定floor及输出/workspace guards通过，最大BF16/FP16 eF为1.62425188506e-5/2.17198745465e-5。memcheck、
racecheck、synccheck限定kns=int8_research，各56配置/112调用零错误，racecheck
零hazard/零警告。Graph78日志逐字取自自身Graph ABBA_b0，不增加独立样本。
实际合作式容量在每个验收进程查询，不从P51继承工具证书。

## 自身profile与保留范围

[自身两核NCU](profile_p56_stats.json)来自冻结二进制首个balanced BF16完整调用，
每核17次replay。准备17.376μs/52regs/65536B/issue15.219711%，consumer
15.936μs/128regs/196736B/issue23.034603%，两核实测local读写sector为零。
consumer寄存器资源上限2CTA/SM，但shared仍限制1CTA/SM；不能宣称实际并发翻倍。
所有profiler进程计时排除于普通/Graph对照，不能加和或与另一轮P51 profile差分
作为完整延迟的因果拆分，issue比例也不能当作单项耗时百分比。

P56凭自身原始门槛、独立普通/Graph改善及fresh双流/guards/限定范围工具验收，
替代P51作为研究普通和Graph参考。P51及更早完整验收版本保留历史回退。
工作四源码/二进制等于冻结P56；没有生产集成，Auto仍Dense，目标未完成。
[audit_p56_hardware.py](audit_p56_hardware.py)、[p56_hardware.json](p56_hardware.json)
和所有manifest记录确切范围。P54/P55历史恢复manifest保持原样；其当时工作路径
哈希在综合审计中显式对应冻结P51，当前P56身份另行校验，不篡改历史记录。
