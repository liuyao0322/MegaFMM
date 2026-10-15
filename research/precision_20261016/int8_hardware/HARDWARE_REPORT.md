# INT8 一阶补偿：硬件精度通过，性能未通过

本阶段将[数值模型](../int8_inside_full_model.json)落实为 H100 内核：七个 FP16/BF16 主乘积，加八个 signed INT8 一阶补偿乘积，保留 inside 幂二平衡。**P7 在已测输入上通过原始 FP64 精度、Graph、双流及限定范围 sanitizer；完整调用仍约94 μs，约为当前 SW 的9倍，因此拒绝性能验收，不进入生产。精度与性能同时达标的总目标仍在进行。**

后续局部优化检查点：P8将A线程内不变的inside缩放决策移出循环，完整调用77.415/76.751μs；P9进一步将重组的乘积类型分支移出16输出循环，完整调用65.711/65.457μs（BF16/FP16）。两者各72项普通精度通过，打印eF与P7相同。这里的Graph、双流、sanitizer验收仍专指P7，不能转移为P9的结果。P9预处理仍约41μs，因此正在研究[片上融合](../fused_int8/DESIGN.md)；[后续融合硬件报告](../fused_int8/REPORT.md)中F13通过72项普通精度，完整调用约224–240μs，仍慢于P9。P10用粗粒度K128 inside降至约54μs且旧72项通过，但新增逐列交替失衡数值模型失败，不能认作精度可接受的替代。P9仍是本轮最快的保留研究原型。

本报告是[调研与误差消融](../REPORT.md)及[针对性设计](../DESIGN.md)的后续硬件证据。早期报告中“尚无 INT8 硬件实现”的描述是当时状态，以此处的后续结果补充。文献依据及其适用限制见[文献记录](../LITERATURE.md)。没有恢复已经停止的 FP8 内核开发。

## 后续尺度粒度消融：P10不能作为可接受提速

P10将逐列inside改为全局K128 inside：256个连续读取的partial-max CTA，打包CTA内完成归约和因子构造。M6旧42配置数值模型通过，旧72项真实硬件FP64检查也全部通过，打印eF与P9逐项相同；完整调用BF16/FP16为54.794/54.421μs，仍慢于SW10.234/10.021μs。FP16预处理诊断30.257μs，其中inside/A/B分别3.768/13.135/11.697μs。所有步骤均在完整调用内，workspace大小不变。

但此前失衡模式的边界正好在512处，不能检查K128内部的尺度变化。M7新增同一slab中逐列交替的互补2^-12失衡，fresh seeds2701–2703，每dtype3组。只有inside粒度改变的成对数值模型中，K128为0/6通过，最大eF FP16 2.1132e-2、BF16 1.5568e-1；逐列为6/6通过，最大6.7070e-6/5.5459e-5。零饱和、零非有限值。旧输入检查通过不能覆盖这一反例。

真实GPU对照采用不同fresh seeds2801/3072/3343，并重建冻结P9/P10设备路径：P9为6/6通过，P10为0/6通过；最大eF FP16/BF16分别为P9 6.65145e-6/5.55399e-5，P10 2.09349e-2/1.54619e-1。原始FP64 AB和生产Dense依赖gate不变，无非有限值。见[硬件原始记录](alternating_hardware.log)和[源码/二进制身份](alternating_hardware_manifest.json)。工作目录已恢复P9源码及比较器二进制；P10及失败证据冻结保留。下一设计是[保持逐列尺度的连续读取归约](EXACT_INSIDE_DESIGN.md)，先保持P9指数逐项一致，再计入额外归约launch的完整成本。粗尺度P10不进入生产，也不提升为推荐实现。

## 边界和计算契约

- H100、g08、既有 SLURM allocation107488；CUDA12.4编译，CUDA12.8 sanitizer/NCU。GPU时钟未锁定。
- 只验证1024³、连续无转置、alpha=1、beta=0、FP16/BF16输入、FP32输出。不推广为其他尺寸或任意动态范围的证明。
- 参考始终为原始、已经舍入的 A/B 的 FP64 AB。门槛保持 `eF <= 3*eF_dense + floor`，FP16 floor=1e-4、BF16 floor=1e-3，且无非有限值。所有新候选都检查新填充的输入。
- 每次完整调用包括当次输入的 inside reduction、A/B变换与量化、四个tensor-map编码及全部主乘积/补偿/输出。调用内无分配或host同步；使用调用方12066176bytes workspace。
- 七个主乘积保留，补偿额外需要八个整数乘积；不能称为只做七个硬件乘积。每个K128整数部分点积独立恢复尺度，再以FP32累积。
- `solution/`、静态库及生产bench均与研究起点哈希一致。Auto/Dense行为不变。独立研究原型没有注册到生产API。

## 从模型到真实 WGMMA

先单独验证 native `m64n32k32` signed INT8 WGMMA。18组测试覆盖六种模式和三种幂二尺度，与CPU int64点积及恢复尺度结果完全一致。最大单个K128绝对值为 `128*127²=2064512`，未发生INT32溢出。原始数据在[primitive.log](primitive.log)。

完整原型将S/T拆为输入dtype高部和FP32残差，再以分块幂二尺度量化高部/低部。A块64×128，B块128×32；B的INT8数据在预处理中转置成K-major。主乘积与补偿使用同一CTA中的TMA/warpgroup管线，最终在寄存器重组FP32输出；同步等待、proxy fence及workspace均计入实现与计时。

独立GPU测试对正常范围的幂二乘法和一般 `ldexpf` 做了1,048,576例核对，覆盖指数−300..300、normal/subnormal边界和正负零，0错误。极端指数保留一般路径。此项仅证明这次实现替换的等价性，不能代替完整精度测试。

## 逐项消融和性能结果

下表每项都是独立研究候选；全部72项普通精度检查通过，打印的eF与P1一致。时间为完整调用的5个配对组均值的中位数，每组四次轮换测量臂。跨候选来自不同进程、未锁时钟，不作跨候选统计显著性声明。

|候选|单一改动/假设|BF16 μs|FP16 μs|结论|
|---|---|---:|---:|---|
|P1|首个完整INT8实现|157.935|159.656|精度达标，严重性能退化|
|P2|常用范围幂二乘法替换ldexp|159.533|159.497|否定“单独ldexp是主因”|
|P3|对合法无别名的packer参数加restrict|156.876|157.985|无实质改善|
|P4|补偿元数据移入warp寄存器、消除动态表访问|149.670|148.159|计算核栈32→0bytes|
|P5|输出反缩放分支和scale构造移出元素循环|129.031|128.861|降低原型开销，仍不达标|
|P6|变换plane在元素循环前编译期分派|109.135|109.815|A打包加快，B共享内存增加并变慢|
|P7|量化scale检查和构造移出元素循环|94.593|94.380|进一步改善，但仍拒绝性能|

每项都有独立源码快照、编译日志、原始测量臂和SHA256 manifest，见[p1](p1/)、[p7](p7/)及[实验记录](EXPERIMENTS.md)。生产优化迭代编号没有增加；这些不是正式iter371。

P7同进程、相同输入下的完整比较：

|模式与dtype|调优DenseLt μs|当前SW μs|直接输入稠密对照 μs|INT8补偿 μs|
|---|---:|---:|---:|---:|
|普通 BF16|11.949|10.192|6.516|94.593|
|普通 FP16|9.596|10.023|6.625|94.380|
|Graph BF16|11.003|9.839|6.151|91.183|
|Graph FP16|9.160|9.896|6.319|89.846|

普通与Graph各自判定，Graph时间不替代普通调用。原始测量见[p7_balanced.json](p7_balanced.json)、[p7_graph.json](p7_graph.json)。直接输入稠密实现仅作硬件对照，不是FMM修复，也未替换生产。当前SW精度未达标，不能作为合格解；它仍是本次“不影响性能”约束的时间参照。

P7诊断性预处理整体约58.20/57.82μs（BF16/FP16），其中inside约9–10μs、A打包约29–30μs、B约18–19μs。诊断数据已包含在完整调用时间中，不再额外相加；单核独立计时也不能直接等同于更改调度后的贡献。

## 精度、同步与内存验证

|测试|范围|结果|
|---|---|---|
|普通调用|12输入模式×3trial×2dtype=72项|全部通过原始FP64 gate，无非有限值|
|Graph更新输入重放|与普通相同72配置|全部通过；不算额外72个独立数值样本|
|独立扩展双流测试|不同seed，72配置，576次交错普通/Graph调用|全部通过，输出前后128float guards完好|
|memcheck|24配置、48次调用，仅`kns=int8_research`|0错误|
|racecheck|同上|0hazards，0错误，0警告|
|synccheck|同上|0错误|

普通/Graph测试最大eF：BF16 **5.09541715534e-4**，FP16 **6.56734752628e-5**，均低于相应固定floor。普通的六类原始输入外，额外覆盖强消去、K失衡、块尺度、宽BF16输入、指数范围与单位阵。双流测试使用独立seed；不能将跨P1–P7重复相同seed的测试数累加成新的样本数。sanitizer只覆盖明确列出的24配置，不宣称覆盖完整72配置或任意用户输入。

结果见[stress.log](stress.log)、[memcheck.log](memcheck.log)、[racecheck.log](racecheck.log)、[synccheck.log](synccheck.log)。[audit.json](audit.json)明确保存 `accuracy_validated_performance_rejected` 和 `goal_achieved=false`，并核对生产文件与所有候选身份。

## 性能归因的证据边界和下一步

P1的单核NCU显示Tensor elapsed约3.80%、issue active约17.75%，profiled核80.96μs；NCU运行中的普通测量臂不用于性能验收。P3的稀疏CTA trace进一步定位到元数据访问、反缩放和阶段间控制开销。trace增加寄存器、扰动调度，阶段中位数不能随意相加。这些是历史P1/P3证据，不能冒充P7当前profile。

P4/P5直接针对上述归因，P6/P7针对打包循环的重复分支。它们将原型开销降下来，但仍有约84μs完整调用差距，尚无证据证明继续局部优化即可赶上10μs。下一步先检查每个元素重复inside缩放决策，随后评估consumer控制与释放slot的时机。更深的方案需要把变换/量化与叶乘法融合，或改变残差尺度与配对累加策略；这些均须先做数值消融，再做真实硬件完整计时，不能仅凭模型宣称可行。

## 复核

在匹配本报告Git版本和P7源文件的checkout中，串行执行：

```bash
bash scripts/on_gpu.sh bash research/precision_20261016/int8_hardware/build.sh
bash scripts/on_gpu.sh research/precision_20261016/int8_hardware/bin/compare
bash scripts/on_gpu.sh research/precision_20261016/int8_hardware/bin/compare --graph
python3 research/precision_20261016/int8_hardware/generate_stress.py
bash scripts/on_gpu.sh bash research/precision_20261016/int8_hardware/build_stress.sh
bash scripts/on_gpu.sh research/precision_20261016/int8_hardware/bin/stress
python3 research/precision_20261016/int8_hardware/audit.py
```

`validate.sh`保存当时三项sanitizer的精确命令及新内核过滤器，并检查P7源码身份以避免覆盖为后续候选。`bin/`及NCU二进制报告被忽略；首次重建后可将compare复制为compare_p7，再核对本地工具链生成的身份。冻结manifest验证的是归档运行本身，重新编译或重新计时不要求得到相同二进制/日志哈希。快照子目录的相对include按原研究目录设计，不能直接在多一层的pN目录原位编译。正式集成与性能不退化验收仍未完成。
