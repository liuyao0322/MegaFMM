# 片上融合检查点：精度通过，性能拒绝

**F8–F11/F13分别通过72项普通原始输入FP64精度检查；逐项优化将完整调用从约0.7ms降至F13约224–240μs，仍慢于独立预处理P9约65μs和当前生产SW约10μs。片上融合的这份实现没有性能收益，不进入生产，整体精度与性能目标仍未完成。** 当前最快的保留研究原型是[独立预处理P13（约55μs）](../int8_hardware/HARDWARE_REPORT.md)。

本阶段按[设计](DESIGN.md)和[逐项实验记录](EXPERIMENTS.md)推进：先做局部缩放/累加顺序模型，再实现新管线，逐项定位共享内存布局错误，最后以真实硬件输出和完整调用计时判定。原始文献和误差归因见[父报告](../REPORT.md)及[文献记录](../LITERATURE.md)。这不是生产优化迭代，未恢复旧SW架构或被停止的FP8开发。

## 数值模型与表示消融

同一组7输入模式、3seeds1701–1703、2dtypes，共42配置，FP64参考始终是原始已经舍入的AB。不同消融复用同组配置，不能累加为新的独立样本。

|消融|固定floor通过|FP16最大eF|BF16最大eF|
|---|---:|---:|---:|
|F1：CTA局部K128 inside与量化尺度|42/42|见原始JSON|见原始JSON|
|F2：再交错K、主乘积与对应补偿累加|42/42|7.2822e-5|5.3049e-4|
|F3：固定低部指数=高部−10/−7，成对累加|39/42|1.2030e-4|9.3263e-4|
|M4：固定低部指数再细一位，为高部−11/−8|42/42|7.2822e-5|5.3049e-4|
|M5：原始象限幅值上界选择高/低尺度|36/42|2.8523e-4|2.1262e-3|
|M6：全局K128 inside、自适应量化，P9 tile|42/42|7.3305e-5|5.3056e-4|

F3三组FP16强消去超固定floor，作为失败筛选保留；未将其冒充实际Dense依赖硬件gate失败。M4未发生本轮低部饱和，支持进一步开发配对整数补偿，但尚无硬件实现，也无任意输入范围的保证。F8/F9采用F2的独立自适应高/低部尺度；M4数值成功不能转移为其性能证据。原始数据在`../int8_hardware/tile_inside_*.json`，冻结源码和身份在F1、F2/F3、M4 manifests。

M5六组失败均为强消去，零高/低部饱和；幅值上界过松使小变换高部量化为零，必要交叉补偿随之消失。CPU常量块构造进一步给出FP16 eF4.6051e-4，而同一模型自适应量化为零误差；这不属于真实GPU验收。M6改变了inside粒度和量化tile，相较F1/F2不能只归因为一个数值因素；其目的为筛选独立预处理P10设计。后续新增K128内部逐列交替失衡揭示模型覆盖缺口：真实GPU P9逐列缩放6/6通过、P10粗缩放0/6通过，见[独立预处理后续](../int8_hardware/HARDWARE_REPORT.md)。P10虽在旧72项中通过且约54μs，也不能取代P9。

## 硬件结果与错误定位

|候选|普通FP64精度|BF16 μs|FP16 μs|判定|
|---|---:|---:|---:|---|
|F6首个可执行融合版本|6/72通过|731.496|724.787|实现错误，计时仅诊断|
|F7修复slot swizzle对齐|12/72通过|732.551|724.294|仍有布局解释错误|
|F8正确转换flagged布局的元素索引|72/72通过|731.992|720.103|精度通过，性能拒绝|
|F9反缩放分支移出16输出循环|72/72通过|715.255|701.239|精度通过，性能仍拒绝|
|F10 inside尺度构造移出逐元素变换|72/72通过|395.751|379.155|精度通过，性能仍拒绝|
|F11量化尺度构造移出逐元素循环|72/72通过|245.429|228.806|精度通过，性能仍拒绝|
|F12缓存残差并完整展开|未测，编译主动中止|—|—|无精度/性能结论|
|F13只改B打包线程映射，起自F11|72/72通过|239.515|224.255|精度通过，性能仍拒绝|

F9同期当前SW约10.182/10.004μs。以上完整调用包括原始输入tensor-map编码、局部缩放、变换、量化、所有七个主乘积及八个补偿、重组输出。普通事件计时取5个配对组均值中位数，保留所有四方法轮换测量臂，100次调用/臂、20次warmup。GPU时钟未锁定，跨候选不是三进程配对显著性结果；显著退化的候选没有执行独立性能胜出认证。

F8–F11/F13打印的eF相同：BF16最大5.09385609333e-4、FP16最大6.31255707396e-5，低于对应固定floor1e-3/1e-4，也通过实际原始gate。范围仅H100、连续无转置1024³、alpha1/beta0、FP16/BF16→FP32，12模式×3trial×2dtypes。调用无分配/host同步，使用199680bytes左右的共享数据（以源码sizeof为准）；没有全局变换workspace，比较器的128byte占位workspace不参与kernel。

两个实现错误分别是：

1. 在slot尾加入尺度后，slot stride仅对齐128bytes，使下一slot的base offset违背所用CuTe SW128描述符的零offset假设；改为1024byte对齐。
2. `GMMA::Layout_*_Atom<T>`带有`smem_ptr_flag_bits`，其swizzle随`make_tensor`移入字节地址pointer。直接把该布局算出的值作为T数组元素索引，缺少类型宽度转换。改用`as_position_independent_swizzle_layout`。单叶P5诊断两种dtype的原始TMA数据、高部、指数、量化字节和4096项整数补偿点积均从大量不符降为0不符；native点积eF约1.86e-7/5.37e-8。该probe只验证一个K128叶，完整72项检查另行运行。

依据是本仓库CUTLASS `pointer_flagged.hpp`、`mma_traits_sm90_gmma.hpp`，以及[NVIDIA异步拷贝文档](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/04-special-topics/async-copies.html)的swizzle周期/base offset说明。不能把这里的1024byte约束说成所有TMA的最低对齐要求。

## 性能证据与剩余工作

单独profile driver只跑首次quick检查，没有计时臂。首个BF16核NCU18passes：933.47μs、DRAM吞吐0.14%、compute16.97%、130997494条执行warp指令，和普通完整调用时间分开报告。Warp分析提示L1TEX dependency等待占平均发射间隔约44.4%；没有足够证据将其直接归结为DRAM瓶颈或某条源语句。

静态BF16 SASS13708指令、980条MUFU.EX2和1369条branch，无LDL/STL/LDG；一般指数路径带来很大的指令体积，但静态出现不等于正常输入实际执行。两个role等待对方时的polling也会增加动态指令数。截至F13的已测普通融合核均124regs、0stack/spill；两份带时间戳的诊断核均122regs、0stack/spill。不能根据时间长就声称有寄存器spill。

F9角色时间戳诊断已分离等待与工作：每个CTA先合计20个叶阶段再取CTA中位数，producer打包687.072μs、consumer等待slot685.344μs；同一CTA的全部native计算合计中位数7.472μs，全部INT8补偿合计中位数15.968μs。producer/consumer时间重叠，不能相加。带时间戳核716.614μs，仅一个新BF16均匀输入通过原始FP64检查，不是完整性能验收。

F10把同一叶的两组inside指数和幂二因子移到循环外，F11再把四组量化尺度判断/因子构造移到量化循环外；一般指数均保留原路径，精度打印值逐项不变。相较F9，完整调用成本明显下降，但尚不足以改变性能拒绝。

F11进一步拆分打包：native A/B生成分别42.784/51.936μs、最大值归约2.688μs、指数选择5.024μs、A/B量化48.192/68.800μs、发布同步1.120μs；总pack中位数220.608μs，外层pack区间225.760μs。各子阶段中位数不必相加等于总中位数。带时间戳核255.563μs，仅同一诊断范围的BF16均匀输入通过FP64检查。量化第二遍仍重读raw tile并重算变换残差，据此设计F12缓存第一遍残差；其编译主动中止，尚无寄存器/溢出和完整调用结果，见下文。不能直接认定片上融合必然达到10μs。

F12完整展开96个残差/producer线程后生成约18MB PTX，编译超过15分钟仍在ptxas优化，主动中止并冻结源码与决定。不能推断有spill、精度失败或任何运行时间。F13回到F11，只把B pack两遍的线程映射改为N16/K8子tile。CPU CuTe枚举确认覆盖和唯一性，源级INT8 B每bank的不同32bit字数从四降为一；实际完整调用仅小幅变化，不把枚举冒充GPU bank-stall测量。

本目录没有Graph、双流、guards或sanitizer验收。先前这些验证只属于独立预处理P7，见[明确范围的审计](../int8_hardware/audit.json)。生产source、静态库和bench哈希均未变。`Auto/Dense`仍使用原调优cuBLASLt；独立稠密6.5μs对照仍只用于比较，不是FMM修复。

## 复核与证据身份

```bash
bash scripts/on_gpu.sh bash research/precision_20261016/fused_int8/build.sh
bash scripts/on_gpu.sh research/precision_20261016/fused_int8/bin/compare
bash scripts/on_gpu.sh bash research/precision_20261016/fused_int8/build_probe.sh
bash scripts/on_gpu.sh research/precision_20261016/fused_int8/bin/probe
python3 research/precision_20261016/fused_int8/audit.py
```

F6–F11/F13源码、编译和原始运行日志分别冻结于同名目录，manifest还保存本地忽略的二进制身份。probe/profile以及F9/F11时间戳诊断源文件和结果有独立manifest。快照目录多一层相对include，不应原位直接编译；重现某候选应检出匹配Git版本。audit复核的是归档运行，重新编译/计时不保证得到相同二进制或日志哈希。NCU原始binary report保留在本地忽略的`ncu/`，文本metric表和SASS是去除行末空白的可读导出。生产集成、不退化性能与最终验收仍未完成。
