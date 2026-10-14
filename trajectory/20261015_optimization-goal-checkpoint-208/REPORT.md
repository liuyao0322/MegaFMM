# 持续优化检查点：iter165–208，目标仍在进行

当前 `solution/` 已通过 `git checkout d8e6261 -- solution/` 原样恢复 iter162，恢复与复测为 iter208。**尚未达到两种 dtype 都超过 Dense 的目标，也没有通过原始 FP64 精度门槛。** 不将本检查点视为完成或暂停目标；不声称达到物理极限或穷尽方案。

## 当前完整调用结果

同一 H10080GB / allocation107488，CUDA12.4；普通调用、1024³、完整动态输入变换，20warmup、5组×100次、3轮fresh-input正确性。每轮独立调优 Dense，同期交错计时；未缩减用户评测。

|dtype|SW μs|Dense μs|Dense/SW|原始eF|gate|验收|
|---|---:|---:|---:|---:|---:|---|
|bf16|11.325440|12.299840|1.086036|0.00601519652935|0.00100255679669|FAIL|
|fp16|11.520320|9.924800|0.861504|0.000754625769821|0.000103752760587|FAIL|

BF16 普通调用快约8.60%；FP16仍慢约16.08%，需从当前SW时间减少约1.596μs（13.85%）。Auto继续Dense，无精度合格SW winner。

Graph并未在208重测：此前162配对证据为SW BF16/FP16=10.95920/11.16416μs，Dense=11.26336/9.50496μs，模式不可混比。原独立72项逐位97检查、3种quick sanitizer和256次双stream/Graph验证适用于恢复的162源码；不借此认证后续被拒候选。

## 本轮候选与结论

- 165–175：多laneTMA、固定product helpers、3Pslots、trywait快路径、N16布局、稀疏free到达、tensor-map地址替换，均未建立胜出。171/172编译失败，173修正为N16/SW32后约16.5μs。
- 176–178：N64分配叶乘法、全局U2/U3交接和3/4产品平衡，约14.8–17.7μs，不能采用。
- 179–185：PTX描述符、常量slot、减少WGMMA fence、分A/B生产者、初始同步拆分、L2promotion，未优于162。
- 186–189：224CTA并行叶乘法，全局P+最后CTA重组。186有真实spill；187去spill仍约32μs；188行优先合并访存改善至约21μs；189等待的TC参与pack改善至约19μs，仍慢。
- 190–193：直接global atomic / TMA reduce。190改变FP32重组求和顺序且非确定，约30μs；191共享指针对齐错误而无有效结果；192修正至约15.7–15.9μs；193复用operand存储与完整tile TMA reduction仍约15–16μs。无采用，精度验收未放宽。
- 194：当前异步前端配float2 P交换，11.33/11.54μs，无可靠收益。
- 195/196：N64、64CTA、3operand slots、单Pslot。195 ready到达计数错误而停住，终止并记录；196修正后14.10/14.30μs。
- 197/198：双TC warpgroup配异步pack；12.08/12.25μs，恢复608pack后11.55/11.82μs，未胜出。
- 199/200/201：七叶+一重组CTA的cluster8 DSM。199启动容量不足；200显式查询得到最多15clusters而需16；201改15持久clusters并由一个处理额外tile后可运行，但约34μs。201 NCU LaunchFailed，nan计数一律排除。
- 202/203/204：完整PTX私有累加器直接寄存器重组。202消除C7515警告但约18.5μs；203将分支提升出片段循环后约13.1μs；204整个TC静态PTX链至11.32/11.63μs，尚未建立优于162的证据。
- 205：双私有累加器、wait1重叠重组，无spill但11.61/11.90μs，不能将更低指令数等同于更低延迟。
- 206/207：回收P空间，独立A/B生产者与A4/B5、A5/B3非对称slot，约11.85/12.10、11.83/12.03μs，未胜出。
- 208：恢复162并完整复测，源文件逐字一致。

详见`iterations165_208.json`及各迭代已提交的源码、stdout、CSV。199–201配置字符串部分沿用旧元数据，实际源码布局与运行检查为准，已在对应日志明确记录。

## 新 profiling 证据与界限

当前162累计阶段诊断（3新数据集、2dtype、10Graph重放×128CTA）与baseline逐位相同，记录7680行。编译64→72regs，无spill/驻留变化；事件扰动很小，但阶段时间仍不是普通完整调用时间。

|dtype|TC operand等待μs|准备与发射μs|显式wait0μs|TC循环μs|
|---|---:|---:|---:|---:|
|bf16|3.136|3.328|0.608|8.608|
|fp16|3.136|3.360|0.608|8.640|

供数等待仍明显，准备/发射包含异步执行重叠，不能将整个区间当作可删除的软件开销。以上为当前162计时，取代131旧前端阶段数；各重叠链时间不可相加为完整调用。

187完整NCU79regs、无local sectors、约1.783M指令；阶段诊断因插桩80regs与16Bstack有约10%扰动，只作链路定性证据：变换与全局发布约6.5μs，变换P数学约10–11μs，最后CTA重组约9–14μs。188行优先重组的实际收益印证重组访存问题。

198 NCU64regs、无spill、约2.92M指令（162约2.52M）；Tensor活跃比例提高仍无完整调用收益。202 NCU80regs、8Bstack、约6144local-load与2048local-store sectors、3.638M指令，控制流与真实spill增加。205 NCU74regs、零local sectors、1.895M指令仍未优于162，说明无需为其盲目降低block尺寸。

下一轮从162当前关键路径继续，重新评估供数/变换与TC时序，避免单独根据指令数或某个等待区间宣称收益。局部输入变换、复用原始tile存储并控制寄存器生命周期仍可研究，尚无胜出实现或物理下限证据。

## 复现与来源

`profile*/`保留原始CSV、NCU报告（若生成成功）、源码和脚本；脚本沿用原`results/opt_goal/...`路径，该目录仍存在。`generators/`是候选生成脚本快照，其中历史输入需从相应Git commit恢复；运行脚本前审阅其路径，不能把生成脚本当作已认证当前实现。正常计时通过逐轮源码恢复后执行`MEGAFMM_ALGORITHM=sw bash scripts/bench.sh iter-N`。

硬件流水与cluster容量参考NVIDIA原始文档：[Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)、[CUDA12.4 occupancy API](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__OCCUPANCY.html)。这些支持设计与容量检查方法，不证明本候选已达硬件极限。
