# 持续优化检查点：iter209–248，目标仍在进行

当前 `solution/` 是通过 `git checkout 62142fc -- solution/` 原样恢复的 iter224，恢复后的普通调用复测为247，Graph复测为248。**尚未达到两种 dtype 普通完整调用均超过 Dense 的目标。两种输入的原始 FP64 精度验收仍失败。** 本检查点不表示完成、暂停、达到物理极限或穷尽方向。

## 当前结果与剩余差距

H10080GB、132SM、g08、allocation107488、CUDA12.4。1024³、量化输入、FP32输出、完整动态输入变换，20warmup、5组×100次、3轮fresh-input正确性，旋转输入。双方同期交错计时，Dense每轮独立调优；评测次数和精度合同未变。

|模式|dtype|SW μs|Dense μs|Dense/SW|原始精度验收|
|---|---|---:|---:|---:|---|
|普通调用247|BF16|10.977600|12.275200|1.118204|FAIL|
|普通调用247|FP16|11.089600|9.975040|0.899495|FAIL|
|Graph248|BF16|10.511360|11.368320|1.081527|FAIL|
|Graph248|FP16|10.877440|9.440640|0.867910|FAIL|

普通FP16仍慢约11.17%，需从SW时间减少1.11456μs，即10.05%，才可追平当前Dense。Graph不能替代普通调用目标。普通与Graph各自采用同模式Dense作分母。

原始数值结果保持：BF16 eF=0.00601519652935，gate=0.00100255679669；FP16 eF=0.000754625769821，gate=0.000103752760587；nonfinite=0。**没有精度合格的当前SW winner，Auto继续Dense。** 性能比较只说明用户明确保留的七乘法无补偿近似路线的速度。

## 保留实现与选型证据

iter215将叶乘法转置为 `B_leaf^T × A_leaf^T`，使LHS使用MN128布局、RHS使用K128布局。iter218在该几何上使用完整静态PTX、私有寄存器重组、直接输出，移除P shared交换。iter224将变换线程608→512，总线程768→672，形成两个均匀批次。

当前128CTA、64×32输出tile、BK256、4个operand slots、196736B shared、64regs、无local sectors；每CTA 1TMA warp、4WGMMA warps、16CUDA变换warps。P1/P2与全局S/T变换重叠，余下P6/P7/P5/P3/P4按原算术顺序重组。每个K块wait0后由4个warp leader及时释放slot。4MiB S/T workspace，无P/C shared存储。

`run()`不分配、不host同步，不缓存跨调用计算值；四个tensor maps每次按当前指针编码。全局发布继续依赖CUDA12.4 cooperative-groups内部的每次调用独立driver workspace。这是现有工具链内部ABI依赖，移植时需复核；未引入持久计算缓存或输入特化。

完整A/B/B/A配对证据位于 `paired/`：

|模式|dtype|162 μs|218 μs|218相对162减时|
|---|---|---:|---:|---:|
|普通|BF16|11.340320|11.015840|2.86%|
|普通|FP16|11.515520|11.113120|3.49%|
|Graph|BF16|10.893440|10.587840|2.81%|
|Graph|FP16|11.266560|10.999520|2.37%|

随后218/224配对：普通BF16 10.979360→10.968480，FP16 11.140160→11.091200μs；Graph BF16 10.684480→10.548160，FP16 10.979200→10.759040μs。224普通调用改善仅约0.10%/0.44%，Graph约1.28%/2.01%，**不宣称224单独取得≥3%改善**。选择224依据小幅配对收益、相同无spill资源和完整独立验证。

## 独立实现检查

`check224/`保留完整源码、运行脚本、编译和检查日志。candidate源与当前224源码仅namespace不同，归一化后逐字一致；`state.json`记录SHA256。baseline为已验证162，不能把基线逐位一致当作原始AB精度通过。

- 6类输入×3新数据集×普通/Graph×2dtype，共72项逐位162检查全部通过；工作区和输出先poison。
- memcheck、racecheck、synccheck各quick路径，0 errors / 0 hazards。
- 两个nonblocking stream、两种dtype、普通/Graph，每类64launch，共256launch，独立workspace，逐位检查全部通过。

这些验证适用于保留224，不替后续被拒候选认证。full FP64数值门槛仍FAIL。

## 新profiling与结论边界

NCU12.8，`--clock-control none --cache-control none`；NCU计时不替代未插桩完整benchmark。完整导出在 `profile218/`、`reassess228/`，最小资源profile在 `profile230/234/238/242/`。

- 218完整NCU：64regs、local读写0、约1.7155M warp instructions、Tensor elapsed utilization约17.2–17.6%、L2 throughput约49.9–50.5%。没有Tensor计算峰值或物理极限证据。
- 221常量descriptor与224同期完整NCU：221约1.468M指令，224约1.700M，两者64regs、local0；221普通调用反而更慢。不能以少指令直接推断关键路径改善。
- 230双独立累加器80regs、local0，仍约11.22/11.43μs；问题不能笼统归因于spill。
- 242双TC共享LHS：64regs、local0，约1.778M指令、profile约15μs；64CTA只使用约半数SM。active-SM Tensor指标约31–32%不能与218的elapsed指标直接比较。

218的稀疏诊断见 `profile218/diagnostics/`：3新数据集、10节点Graph、每次只取15个稀疏样本，90次逐位检查通过；115200行记录。traced72regs、base64regs、local0、均1CTA/SM。使用同指针10节点Graph，**不同于普通100-call旋转输入benchmark**，且插桩事件可更快，不能宣称无扰动。

每阶段样本中位数之和：operand-ready约3.424BF/3.360FPμs，准备/发射约2.368μs，显式drain约1.344μs。突出等待在首个raw块和首个变换块（约1.1–1.2μs），其余多数K块ready只约0.03–0.12μs。TMA进入到packwait前约1.440μs，packwait约0.800μs，全局发布约1.408BF/1.344FPμs。**这些是不同阶段的稀疏样本统计，存在重叠，不能相加成同一次完整调用总时间。**

235 splitK稀疏诊断见 `trace235/`：30次bitexact检查通过，traced/untraced126regs、stack0。TC链约7.78μs，fence+barrier约0.90μs，counter+barrier约0.64μs，last-CTA线性merge约7.82μs。测试制度不同，不能将这些诊断作为标准benchmark或声称无扰动。它支持优化merge；扩大merge线程到384后236仍约15.82μs，未胜过224。

## 被拒方向与无效数据

逐轮源码、stdout和CSV都在对应 `trajectory/*_iter-N/`，`iterations209_248.json`汇总40轮原始行；所有实验都先记录再提交。

- 209/210：本地shared变换、无全局S/T发布，但约20μs。210修复209的C21寄存器算术顺序；不能采用209的错误结果。210独立验证通过不代表性能合格。
- 211–214：float4 P、双TC、4Pslots等未胜出；212生成器失败而benchmark沿用211，已明确标记，不作新候选证据。
- 215/218/224：上述转置、私有PTX、均匀变换工作分配，为本轮保留路线。
- 216/217/219：K192三块、N16两CTA/SM、K128九slots，分别约12.5–14μs、13.8μs、11.6–11.7μs，均未胜出。
- 220/221：低32位/常量descriptor。220首轮FP16 Dense不稳定，约12.83→10.09μs的CSV分母导致假胜出，不算达到目标；稳定配对更慢。221少指令但11.18/11.25μs，未保留。
- 222/223/225：提前预取变换批次、cluster2层级发布、持久tensor-map metadata，未改善。225额外设备metadata和plan生命周期逻辑已移除。
- 226编译失败（N64累加器向量未扩为32）；227修复但约13.56/13.66μs。226的一次资源读取误插在日志前已在当轮记录纠正，没有据此跳过失败轮。
- 228：K512双slots、LHS分两次TMA，约12.09/12.41μs。229 64ns发布退避、230双独立累加器、231静态TMA生产、232无符号索引、233 S/T分工，没有可靠收益。
- 234–238：splitK2的128CTA，改变FP32分组。234约41μs；235向量merge约21μs；236扩大384merge线程约15.82μs；237直接FP32 global atomic约13.5μs，但global FP32 atomic有FTZ语义变化；238有序逐象限flag交接约42μs。均拒绝。234/238 local0，不能归因于spill。
- 239生成inline PTX共享地址输入转义错误而编译失败；240修复、4个final-C shared slots约11.03/11.20μs；241回收成两槽约11.23/11.46μs，均未优于私有直接输出。
- 242：64×64、两个TC group共享LHS，1024线程，3operand slots，64CTA，约14.34/14.45μs。243保留128CTA、两个N16 TC group，约12.36/12.56μs。并行group成本未被隐藏，均拒绝。
- 244：当前私有输出架构4operand slots→3，shared147584，约11.16/11.41μs，无收益。
- 245：同一累加器发射两个K group，wait1/free再wait0/free，约11.14/11.29μs；246合成一个group再释放两个槽，约11.16/11.37μs。少等待不足以抵消延迟释放的代价。
- 247/248：原样恢复224，普通和Graph分别完整复测，未更改评测或精度标准。

## 后续工作与重现

目标保持active。当前证据支持继续分析首个raw TMA、首次变换发布与TC发射的关键路径，不支持盲目加深预取、增加计算组或用更少指令作胜出依据。下一轮应围绕依赖范围更小的S/T发布（需解决每次调用计数初始化与跨CTA内存顺序），或显著降低本地变换指令成本，先给出正确同步协议再测。尚未声称这些方向已穷尽。

本目录保留135个证据文件及SHA256、逐轮CSV行、被测源码hash；大CSV按gzip压缩。源NCU报告仍在 `results/opt_goal/`，提交的raw导出保留可审计指标。这里的生成器是历史实验重现材料，不是当前solution入口。

```bash
MEGAFMM_ALGORITHM=sw bash scripts/bench.sh iter-N
# 下一次工具调用必须日志，随后必须git commit；即便数值FAIL/编译失败。
python3 scripts/log_iteration.py N --title ... --hypothesis ... --changes ... --analysis ... --next ... --status experiment
# Graph用MEGAFMM_GRAPH=1，并独立记录。
```

下一迭代编号249。原始数值门槛不变，Auto继续Dense。
