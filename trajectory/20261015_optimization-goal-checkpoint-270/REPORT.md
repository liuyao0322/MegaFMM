# 持续优化检查点：249–270，目标仍在进行

当前保留 **266（18f9a6f）**，269原样恢复，270独立Graph复测。用户要求的1024³ BF16和FP16普通完整调用都超过独立调优Dense尚未达成。原FP64精度验收两者仍FAIL，Auto仍Dense；不将近似路线的性能排名称为合格winner。没有物理下限或方向耗尽证据，继续优化。

## 完整调用性能

全部正式基准保留20 warmup、5组各100调用、3组fresh正确性输入、轮换两个输入集。同期交错SW与每个dtype独立调优的cuBLASLt。普通与Graph分开，二进制不同版本按A/B/B/A顺序运行；解锁时钟，结果有噪声，不能把单组最小值当稳定收益。

224→254的完整配对证明新的PDL架构有效：普通BF16 10.95968→10.52608us（3.96%），FP16 11.10080→10.44272us（5.93%）；Graph10.56000→10.09136、10.84160→10.10256us。详见 `paired/compare_iter224_iter254/`。

254→266配对见 `paired/compare_iter254_iter266/`：

| 模式 | dtype | 254 SW us | 266 SW us | 266同期Dense us |
|---|---|---:|---:|---:|
| 普通 | BF16 |10.51568|10.50944|12.32128|
| 普通 | FP16 |10.42944|10.36512|9.80560|
| Graph | BF16 |10.15168|10.03136|11.32624|
| Graph | FP16 |10.07216|10.13264|9.34624|

266普通BF基本持平、FP改善0.62%；Graph BF改善1.19%、FP退化0.60%。选择266针对普通调用目标的小幅改进，不能宣称跨模式普遍更快，更不是3%新突破。普通FP仍需减少约0.55952us（当前SW的5.40%）；BF已快于Dense约1.1724倍。

新鲜普通269：SW BF/FP10.54912/10.40128us，Dense12.39040/9.82944us。270 Graph：SW10.03968/10.12256us，Dense11.28544/9.39648us。精度eF仍约.0060151965/.00075462577，原门槛约.0010025568/.00010375276；无nonfinite。不得放宽门槛。

## 选中架构和验收

primary `sw_transform_pdl`128CTA×512线程生成原顺序舍入的8个S/T高位平面，4MiB workspace；入口触发PDL，末尾保留async proxy fence。

secondary128CTA、每CTA256线程，TMA warp0、TC128–255；4个BK256紧凑operand槽，196736B动态shared。反向叶乘法B_leaf转置乘A_leaf转置，64×32输出；TC私有静态PTX累加、原顺序重组直接写D。每K块wait0后四warp leader及时释放。TMA生产者仅在首次变换块前调用cudaGridDependencySynchronize，原始P1/P2可提前运行。两个kernel都设置100%shared carveout，实际233472B配置；缺少此配对设置的252慢约12.2us。

266在未使用的warp1 leader中于CTA初始化barrier前预取四个本次tensor descriptor。SASS确有四条UTMACCTL.PF，未缓存输入或计算值。run无分配、无host同步，每次编码当前指针；依赖的是官方PDL，不再使用224的private CUDA driver workspace publication ABI。make中的旧cooperative-capability检查仍存在，H100满足；不据此称当前launch为cooperative。

254和266分别独立通过：72组针对162保留算术的逐位输出比较（6种模式、3fresh dataset、2dtype、普通/Graph）、带poison workspace，quick memcheck/racecheck/synccheck各0错误/0hazard，以及256次双nonblocking stream/Graph调用、不同workspace。这些检查验证实现和同步，不证明原FP64精度合格。266检查源码经namespace还原与当前solution逐字一致，见state.json与sha256.json。

254 full NCU：primary32regs、compute60regs，动态local读写0，compute约1.06368M warp指令，Tensor elapsed约20–20.5%，L2约50%、不是HBM饱和。266最小指标NCU仍compute60regs、local读写0，primary32regs、local0，compute约1.0697M指令。266最小采集与254 full replay的duration不能直接作加速证据；正式完整基准才是收益依据。

## 当前瓶颈证据和适用范围

`nsys252/`、`nsys254/`普通单次capture均观察到primary结束后secondary启动；这一观察不能推广到连续调用。`trace254/`的单次普通时间戳也没有重叠，单节点Graph有约1us重叠。

`batchtrace254/`改用10次连续调用/10节点Graph，轮换两个输入集，3数据集×5重复×两模式×两dtype，60批、120最终输出逐位一致，76800条CTA行。普通稳态TC最早启动比各CTA pack lane0末时间的最大值早约.256us，Graph早约.544us。故普通连续调用确有重叠，不能把主机单次提交间隙说成全部瓶颈。pack lane0 finish不是整个CTA结束，量化是参与线程的界限；traced64regs vsbase60，不能声称无扰动。

`phasetrace254/`在首raw块、首变换块与最后块加入稀疏PTX时间戳，同样120个最终输出逐位一致，traced72regs vsbase60、cuobjdump stack0；未测此诊断的动态local traffic，不称其已证明无spill。普通BF/FP中位阶段：

| 诊断块 | ready us | issue us | drain us |
|---|---:|---:|---:|
| 首raw |1.024|.192|.128|
| 首S/T |.256|.192|.096|
| 最后 |.128|.160|.096|

首块供数的暴露等待大于单块MMA完成等待；后续槽循环依旧占时。阶段和不同CTA的中位数不能相加当完整调用，不把issue全部归于软件，也不能把旧218/97的阶段时间称为当前266的精确分解。没有硬件物理下限证据。

## 失败方向与后续

249细粒度计数器12.8/13.0us，251padding+减少fence仍12.5us；249时间戳定位独立初始化grid barrier约1.8–1.9us暴露等待，而pack wait约.032us，不能误说主要等待pack。250只变换所有权32×64基本持平。

252 PDL未配对carveout约12.2us；253 Graph也慢。254两个kernel共同100%carveout带来真实配对改善。255移除PDL改普通双kernel约13.0/13.2us，不保留，不能因单次NSys不重叠而说PDL无关。

256 N64半grid11.6us、257 N48 10.84FP、258 K512双slot11.45FP、259变换256×25610.49FP、260变换64×51211.89FP，均不优于254。261新joint-free cluster2 multicast19us，未进行独立同步验收，不选用。262 pairwait1约10.84FP，263单CTA leader trigger约10.43FP未证实普遍收益。264主机descriptor模板+ReplaceAddress10.46FP，设备不变但未见完整调用收益；不同于225设备持久descriptor。265静态TMA10.64FP；267提前首TMA10.59FP；268移除闲置warp改160线程10.46FP，均未选用。

后续继续当前PDL路线。可调查每个primary CTA自己初始化发布标志、fence后触发PDL，再按所需32×64输入块发布的细粒度供数；这与249的独立全grid初始化不同，尚未实施，必须先证明每调用重新初始化的可见性与内存序，并独立验证，不能因数值看似一致便宣称协议正确。CUDA12.4官方runtime说明trigger本身没有可见性保证，可用合适scope的fence建立可见性；仍需为具体协议给出证明。

证据保留146项源码、驱动、日志、原始CSV导出（大CSV gzip）、摘要、配对记录和hash。NCU/NSys原始二进制报告留在results，不重复进入Git；这里保留可审查导出与复现脚本。前一检查点248为历史记录，当前不会还原补偿版。下一实验271，目标active。
