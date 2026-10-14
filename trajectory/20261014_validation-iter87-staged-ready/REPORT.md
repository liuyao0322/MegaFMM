# iter87：固定 slot 与分阶段供数验证

源码提交 `c9380c3`；起点为 iter85=`2870c75`。本次用户同意先优化固定 slot/descriptor 复用，收益不足再尝试高低部分分阶段就绪。最终保留 iter87；本次完成两个有界实验，不恢复已暂停的无限优化目标。

仅 H100 80GB HBM3 / g08 / allocation107488 / CUDA12.4，1024³，BF16/FP16输入、FP32累加及输出。所有计时包含每次输入变换、grid barrier、15项叶乘法、补偿和输出重组。

## 实现

iter86 在 TC producer 中展开两个 K 块（q=0/1），让 operand slot 的 shared 地址固定。此前 SASS 只有48条静态64x32 WGMMA，说明该循环保留；展开后96条。原有内层K=16展开、低位descriptor偏移、运算及同步均保留。

iter87 在iter86上增加每slot的 `low_ready` transaction barrier。TMA先登记高、低部分各自的byte count，先发射全部 A_hi/B_hi，再发射需要的 A_lo/B_lo；TC等高部分后执行 `hi×hi`，再等低部分并按原顺序执行 `lo×hi`、`hi×lo`。完成所有WGMMA之后才释放operand slot。
所有stage都推进两个ready barrier的phase，包括低部分byte count为0的P1/P2；TC只在存在低部分时等待low_ready。P顺序、重组算术、free barrier参与者及phase规则保持不变。

新增16字节barrier使用原结构体的尾部padding，动态shared仍为213120字节。每CTA512线程，128CTA，8MiB全局workspace；无额外kernel、run内分配或host同步。

## 完整调用性能

20 warmup、5组×100次、3组fresh输入，warm_cache，rotate_inputs=1，原精度门限不变。下表平均两次独立运行的行中位数，不是单次调用分位数。无锁频；短kernel仍有时钟/调度噪声。

按 A/B/C/C/B/A 顺序比较iter85/86/87，launch与Graph各一组：

| 模式 | dtype | iter85 μs | iter86 μs | iter87 μs | 85→87下降 | 86→87下降 |
|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | 20.58592 | 20.43168 | 19.28576 | 6.32% | 5.61% |
| launch | fp16 | 20.70080 | 20.48784 | 19.30112 | 6.76% | 5.79% |
| graph | bf16 | 20.25632 | 19.68464 | 18.75104 | 7.43% | 4.74% |
| graph | fp16 | 20.60000 | 19.79200 | 18.79936 | 8.74% | 5.02% |

正式独立bench（standard_launch.csv）SW BF16/FP16=19.31584/19.28320μs，调优cuBLASLt Dense=12.30272/9.94944μs，SW仍慢1.57/1.94倍。Auto继续Dense；仅能称为通过精度的SW改进，不能称为击败Dense的性能winner。

## NCU 与分段计时

NCU2025.1.1，full set，39 passes/dtype，clock-control none、cache-control none。重放耗时不代替普通benchmark。实际64x32 WGMMA每dtype始终245760次，15项叶乘法未减少。

| 版本 | R2UR / dtype | BF16 / FP16动态warp指令 | registers/thread |
|---|---:|---:|---:|
| iter85 | 995968 | 5395148 / 5419281 | 110 |
| iter86 | 487424 | 4794791 / 4810915 | 72 |
| iter87 | 178944 | 4673556 / 4681806 | 79 |

固定slot后R2UR减少51.06%；分阶段版本相对iter85减少82.03%。iter87的barrier和调度改变也影响descriptor生成及复用，因此不能把整个性能收益仅归因于隐藏TMA延迟。R2UR计数是生成代码证据，并不单独证明它的全部执行延迟在关键路径上。

所有P指令（每dtype16条STS、42条LDS）仍1-way、0 excessive wavefront，summary.json保留逐指令证据。无local或stack，驻留仍为1CTA/SM；降低寄存器未突破shared对occupancy的限制。

隔离副本使用globaltimer记录同一调用内的累计区间，每dtype3组fresh×10次kernel×128CTA=3840个样本。所有trace输出与各自未插桩副本bitwise一致。iter87统计operand等待时包含高、低两个barrier，低部分等待从prepare/issue区间扣除。

| 版本 | dtype | operand累计等待 μs | prepare/issue μs | 显式wait0 μs | TC循环 μs | 插桩event扰动 |
|---|---|---:|---:|---:|---:|---:|
| iter86 | bf16 | 2.976 | 6.720 | 1.120 | 12.416 | +0.00% |
| iter86 | fp16 | 2.944 | 6.688 | 1.120 | 12.352 | +0.38% |
| iter87 | bf16 | 2.368 | 6.816 | 0.960 | 11.744 | +0.11% |
| iter87 | fp16 | 2.368 | 6.784 | 0.960 | 11.744 | +1.55% |

分段采样不是三版本完整调用交替对照；不锁频且有插桩扰动，只用于定位。各区间中位数不必相加等于循环中位数。prepare/issue包含异步WGMMA重叠执行，不能全部当作软件开销。

相对iter86，operand总等待约2.95→2.37μs，TC循环约12.4→11.74μs；prepare/issue基本持平，显式wait0略降。这支持分阶段发布改善供数等待。现有采样只统计高、低等待之和，没有独立量化两者各自的耗时，也未通过独立ablation区分供数重叠与编译变化的收益。

## 正确性

正式bench及每次比较均使用原始量化输入的FP64普通AB参考，所有3个fresh trial通过原门限，无nonfinite。BF16 eF=7.394587e-6，limit=1.002557e-3；FP16 eF=2.371355e-6，limit=1.037528e-4。
独立6种模式×2dtype×3fresh×launch/Graph共72项均通过，包含fresh Graph重放、NaN输出污染及API拒绝检查。3个CPU数学DAG测试通过。

最终版本quick memcheck/racecheck/synccheck全部通过，0 errors、0 hazards、0 warnings。对应日志与编译资源证据已保留。

## 证据与复现

- standard_launch.csv、build.log：正式整调用评测和构建。
- compare_iter*/：交替运行的原始CSV、日志和执行脚本。
- production_chain*/：固定git提交的源码生成器、插桩源码、CTA原始计时、bitwise校验和编译资源。measure.cu的role5为累计计时，先创建输出目录，再执行measure <output> 5。
- sw_full.ncu-rep、sw_raw.csv、sw_sass.csv、summary.json：NCU原始报告、导出和聚合证据。
- stress.log及最终版本sanitizer日志。

原始精度门限、参考输入、输出类型及测量范围均未放宽。

复现三版本对照：在有效H100 allocation下执行 `python3 trajectory/20261014_validation-iter87-staged-ready/reproduce_compare.py`。脚本在results独立重建固定commit85/86/87，通过现有on_gpu.sh运行全部1024³对照，不修改生产solution。脚本经过语法检查；归档数据来自本轮原始run.py执行，未重复运行复现脚本。
