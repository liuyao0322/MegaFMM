# SW megakernel 的 NCU profiling：1024³ / iter76

2026-10-14。用户暂停持续优化后单独要求 NCU profiling。本次只采集、分析和保存证据，
没有继续修改生产kernel。生产solution当前为iter79；本次分析的是已通过原精度门限且
当前最快的iter76（456ce9d），在独立目录构建，kernel符号为sw_cooperative_hilo。

**结论：当前warm-cache输入下没有跑满HBM或Tensor Core。大量warp在等待产物，
P写入shared存在确定的8路bank conflict。110寄存器/线程和213120字节动态shared同时
限制每SM只能驻留1 CTA。需要结合流水分工、P布局及15项补偿乘法分析，单纯提高occupancy
不能解释或解决相对Dense的差距。**

## 采集方法与完整性

- H10080GBHBM3，132SM，g08，allocation107488；driver575.51.03。
- CUDA12.4 Release构建；NCU2025.1.1（/share/app/cuda/cuda-12.8/bin/ncu）。
- 只采1024³、BF16/FP16、FP32输出、连续无转置、alpha1/beta0。
- NCU --set full，kernel replay，--profile-from-start off；仅cudaProfilerStart/Stop范围
  内的完整一次run。SW每dtype一个kernel，每个kernel39 passes。
- --clock-control none、--cache-control none；不清空缓存，接近当前warm-cache任务，但replay
  不保证每pass缓存状态完全一致。时钟未锁定，短kernel采样有波动，所有百分比只表示本次捕获。
- 第二份SW构建增加-lineinfo，用于CUDA/SASS关联；其110寄存器、occupancy、Tensor活跃度及
  冲突结论与普通Release采集一致。没有修改SW计算内容。
- 6个ctc收发指标不可访问；其余本报告使用的指标成功取得。不是所有可能硬件指标均可采集。
- 初次采集因/tmp/nsight-compute-lock无法打开而退出9。确认无其他ncu进程后，采用NVIDIA
  文档的私有TMPDIR方案成功重试，没有删除或修改他人的lock。

NCU timing只用来描述捕获，不代替正常benchmark。iter76普通完整调用原始结果为
BF16：Dense12.272μs / SW21.245μs；FP16：Dense9.908μs / SW21.883μs。
已有72项fresh-input压力检查通过；本次--profile分支不执行数值验收，不能将profile成功当作精度证明。

## 实测指标

普通Release SW见sw_raw.csv，固定Dense见dense_fixed_raw.csv。Tensor一行是
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed，表示管线活跃周期指标，
不能直接等同于有效FLOP占理论峰值比例。

| 指标 | SW BF16 | SW FP16 | Dense BF16 | Dense FP16 |
|---|---:|---:|---:|---:|
| NCU duration，μs | 21.856 | 21.760 | 13.312 | 10.560 |
| Tensor管线活跃 / elapsed | 18.75% | 18.85% | 16.18% | 32.14% |
| achieved occupancy | 24.93% | 24.93% | 13.49% | 6.24% |
| eligible warps / scheduler / active cycle | 0.316 | 0.318 | 0.224 | 0.277 |
| scheduler issue active | 27.57% | 27.85% | 19.37% | 27.69% |
| 寄存器 / 线程 | 110 | 110 | 168 | 150 |
| HBM active / elapsed | 0.531% | 0.475% | 0.062% | 0.059% |
| DRAM字节吞吐，GB/s | 17.78 | 15.88 | 2.08 | 1.96 |
| L2吞吐 / peak指标 | 48.11% | 47.68% | 13.99% | 31.00% |
| L2 hit rate | 96.64% | 93.70% | 98.05% | 90.41% |

低HBM吞吐与高L2命中只支持本次warm-cache输入下HBM带宽未饱和；不支持推广到cold-cache，
也不能据此排除L2访问延迟。SW的L2流量成本仍然明显存在。

Dense对照固定为正常benchmark中独立调优得到的确切配置：

- BF16：id39、tile18、split_k=-1、reduction2、swizzle0、stage35、custom0、inner0、cluster10。
- FP16：id21、tile18、split_k1、reduction0、swizzle0、stage18、custom0、inner0、cluster0。

单独的profiling harness逐字段匹配以上配置，并在范围外预热20次，不在NCU环境重新排名。
它仅对独立快照plan.cpp增加环境变量开关MEGAFMM_NCU_FIXED_DENSE，未改生产plan.cpp。
最初尝试在NCU下重新tune时，BF16选中了id24/splitK4，并捕获主核与reduce两个kernel；
该初始dense_full捕获不用于上表或性能结论。失败对照日志仍保留，避免混淆。

## 热点与结论

### 1. 多数CUDA warp在等待产物，Long Scoreboard不是HBM瓶颈的直接证据

SW每scheduler平均4.01个active warps，但只有约0.32个eligible，约72%的active周期没有
eligible warp。Warp State表中Long Scoreboard约8.25/8.23 cycles per issued instruction，
NCU按其warp状态口径给出56.7%/57.5%；这不是kernel wall time的分段占比。

SASS顶部热点是SYNCS.PHASECHK.TRANS64.TRYWAIT之后依赖其predicate的分支，地址对应
product_ready双槽barrier（shared偏移0x34020/0x34028）。CUDA源码对应
source/sw_pipeline.cu:116，CUDA workers在这里等待七个完整P。
另有输入载入后BF16/FP16→FP32转换处的Long Scoreboard样本，不能忽略packing阶段。

所以，直接将全部Long Scoreboard称为“显存慢”会误判。当前512线程中11个CUDA warp
在pack后逐P等待，只有4个warp组成一个WGMMA group。这与消费端空等及低Tensor整体活跃
一致，但采样不是每warp精确时间线，不能从中推导某个阶段的精确微秒占比。

### 2. P写入shared的8路bank conflict已被具体指令证实

source/sw_pipeline.cu:154把GMMA CLayout映射到普通row-major的P缓冲区。
NCU定位到16条STS指令，每条统计相同：

- shared N-Way = 8。
- ideal wavefronts = 3584。
- actual wavefronts = 28672。
- excessive wavefronts = 25088。

合计401408个excessive wavefronts，两dtype相同；约占该采集全部shared wavefronts的20%。
shared load bank-conflict计数为0，问题集中在存储端。

这个20%是wavefront比例，不是20%的kernel时间，也不是承诺修正后能快20%。
NCU给出的自动estimated speedup同样只是估计，未做修改前后实测验证。
下一步最直接的验证方向是让P按WGMMA fragment的写出顺序组织，再让CUDA重组沿同一
物理布局读取，或测试合适的padding/swizzle；本次未实施。

### 3. shared和寄存器同时限制驻留，但occupancy不是独立目标

SW动态shared213120字节，加driver shared1024字节；110寄存器/线程（分配粒度后112），
512线程/CTA。NCU两项block limit都为1，理论16 warps/SM即25%，实测约24.93%。
只有压缩shared或只有降低寄存器，另一项仍会限制驻留。

128个CTA对应132个SM，grid本身最多空出4个SM，约3%的规模差异不足以解释当前约2倍差距。
Dense的occupancy实际上更低却更快，因此不能把“提高occupancy到50%”当作优化成功条件。

### 4. 当前数值补偿已增加Tensor工作量

原始Strassen–Winograd为7个叶乘法；为通过原始精度门限，当前hi/lo实现P1/P2各1项、
P3/P4各2项、P5/P6/P7各3项，共15项，同样512分块的普通Dense为8项。
这使得“用了SW所以乘法一定更少”不适用于当前精度策略。即使BF16 Dense的Tensor活跃
指标低于SW，也可能因为它完成的总工作量更小而更快，不能只比较利用率百分比。

## 后续验证优先级（尚未实施）

1. 修正P写入布局的8路冲突，保持原数值DAG与门限，实测完整调用差异。
2. 重新安排等待产物的CUDA warps与其寄存器驻留，减少空等，但避免把P寄存器压力变成spill。
3. 联合缩减shared和寄存器需求，再验证是否真能用更多独立Tensor工作覆盖等待；不能只追occupancy。
4. 研究在原精度门限下减少15项补偿乘法的数学方案，先证明数值通过，再评价性能。

## 文件与复现

- sw_full.ncu-rep：普通Release的BF16/FP16完整采集。
- sw_lineinfo_full.ncu-rep：含CUDA/SASS源码关联的完整采集，推荐用ncu-ui打开。
- dense_fixed_full.ncu-rep：与benchmark相同算法配置的Dense对照。
- sw_raw.csv、sw_lineinfo_raw.csv、dense_fixed_raw.csv：原始metrics导出。
- summary.json：选择的指标、单位和采集限制；hotspots.json：精确SASS热点及冲突指标。
- source/：已测iter76 kernel及profiling-only Dense绑定辅助代码；源hash见summary.json。
- 完整source CSV及大型details文本保留于results/ncu_iter76/，可从.ncu-rep重新导出。

采集的关键命令（已在现有独立构建上执行）：

```bash
bash scripts/on_gpu.sh bash -c '
  TMPDIR=/tmp/megafmm-ncu-1052 /share/app/cuda/cuda-12.8/bin/ncu \
    --target-processes all --set full --profile-from-start off \
    --kernel-name regex:sw_cooperative_hilo --launch-count 2 \
    --clock-control none --cache-control none --force-overwrite \
    -o results/ncu_iter76/sw_full \
    ./results/ncu_iter76/build/megafmm_bench \
    --algorithm sw --shapes 1024,1024,1024 --dtype both --profile
'
```

使用私有TMPDIR前需确认节点没有并发NCU；不要用私有锁绕过实际的并发profiling。
Dense对照使用build_fixed可执行文件，加MEGAFMM_NCU_FIXED_DENSE=1和--profile-dense，
不加kernel过滤或launch-count上限，以捕获范围内的所有kernel。

指标解释及TMPDIR处理参考：
[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
本报告明确区分采集事实、源码推断和待验证的优化方向；持续优化仍为paused。
