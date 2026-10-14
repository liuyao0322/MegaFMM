# 当前七乘法版本的瓶颈：重新 profiling

当前solution按用户要求保留iter97七乘法无补偿实现，选择提交0d5d74d，内核源码逐文件等同503c7de。本轮只profiling和诊断，没有修改solution，也没有切回15项补偿版。仅1024³、BF16/FP16输入、FP32输出。

## 结论

当前生产链最大的已测等待是TC等待operand就绪，准备与发射已明显缩短。瓶颈更偏向双operand slot复用、TMA发布和TC执行之间的供数延迟与同步依赖，同时仍有输入变换及全grid同步成本。没有证据表明HBM带宽、P shared bank conflict或寄存器spill是主限制。不能只因TMA等待free-slot累计时间大就认定这些等待都可以去掉：其中包含安全复用必须等待的计算阶段。

当前TC生产循环约7.94μs，输入变换+grid约2.62μs（含初始shared初始化/同步）。消费者多在等P，TC等待消费者释放P slot较少；消费者目前没有表现出明显反压。但未对本次七乘法版本做consumer数量消融，不能直接套用此前15项版本的消融结论。

## 当前无profiler完整调用

20warmup、5组×100调用、3组新输入精度检查，单位μs。这是正常benchmark结果，不是NCU replay或插桩时间。

| 模式 | dtype | SW | 调优Dense |
|---|---|---:|---:|
| launch | BF16 | 14.16448 | 12.44352 |
| launch | FP16 | 14.31584 | 9.94624 |
| Graph | BF16 | 13.85632 | 11.33568 |
| Graph | FP16 | 14.10080 | 9.56256 |

原门限未更改。SW BF16/FP16 eF仍约.00601520/.000754626，两者验收FAIL；benchmark预期返回2。无NaN/Inf。这里是按用户要求保留的性能实验，不是通过精度验收的winner。

## 独立诊断：谁等谁

使用当前内核生成独立baseline及分角色插桩版本，通过%globaltimer测量；3组新随机输入，每组每个采样点10次Graph执行×128CTA。计时采用CTA记录的中位数，单位μs。

| 同一次调用TC累计区间 | BF16 | FP16 |
|---|---:|---:|
| 等待operand ready | 2.880 | 2.912 |
| 准备、描述符及WGMMA发射 | 2.368 | 2.368 |
| 显式WGMMA wait0 | 1.408 | 1.408 |
| TC生产循环整体 | 7.936 | 7.936 |

operand等待约占TC生产循环36%，准备发射约30%，显式完成等待约18%。这是生产循环内部比例，不是完整调用比例。剩余区间包含P发布、slot同步、循环控制等；中位数不严格可加。准备发射与异步Tensor Core执行重叠，不能当成纯软件成本。

独立稀疏逐阶段交叉检查：TC operand等待各阶段中位数之和2.688μs、发射2.368μs、wait0约1.312μs，两dtype相同。与累计测量接近，差异来自插桩及取中位数方式。

| 其他角色区间，各阶段中位数之和 | BF16 | FP16 |
|---|---:|---:|
| TMA等待operand free-slot | 4.992 | 5.056 |
| TMA发射区间 | 1.728 | 1.728 |
| CUDA consumer等待P ready | 4.448 | 4.448 |
| CUDA consumer读P、重组及早期输出 | 2.464 | 2.464 |
| TC等待P slot可覆盖 | 0.384 | 0.384 |
| TC写P区间 | 0.416 | 0.416 |

这些角色相互重叠，不能将整表相加或视为可节省耗时。consumer记录中的读P/重组区间与随后发布P-free、部分输出区间分开，2.464μs不是完整consumer工作总量。所有role均以tid32或tid128等代表线程采样，不代表每条warp的完整分布。

依赖链为：TMA等待旧slot安全释放→发布下一批数据→TC等待ready、发射WGMMA并wait0→释放slot、生成P→CUDA consumer重组。TC既等数据，又限制TMA何时覆盖下一批数据；这一循环是当前优先检查的对象。仅凭当前观察不能断言增加buffer一定加速。

与历史补偿版的诊断相比，发射从约6.66μs降到2.37μs，而operand等待约2.34→2.9μs，显式完成等待约.99→1.41μs。历史采集不在本轮同一时钟状态，不能据此精确分解删除补偿的每一份收益，但当前阶段排序已有本轮两种计时方法支持。

## NCU硬件证据

NCU2025.1.1，CUDA12.8工具，实际内核由CUDA12.4 Release构建。使用--profile-from-start off尊重benchmark的CUDA profiler start/stop边界；raw CSV核实正好一条BF16、一条FP16完整内核，每dtype39次replay。未锁频，未清cache，符合本轮warm-cache问题。NCU replay时间仅作诊断。

| 指标 | BF16 | FP16 |
|---|---:|---:|
| NCU内核时间 μs | 14.592 | 14.464 |
| Tensor管线活动率，elapsed口径 | 13.22% | 12.93% |
| scheduler issue active | 28.61% | 28.62% |
| 实测active warp occupancy | 24.98% | 24.98% |
| HBM吞吐/峰值 | 0.286% | 0.345% |
| L2吞吐/峰值 | 42.26% | 42.74% |
| L2 sector hit rate | 97.92% | 97.38% |
| 寄存器/线程 | 56 | 56 |
| 动态shared/CTA | 213120 B | 213120 B |
| shared限制CTA/SM | 1 | 1 |
| 寄存器限制CTA/SM | 2 | 2 |

在这一warm-cache测试中，HBM不是带宽瓶颈；L2也未显示全程带宽饱和。低平均吞吐不能排除单阶段瞬时拥塞或访问延迟。Tensor活动率低说明未持续保持计算，不能解释成纯算力已满。当前每CTA16warp，其中4warp执行WGMMA；每SM一CTA，测得约25%warp occupancy。

每dtype实际64×32 WGMMA动态warp指令114688，符合15→7的工作量变化；R2UR41216。动态warp指令总数约320万。P shared全部16条STS及42条LDS仍1-way、0额外wavefront，bank conflict修复未退化。local/stack与spill均为0。

硬件stall采样可见long-scoreboard与barrier等待热点。若干采样落在barrier检测后的控制分支处，不能简单把long-scoreboard等同HBM带宽瓶颈；raw源码页和stall_hotspots.json保留了原始位置。本轮结论主要依据角色分段计时与硬件吞吐交叉判断。

## Shared布局遗留与下一步优先级

删除补偿后仍保留高低部分的旧布局间距，因此213120字节shared中仍有98304字节原operand低部分空间未使用，约占46%。当前真正需要的双slot高部分数据为98304字节，P双slot16384字节，另有barrier与padding。

1. 优先研究在安全释放operand slot的前提下，利用回收的shared空间改善预取深度和供数重叠；例如测试更深的operand流水，而非直接延后wait0。三slot是否有效仍需完整调用实验。
2. 对七乘法版重新选择tile、K分块和发射/回收节奏，降低约2.9μs的ready等待及约1.4μs的显式完成等待。
3. 继续拆分并降低输入变换与grid同步的2.62μs前置成本。

压缩shared本身不保证性能提升。当前grid只有128CTA，而H100有132SM；即使shared允许每SM两CTA，也不会自动产生足够CTA填满两层驻留，需要同时评估grid/tile划分。不能据此直接承诺occupancy翻倍或固定百分比加速。

## 测量可信度与复现

独立baseline与生产二进制的SASS逐条指令和操作数完全一致，两dtype均2416条静态指令。诊断270条timing记录均与各自baseline bitwise一致，涵盖3组新输入；这是诊断不改结果的检查，不代表通过FP64精度门限。

插桩会改变代码与调度。各角色诊断Graph事件中位数约12.61–13.07μs，独立baseline约12.74–12.85μs。诊断使用10节点Graph，与正式benchmark的调用环境、数据生成及样本不同；其绝对耗时不能替代正式benchmark约14μs的结果。globaltimer样本量化可见32ns步长，单次相邻stamp读数为0也不能解释成无测量开销。

- ncu/：完整ncu-rep、raw CSV、SASS CSV、汇总及stall热点。
- diagnostics/：实际执行的生成器、baseline、插桩源码、harness、压缩原始记录、timing及资源日志。
- sass_identity.json：生产内核与独立baseline一致性检查。
- launch.csv、graph.csv及日志：本轮无profiler完整调用。
- run_ncu.sh、run_diagnostic.sh、run_bench.sh、check_sass.sh：实际执行命令，路径以仓库根目录为工作目录。重现时先生成results/profile_iter97目录，运行scripts/build.sh，再依次经scripts/on_gpu.sh执行；diagnostics/generate.py固定读取实验提交503c7de。

NCU首次遇到系统/tmp锁文件权限问题，按[NVIDIA官方FAQ](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)设置到本任务可写的TMPDIR后成功。没有修改他人的锁文件。诊断与NCU按顺序独占本任务GPU运行。
