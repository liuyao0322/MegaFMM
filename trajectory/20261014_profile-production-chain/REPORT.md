# iter82 生产链条拆分：TMA 供数与 WGMMA 等待

本次按用户要求做诊断，不改生产代码。分析对象是iter82（77e1489）的1024³ BF16/FP16 kernel。

**结论：在两种显式等待中，operand就绪等待明显长于WGMMA全量等待；但生产链条最大的测得区间，是operand就绪之后的描述符准备、WGMMA指令提交及其间执行。不能把当前主要瓶颈归因于每个K块末尾的wait<0>。**

此前建议优先处理wait<0>只是待验证假设，本次计时不支持把它作为当前首要优化方向。

## 计时口径

独立生成baseline与instrumented两个命名空间的kernel，不修改solution/、公开API或workspace合同。baseline来自固定提交77e1489。两种dtype各自2800条静态SASS指令与正式build/libmegafmm.a中的生产kernel逐条比较，指令及操作数完全一致，见verification.json。

计时使用H100的64位%globaltimer，单位ns；本机记录表现出约32ns步进，短区间受计时与指令调度开销影响。官方将其定义为目标相关的全局纳秒计时寄存器，不应推广到其他GPU：[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)。

两个交叉测量：

1. **逐块稀疏采样**：一次调用仅对一个选定K块读四次时间，计时结果保留在寄存器，到kernel末尾写回。遍历14个K块；分别独立测TMA线程、Tensor warpgroup、CUDA consumer及P发布。没有在被测区间内新增trace global store。
2. **同次调用累计**：同一个Tensor warpgroup leader在14个K块中累计三个连续区间，再记录整个生产阶段时间。用这个结果给出正式分解；稀疏采样验证其量级。

Tensor线程分解边界：

```text
记录t0
wait(operand_ready)
记录t1
准备shared tensor/GMMA descriptor，warpgroup fence
提交本块16/32/48条WGMMA，commit
记录t2
warpgroup_wait<0>()
记录t3
释放operand槽，继续下一块
```

- **operand_wait**：t1-t0，包含ready检查/等待和计时开销；不是TMA一次搬运的完整时长，也不能单独解释为HBM带宽不足。
- **prepare_issue**：t2-t1，包含描述符准备、发射、必要的指令依赖/发射背压；已经与异步Tensor Core执行重叠，不等于纯软件地址计算耗时。
- **explicit_wait0**：t3-t2，提交完后仍需等待的残余时间；不是全部Tensor Core计算时间。
- **producer_elapsed**：Tensor warpgroup进入P循环到最后发布P的时间，不包含前面的输入变换/grid barrier，也不包含末尾CTA barrier。

查看反汇编确认了计时读取与`WARPGROUP.DEPBAR.LE`的顺序。不能将long-scoreboard采样百分比当作这些区间的时间占比。

## 同次调用累计结果

每种dtype：3组新量化输入×10次实测调用×128个CTA，共3840个CTA样本。每个样本是一个Tensor warpgroup leader在同一次调用中14个K块的累计值。下表为CTA样本分布，不是所有SM/warp时间相加，也不是端到端kernel的百分比。

| 区间 | BF16 P10 / P50 / P90 | FP16 P10 / P50 / P90 |
|---|---|---|
| 等operand就绪 | 1.952 / **2.432** / 2.944 us | 2.016 / **2.464** / 2.976 us |
| 描述符准备与WGMMA提交 | 8.352 / **8.544** / 8.768 us | 8.384 / **8.576** / 8.800 us |
| 显式wait<0>残余等待 | 0.288 / **0.352** / 0.416 us | 0.288 / **0.352** / 0.416 us |
| 整个P生产循环 | 13.024 / **13.344** / 13.632 us | 13.088 / **13.408** / 13.728 us |

按每个CTA的同次调用区间除以该CTA生产循环时间，再取中位数：准备/提交约64%，operand等待约18%，显式wait0约2.7%；未覆盖的循环控制、free到达、P槽等待及写入发布等约15%。这些比例只针对被测P生产循环，不能称为整个kernel的占比。

在两类等待中，operand_wait约为explicit_wait0的7倍。但最主要区间是prepare_issue，约为operand_wait的3.5倍。

累计版在wait前还执行少量计数算术，可能进一步隐藏残余等待；稀疏版不做这些累计，14块的逐块中位数之和为：

| dtype | operand_wait | prepare_issue | explicit_wait0 |
|---|---:|---:|---:|
| BF16 | 2.304 us | 8.608 us | 0.448 us |
| FP16 | 2.304 us | 8.640 us | 0.448 us |

跨采样逐块中位数之和不等于一次真实调用的总时间，只用于交叉检查。两种方法都显示wait0残余量级低于0.5us，准备/提交约8.5–8.6us。

## 等待集中在哪里

稀疏采样按P顺序P1,P2,P6,P7,P5,P3,P4推进，每个P有两个BK256块。

BF16的operand_wait逐块中位数：

| P | 首个K块 | 第二个K块 |
|---|---:|---:|
| P1 | 0.416 us | 0.064 us |
| P2 | 0.256 us | 0.064 us |
| P6 | 0.704 us | 0.064 us |
| P7 | 0.128 us | 0.064 us |
| P5 | 0.096 us | 0.064 us |
| P3 | 0.064 us | 0.064 us |
| P4 | 0.192 us | 0.064 us |

大部分第二个K块的检查很快，较大的供数等待集中在起步及部分P切换，尤其第一次进入双边hi/lo的P6。不能据此断言TMA整体带宽饱和。

其他角色稀疏采样的逐阶段中位数之和，均只是交叉参考：

- TMA线程等operand空槽：BF16/FP16约5.760/5.792us，说明它也经常受Tensor消费者释放槽位的限制；这段时间与Tensor计算重叠，不能再加到Tensor生产循环时间上。
- CUDA consumer等P：约9.376/9.360us，仍是等待P生产侧。
- Tensor生产者等P空槽：约0.384/0.384us，重组消费侧没有表现出显著回压。该值包含快速检查与计时开销。

## 为什么wait0很短

WGMMA是异步执行。发射一个K块的所有指令、准备后续descriptor的过程中，已提交的计算可以同时执行；因此，发射结束后剩下的等待可能很短。`wait_group(0)`保证已提交组完成，但其计时不代表整段矩阵乘法时间：[NVIDIA WGMMA指南](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html)。

现有iter82 NCU的BF16动态指令记录有约99.6万条R2UR，对应约24.6万条真实HGMMA。SASS中反复出现descriptor位运算、地址构造及普通寄存器到uniform寄存器的搬运。这为下一步检查descriptor生成和发射效率提供了直接线索，但不能把8.5us全部归因于这些软件指令；区间还包含硬件发射节奏和异步执行影响。

## 插桩扰动与正确性

- baseline寄存器110/thread；累计版118/thread，无local/stack spill。稀疏版112–118/thread；所有版本仍每SM1个CTA、动态shared213120字节，支持128个合作CTA。
- 累计版相对同数据集baseline的Graph事件时间变化：BF16均值+0.53%（三组-0.11%到+0.99%），FP16均值+0.73%（+0.39%到+1.19%）。未锁频，不把这些差异当作优化收益。
- 稀疏版各角色的事件时间中位数变化约+0.6%到+1.6%；没有出现大幅插桩减速。
- 每个稀疏采样case及累计case的最终FP32输出，均与固定源码baseline逐位一致；三组fresh输入、两种dtype全通过。
- 正式生产binary另外重跑标准Graph benchmark：原始输入FP64参考、原门限、20 warmup、5组×100次、3轮fresh correctness，全通过。此次BF16/FP16为20.470/20.832us；不同输入、调用方式、GPU状态下，不能与诊断harness的baseline绝对时长混作优化比较。
- solution/没有修改。诊断测量不是新的优化iteration。

## 优先级修正

1. 先检查prepare_issue区间里的重复descriptor构造、R2UR搬运、指令依赖和发射效率。
2. 再处理P切换处的operand就绪缺口，明确是TMA发起、传输还是槽位调度造成。
3. 当前数据不支持先以“去掉每块wait0”为主要提速手段；安全释放operand槽和读取accumulator仍必须等待相应异步计算完成。

本次只做测量与归因，没有实施这些优化，也没有证明某个后续优化一定能收回整个8.5us。

## 文件与复现

- `records.csv.gz`：逐块稀疏采样原始时间戳；`aggregate/records.csv.gz`：累计样本，role5字段t0/t1/t2分别是三项累计ns，t3是生产循环ns，不是绝对时间戳。
- `summary.json`、`aggregate_summary.json`、`aggregate/summary.json`：逐块与累计统计、插桩事件时间变化。
- `source/`：固定baseline、instrumented源码、harness和schema；`generate.py`：从固定提交生成两个命名空间的源码；`analyze.py`：分析脚本。
- `instrumented_sass.txt.gz`：最终插桩版本全部反汇编；`production_sass.txt.gz`：生产binary反汇编；`verification.json`：baseline与生产SASS逐条一致性。
- build/run/timing日志、正式Graph CSV与输出一并保留。
- 在仓库根目录运行`bash trajectory/20261014_profile-production-chain/run.sh`复现全部模式。默认输出到results/production_chain_replay；GPU分配沿用scripts/on_gpu.sh。
