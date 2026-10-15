# 精度与性能联合研究检查点，2026-10-15

目标仍在进行：找出同时满足 FP16/BF16 原始 FP64 精度合同、且不显著增加完整调用时间的实现。当前证据没有证明这一目标完成。

这轮没有修改 solution、bench 或原始验收门槛。当前 solution/sw_pipeline.cu SHA256 为 `748e7ace9de0b6469d47922cb7cb531bbb56a9d55c9915b0194c0e0a72e916e7`；当前分支保留的内核来自 iter330，不能把 README 中选择282的历史描述当作新的源码核验。

## 一、准确七乘法公式仍不足以解决 FP16 变换量化

源文献：Dumas/Pernet/Sedoglavic, *Towards automated generation of fast and accurate algorithms for recursive matrix multiplication*, https://arxiv.org/html/2506.19405 ，公式98、101、102。来源仅用于公式与背景；下面数字来自本仓库独立实验。

`screen.py` 录入三种公式及经典 Strassen、当前 Winograd。先验证4×4×4矩阵乘法张量恒等式，最大残差分别为1.67e-16、0、2.22e-16；经典Strassen与Winograd为0。所有主实验仍使用1024³、原始FP16/BF16量化输入、FP64 AB参考。变换在FP64中形成、送入叶乘法前量化到指定格式，叶乘法及重组使用FP64。因此这是变换量化诊断，不是完整GPU验收，也没有完全模拟FP32变换的每一步舍入。

三种独立seed=1,2,3，FP16输入和FP16叶操作数的误差范围：

| 公式 | 相对Frobenius误差 |
|---|---:|
| Winograd | 7.5214e-4–7.5403e-4 |
| 经典Strassen | 5.4904e-4–5.5001e-4 |
| DPS98 | 4.9229e-4–4.9386e-4 |
| DPS101 | 5.3353e-4–5.3434e-4 |
| DPS102 | 4.9927e-4–5.0205e-4 |

均高于固定FP16 floor=1e-4。公式优化有改善，但这三种公式不值得仅为精度目的直接移植当前GPU流水线。

假想增加尾数预算的Winograd诊断（不是硬件可用格式）：增加1位约3.47e-4；增加2位约1.21e-4；增加3位约3.98e-5。说明在这些输入上，需要比当前FP16变换明显更多的信息；不能把普通TF32误当成增加尾数位的解决方案。该实验不是信息论下界。

BF16输入改用FP16变换操作数在有界输入上仍改善明显，但不能独立解决FP16要求，也未处理一般BF16动态范围，不作为完成方案。

## 二、FP8残差修正：数值有余量，性能尚未解决

使用一阶修正 `Sl*Th + Sh*Tl`，保留当前Winograd的八个标准交叉项，省略低×低二阶项。所有FP8操作数按各自amax选取2的幂缩放；缩放量每次依赖当前输入，没有缓存计算结果。

`fp8_correction.py` 使用FP64叶乘法模拟FP8 E4M3/E5M2输入。在seed1,2,3上：

| 输入 | E4M3修正 eF | E5M2修正 eF |
|---|---:|---:|
| FP16 | 约1.99e-5 | 约4.03e-5 |
| BF16 | 约1.60e-4 | 约3.24e-4 |

这些数字支持用更低精度计算小修正量，而不是用同样FP16/BF16格式承担全部补偿成本。

`hardware_fp8.py` 在allocation107488/g08 H10080GB上，用真正E4M3 Tensor Core乘法交叉验证，seed101,102,103，与数值筛选的seed不同。主七项用禁用TF32的FP32 cuBLAS乘法；修正用torch._scaled_mm、FP32输出、use_fast_accum=False；参考为原始量化输入的FP64 AB。全部结果无nonfinite。

| 输入 | 全稠密E4M3修正 eF 范围 | 全2:4剪枝修正 eF 范围 | 四稠密+四剪枝 eF 范围 |
|---|---:|---:|---:|
| FP16 | 1.9915e-5–1.9993e-5 | 1.6935e-4–1.6955e-4 | 约9.02e-5–9.07e-5 |
| BF16 | 1.6068e-4–1.6125e-4 | 1.3579e-3–1.3582e-3 | 约7.23e-4–7.27e-4 |

全稠密修正低于两种固定floor，但仍不是生产内核验收：主乘法路径不同，尚未集成原始完整调用benchmark。

## 三、残差2:4稀疏的收益与限制

FP16变换残差有约33%–51%的精确零值，但没有天然符合每连续4项仅2项非零的保证。对每组保留绝对值最大的2项，会丢失约4.6%–6.5%的残差能量，导致全稀疏修正不通过固定floor。

`sparse_mix.py` 在seed201,202,203上枚举256个稠密/2:4剪枝修正子集，仍用FP64叶乘法、全局幂二缩放E4M3操作数。

| 稠密修正数 | 稀疏修正数 | FP16最坏eF | 算术模型中的FP16等价叶乘积总数 |
|---:|---:|---:|---:|
| 0 | 8 | 1.6977e-4 | 9.00 |
| 2 | 6 | 1.1426e-4 | 9.50 |
| 3 | 5 | 1.0291e-4 | 9.75 |
| 4 | 4 | 9.0414e-5 | 10.00 |
| 8 | 0 | 1.9942e-5 | 11.00 |

三稠密方案高于固定floor，仅接近历史实际gate，余量太小，不作为优先实现。四稠密方案采用索引3,4,5,6，即 `P5_Tlow, P6_Slow, P6_Tlow, P7_Slow`，其余四项剪枝，并在独立硬件seed101–103上再次验证了数值可行性。

注意：硬件脚本通过普通稠密FP8乘法计算已经剪枝的数据，尚未使用稀疏Tensor Core指令/metadata。因此只验证了剪枝后的数值，未验证硬件稀疏吞吐、压缩布局或metadata开销。所谓等价乘积数仅基于FP8相对FP16的理论吞吐以及稀疏2倍吞吐，是算术模型，不是完整调用速度预测；10/7也不能直接用于预测延迟。

## 四、单独追加修正路径的实测成本仍太高

所有下表计时都只计算已经量化的八个512³修正乘积，不包含变换、amax、缩放、量化、原七项主乘法或最终重组。不能作为生产方案完整调用时间。GPU未锁频，跨库测量不是严格交错配对排名。

| 实现 | Graph组中位数 |
|---|---:|
| PyTorch/cuBLAS12.8，八个独立FP8 GEMM | 约26.16µs |
| 独立C++/cuBLASLt12.4，八次调用 | 41.28µs |
| 独立C++/cuBLASLt12.4，一次strided batch8 | 10.01µs |
| 单Triton kernel，十组配置中最快 | 9.42µs |

C++batch实验每种模式枚举heuristic返回的8个算法候选，用独立stream捕获Graph，20次warmup、5组×100次调用，128个输出点对原FP8数据的FP64点积检查maxabs=0。输入为确定性量化数列，仅用于乘法原语，不代表完整应用分布。

Triton最佳配置BM64/BN64/BK128、4warps、3stages，输出相对于FP64参考的eF约3.76e-9。所有十种配置均通过其独立FP64检查。主调用没有修改。

额外发现：随机FP8乘积上，cuBLAS12.8 _scaled_mm 即使use_fast_accum=False，相对FP64仍有约1.259e-4误差，Triton max_num_imprecise_acc=0 的结果更接近FP64。最初把cuBLAS结果作为逐项严格golden会误判Triton；已改为直接检查FP64，未修改生产gate。该库差异没有改变上面的完整修正数值实验结论，但在新内核中仍需使用FP64参考。

结论：已测的追加式修正实现均不适合当前约10µs的主路径。单个Triton kernel也不能仅靠减少launch解决此问题。这不是对融合实现或所有算法的物理下界证明。

## 五、下一实验的具体方向

保留七乘法主路径，将FP8修正融入主乘法流水线而不是追加独立输出张量：

1. 输入变换时同时生成高部分与缩放残差，复用原输入读取。先采用全稠密E4M3修正，避免稀疏压缩复杂性掩盖基础流水线成本。
2. 让修正消费与对应主乘积相同的输入tile；优先复用片上数据，避免预量化八套矩阵和八个FP32修正输出的显存往返。
3. 修正累加器直接缩放并入对应输出重组，避免单独写回八个512² FP32修正矩阵。
4. 在独立原型中测试Hopper FP8 WGMMA的布局、K32和缩放累加，先完成原始FP64验收；只有精度通过才进行完整调用ABBA对比。
5. 如果全稠密融合开销仍大，再评估四稠密+四稀疏；原型必须包含真实metadata生成、压缩、缩放和全部同步成本。

尚未验证：融合修正的完整延迟、一般输入动态范围的安全缩放、workspace与shared占用、Graph与双stream安全、真实稀疏指令。任何一项缺失都不能宣称目标完成。

## 复现

在仓库根目录，使用现有H100 allocation；先确认107488仍RUNNING。以下命令均不修改生产源码。

```bash
bash scripts/on_gpu.sh env OPENBLAS_NUM_THREADS=4 /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261015/screen.py
bash scripts/on_gpu.sh env OPENBLAS_NUM_THREADS=4 /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261015/fp8_correction.py
bash scripts/on_gpu.sh env OPENBLAS_NUM_THREADS=4 /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261015/sparse_mix.py
bash scripts/on_gpu.sh env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261015/hardware_fp8.py
bash scripts/on_gpu.sh env -u LD_LIBRARY_PATH TRITON_CACHE_DIR=/tmp/liuyao-megafmm-precision-triton /share/home/liuyao/miniconda3/envs/kernelagent/bin/python research/precision_20261015/grouped_fp8.py
bash scripts/on_gpu.sh nvcc -O3 -std=c++17 -arch=sm_90a research/precision_20261015/batch_fp8.cu -lcublasLt -o research/precision_20261015/batch_fp8
bash scripts/on_gpu.sh research/precision_20261015/batch_fp8
```

PyTorch2.10.0+cu128与Triton3.6.0使用自身配套cuBLAS，必须避免scripts/on_gpu.sh加载的CUDA12.4 LD_LIBRARY_PATH覆盖；首次失败日志保留供核对。C++batch实验有意使用仓库CUDA12.4。没有安装依赖或下载模型。

本研究包含AI辅助文献提取、代码编写与实验分析。公式恒等式及数值、性能结论均由本地脚本输出核对；原始日志与JSON随目录保存。
