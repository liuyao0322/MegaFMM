# MegaFMM 实验与验收记录

本文件由 `scripts/summarize_results.py` 生成，时间 2026-10-13T04:34:16+00:00。

**未获得数值认证的强制 SW 路径只能作为研究候选，不能进入 Auto。Auto 保持调优普通 GEMM；性能结果不能替代数值验收。未经正确性检查的 `unchecked` 行也不能作为启用快速路径的依据。**

每个 speedup 都由同一 CSV 行的调优普通 GEMM 与实际执行后端的完整调用计算，精度、形状、launch/Graph 模式、warm-cache/轮换输入模式相同。不同 dtype、执行模式和 SW/dense fallback 后端分别展示，未跨模式汇总或计算总体平均加速比。

已发现最终命名数据集 9 个、94 行；这些命名表示复测来源，不自动表示性能或精度验收通过。

SW 已检查 88 行：0 行固定数值门槛通过，88 行未通过。Dense fallback 已检查 6 行：6 行 fallback 通过，0 行失败；这些通过均不是 SW 认证。普通/其他后端已检查 0 行；另有 0 行 unchecked，不参与 SW 验收。

满足固定 SW 数值门槛、完整 `100/5/3` 或更充分计时配置且单行 speedup ≥ 1.05 的候选观察为 0 行。单行候选还需要预先指定的相邻大尺寸和超出噪声的重复证据，不能直接视为性能验收通过。

原始输入先舍入至 BF16/FP16，再直接转 FP64 计算普通 GEMM 参考，参考输入不经过 S/T 变换。`actual` 字段决定后端分类；请求 SW 后实际执行 `dense_cublaslt` 的边界、未知范围或资源 fallback 结果只验证普通 GEMM 路由。

固定门槛为每次独立试验 `eF_SW ≤ 3*eF_dense + floor`，BF16 floor 为 `1e-3`，FP16 为 `1e-4`。门槛未为某个候选实现放宽；CSV 的误差和 gate_limit 是跨正确性试验的最大值，`numerical_pass` 是逐试验验收的合取。最大误差试验与最大门槛试验可能不同，不能仅用两个最大值替代逐试验验收。

## 原始数据与计时配置

| 来源 | 行数 | 配置 trials/groups/correct | 模式 | 结果状态 |
| --- | ---: | --- | --- | --- |
| [final_lt_launch](<final_lt_launch.csv>) | 32 | 100/5/3 | launch/warm_cache/pairs=1 | 完整计时配置；SW 数值失败；fallback 通过 6 行（非 SW 认证） |
| [final_lt_graph](<final_lt_graph.csv>) | 10 | 100/5/3 | graph/warm_cache/pairs=1 | 完整计时配置；SW 数值失败 |
| [final_lt_rotating](<final_lt_rotating.csv>) | 4 | 100/5/3 | launch/rotating_inputs/pairs=4 | 完整计时配置；SW 数值失败 |
| [final_lt_rotating_graph](<final_lt_rotating_graph.csv>) | 4 | 100/5/3 | graph/rotating_inputs/pairs=4 | 完整计时配置；SW 数值失败 |
| [final_hopper3_launch](<final_hopper3_launch.csv>) | 10 | 100/5/3 | launch/warm_cache/pairs=1 | 完整计时配置；SW 数值失败 |
| [final_hopper4_launch](<final_hopper4_launch.csv>) | 10 | 100/5/3 | launch/warm_cache/pairs=1 | 完整计时配置；SW 数值失败 |
| [final_hopper3_graph](<final_hopper3_graph.csv>) | 10 | 100/5/3 | graph/warm_cache/pairs=1 | 完整计时配置；SW 数值失败 |
| [final_hopper4_graph](<final_hopper4_graph.csv>) | 10 | 100/5/3 | graph/warm_cache/pairs=1 | 完整计时配置；SW 数值失败 |
| [final_mega3_launch](<final_mega3_launch.csv>) | 4 | 100/5/3 | launch/warm_cache/pairs=1 | 完整计时配置；SW 数值失败 |

`current.csv` 是可变文件，与轨迹快照重复，因此自动发现时不纳入。默认自动发现包括全部实验轨迹，不代表最终候选选择；发布最终验收表时使用 `--files` 明确列出已经完成的最终 CSV。轨迹目录保存各自源码快照和输出；跨来源时钟/共享干扰可能不同，同形状跨来源的绝对时间不能作为受控配对实验。

## 完整调用的逐行性能与精度

### BF16 / launch / warm_cache / input pairs=1 / dense_fallback

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | 实际后端 median [p10,p90] ms | 同一行时间比 | CV dense/实际后端 | eF dense/实际后端 | fixed floor / max gate | 分类与数值状态 | workspace dense/实际后端 GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_launch / dense_cublaslt v0 | 1,1,1 | 0.0029 [0.0029,0.0029] | 0.0029 [0.0029,0.0030] | 0.9976× | 0.45% / 2.36% | 0.000e+00 / 0.000e+00 | 1.00e-03 / 1.000e-03 | Dense fallback 通过；非 SW 认证 | 0.000 / 0.000 |
| final_lt_launch / dense_cublaslt v0 | 514,770,258 | 0.0087 [0.0087,0.0087] | 0.0087 [0.0087,0.0087] | 0.9989× | 0.14% / 0.15% | 1.897e-07 / 1.897e-07 | 1.00e-03 / 1.001e-03 | Dense fallback 通过；非 SW 认证 | 0.006 / 0.006 |
| final_lt_launch / dense_cublaslt v0 | 1025,1537,769 | 0.0373 [0.0373,0.0373] | 0.0374 [0.0374,0.0374] | 0.9977× | 0.09% / 0.08% | 3.645e-07 / 3.645e-07 | 1.00e-03 / 1.001e-03 | Dense fallback 通过；非 SW 认证 | 0.023 / 0.023 |

### FP16 / launch / warm_cache / input pairs=1 / dense_fallback

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | 实际后端 median [p10,p90] ms | 同一行时间比 | CV dense/实际后端 | eF dense/实际后端 | fixed floor / max gate | 分类与数值状态 | workspace dense/实际后端 GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_launch / dense_cublaslt v0 | 1,1,1 | 0.0029 [0.0029,0.0030] | 0.0029 [0.0029,0.0029] | 0.9978× | 1.77% / 0.26% | 0.000e+00 / 0.000e+00 | 1.00e-04 / 1.000e-04 | Dense fallback 通过；非 SW 认证 | 0.000 / 0.000 |
| final_lt_launch / dense_cublaslt v0 | 514,770,258 | 0.0087 [0.0087,0.0087] | 0.0087 [0.0087,0.0087] | 0.9997× | 0.17% / 0.09% | 3.410e-07 / 3.410e-07 | 1.00e-04 / 1.010e-04 | Dense fallback 通过；非 SW 认证 | 0.006 / 0.006 |
| final_lt_launch / dense_cublaslt v0 | 1025,1537,769 | 0.0365 [0.0363,0.0365] | 0.0364 [0.0364,0.0365] | 1.0019× | 0.22% / 0.16% | 9.333e-07 / 9.333e-07 | 1.00e-04 / 1.028e-04 | Dense fallback 通过；非 SW 认证 | 0.012 / 0.012 |

### BF16 / graph / rotating_inputs / input pairs=4 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_rotating_graph / winograd_lt_v1 v0 | 8192,8192,8192 | 1.6096 [1.6047,1.8002] | 1.9345 [1.9251,1.9836] | 0.8320× | 5.83% / 1.41% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_lt_rotating_graph / winograd_lt_v1 v0 | 16384,16384,16384 | 12.7177 [12.6929,12.7385] | 12.9263 [12.9152,12.9755] | 0.9839× | 0.18% / 0.21% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 3.750 |

### FP16 / graph / rotating_inputs / input pairs=4 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_rotating_graph / winograd_lt_v1 v0 | 8192,8192,8192 | 1.7356 [1.7252,1.8124] | 2.0150 [1.9604,2.0622] | 0.8613× | 2.49% / 2.30% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.812 |
| final_lt_rotating_graph / winograd_lt_v1 v0 | 16384,16384,16384 | 13.2999 [13.2481,13.3154] | 13.4817 [13.4494,13.5123] | 0.9865× | 0.23% / 0.21% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.765 |

### BF16 / graph / warm_cache / input pairs=1 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_hopper3_graph / winograd_hopper_v2 v3 | 4096,4096,4096 | 0.2014 [0.1963,0.2109] | 0.3188 [0.3173,0.3261] | 0.6318× | 3.29% / 1.61% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.172 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 4096,4096,4096 | 0.2019 [0.1980,0.2073] | 0.3185 [0.3098,0.3235] | 0.6339× | 2.14% / 2.04% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.172 |
| final_lt_graph / winograd_lt_v1 v0 | 4096,4096,4096 | 0.2022 [0.1969,0.2109] | 0.3319 [0.3119,0.3413] | 0.6092× | 3.37% / 4.02% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.172 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 8192,8192,8192 | 1.6751 [1.6206,1.7744] | 1.8825 [1.8581,1.8990] | 0.8899× | 4.24% / 1.00% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 8192,8192,8192 | 1.7038 [1.6508,1.7570] | 1.8547 [1.8469,1.8584] | 0.9187× | 2.74% / 0.28% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_lt_graph / winograd_lt_v1 v0 | 8192,8192,8192 | 1.7114 [1.6102,1.7230] | 1.9469 [1.9419,1.9984] | 0.8790× | 3.18% / 1.50% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 12288,12288,12288 | 5.2778 [5.1895,5.4223] | 6.0865 [6.0256,6.2000] | 0.8671× | 2.02% / 1.27% | 1.321e-05 / 6.024e-03 | 1.00e-03 / 1.040e-03 | SW 数值失败 | 0.012 / 1.559 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 12288,12288,12288 | 5.3982 [5.3301,5.5506] | 5.8670 [5.7829,5.9294] | 0.9201× | 1.84% / 1.16% | 1.321e-05 / 6.024e-03 | 1.00e-03 / 1.040e-03 | SW 数值失败 | 0.012 / 1.559 |
| final_lt_graph / winograd_lt_v1 v0 | 12288,12288,12288 | 5.3328 [5.2731,5.4531] | 6.1208 [5.9975,6.1855] | 0.8713× | 1.49% / 1.38% | 1.321e-05 / 6.024e-03 | 1.00e-03 / 1.040e-03 | SW 数值失败 | 0.012 / 2.109 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 14336,14336,14336 | 8.6071 [8.6018,8.6251] | 9.7610 [9.6818,9.8094] | 0.8818× | 0.13% / 0.58% | 1.555e-05 / 6.026e-03 | 1.00e-03 / 1.047e-03 | SW 数值失败 | 0.012 / 2.117 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 14336,14336,14336 | 8.5477 [8.5358,8.5765] | 8.9072 [8.8962,8.9329] | 0.9596× | 0.21% / 0.18% | 1.555e-05 / 6.026e-03 | 1.00e-03 / 1.047e-03 | SW 数值失败 | 0.012 / 2.117 |
| final_lt_graph / winograd_lt_v1 v0 | 14336,14336,14336 | 8.5868 [8.5718,8.5973] | 9.2185 [9.2038,9.3391] | 0.9315× | 0.13% / 0.89% | 1.555e-05 / 6.026e-03 | 1.00e-03 / 1.047e-03 | SW 数值失败 | 0.012 / 2.871 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 16384,16384,16384 | 12.7398 [12.6682,12.7639] | 14.3374 [14.2827,14.4129] | 0.8886× | 0.34% / 0.41% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 2.762 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 16384,16384,16384 | 12.7695 [12.7464,12.8238] | 13.2510 [13.2285,13.3534] | 0.9637× | 0.31% / 0.43% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 2.762 |
| final_lt_graph / winograd_lt_v1 v0 | 16384,16384,16384 | 12.7270 [12.6986,12.7378] | 12.9819 [12.9256,13.0092] | 0.9804× | 0.15% / 0.29% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 3.750 |

### FP16 / graph / warm_cache / input pairs=1 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_hopper3_graph / winograd_hopper_v2 v3 | 4096,4096,4096 | 0.2074 [0.2067,0.2146] | 0.3436 [0.3144,0.3620] | 0.6036× | 2.14% / 6.19% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.297 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 4096,4096,4096 | 0.2093 [0.2080,0.2114] | 0.3330 [0.3179,0.3423] | 0.6284× | 0.74% / 3.63% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.297 |
| final_lt_graph / winograd_lt_v1 v0 | 4096,4096,4096 | 0.2094 [0.2078,0.2175] | 0.3684 [0.3539,0.3768] | 0.5684× | 2.09% / 2.82% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.297 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 8192,8192,8192 | 1.7473 [1.7373,1.8317] | 1.9798 [1.9688,2.0065] | 0.8826× | 3.12% / 0.93% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.702 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 8192,8192,8192 | 1.7686 [1.7132,1.8478] | 1.9313 [1.9237,1.9917] | 0.9158× | 3.66% / 1.69% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.702 |
| final_lt_graph / winograd_lt_v1 v0 | 8192,8192,8192 | 1.7588 [1.7043,1.7960] | 2.0178 [1.9757,2.0696] | 0.8716× | 2.45% / 2.09% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.812 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 12288,12288,12288 | 5.5977 [5.5040,5.7266] | 6.2175 [6.1688,6.4161] | 0.9003× | 1.79% / 1.79% | 1.452e-05 / 7.532e-04 | 1.00e-04 / 1.436e-04 | SW 数值失败 | 0.012 / 1.559 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 12288,12288,12288 | 5.6250 [5.5050,5.7202] | 6.0706 [5.9568,6.1576] | 0.9266× | 1.71% / 1.48% | 1.452e-05 / 7.532e-04 | 1.00e-04 / 1.436e-04 | SW 数值失败 | 0.012 / 1.559 |
| final_lt_graph / winograd_lt_v1 v0 | 12288,12288,12288 | 5.6430 [5.5589,5.7314] | 6.3048 [6.2590,6.3875] | 0.8950× | 1.41% / 0.94% | 1.452e-05 / 7.532e-04 | 1.00e-04 / 1.436e-04 | SW 数值失败 | 0.012 / 1.828 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 14336,14336,14336 | 8.9526 [8.9371,8.9747] | 10.0044 [9.8697,10.1163] | 0.8949× | 0.19% / 1.10% | 1.693e-05 / 7.530e-04 | 1.00e-04 / 1.508e-04 | SW 数值失败 | 0.012 / 2.117 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 14336,14336,14336 | 8.9616 [8.9103,8.9805] | 9.2264 [9.2125,9.2787] | 0.9713× | 0.35% / 0.33% | 1.693e-05 / 7.530e-04 | 1.00e-04 / 1.508e-04 | SW 数值失败 | 0.012 / 2.117 |
| final_lt_graph / winograd_lt_v1 v0 | 14336,14336,14336 | 8.9397 [8.9167,8.9495] | 9.1373 [9.1143,9.1535] | 0.9784× | 0.17% / 0.19% | 1.693e-05 / 7.530e-04 | 1.00e-04 / 1.508e-04 | SW 数值失败 | 0.012 / 2.488 |
| final_hopper3_graph / winograd_hopper_v2 v3 | 16384,16384,16384 | 13.3058 [13.2527,13.3398] | 14.7822 [14.6001,14.8865] | 0.9001× | 0.29% / 0.88% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.750 |
| final_hopper4_graph / winograd_hopper_v2 v4 | 16384,16384,16384 | 13.2930 [13.2678,13.3101] | 13.7487 [13.7333,13.7665] | 0.9668× | 0.15% / 0.11% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.750 |
| final_lt_graph / winograd_lt_v1 v0 | 16384,16384,16384 | 13.2751 [13.2701,13.3150] | 13.4714 [13.4355,13.4792] | 0.9854× | 0.17% / 0.16% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.765 |

### BF16 / launch / rotating_inputs / input pairs=4 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_rotating / winograd_lt_v1 v0 | 8192,8192,8192 | 1.6500 [1.5903,1.7914] | 1.9404 [1.9158,1.9977] | 0.8503× | 5.60% / 2.06% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_lt_rotating / winograd_lt_v1 v0 | 16384,16384,16384 | 12.7074 [12.6823,12.7528] | 12.9631 [12.9338,12.9874] | 0.9803× | 0.26% / 0.20% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 3.750 |

### FP16 / launch / rotating_inputs / input pairs=4 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_rotating / winograd_lt_v1 v0 | 8192,8192,8192 | 1.7416 [1.6986,1.8034] | 2.0019 [1.9644,2.0767] | 0.8700× | 2.84% / 2.52% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.812 |
| final_lt_rotating / winograd_lt_v1 v0 | 16384,16384,16384 | 13.2599 [13.2333,13.3185] | 13.4799 [13.4568,13.4938] | 0.9837× | 0.32% / 0.13% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.765 |

### BF16 / launch / warm_cache / input pairs=1 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_launch / winograd_lt_v1 v0 | 512,512,512 | 0.0059 [0.0059,0.0059] | 0.0438 [0.0438,0.0439] | 0.1341× | 0.19% / 0.04% | 3.941e-07 / 6.049e-03 | 1.00e-03 / 1.001e-03 | SW 数值失败 | 0.008 / 0.010 |
| final_lt_launch / winograd_lt_v1 v0 | 1024,1024,1024 | 0.0125 [0.0125,0.0125] | 0.0538 [0.0537,0.0538] | 0.2321× | 0.12% / 0.08% | 8.523e-07 / 6.015e-03 | 1.00e-03 / 1.003e-03 | SW 数值失败 | 0.031 / 0.042 |
| final_lt_launch / winograd_lt_v1 v0 | 1024,8192,512 | 0.0260 [0.0260,0.0261] | 0.1104 [0.1104,0.1105] | 0.2358× | 0.18% / 0.06% | 3.933e-07 / 6.025e-03 | 1.00e-03 / 1.001e-03 | SW 数值失败 | 0.000 / 0.095 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,1024,512 | 0.0257 [0.0256,0.0267] | 0.1258 [0.1257,0.1270] | 0.2040× | 2.62% / 0.51% | 3.923e-07 / 6.057e-03 | 1.00e-03 / 1.001e-03 | SW 数值失败 | 0.000 / 0.095 |
| final_lt_launch / winograd_lt_v1 v0 | 2048,2048,2048 | 0.0311 [0.0310,0.0311] | 0.1296 [0.1290,0.1296] | 0.2402× | 0.16% / 0.23% | 1.857e-06 / 6.025e-03 | 1.00e-03 / 1.006e-03 | SW 数值失败 | 0.000 / 0.074 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 4096,4096,4096 | 0.2006 [0.1976,0.2091] | 0.3377 [0.3228,0.3443] | 0.5942× | 3.15% / 2.95% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.172 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 4096,4096,4096 | 0.1990 [0.1965,0.2105] | 0.3307 [0.3204,0.3427] | 0.6018× | 3.34% / 3.08% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.172 |
| final_lt_launch / winograd_lt_v1 v0 | 4096,4096,4096 | 0.1978 [0.1954,0.2080] | 0.3518 [0.3446,0.3654] | 0.5621× | 3.37% / 2.65% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.172 |
| final_mega3_launch / megakernel_m3_pipeline v3 | 4096,4096,4096 | 0.1970 [0.1966,0.1972] | 2.6663 [2.6461,2.7097] | 0.0739× | 0.13% / 1.07% | 4.024e-06 / 6.027e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.000 |
| final_lt_launch / winograd_lt_v1 v0 | 4096,8192,8192 | 0.8221 [0.8128,0.8956] | 1.0413 [1.0170,1.0684] | 0.7895× | 4.59% / 2.40% | 8.551e-06 / 6.026e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 0.500 / 0.906 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,4096,8192 | 0.8255 [0.8208,0.8593] | 1.0657 [1.0278,1.1135] | 0.7746× | 2.38% / 3.96% | 8.551e-06 / 6.025e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 0.500 / 0.906 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,8192,4096 | 0.8707 [0.8296,0.9205] | 1.1251 [1.1015,1.1608] | 0.7739× | 4.68% / 2.58% | 4.025e-06 / 6.024e-03 | 1.00e-03 / 1.012e-03 | SW 数值失败 | 0.000 / 0.563 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 8192,8192,8192 | 1.6844 [1.6322,1.7484] | 1.9240 [1.9218,1.9288] | 0.8755× | 3.05% / 0.18% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 8192,8192,8192 | 1.6724 [1.6134,1.7525] | 1.8785 [1.8455,1.9358] | 0.8903× | 3.96% / 2.40% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,8192,8192 | 1.7175 [1.6214,1.7214] | 1.9354 [1.9307,1.9846] | 0.8874× | 2.85% / 1.37% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 1.688 |
| final_mega3_launch / megakernel_m3_pipeline v3 | 8192,8192,8192 | 1.5897 [1.5522,1.5955] | 21.6708 [21.6180,21.6990] | 0.0734× | 1.34% / 0.17% | 8.551e-06 / 6.023e-03 | 1.00e-03 / 1.026e-03 | SW 数值失败 | 1.000 / 0.000 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 12288,12288,12288 | 5.4438 [5.3491,5.5684] | 6.0868 [6.0212,6.2423] | 0.8944× | 1.82% / 1.65% | 1.321e-05 / 6.024e-03 | 1.00e-03 / 1.040e-03 | SW 数值失败 | 0.012 / 1.559 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 12288,12288,12288 | 5.4239 [5.3531,5.5366] | 5.8670 [5.8166,5.9436] | 0.9245× | 1.53% / 0.98% | 1.321e-05 / 6.024e-03 | 1.00e-03 / 1.040e-03 | SW 数值失败 | 0.012 / 1.559 |
| final_lt_launch / winograd_lt_v1 v0 | 12288,12288,12288 | 5.3728 [5.2755,5.4517] | 6.0706 [6.0040,6.1713] | 0.8850× | 1.45% / 1.41% | 1.321e-05 / 6.024e-03 | 1.00e-03 / 1.040e-03 | SW 数值失败 | 0.012 / 2.109 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 14336,14336,14336 | 8.5743 [8.5586,8.6008] | 9.8438 [9.6641,9.8759] | 0.8710× | 0.26% / 1.12% | 1.555e-05 / 6.026e-03 | 1.00e-03 / 1.047e-03 | SW 数值失败 | 0.012 / 2.117 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 14336,14336,14336 | 8.5594 [8.5320,8.5805] | 8.9314 [8.9133,8.9564] | 0.9584× | 0.27% / 0.24% | 1.555e-05 / 6.026e-03 | 1.00e-03 / 1.047e-03 | SW 数值失败 | 0.012 / 2.117 |
| final_lt_launch / winograd_lt_v1 v0 | 14336,14336,14336 | 8.5699 [8.5602,8.5884] | 9.2260 [9.2145,9.3950] | 0.9289× | 0.16% / 0.93% | 1.555e-05 / 6.026e-03 | 1.00e-03 / 1.047e-03 | SW 数值失败 | 0.012 / 2.871 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 16384,16384,16384 | 12.7681 [12.7309,12.7914] | 14.3852 [14.3528,14.4487] | 0.8876× | 0.23% / 0.30% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 2.762 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 16384,16384,16384 | 12.7486 [12.7461,12.7807] | 13.2942 [13.2613,13.3338] | 0.9590× | 0.13% / 0.26% | 1.791e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 2.762 |
| final_lt_launch / winograd_lt_v1 v0 | 16384,16384,16384 | 12.8424 [12.8156,12.9201] | 12.9445 [12.9143,12.9705] | 0.9921× | 0.37% / 0.21% | 1.790e-05 / 6.023e-03 | 1.00e-03 / 1.054e-03 | SW 数值失败 | 0.012 / 3.750 |

### FP16 / launch / warm_cache / input pairs=1 / sw

| 来源 / 实际算法 | M,N,K | Dense median [p10,p90] ms | SW median [p10,p90] ms | 同一行时间比 | CV dense/SW | eF dense/SW | fixed floor / max gate | 分类与数值状态 | workspace dense/SW GiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| final_lt_launch / winograd_lt_v1 v0 | 512,512,512 | 0.0059 [0.0059,0.0059] | 0.0413 [0.0412,0.0413] | 0.1433× | 0.13% / 0.05% | 6.443e-07 / 7.569e-04 | 1.00e-04 / 1.019e-04 | SW 数值失败 | 0.004 / 0.007 |
| final_lt_launch / winograd_lt_v1 v0 | 1024,1024,1024 | 0.0098 [0.0098,0.0099] | 0.0525 [0.0525,0.0526] | 0.1874× | 0.08% / 0.05% | 1.251e-06 / 7.546e-04 | 1.00e-04 / 1.038e-04 | SW 数值失败 | 0.031 / 0.042 |
| final_lt_launch / winograd_lt_v1 v0 | 1024,8192,512 | 0.0248 [0.0248,0.0249] | 0.1120 [0.1120,0.1121] | 0.2214× | 0.17% / 0.07% | 6.435e-07 / 7.532e-04 | 1.00e-04 / 1.019e-04 | SW 数值失败 | 0.062 / 0.126 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,1024,512 | 0.0252 [0.0251,0.0253] | 0.1098 [0.1096,0.1106] | 0.2293× | 0.37% / 0.44% | 6.434e-07 / 7.549e-04 | 1.00e-04 / 1.019e-04 | SW 数值失败 | 0.000 / 0.079 |
| final_lt_launch / winograd_lt_v1 v0 | 2048,2048,2048 | 0.0317 [0.0316,0.0317] | 0.1107 [0.1105,0.1108] | 0.2858× | 0.11% / 0.14% | 2.458e-06 / 7.536e-04 | 1.00e-04 / 1.074e-04 | SW 数值失败 | 0.031 / 0.074 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 4096,4096,4096 | 0.2092 [0.2085,0.2185] | 0.3682 [0.3591,0.3818] | 0.5683× | 2.77% / 2.99% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.297 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 4096,4096,4096 | 0.2088 [0.2074,0.2177] | 0.3535 [0.3487,0.3724] | 0.5906× | 2.55% / 3.77% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.297 |
| final_lt_launch / winograd_lt_v1 v0 | 4096,4096,4096 | 0.2090 [0.2069,0.2150] | 0.3737 [0.3649,0.3866] | 0.5594× | 1.79% / 2.81% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.297 |
| final_mega3_launch / megakernel_m3_pipeline v3 | 4096,4096,4096 | 0.2082 [0.2078,0.2089] | 2.7022 [2.6621,2.7370] | 0.0770× | 0.24% / 1.23% | 4.877e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.125 / 0.000 |
| final_lt_launch / winograd_lt_v1 v0 | 4096,8192,8192 | 0.8609 [0.8453,0.9283] | 1.1141 [1.0922,1.1538] | 0.7727× | 4.33% / 2.88% | 9.690e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.007 / 0.469 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,4096,8192 | 0.8644 [0.8416,0.8955] | 1.1067 [1.0605,1.1636] | 0.7811× | 3.00% / 4.25% | 9.690e-06 / 7.532e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.007 / 0.469 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,8192,4096 | 0.8875 [0.8534,0.9058] | 1.1690 [1.1473,1.2228] | 0.7592× | 2.74% / 3.38% | 4.878e-06 / 7.531e-04 | 1.00e-04 / 1.146e-04 | SW 数值失败 | 0.500 / 1.062 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 8192,8192,8192 | 1.7214 [1.7002,1.7899] | 2.0071 [1.9776,2.0504] | 0.8577× | 2.43% / 1.66% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.702 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 8192,8192,8192 | 1.7385 [1.6642,1.8588] | 1.9542 [1.9008,2.0177] | 0.8896× | 4.97% / 2.93% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.702 |
| final_lt_launch / winograd_lt_v1 v0 | 8192,8192,8192 | 1.7720 [1.6973,1.8320] | 1.9849 [1.9178,2.0963] | 0.8927× | 3.64% / 3.97% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.812 |
| final_mega3_launch / megakernel_m3_pipeline v3 | 8192,8192,8192 | 1.6541 [1.6135,1.6783] | 21.8080 [21.7409,21.8426] | 0.0758× | 1.82% / 0.21% | 9.689e-06 / 7.534e-04 | 1.00e-04 / 1.291e-04 | SW 数值失败 | 0.015 / 0.000 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 12288,12288,12288 | 5.6402 [5.5096,5.7031] | 6.2706 [6.2208,6.4419] | 0.8995× | 1.67% / 1.59% | 1.452e-05 / 7.532e-04 | 1.00e-04 / 1.436e-04 | SW 数值失败 | 0.012 / 1.559 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 12288,12288,12288 | 5.5793 [5.4905,5.7208] | 6.0522 [5.9699,6.1782] | 0.9219× | 1.83% / 1.53% | 1.452e-05 / 7.532e-04 | 1.00e-04 / 1.436e-04 | SW 数值失败 | 0.012 / 1.559 |
| final_lt_launch / winograd_lt_v1 v0 | 12288,12288,12288 | 5.6722 [5.5414,5.7002] | 6.3147 [6.2286,6.4314] | 0.8982× | 1.30% / 1.41% | 1.452e-05 / 7.532e-04 | 1.00e-04 / 1.436e-04 | SW 数值失败 | 0.012 / 1.828 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 14336,14336,14336 | 8.9269 [8.9143,8.9646] | 10.1963 [9.9872,10.2578] | 0.8755× | 0.26% / 1.32% | 1.693e-05 / 7.530e-04 | 1.00e-04 / 1.508e-04 | SW 数值失败 | 0.012 / 2.117 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 14336,14336,14336 | 8.9340 [8.8985,8.9831] | 9.2364 [9.2313,9.2976] | 0.9673× | 0.42% / 0.34% | 1.693e-05 / 7.530e-04 | 1.00e-04 / 1.508e-04 | SW 数值失败 | 0.012 / 2.117 |
| final_lt_launch / winograd_lt_v1 v0 | 14336,14336,14336 | 8.9576 [8.9125,8.9857] | 9.1380 [9.1116,9.1517] | 0.9803× | 0.38% / 0.20% | 1.693e-05 / 7.530e-04 | 1.00e-04 / 1.508e-04 | SW 数值失败 | 0.012 / 2.488 |
| final_hopper3_launch / winograd_hopper_v2 v3 | 16384,16384,16384 | 13.2990 [13.2830,13.3092] | 14.9650 [14.9243,15.0078] | 0.8887× | 0.09% / 0.25% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.750 |
| final_hopper4_launch / winograd_hopper_v2 v4 | 16384,16384,16384 | 13.2802 [13.2386,13.3062] | 13.7397 [13.7213,13.7581] | 0.9666× | 0.23% / 0.12% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.750 |
| final_lt_launch / winograd_lt_v1 v0 | 16384,16384,16384 | 13.3023 [13.2383,13.3079] | 13.4726 [13.4578,13.4863] | 0.9874× | 0.25% / 0.10% | 1.933e-05 / 7.531e-04 | 1.00e-04 / 1.580e-04 | SW 数值失败 | 0.000 / 2.765 |

## 静态导出图

[performance.svg](<plots/performance.svg>)
[performance.png](<plots/performance.png>)
[numerical.svg](<plots/numerical.svg>)
[numerical.png](<plots/numerical.png>)

性能图使用逐行加速比，误差条端点为 `dense_p10 / solution_p90` 与 `dense_p90 / solution_p10`，用于表示组计时离散度，不能解释为置信区间。数值图实线为每来源最差正确性试验 eF，黑虚线为固定 floor，灰点线为各来源最大试验 gate_limit；精确零误差仅为对数显示而绘制于 `1e-12`。

## 计时和误差解释

计时采用预分配、预调优的完整调用；实际执行 SW 的行每次重新执行依赖 A/B 的变换、全部七个叶乘法和 FP32 重组，fallback 行完整执行普通 GEMM。库调优、FP64 参考、输入更新在计时区间之外。Graph 捕获包含实际后端的完整调用；Graph 与普通 launch 的结果分别报告。beta 非零时 C0 语义在功能测试中独立覆盖。

CSV 的 median、p10、p90 和 CV 来自独立重复组的每调用平均时间（`quantile_unit=group_mean`），而非单次 invocation 的延迟分布。speedup 是两组 median 的比值。只有五组时尾分位数和 CV 的估计有限；接近 1.05× 的收益必须结合配对重复结果、时钟和噪声验证。

warm_cache 重复使用同一对地址，仍每次重算变换。rotating_inputs 轮换多个不同的 A/B 地址对，记录实际输入工作集；应根据工作集与 L2 的大小判断缓存影响，不能仅凭轮换标签断言完全冷缓存。轮换数据不与 warm-cache 合并。

原始 CSV 还保留 maxabs、Frobenius eF、`|error|/(atol+rtol*|reference|)` 最大值、`||A||F||B||F` 归一化误差、近零参考的绝对误差、nonfinite 数量、主机 wall time、实际 workspace 与初始化/调优成本。等效 TFLOP/s 使用 `2MNK/T`，不能等同七叶的实际 Tensor Core 算术吞吐。

## 工具与独立证据

结构和接口测试通过：[final_correctness.txt](<final_correctness.txt>)。结构正确性与混合精度数值认证是不同关口。

精确 DAG 测试通过：[final_dag.txt](<final_dag.txt>)。

memcheck 实际报告 0 errors：[final_memcheck.txt](<final_memcheck.txt>)。结果只覆盖实际运行的用例，不替代异步协议审查，也不认证所有可能输入。

racecheck 实际报告 0 hazards、0 errors、0 warnings：[final_racecheck.txt](<final_racecheck.txt>)。日志包含 18 个 dtype/algorithm/variant 组合、44 个候选路径用例，覆盖 beta-zero、alpha/beta 原位 alias 和矩形 fallback。结果只覆盖实际运行的用例，不替代异步协议审查，也不认证所有可能输入。

synccheck 实际报告 0 errors：[final_synccheck.txt](<final_synccheck.txt>)。日志包含 18 个 dtype/algorithm/variant 组合、44 个候选路径用例，覆盖 beta-zero、alpha/beta 原位 alias 和矩形 fallback。结果只覆盖实际运行的用例，不替代异步协议审查，也不认证所有可能输入。

最终全候选严格数值压力组：[final_strict_numerics.txt](<final_strict_numerics.txt>)；退出状态 `2`：[final_strict_numerics.exit](<final_strict_numerics.exit>)。从实际 `NUMERICAL_GATE` 行解析得到 300 个试验、180 个门槛通过、120 个失败，nonfinite 元素合计 0。其中 60 个试验声明范围未知，60 个通过由普通 GEMM fallback 获得，不是 SW 认证。压力组中的精确 cancellation 通过也不能消除普通随机/动态范围样本的失败。

| pattern | 试验 | 通过 | 失败 | nonfinite | 范围未知/fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| cancellation | 60 | 60 | 0 | 0 | 0 |
| fp16_upper_range | 60 | 60 | 0 | 0 | 60 |
| positive | 60 | 60 | 0 | 0 | 0 |
| range | 60 | 0 | 60 | 0 | 0 |
| uniform | 60 | 0 | 60 | 0 | 0 |

最终测量来源与退出状态：[final_runs.csv](<final_runs.csv>)。目前记录 9 次调用、source commit `d3cff919a799fc665408b33c080ce66f91d766dd`、exit code `2`。benchmark 的 exit 2 表示数值门槛失败，不能忽略该状态而仅保留性能表。

最终测量前后 GPU telemetry 证据：

| 调用标签 | before | after |
| --- | --- | --- |
| hopper3_graph | [final_hopper3_graph_telemetry_before.csv](<final_hopper3_graph_telemetry_before.csv>) | [final_hopper3_graph_telemetry_after.csv](<final_hopper3_graph_telemetry_after.csv>) |
| hopper3_launch | [final_hopper3_launch_telemetry_before.csv](<final_hopper3_launch_telemetry_before.csv>) | [final_hopper3_launch_telemetry_after.csv](<final_hopper3_launch_telemetry_after.csv>) |
| hopper4_graph | [final_hopper4_graph_telemetry_before.csv](<final_hopper4_graph_telemetry_before.csv>) | [final_hopper4_graph_telemetry_after.csv](<final_hopper4_graph_telemetry_after.csv>) |
| hopper4_launch | [final_hopper4_launch_telemetry_before.csv](<final_hopper4_launch_telemetry_before.csv>) | [final_hopper4_launch_telemetry_after.csv](<final_hopper4_launch_telemetry_after.csv>) |
| lt_graph | [final_lt_graph_telemetry_before.csv](<final_lt_graph_telemetry_before.csv>) | [final_lt_graph_telemetry_after.csv](<final_lt_graph_telemetry_after.csv>) |
| lt_launch | [final_lt_launch_telemetry_before.csv](<final_lt_launch_telemetry_before.csv>) | [final_lt_launch_telemetry_after.csv](<final_lt_launch_telemetry_after.csv>) |
| lt_rotating | [final_lt_rotating_telemetry_before.csv](<final_lt_rotating_telemetry_before.csv>) | [final_lt_rotating_telemetry_after.csv](<final_lt_rotating_telemetry_after.csv>) |
| lt_rotating_graph | [final_lt_rotating_graph_telemetry_before.csv](<final_lt_rotating_graph_telemetry_before.csv>) | [final_lt_rotating_graph_telemetry_after.csv](<final_lt_rotating_graph_telemetry_after.csv>) |
| mega3_launch | [final_mega3_launch_telemetry_before.csv](<final_mega3_launch_telemetry_before.csv>) | [final_mega3_launch_telemetry_after.csv](<final_mega3_launch_telemetry_after.csv>) |

这些文件是每次测量前后的 UUID、时钟、功耗、温度和利用率快照，不是连续采样，不能据此声称受测 kernel 期间频率固定或完全没有共享干扰。

独立精确叶乘法量化反例：[mixed_precision_witness.json](<mixed_precision_witness.json>)。即使叶乘法无额外累加误差，最终 S/T 量化也可能超过固定 floor。

Nsight Compute probe 失败：`/tmp/nsight-compute-lock` 无法创建/打开，报告 `InterprocessLockFailed`，未获得 kernel 指标。本会话未重新 probe。证据：[profile_baseline.txt](<profile_baseline.txt>)。

Nsight Systems 使用 cudaProfilerApi 范围，只捕获调优后的最终完整调用。捕获对象：`8192,8192,8192 bf16 winograd_lt_v1 workspace=1811939328`。证据：[timeline_lt.txt](<timeline_lt.txt>)。

| 阶段 | GPU kernel 时间占比 | 累计 ms | instances |
| --- | ---: | ---: | ---: |
| leaves | 74.9% | 1.331391 | 7 |
| pack | 11.7% | 0.207520 | 2 |
| combine | 13.4% | 0.238784 | 1 |

阶段比例约为叶乘法 75%、两侧 packing 12%、combine 13%，对应一次被 profiler 捕获的 BF16 8192 调用。比例分母为 kernel 执行时间，不包含所有主机调度间隙；不能将 profiler replay/采样延迟作为 steady-state 性能成绩，也不能将该比例推广至全部尺寸或 FP16。

历史初始 Release probe 资源核验：[hopper_release_resources.txt](<hopper_release_resources.txt>)。该文件覆盖当时 Hopper variants 0/1/2 的 BF16/FP16 六个普通 leaf 核：REG 168、STACK/LOCAL 0，SASS LDL/STL 计数均为 0；warpgroup 存在 232/40 的寄存器重新分配。它不代表当前全部候选，尤其不覆盖后加的 Hopper variants 3/4。早期 probe 编译日志的 spill 数不代表当时实际 Release 对象；静态 SHARED=1024 之外还有动态 shared storage。

Megakernel pipeline 的编译与 SASS 证据：[mega_pipeline_resources.txt](<mega_pipeline_resources.txt>)。记录 iter 4 对象 v2 的 248 registers/thread、12 KiB 动态 shared，v3 的 233 registers/thread、24 KiB 动态 shared；两种 dtype 均无 stack/spill 和 LDL/STL。该记录检查了七个产品的下一 stage load/transform/store/publish 在对应 wait 前发出，以及未过早访问未完成的 accumulator。它证明的是静态编译指令顺序与所列用例的 sanitizer 状态，不提供实际重叠时长、动态指令数、stall 或 Tensor Core 利用率；非 v3 全 tile/对齐/beta-zero 范围使用 v2。

SubCuber 主源码兼容性审计：[subcuber_compatibility.json](<subcuber_compatibility.json>)。固定作者 commit `c7b6fd8c72db4fc83982b26ea6e80e47567d674f`，状态 `not_built_or_run`，没有 H100 编译/运行、性能测量或加速声明。未改动的作者 runner 没有 BF16 入口，所审查 FP16 kernel/runner 输出和部分中间 postsum 为 FP16；选定 global-presum presets 在 warmup 后复用 B 变换，不能直接比较本项目 FP32 输出、动态 A/B 的完整调用。该缓存问题不覆盖所有 L1 presets。公平对照需要独立 BF16/FP16 输入及 FP32 输出适配、每次动态变换计时、资源重新核算和本项目原始输入 FP64 参考认证；审计中的 deferred commands 未执行。

本次最终结果校验和与复测协议：[final_manifest.json](<final_manifest.json>)。记录测量源码 commit、solution tree、源文件/二进制/CSV SHA256、设备 UUID、固定门限与完整采样配置；该证据文件描述这次复测，后续重新运行需重新核对来源。

环境/依赖记录：[environment.txt](<environment.txt>)。

环境/依赖记录：[dependency.txt](<dependency.txt>)。

## 后续验收

需要同时满足固定数值门槛、完整调用性能和独立功能/内存检查，才能为特定 dtype 与 shape 启用快速路径。SW 的数值失败行、所有 unchecked 行、所有 dense fallback 行均不能用于 SW 性能验收或 SW 认证。fallback 通过只说明普通 GEMM 路由正常。

性能目标需要预先指定的至少两个相邻大尺寸，在同一 dtype、执行模式、缓存/轮换合同、算法版本和 variant 下，各自 median speedup ≥ 1.05，并由完整 trials≥100、groups≥5、correct_trials≥3 配置与超出噪声的重复数据支持。各尺寸也须通过冻结的逐试验精度门槛。单行、短测或跨精度/Graph 模式拼接的结果不足以完成验收。若约 5% 的收益与 CV/组分布变化相近，保留原始分组日志并增加独立重复确认；这里不从五组数据声称置信区间，也不自动宣布胜者。

最终报告通过 `--files` 选择完整最终数据集；默认含全部轨迹的报告用于实验回顾。后续可独立评估保持原输入 FP32 信息的叶策略、计入时间的补偿/缩放与不同量化合同，任何合同变更必须明确记录并重新建立同精度基线。
