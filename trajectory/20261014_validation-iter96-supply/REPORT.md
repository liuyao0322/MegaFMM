# Iter88–96：供数、输入变换与 slot 释放优化

当前源码 iter96（e3837a8）逐文件恢复自 iter92（bf34e5a），基线为 iter87（c9380c3）。保留32位成对输入读写，以及A侧四个K64 TMA合并为一个K256搬运。仅验证H100 80GB、1024³、BF16/FP16输入、FP32累加和输出。完整调用含S/T变换、grid barrier、15项叶乘法和重组。Auto继续使用调优cuBLASLt Dense，SW尚未成为性能winner。

## 完整调用性能

下表来自最后一组同期87/92/95的A/B/C/C/B/A对照。每个版本重新构建，20次warmup、5组×100次调用、3组新输入精度检查；计时包含完整调用，不取独立阶段相加。单位μs。

| 模式 | 输入 | iter87 | 保留iter92 | 耗时下降 | 舍弃iter95 |
|---|---|---:|---:|---:|---:|
| launch | BF16 | 19.23712 | 18.93904 | 1.55% | 19.03984 |
| launch | FP16 | 19.20480 | 19.01120 | 1.01% | 19.26720 |
| Graph | BF16 | 18.89248 | 18.75664 | 0.72% | 18.59568 |
| Graph | FP16 | 19.08192 | 18.85904 | 1.17% | 18.54128 |

四轮同期对照中，92相对87的普通调用BF16下降1.29%–1.73%，FP16下降0.77%–1.51%；Graph BF16下降0.54%–1.60%，FP16范围为回退0.15%至下降1.17%。这是小幅收益，不能声称普遍达到3%或Graph总是改善。GPU未锁频，结果有时钟和运行波动。iter95 Graph更快但普通调用回退，本轮以普通完整调用为主要保留依据。

最终独立正式基准（final标签，非同期旧新版本对照）：

| 输入 | SW μs | cuBLASLt Dense μs | SW eF | 原门限 | 数值验收 |
|---|---:|---:|---:|---:|---|
| bf16 | 18.96224 | 12.26752 | 7.39458724362e-06 | 0.00100255679669 | PASS |
| fp16 | 18.95360 | 9.94816 | 2.37135524364e-06 | 0.000103752760587 | PASS |

## 新的因果证据：完成等待减少，供数等待反而增加

iter94允许两个WGMMA group在途：q0提交后不立即wait0，q1高部分提交后wait1再释放slot0，最后补偿项结束wait0释放slot1。算术和精度均通过，但普通完整调用BF16/FP16约20.46/20.37μs，较92约18.93/19.19μs回退。

| 每CTA累计区间中位数 μs | 92 BF16/FP16 | 94 BF16/FP16 |
|---|---:|---:|
| operand ready等待（高+低） | 2.336 / 2.336 | 4.192 / 4.352 |
| 准备与发射 | 6.656 / 6.656 | 6.624 / 6.496 |
| 显式WGMMA完成等待 | 0.992 / 0.992 | 0.544 / 0.544 |
| TC生产循环 | 11.552 / 11.552 | 12.896 / 12.832 |

减少完成等待本身不足以加速：slot0延后释放，使TMA更晚覆盖下一批数据，增加了后续operand等待。测量与这一机制一致；没有通过独立消融证明所有回退均来自同一因素。当前关键约束是operand slot复用、TMA发布和TC执行的相互依赖，不能仅由某一等待指令的时间判断瓶颈。

92稀疏计时：TMA各阶段等待free-slot的中位数之和约6.688/6.624μs；CUDA consumer等待P约6.816/6.784μs；TC等P slot空闲约0.384μs。这些分别属于不同角色、可重叠，且为分阶段中位数之和，不能相加当作完整调用时间。消费者轮询多并不证明消费者处于关键路径，缩减消费者反而回退。

## 实验去留

| 迭代 | 改动 | 结论 |
|---|---|---|
| 88 | CUDA consumer由11 warp减至5 | launch约回退4.2%–4.6%，舍弃 |
| 89 | consumer减至8 warp | launch约回退1.1%–1.9%，舍弃 |
| 90 | 相邻两个16位输入以32位读写 | 保留；输入变换+grid约3.97/4.00→3.68/3.68μs |
| 91 | 扩成64位四元素读写 | 寄存器和打包开销增加，完整调用回退，舍弃 |
| 92 | A四个K64 TMA合并为一个K256 | 与90合并保留；收益小，见同期对照 |
| 93 | restrict别名提示 | 增益很小且Graph FP16回退，舍弃 |
| 94 | 两group在途，延后slot0释放 | 完成等待下降但供数等待上升，整体回退，舍弃 |
| 95 | 变换阶段预取4个tensor-map descriptor | Graph改善但普通调用回退，舍弃 |
| 96 | 逐文件恢复92 | 完整精度及sanitizer通过，当前版本 |

88/89的历史benchmark配置字符串仍显示11个consumer warp；实际实验源码分别为160/256个consumer线程，P-free计数同步修改。以源码和本表为准。最终96字符串与11个consumer warp一致。

## NCU与生成指令

32位成对读写使输入LDG warp指令65536次U16降至32768次E32，S/T store由131072次U16降至65536次E32。FP32变换运算、目标dtype舍入和残差顺序保持不变；打包指令开销使总动态指令并不按载入数量同比下降。

92 A侧原矩阵用rank3 tensor map，变换矩阵用rank4 map，将K64分块维搬到tensor-map维度中。逻辑地址及shared排列与原来一致。每CTA实际TMA命令由110降至44。必须使用Predicated-On Thread Instructions Executed统计实际发布：warp UTMA计数含谓词关闭指令，92为46/CTA，不能直接当成44条实际命令。

92/96寄存器79/线程，无local/stack spill，动态shared213120字节，仍一CTA/SM。R2UR180992/dtype，略高于87的178944；实际64×32 WGMMA次数245760/dtype保持一致。NCU BF16/FP16动态warp指令约4783889/4750390，Tensor活动21.26%/21.17%。P全部16条STS、42条LDS均1-way、0额外wavefront。当前供数优化没有改变P布局。

NCU时间仅作诊断，不能替代无profiler的完整调用对照。插桩计时也可能影响寄存器和调度；准备/发射区间包含异步TC执行重叠，不能全部解释为可消除的descriptor成本。NCU采样为92，当前96的solution文件与92完全一致。

## 精度及独立验证

参考保持为原始量化A/B的普通AB，以FP64计算。验收保持eF_SW≤3×eF_dense+floor；BF16 floor=1e-3，FP16=1e-4，非有限值必须为0。没有删除8项残差补偿；总计15项叶乘法，运算量为普通Dense的15/8。

最终96通过72项独立检查：两dtype、六类输入、三组新数据、launch/Graph；包含NaN输出污染、独立Graph及API拒绝检查。quick memcheck、racecheck、synccheck均0错误/危害/警告。数学DAG的3项CPU测试通过。90/91/92各在3组新随机输入、两dtype上与87 bitwise一致；92比较使用各自独立编码的新旧tensor map，不能外推为所有输入逐位一致。

## 可复核证据

- `final_launch.csv`、`final_driver.log`：最终正式基准；`restored_launch.csv`为iter96恢复时的独立结果。
- `summary.json`：所有同期对照、NCU和诊断汇总。
- `paired/iter88`至`iter95`：实际执行的对照脚本、CSV和日志。
- `ncu/`：raw CSV、SASS统计、完整ncu-rep。
- `diagnostics/`：插桩源码、生成器、日志、压缩原始记录。94仅累计角色4/5有验证，不主张完整稀疏role0诊断。
- `bitwise/`、`checks/`：独立一致性、压力测试、资源及sanitizer日志。
- `reproduce_compare.py`：重建87/92/95的汇总复现入口，已经AST语法检查；本轮未再次执行该汇总入口，实际执行证据在paired目录。

查阅的技术依据：[CUDA编程指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html)、[tensor-map Driver API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)、[CUTLASS Hopper多级流水实现](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_multistage_gmma_ss_warpspecialized.hpp)。性能和因果判断以上述本机实验为依据。

本轮已验证consumer数量、IO宽度、TMA命令合并、别名提示、WGMMA在途深度、descriptor预取六个方向；保留最稳定的小幅普通调用收益。尚无物理下限证据，不声称优化空间耗尽。
