# iter82：流式重组与提前输出

用户要求尝试上一轮提出的下一版流水线。本轮只实现第一步：重排七个 P，边接收边重组并提前输出。源码提交77e1489；基线为iter81（58743df）。没有改变BK、stage数、warp数或WGMMA等待策略，没有恢复无限优化任务。

结果：原精度门限、72项独立launch/Graph检查及三个sanitizer均通过。同期对照普通调用约改善1.5%–1.6%，Graph约改善0.8%–1.4%；低于3%的明显提升判据，仍慢于调优cuBLASLt Dense。保留iter82作为当前通过验证的候选，不能称为性能winner。

## 改动与同步

- P顺序从1,2,3,4,5,6,7改成1,2,6,7,5,3,4；TMA与WGMMA共用同一映射。
- operand sequence按`step*(512/BK)+q`推进。product slot与phase按step推进，不再按逻辑P编号。
- CUDA worker从保留七个P数组改为两个中间状态数组及当前P数组。
- 读完当前P后立即arrive product_free，再执行重组和global store；下一次复用该槽仍须等所有352个CUDA worker读完。
- P2到达后输出D11；P5后输出D22；P3后输出D12；P4后输出D21。
- 重组保留原有FP32加法/减法表达式与顺序：U1=P1+P2，U2=P1+P6，U3=U2+P7，U4=U2+P5；D11=U1，D12=U4+P3，D21=U3-P4，D22=U3+P5。
- 每个P内K循环及hi/lo补偿顺序不变。BK256、两个operand槽、两个P槽、P行内XOR swizzle、512线程、1TMA+11CUDA+4WGMMA warp分工不变。

## 正确性

仅测试1024^3，BF16/FP16输入、FP32累加与输出。参考为原始量化A/B的FP64普通AB，门限仍为`eF_SW <= 3*eF_dense + floor`，BF16 floor=1e-3，FP16 floor=1e-4。

| dtype | 标准CSV SW eF | gate limit | 三轮fresh输入 |
|---|---:|---:|---|
| BF16 | 7.394587e-6 | 1.002557e-3 | 全部通过，无nonfinite |
| FP16 | 2.371355e-6 | 1.037528e-4 | 全部通过，无nonfinite |

独立test_sw1024：六类输入×两种dtype×三轮fresh输入×launch/同一Graph重放，共72项，全部通过；API拒绝检查也通过。quick memcheck=0 errors，racecheck=0 hazards，synccheck=0 errors。3项CPU Winograd DAG测试通过。同期性能对照每次调用也执行完整三轮fresh精度检查，全部通过。

## 完整调用性能

所有性能均包含S/T变换、grid同步、15项补偿叶乘法、P传递、重组和输出。H10080GBHBM3，CUDA12.4 Release。每次benchmark均20 warmup、每组100次、5组、3轮fresh精度检查。每份CSV报告五个组均值的中位数。

正式iter82标准benchmark：

| dtype | iter82 SW | 同期调优cuBLASLt Dense | SW/Dense耗时比 |
|---|---:|---:|---:|
| BF16 | 21.350 us | 12.194 us | 1.75 |
| FP16 | 21.462 us | 9.978 us | 2.15 |

为减少历史测试间GPU状态变化的干扰，单独从提交58743df重建iter81，CMake Release与CUDA编译选项相同。按A=iter81、B=iter82执行launch A1,B1,B2,A2；然后Graph A1,B1。两个二进制均使用原版plan，未引入任何profiling-only Dense配置补丁。

下表launch是两份CSV中位数的算术平均，Graph为一份CSV；它们不是合并全部样本后重新计算的中位数。精确原始值见paired/。

| 模式 | dtype | iter81 | iter82 | 耗时降低 |
|---|---|---:|---:|---:|
| launch | BF16 | 21.665 us | 21.318 us | 1.60% |
| launch | FP16 | 21.755 us | 21.429 us | 1.49% |
| Graph | BF16 | 21.083 us | 20.905 us | 0.84% |
| Graph | FP16 | 21.175 us | 20.870 us | 1.44% |

这轮仅得到小幅改善；未锁频，样本量不支持把它宣称为显著架构提速。Auto继续Dense。

## NCU与资源

NCU2025.1.1（CUDA12.8工具），每种dtype39 passes；kernel replay、clock-control none、cache-control none。六个CTC link metrics不可读取的警告不影响下述指标。NCU时长只作诊断，性能结论使用上述普通benchmark。

| 指标 | iter81 BF16/FP16 | iter82 BF16/FP16 |
|---|---|---|
| P STS：16条静态指令 | 全部1-way，0 excessive | 全部1-way，0 excessive |
| P LDS：42条静态指令 | 全部1-way，0 excessive | 全部1-way，0 excessive |
| Tensor Core elapsed activity | 18.91% / 19.14% | 18.88% / 18.97% |
| eligible warps/scheduler | 0.434 / 0.447 | 0.480 / 0.494 |
| 动态执行warp指令 | 572.1万 / 574.5万 | 612.0万 / 614.4万 |
| 编译寄存器/thread | 110 / 110 | 110 / 110 |
| 动态shared/CTA | 213120 bytes | 213120 bytes |
| stack/local | 0 / 0 | 0 / 0 |

CUDA重组源码中的活跃P状态减少，但整个kernel的编译寄存器分配没有下降，仍然只能驻留一个CTA/SM。动态指令增加约7%，包含控制与等待循环变化，不能全归因于某一个阶段。Tensor Core活跃度基本不变；本轮尚未处理每个K块的wait<0>。这些结果与端到端仅有小幅变化一致。

本轮没有添加阶段计时，不能从聚合计数定量断言尾部减少了多少或各阶段具体占比。

## 归档

- 源码：../20261014_212022_iter-82/，提交77e1489。
- 正式benchmark：standard_launch.csv；同期对照：paired/，包含构建日志、run.py、六次完整benchmark的CSV与输出。
- 独立验证：stress.log、memcheck.log、racecheck.log、synccheck.log、resources.txt。
- NCU原始报告：sw_full.ncu-rep；导出：sw_raw.csv、sw_sass.csv；P指令：p_instructions.json；对照指标与源码SHA256：summary.json。
