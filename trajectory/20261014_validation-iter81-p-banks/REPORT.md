# P shared bank conflict 修复与验证（iter81）

按用户“先解决 P 写回 shared 时存在 8 路 bank conflict”的要求，本轮仅修复 P 的 shared 布局。以此前 NCU 实测的 iter76（456ce9d）为基线；最终源码提交 58743df。持续优化目标仍暂停。

## 实现

原 P 按 `row*32+col` 存储。WGMMA 的同一 fragment 元素在一个 warp 内分布到 8 行、每行 4 列；这些行的 stride 都是 32 个 float，导致 8 个线程落到同一个 bank。

新布局在每行内执行 XOR 置换：

```cpp
physical = row*32 + (col ^ ((row & 1) | ((row & 6) << 2)));
```

fragment lane 变化的列位是 bit1、bit2，行位是 bit0、bit1、bit2。把这三个行位 XOR 到列 bit0、bit3、bit4，就让一个 warp 的 bank 地址覆盖全部 32 个 bank。CUDA warp 仍按逻辑 row-major 顺序读取，使用同一个索引函数，因此每行读取也是 32 个互异 bank，最终全局输出保持合并访问。

P 的元素数量、两个 product slot、barrier、BK256、计算顺序及精度策略均与 iter76 一致。共享内存仍为 213120 字节/CTA，workspace 8 MiB，寄存器 110/thread。

iter80 曾尝试 native fragment 连续存储/读取，72 项精度检查通过，但完整调用为 23.93/24.14 us，发生回退；最终采用 iter81 行内 swizzle。

## NCU 指令级验收

H100 80GB HBM3（132 SM），CUDA12.4 编译，NCU2025.1.1；两种 dtype 各采集一次完整 SW kernel，每个 kernel 39 passes。与旧报告同为 kernel replay、clock-control none、cache-control none。原始 `.ncu-rep`、raw CSV、SASS CSV、选定 P 指令及汇总 JSON 一并归档。

下表的 wavefront 总量均按一次 kernel、每种 dtype 分别计数；BF16 与 FP16 结果相同。

| P 指令指标 | iter76 | iter81 |
|---|---:|---:|
| 16 条静态 STS 的 Shared N-Way | 全部 8 | 全部 1 |
| P 写入实际 wavefront | 458752 | 57344 |
| P 写入理想 wavefront | 57344 | 57344 |
| P 写入 excessive wavefront | 401408 | 0 |
| P 读取 Shared N-Way | 1 | 1（42 条静态 LDS） |
| P 读取 excessive wavefront | 0 | 0 |

每条 STS 的实际/理想 wavefront 从 28672/3584 变为 3584/3584。原问题已在 SASS 指令级消除。

Kernel 级硬件计数仍有少量非零：shared store bank-conflict 计数 BF16/FP16 为 3111/3282，load 为 180/180；旧 store 为 414930/410271。这个计数与按指令统计的 excessive wavefront 口径不同，本报告不将其剩余部分归因给某条未确认指令，也不声称整个 kernel 的所有硬件计数归零。已定位的 P STS/LDS 都是 1-way、0 excessive wavefront。

六个 CTC link metrics 不可读取的 NCU 警告不涉及上述计数。短 kernel、未锁频和 replay 会影响采样及 profiler 时长；性能结论使用普通完整调用 benchmark。

## 精度和性能

仅测试 M=N=K=1024，BF16/FP16 输入、FP32 累加及输出。完整调用包含 S/T 生成、同步、补偿乘法、P 传递、重组和输出。

普通 benchmark：20 warmup、每组100次、5组、3组 fresh-input correctness；结果为组均值的中位数。

| dtype | iter76 SW（历史对照） | iter81 SW | iter81 同期 cuBLASLt Dense | SW/Dense 耗时比 |
|---|---:|---:|---:|---:|
| BF16 | 21.245 us | 21.216 us | 12.381 us | 1.71 |
| FP16 | 21.883 us | 21.661 us | 9.992 us | 2.17 |

相对历史对照变化约 -0.14%/-1.01%，不足以证明显著整体提速；修复了 P 的 bank conflict，但完整调用仍约 21–22 us，仍慢于 Dense。Auto 继续使用 Dense。

原始量化 A/B 的 FP64 普通 AB 参考及门限保持不变：`eF_SW <= 3*eF_dense+floor`，BF16 floor=1e-3，FP16 floor=1e-4。

| dtype | 标准 CSV 记录的 SW eF | gate limit | 三轮 fresh 输入 |
|---|---:|---:|---|
| BF16 | 7.394587e-6 | 1.002557e-3 | 全部通过，无 nonfinite |
| FP16 | 2.371355e-6 | 1.037528e-4 | 全部通过，无 nonfinite |

另执行独立 `test_sw1024`：两种 dtype、六类输入、三轮新输入、普通 launch 和同一 Graph 重放，共72项，通过且无 nonfinite；API 拒绝检查同时通过。memcheck、racecheck、synccheck 的 quick 1024 测试分别为0 errors、0 hazards、0 errors。Winograd DAG 的3项 CPU 测试通过。

## 证据

- 源码快照：`../20261014_203116_iter-81/`
- 标准 benchmark：`launch.csv`
- NCU：`sw_full.ncu-rep`、`sw_raw.csv`、`sw_sass.csv`、`p_instructions.json`、`summary.json`、`ncu.log`
- 独立验证：`stress.log`、`memcheck.log`、`racecheck.log`、`synccheck.log`、`resources.txt`
- 旧 NCU 对照：`../20261014_ncu_iter76/`
