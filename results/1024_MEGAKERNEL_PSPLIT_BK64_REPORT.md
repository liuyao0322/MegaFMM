# 1024 单launch P3+4 BK64 实测（iteration35）

新Mega38未超过同binary本轮已测29/36/38中最快native Mega36；普通launch下BF16耗时变化+0.862%、FP16耗时变化+2.301%；Graph下BF16耗时变化+1.697%、FP16耗时变化+3.022%。

BF16新Mega38完整调用23.765121µs，同行cuBLASLt 12.318720µs，Dense/SW 0.518353×；相对同binary Mega29自身耗时变化-6.068%。FP16新Mega38完整调用24.090879µs，同行cuBLASLt 10.020159µs，Dense/SW 0.415932×；相对同binary Mega29自身耗时变化-5.341%。

预声明16行，实际16行、160条TIME、48个fresh CHECK；完整矩阵=是。原始门限0 PASS /48 FAIL，nonfinite=0，实际记录行0/16通过。

实际原始数值门限存在FAIL或nonfinite；性能与结构检查不构成原始精度认证。

完整调用性能、fresh gate和strict执行范围分别报告；本报告不宣布目标完成或提升Auto策略。

测量源码checkpoint `6d3a95ec69892828a656453ec337f5c85520cc0b`；bench SHA256 `c7e60233d9413703d0a2abbdfc8951370a3e9866bb68ce4778e8cb490407aa64`。七源、双binary与原始证据见[small_psplit_bk64_profile_iter35.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk64_profile_iter35.json)，展开测量见[small_psplit_bk64_measurements_iter35.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk64_measurements_iter35.json)。

## 测量协议

完整目标为新38/retained36/29/Lt1 × BF16/FP16 × ordinary/Graph。每行20warmup、100calls/group、5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates和共同8GiB workspace cap。BF16/FP16输入、FP32输出。median与线性p10/p90来自五个组均值，CV使用总体标准差/均值。每行cuBLASLt独立调优，调优发生在capture前，没有Graph专用调优；Dense和SW按同行相同模式计时。Dense/SW使用同行Dense event median / 完整SW event median。quick仅作筛选：20warmup、20calls/group、5groups、1fresh，不能替代这里的mandatory full测量。

新38采用BM64/BN32/BK64、两bank、384线程，将P3+4分给两个独立consumer WG。每WG完成升序whole-K，每BK64包含四个升序K16步骤，wait0/release；drain结束后复用typed shared union，发布全部七P后计算固定U与原alpha/beta epilogue。没有将K拆成浮点分段归约。最终名称、producer/consumer配额及launch bounds按冻结plan填写；本轮初始与最终配额均64/96/96；归一化相对保留36仅BK几何、符号名称与注释发生变化。正式fresh compiled gate绑定最终d25源码和最终两个binary；初始e3构建当时尚未完成test绑定，不单独称其通过最终门限。

新38最终配置：actual=`megakernel_wgmma_psplit_bk64_64x32`、quota=64/96/96、min_ctas=1。BK64 native要求父矩阵K能被128整除；partial K64/192/320走回退，不能把这些fallback的PASS解释为native认证。sameatom检查的新38参考是36的N32 raw U与独立host RNalpha/betaFMA，不采用N16 oracle。

本轮实际测得的native Mega29/Mega36/Mega38 每次读取当前原始A/B，重新构造输入变换并完成七个P、固定U和epilogue，完整调用为一个kernel、零native矩阵workspace、无host descriptor encode。Lt1是pack→七batch leaf GEMM→combine的完整多launch控制，不能替代单launch目标。

Graph捕获固定地址，replay读取当前A/B内容并重做变换；三次fresh检查更换内容。setup/tuning/capture不计入稳态。event覆盖完整GPU工作，可能包含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，再取五组median；不是独立同步单调用CPU wall。clock设置按本轮环境记录，环境快照不认证组内clock。

环境记录：GPU=NVIDIA H10080GB HBM3，SMs=132，host=g08，CUDA=12.4.99，CUTLASS=3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee；记录作用域见profile。

## 同行Dense组均值的波动

下列行的Dense总体CV达到2%。全部五组保留，median、p10/p90和CV均使用这些组；没有剔除较慢组。该波动会改变同行Dense/SW分母，不能据加速比变化单独认定优化收益。

Mega38 P3+4 BK64 64/96/96 min1 launch bf16：Dense CV=2.398301%，SW CV=0.106007%；stdout五个Dense event组均值µs=[12.8554, 12.2877, 12.9446, 12.3187, 12.264]；[CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_launch_iter35_full1024.csv:2)。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega38 P3+4 BK64 64/96/96 min1 | bf16 | 12.3187 [12.2735,12.9089] | 23.7651 [23.7632,23.8060] | 0.518353 | 2.3983/0.1060 | 12.40344/23.87097 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_launch_iter35_full1024.csv:2) |
| Mega38 P3+4 BK64 64/96/96 min1 | fp16 | 10.0202 [10.0076,10.0262] | 24.0909 [24.0850,24.1108] | 0.415932 | 0.0829/0.0488 | 10.10340/24.18414 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_launch_iter35_full1024.csv:3) |
| Mega36 P3+4 BK32 64/96/96 min1 | bf16 | 12.3693 [12.3644,12.7446] | 23.5619 [23.5504,23.5789] | 0.524969 | 1.7355/0.0556 | 12.45514/23.65085 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter35_full1024.csv:2) |
| Mega36 P3+4 BK32 64/96/96 min1 | fp16 | 9.9498 [9.9290,9.9871] | 23.5491 [23.5237,23.5825] | 0.422511 | 0.2820/0.1104 | 10.03489/23.65636 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter35_full1024.csv:3) |
| Mega29 wait0 BK32 | bf16 | 12.2634 [12.2444,12.2831] | 25.3005 [25.2335,25.3137] | 0.484709 | 0.1551/0.1494 | 12.34568/25.37807 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter35_full1024.csv:2) |
| Mega29 wait0 BK32 | fp16 | 9.9462 [9.9254,9.9593] | 25.4502 [25.4308,25.4593] | 0.390811 | 0.1529/0.0547 | 10.02718/25.52931 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_launch_iter35_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2378 [12.2330,12.2410] | 19.1978 [19.1498,19.2120] | 0.637458 | 0.0293/0.1686 | 12.31571/19.28621 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter35_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.9651 [9.9542,9.9700] | 19.7805 [19.7723,19.8175] | 0.503786 | 0.0826/0.1043 | 10.04343/19.85968 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter35_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega38 P3+4 BK64 64/96/96 min1 | bf16 | 11.3446 [11.3375,11.3534] | 23.4915 [23.4807,23.4927] | 0.482925 | 0.0671/0.0248 | 11.42736/23.56978 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_graph_iter35_full1024.csv:2) |
| Mega38 P3+4 BK64 64/96/96 min1 | fp16 | 9.4483 [9.4412,9.4740] | 23.8810 [23.8647,23.8915] | 0.395642 | 0.1973/0.0500 | 9.52456/23.96241 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_graph_iter35_full1024.csv:3) |
| Mega36 P3+4 BK32 64/96/96 min1 | bf16 | 11.3722 [11.3637,11.3804] | 23.0995 [23.0909,23.1232] | 0.492312 | 0.0697/0.0632 | 11.45335/23.19555 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter35_full1024.csv:2) |
| Mega36 P3+4 BK32 64/96/96 min1 | fp16 | 9.3952 [9.3750,9.3966] | 23.1805 [23.1552,23.1828] | 0.405307 | 0.1092/0.0582 | 9.47340/23.25995 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter35_full1024.csv:3) |
| Mega29 wait0 BK32 | bf16 | 11.3379 [11.3270,11.3521] | 24.7472 [24.7380,24.7782] | 0.458150 | 0.0986/0.0740 | 11.41406/24.82652 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter35_full1024.csv:2) |
| Mega29 wait0 BK32 | fp16 | 9.3952 [9.3701,9.4369] | 25.0797 [25.0235,25.1041] | 0.374614 | 0.3157/0.1463 | 9.46649/25.15779 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_pipeline29_graph_iter35_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.3443 [11.3364,11.3478] | 16.9955 [16.9942,17.0029] | 0.667489 | 0.0507/0.0239 | 11.42282/17.08103 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter35_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.5846 [9.5730,9.5914] | 17.6374 [17.6275,17.6474] | 0.543426 | 0.0864/0.0503 | 9.66549/17.71678 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter35_full1024.csv:3) |

## 新38相对两条保留Mega的自身耗时

| 模式 | dtype | 对照 | 对照 µs | 新38 µs | 对照/38自身延迟比 | 38自身耗时变化 | 对照/38 batchwall µs/call |
|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega29 | 25.300479 | 23.765121 | 1.064606× | -6.068% | 25.37807/23.87097 |
| launch | bf16 | Mega36 | 23.561921 | 23.765121 | 0.991450× | +0.862% | 23.65085/23.87097 |
| launch | fp16 | Mega29 | 25.450239 | 24.090879 | 1.056426× | -5.341% | 25.52931/24.18414 |
| launch | fp16 | Mega36 | 23.549120 | 24.090879 | 0.977512× | +2.301% | 23.65636/24.18414 |
| graph | bf16 | Mega29 | 24.747200 | 23.491521 | 1.053452× | -5.074% | 24.82652/23.56978 |
| graph | bf16 | Mega36 | 23.099520 | 23.491521 | 0.983313× | +1.697% | 23.19555/23.56978 |
| graph | fp16 | Mega29 | 25.079679 | 23.880961 | 1.050196× | -4.780% | 25.15779/23.96241 |
| graph | fp16 | Mega36 | 23.180480 | 23.880961 | 0.970668× | +3.022% | 23.25995/23.96241 |

38对36同为BN32，BK32→BK64是预声明比较轴；最终配额和其他源码/编译差异需独立审核。本生成器不自动将它认证为只改变BK的受控实验，不单独归因动态occupancy、stall或traffic。收益判断使用SW自身耗时；缺失的新38不能由其他binary或历史结果补齐。

## 编译资源与实际检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 38 | fp16 | 64×32 /64 | 384 | 172160 | 96 | 36864 | 64/96/96 | 32768 | 0/0/0 |
| 38 | bf16 | 64×32 /64 | 384 | 172160 | 96 | 36864 | 64/96/96 | 32768 | 0/0/0 |

动态shared、ELF资源和spill来自实际编译记录。172160字节的预测本身不能作为证明；本轮新host/ABI/ELF证据已经确认dynamic172160/static1024字节，未把静态资源当成动态occupancy、traffic或物理floor。编译安全状态、最终对象/linked binary关系见独立编译审查。

CPU DAG实际归档exit=0，tests=3，[psplit_bk64_cpu_dag_iter35.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bk64_cpu_dag_iter35.txt)。

CPU stage实际归档exit=0，tests=6，[psplit_bk64_cpu_stage_iter35.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bk64_cpu_stage_iter35.txt)。

CPU几何与抽象模型实际归档exit=0，status=PASS_CPU_SOURCE_GEOMETRY_AND_ABSTRACT_PROTOCOL_ONLY；[psplit_bk64_cpu_geometry_model_iter35.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bk64_cpu_geometry_model_iter35.txt)。只检查相应抽象/源码模型范围，不认证GPU memory-order、forward-progress或最终修复配额的硬件ownership。

default：exit=0，2354 CASE /1494 PASS；[correctness_psplit_bk64_iter35.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_bk64_iter35.txt)。
selected38：exit=0，56 CASE /135 PASS；[correctness_psplit_bk6438_targeted_iter35.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_bk6438_targeted_iter35.txt)。
disabled37_cli：exit=1，0 CASE /0 PASS；[disabled_psplit_bk6437_cli_iter35.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_bk6437_cli_iter35.txt)。

新Graph configs/replays=36/108；全部Graph=222/666。Mega37继续disabled；结构PASS不能替代原始FP64数值认证。

## 原始精度与sanitizer

本轮strict实际30 gates：18 PASS /12 FAIL /nonfinite 0；24 native、6 Dense fallback。BF16/FP16原floor保持0.001/0.0001，门限保持3×Dense_eF+floor；回退PASS不认证native。

历史实际strict按各自身份保留：source29 full1290，source30/31/32/33各新增60，source34仅enabled36新增30。历史full1350/1410/1470/1530/1560与当前full1590均为未重跑的源码域计数；本轮仅报告实际归档new38范围。

实际归档3项CUDA Compute Sanitizer，其中0项exit/errors/warnings/hazards非零或未记录；memcheck leak bytes=[0]。race/synccheck未测泄漏字段保持null，实际summary见profile；没有替Mega37执行sanitizer。

旧CUDA保留31个，历史证据784个绑定，onsite-only 4个按现场SHA保留，不声称成为Git对象。本生成器填报实际原件；统计、来源、compiled gate及最终Git绑定由独立审计核验。
