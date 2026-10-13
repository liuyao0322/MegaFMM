# 1024 两条输入路线与RS-A megakernel实测（iteration36）

普通launch BF16：单P输入融合10完整2launch 38.333120µs，同行cuBLASLt 12.443200µs，Dense/SW 0.324607×；S/T构造一次12完整3launch 34.119999µs，同行cuBLASLt 12.244160µs，Dense/SW 0.358856×；融合路线相对构造一次路线自身耗时变化+12.348%。
普通launch FP16：单P输入融合10完整2launch 38.501761µs，同行cuBLASLt 9.919360µs，Dense/SW 0.257634×；S/T构造一次12完整3launch 34.502721µs，同行cuBLASLt 9.971840µs，Dense/SW 0.289016×；融合路线相对构造一次路线自身耗时变化+11.591%。
Graph BF16：单P输入融合10完整2launch 36.454079µs，同行cuBLASLt 11.360000µs，Dense/SW 0.311625×；S/T构造一次12完整3launch 31.372161µs，同行cuBLASLt 11.306241µs，Dense/SW 0.360391×；融合路线相对构造一次路线自身耗时变化+16.199%。
Graph FP16：单P输入融合10完整2launch 36.701441µs，同行cuBLASLt 9.407040µs，Dense/SW 0.256313×；S/T构造一次12完整3launch 32.098880µs，同行cuBLASLt 9.562880µs，Dense/SW 0.297919×；融合路线相对构造一次路线自身耗时变化+14.339%。

新Mega39相对同binary保留的BK32参照 Mega36（上轮已测最快）：普通launch下BF16耗时变化-1.330%、FP16耗时变化-1.591%；Graph下BF16耗时变化-0.785%、FP16耗时变化-1.596%。

新Mega39相对同binary matched BK64 SS Mega38：普通launch下BF16耗时变化-2.250%、FP16耗时变化-4.114%；Graph下BF16耗时变化-2.362%、FP16耗时变化-4.326%。

BF16新Mega39完整调用23.234880µs，同行cuBLASLt 12.184640µs，Dense/SW 0.524412×；相对同binary保留BK32参照 Mega36自身耗时变化-1.330%。FP16新Mega39完整调用23.105919µs，同行cuBLASLt 9.885120µs，Dense/SW 0.427818×；相对同binary保留BK32参照 Mega36自身耗时变化-1.591%。

原16行计划在首次timing前由ROOT授权追加8行，最终mandatory24；实际24行、240条TIME、72个fresh CHECK；完整矩阵=是。原始门限0 PASS /72 FAIL，nonfinite=0，实际记录行0/24通过。

实际原始数值门限存在FAIL或nonfinite；性能与结构检查不构成原始精度认证。

完整调用性能、fresh gate和strict执行范围分别报告；本报告不宣布目标完成或提升Auto策略。

测量源码checkpoint `8f355e1ccc73d73e59f3923631faf5f1779f3ffc`；bench SHA256 `700c767cf42d6f13eadf68df1ec2716c6c5f7e2a2d428164a4348f1c0f3f0012`。七源、双binary与原始证据见[small_psplit_rsa_profile_iter36.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_profile_iter36.json)，展开测量见[small_psplit_rsa_measurements_iter36.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_measurements_iter36.json)。

## 测量协议

完整目标为新39/retained36/BK64 SS38/Lt1/单P输入融合hopper10/S/T构造一次hopper12 × BF16/FP16 × ordinary/Graph。原16计划独立保留，timing前追加两条路线8行；旧16或39blocked时的旧12范围不能充当本轮完整范围。每行20warmup、100calls/group、5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates和共同8GiB workspace cap。BF16/FP16输入、FP32输出。median与线性p10/p90来自五个组均值，CV使用总体标准差/均值。每行cuBLASLt独立调优，调优发生在capture前，没有Graph专用调优；Dense和SW按同行相同模式计时。Dense/SW使用同行Dense event median / 完整SW event median。quick仅作筛选：20warmup、20calls/group、5groups、1fresh，不能替代这里的mandatory full测量。

新39采用RS-A路线，两个独立consumer WG按whole-K计算P3+4。最终几何、名称、role配额、launch bounds、barrier与shared配置见下列冻结plan记录；初始CPU模型和候选预算只证明各自归档的抽象模型范围，不能充当最终编译fit或硬件执行证据。

最终route_configurations：

```json
{
  "39": {
    "actual_name": "megakernel_wgmma_psplit_rsa_64x32",
    "producer_quota": 64,
    "consumer_quotas": [
      192,
      192
    ],
    "launch_bounds_min_ctas": 1,
    "enabled": true,
    "BM": 64,
    "BN": 32,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 123008,
    "predicted_dynamic_shared_bytes": 123008,
    "static_shared_bytes": 1024,
    "drain_barrier": "nonaligned barrier.sync6,384; CTAepochs0/6/0",
    "compiled_gate": {
      "path": "results/psplit_rsa_static_review_initial_iter36.json",
      "sha256": "46db33836552063d31dd3cfde249ddb472cf0bdd252e2a9e4f790690362a1488",
      "status": "PASS_COMPILED_CAN_LAUNCH_39_ONLY_37_REMAINS_BLOCKED"
    },
    "initial_registers_per_thread": 168,
    "initial_owned_CTA_pool_registers": 64512,
    "requested_role_pool_registers": 57344,
    "producer_gpr_highest_including_implicit": 52,
    "wg0_gpr_highest_including_implicit": 185,
    "wg1_gpr_highest_including_implicit": 186
  },
  "38": {
    "actual_name": "megakernel_wgmma_psplit_bk64_64x32",
    "producer_quota": 64,
    "consumer_quotas": [
      96,
      96
    ],
    "launch_bounds_min_ctas": 1,
    "enabled": true,
    "BM": 64,
    "BN": 32,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 172160,
    "static_shared_bytes": 1024,
    "drain_barrier": "nonaligned barrier.sync6,384; 3 collective epochs0/6/0",
    "compiled_gate": "Independent fresh CAN_LAUNCH38 actualsource d25/full3role/newobjectbenchtest"
  },
  "36": {
    "actual_name": "megakernel_wgmma_psplit_64x32",
    "producer_quota": 64,
    "consumer_quotas": [
      96,
      96
    ],
    "launch_bounds_min_ctas": 1,
    "enabled": true,
    "BM": 64,
    "BN": 32,
    "BK": 32,
    "threads": 384,
    "dynamic_shared_bytes": 86144,
    "static_shared_bytes": 1024
  },
  "10": {
    "algorithm": "hopper",
    "label": "fused10",
    "variant": 10,
    "enabled": true,
    "complete_run_kernel_launches": 2,
    "scope": "Retained immutable control, actual algorithm name from measured CSV; inputtransform/packing and combine inside each complete timed run.",
    "actual_name": "winograd_fused_input_32x64",
    "launch_count": 2
  },
  "12": {
    "algorithm": "hopper",
    "label": "matched12",
    "variant": 12,
    "enabled": true,
    "complete_run_kernel_launches": 3,
    "scope": "Retained immutable control, actual algorithm name from measured CSV; inputtransform/packing and combine inside each complete timed run.",
    "actual_name": "winograd_packed_input_32x64",
    "launch_count": 3
  }
}
```

RS-A39与BK64 SS38是预声明比较轴。仅共享BK或BN配置不能自动证明受控因果关系；是否只改变A操作数路径由独立源码与compiled scope审核。所有性能数字只取本轮实际完整调用。native资格、partial-K回退与sameatom reference范围见本轮实际测试coverage/profile；fallback PASS不认证native。

本轮实际测得的native Mega36/Mega38/Mega39 每次读取当前原始A/B，重新构造输入变换并完成七个P、固定U和epilogue，完整调用为一个kernel、零native矩阵workspace、无host descriptor encode。Lt1是pack→七batch leaf GEMM→combine的完整多launch控制，不能替代单launch目标。

单P输入融合hopper10完整调用为2launch：输入融合到七P的leaf，再combine；S/T构造一次hopper12完整调用为3launch：pack、七P的leaf、combine。两者均将所有输入变换/packing、leaf计算和combine计入每次端到端调用；workspace按实际CSV填报，不能叫单kernel。

Graph捕获固定地址，replay读取当前A/B内容并重做变换；三次fresh检查更换内容。setup/tuning/capture不计入稳态。event覆盖完整GPU工作，可能包含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，再取五组median；不是独立同步单调用CPU wall。clock设置按本轮环境记录，环境快照不认证组内clock。

计划扩展证据：[psplit_rsa_measurement_plan_addendum_iter36.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_measurement_plan_addendum_iter36.json) SHA256 `649d38d6057f2787166413071e424d40c1511b09e958775415b87112a7ba4199`；原16计划 [psplit_rsa1024_plan_iter36.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa1024_plan_iter36.json) SHA256 `e484a3eb525b46fbb1e21f778510d0458064592332ef7c0624ebf7bc4ad2c44a`；最终24计划 [psplit_rsa1024_expanded_plan_iter36.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa1024_expanded_plan_iter36.json) SHA256 `b3054f6da2addbf872d066e0c925c73c89c2639208c11b5d1c7c3b21f99046e8`。扩展前timing数与授权时序按addendum归档；原pre-gate身份不追溯改写。

环境记录：GPU=NVIDIA H100 80GB HBM3，SMs=132，host=g08，CUDA=12.4.99，CUTLASS=3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee；记录作用域见profile。

## 同行Dense组均值的波动

下列行的Dense总体CV达到2%。全部五组保留，median、p10/p90和CV均使用这些组；没有剔除较慢组。该波动会改变同行Dense/SW分母，不能据加速比变化单独认定优化收益。

Hopper10 单P输入融合（完整2launch） launch bf16：Dense CV=2.100053%，SW CV=0.094558%；stdout五个Dense event组均值µs=[12.7939, 12.3514, 13.0208, 12.4432, 12.3789]；[CSV](/share/home/liuyao/workspace/MegaFMM/results/small_fused10_launch_iter36_full1024.csv:2)。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega39 RS-A BK64 64/192/192 min1 | bf16 | 12.1846 [12.1812,12.2093] | 23.2349 [23.2166,23.2416] | 0.524412 | 0.1088/0.0563 | 12.26051/23.31580 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter36_full1024.csv:2) |
| Mega39 RS-A BK64 64/192/192 min1 | fp16 | 9.8851 [9.8712,9.9036] | 23.1059 [23.1033,23.1324] | 0.427818 | 0.1544/0.0589 | 9.96339/23.19059 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter36_full1024.csv:3) |
| Mega36 BK32 reference, prior best BK32 64/96/96 min1 | bf16 | 12.1664 [12.1495,12.1734] | 23.5482 [23.5232,23.6104] | 0.516660 | 0.0877/0.1664 | 12.23861/23.62070 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter36_full1024.csv:2) |
| Mega36 BK32 reference, prior best BK32 64/96/96 min1 | fp16 | 9.8355 [9.8143,9.9099] | 23.4794 [23.4601,23.4990] | 0.418901 | 0.4408/0.0744 | 9.90690/23.55260 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter36_full1024.csv:3) |
| Mega38 BK64 SS matched BK64 64/96/96 min1 | bf16 | 12.2211 [12.1985,12.2250] | 23.7696 [23.7500,23.7782] | 0.514149 | 0.1006/0.0534 | 12.29426/23.84081 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_launch_iter36_full1024.csv:2) |
| Mega38 BK64 SS matched BK64 64/96/96 min1 | fp16 | 9.9347 [9.9169,9.9482] | 24.0973 [24.0935,24.1039] | 0.412276 | 0.1388/0.0200 | 10.00661/24.17451 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_launch_iter36_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.3155 [12.3096,12.3300] | 19.4086 [19.4018,19.4263] | 0.634538 | 0.0820/0.0582 | 12.39236/19.48587 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter36_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.9875 [9.9615,9.9940] | 19.6096 [19.6022,19.6227] | 0.509318 | 0.1542/0.0477 | 10.06490/19.68664 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter36_full1024.csv:3) |
| Hopper10 单P输入融合（完整2launch） | bf16 | 12.4432 [12.3624,12.9300] | 38.3331 [38.3145,38.3946] | 0.324607 | 2.1001/0.0946 | 12.52944/38.48349 | FAIL finite | 7 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_fused10_launch_iter36_full1024.csv:2) |
| Hopper10 单P输入融合（完整2launch） | fp16 | 9.9194 [9.9023,9.9425] | 38.5018 [38.4354,38.5155] | 0.257634 | 0.1931/0.0955 | 9.99795/38.57716 | FAIL finite | 7 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_fused10_launch_iter36_full1024.csv:3) |
| Hopper12 S/T构造一次（完整3launch） | bf16 | 12.2442 [12.2294,12.2482] | 34.1200 [34.0719,34.2089] | 0.358856 | 0.0701/0.1834 | 12.32280/34.21039 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_matched12_launch_iter36_full1024.csv:2) |
| Hopper12 S/T构造一次（完整3launch） | fp16 | 9.9718 [9.8981,10.0148] | 34.5027 [34.4611,34.6316] | 0.289016 | 0.5252/0.2250 | 10.05238/34.58032 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_matched12_launch_iter36_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega39 RS-A BK64 64/192/192 min1 | bf16 | 11.2442 [11.2345,11.2527] | 22.9414 [22.9292,22.9435] | 0.490124 | 0.0717/0.0289 | 11.32537/23.02560 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter36_full1024.csv:2) |
| Mega39 RS-A BK64 64/192/192 min1 | fp16 | 9.5715 [9.5618,9.5891] | 22.8445 [22.8307,22.8908] | 0.418986 | 0.1457/0.1416 | 9.65176/22.92517 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter36_full1024.csv:3) |
| Mega36 BK32 reference, prior best BK32 64/96/96 min1 | bf16 | 11.3277 [11.3226,11.3372] | 23.1229 [23.0950,23.1349] | 0.489891 | 0.0588/0.0828 | 11.40466/23.20564 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter36_full1024.csv:2) |
| Mega36 BK32 reference, prior best BK32 64/96/96 min1 | fp16 | 9.5232 [9.5121,9.5369] | 23.2150 [23.2019,23.2388] | 0.410217 | 0.1234/0.0711 | 9.60215/23.29545 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter36_full1024.csv:3) |
| Mega38 BK64 SS matched BK64 64/96/96 min1 | bf16 | 11.2781 [11.2650,11.2803] | 23.4963 [23.4725,23.5030] | 0.479993 | 0.0636/0.0744 | 11.35277/23.57162 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_graph_iter36_full1024.csv:2) |
| Mega38 BK64 SS matched BK64 64/96/96 min1 | fp16 | 9.4630 [9.4497,9.5129] | 23.8774 [23.8584,23.8861] | 0.396317 | 0.3054/0.0575 | 9.53732/23.95519 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bk6438_graph_iter36_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.3763 [11.3590,11.3768] | 17.1536 [17.1480,17.1552] | 0.663203 | 0.0821/0.0215 | 11.45484/17.23654 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter36_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.5827 [9.5758,9.5836] | 17.4970 [17.4951,17.5088] | 0.547679 | 0.0437/0.0363 | 9.66273/17.58022 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter36_full1024.csv:3) |
| Hopper10 单P输入融合（完整2launch） | bf16 | 11.3600 [11.2930,11.3712] | 36.4541 [36.4340,36.4829] | 0.311625 | 0.3182/0.0599 | 11.43913/36.53479 | FAIL finite | 7 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_fused10_graph_iter36_full1024.csv:2) |
| Hopper10 单P输入融合（完整2launch） | fp16 | 9.4070 [9.3967,9.4131] | 36.7014 [36.6811,36.7211] | 0.256313 | 0.0804/0.0482 | 9.49135/36.78116 | FAIL finite | 7 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_fused10_graph_iter36_full1024.csv:3) |
| Hopper12 S/T构造一次（完整3launch） | bf16 | 11.3062 [11.2942,11.3131] | 31.3722 [31.3369,31.4013] | 0.360391 | 0.0843/0.0983 | 11.38492/31.44981 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_matched12_graph_iter36_full1024.csv:2) |
| Hopper12 S/T构造一次（完整3launch） | fp16 | 9.5629 [9.4970,9.5884] | 32.0989 [31.9672,32.1140] | 0.297919 | 0.4285/0.2513 | 9.65056/32.17780 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_matched12_graph_iter36_full1024.csv:3) |

## 单P输入融合与S/T构造一次的完整调用比较

| 模式 | dtype | S/T一次12 µs（3launch） | 输入融合10 µs（2launch） | 12/10自身延迟比 | 10自身耗时变化 | 12/10 batchwall µs/call | 12/10原始Gate |
|---|---|---:|---:|---:|---:|---:|---|
| launch | bf16 | 34.119999 | 38.333120 | 0.890092× | +12.348% | 34.21039/38.48349 | FAIL/FAIL |
| launch | fp16 | 34.502721 | 38.501761 | 0.896134× | +11.591% | 34.58032/38.57716 | FAIL/FAIL |
| graph | bf16 | 31.372161 | 36.454079 | 0.860594× | +16.199% | 31.44981/36.53479 | FAIL/FAIL |
| graph | fp16 | 32.098880 | 36.701441 | 0.874595× | +14.339% | 32.17780/36.78116 | FAIL/FAIL |

上述配对按同一binary与相同dtype/mode比较完整调用自身耗时，保留各自独立调优的同行Dense基准与原始精度结果。

## 新39相对保留BK32参照36及matched38的自身耗时

| 模式 | dtype | 对照 | 对照 µs | 新39 µs | 对照/39自身延迟比 | 39自身耗时变化 | 对照/39 batchwall µs/call |
|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega36 | 23.548160 | 23.234880 | 1.013483× | -1.330% | 23.62070/23.31580 |
| launch | bf16 | Mega38 | 23.769600 | 23.234880 | 1.023014× | -2.250% | 23.84081/23.31580 |
| launch | fp16 | Mega36 | 23.479359 | 23.105919 | 1.016162× | -1.591% | 23.55260/23.19059 |
| launch | fp16 | Mega38 | 24.097281 | 23.105919 | 1.042905× | -4.114% | 24.17451/23.19059 |
| graph | bf16 | Mega36 | 23.122880 | 22.941439 | 1.007909× | -0.785% | 23.20564/23.02560 |
| graph | bf16 | Mega38 | 23.496320 | 22.941439 | 1.024187× | -2.362% | 23.57162/23.02560 |
| graph | fp16 | Mega36 | 23.215041 | 22.844479 | 1.016221× | -1.596% | 23.29545/22.92517 |
| graph | fp16 | Mega38 | 23.877439 | 22.844479 | 1.045217× | -4.326% | 23.95519/22.92517 |

39对36用于比较保留的BK32参照（上轮已测最快）；39对38用于观察matched BK64 SS路线的自身延迟差异。RS_A_vs_SS_BK64_axis_comparison实际保留4个轴metadata记录，controlled_axis_claimed默认false，不重复添加第二组性能表行。本生成器不单独归因动态occupancy、stall或traffic；收益判断使用SW自身耗时。缺失的新39不能由历史结果或其他binary补齐。

## 编译资源与实际检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 39 | fp16 | 64×32 /64 | 384 | 123008 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |
| 39 | bf16 | 64×32 /64 | 384 | 123008 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

动态shared、ELF资源和spill只引用当前实际compiled metadata。候选预算或CPU预测不是actualfit；静态资源不代表实测occupancy、traffic或物理floor。最终对象和linked binary关系见独立编译审查。

本轮measured identity绑定的fresh compiled gate：status=PASS_COMPILED_CAN_LAUNCH_39_ONLY_37_REMAINS_BLOCKED，[psplit_rsa_static_review_initial_iter36.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_static_review_initial_iter36.json)，SHA256 `46db33836552063d31dd3cfde249ddb472cf0bdd252e2a9e4f790690362a1488`。生成器仅检查绑定；CAN_LAUNCH内容及source/object/linked-bin时序由独立审计核验。

初始与最终证据时序/模型作用域按本轮实际metadata保留：

```json
{
  "scope": "Original CPU-only RS-A prototype and models are frozen inputs, with all candidate initial quotas hypothetical until a fresh compiled source/binary gate. Final budgets/name/initial allocation come from the frozen plan and actual gate. Chronological archived build snapshots preserve their actual exit/status without retrospectively certifying pending scopes; retained unsafe37 belongs to iteration34.",
  "final_route_configurations": {
    "39": {
      "actual_name": "megakernel_wgmma_psplit_rsa_64x32",
      "producer_quota": 64,
      "consumer_quotas": [
        192,
        192
      ],
      "launch_bounds_min_ctas": 1,
      "enabled": true,
      "BM": 64,
      "BN": 32,
      "BK": 64,
      "threads": 384,
      "dynamic_shared_bytes": 123008,
      "predicted_dynamic_shared_bytes": 123008,
      "static_shared_bytes": 1024,
      "drain_barrier": "nonaligned barrier.sync6,384; CTAepochs0/6/0",
      "compiled_gate": {
        "path": "results/psplit_rsa_static_review_initial_iter36.json",
        "sha256": "46db33836552063d31dd3cfde249ddb472cf0bdd252e2a9e4f790690362a1488",
        "status": "PASS_COMPILED_CAN_LAUNCH_39_ONLY_37_REMAINS_BLOCKED"
      },
      "initial_registers_per_thread": 168,
      "initial_owned_CTA_pool_registers": 64512,
      "requested_role_pool_registers": 57344,
      "producer_gpr_highest_including_implicit": 52,
      "wg0_gpr_highest_including_implicit": 185,
      "wg1_gpr_highest_including_implicit": 186
    },
    "38": {
      "actual_name": "megakernel_wgmma_psplit_bk64_64x32",
      "producer_quota": 64,
      "consumer_quotas": [
        96,
        96
      ],
      "launch_bounds_min_ctas": 1,
      "enabled": true,
      "BM": 64,
      "BN": 32,
      "BK": 64,
      "threads": 384,
      "dynamic_shared_bytes": 172160,
      "static_shared_bytes": 1024,
      "drain_barrier": "nonaligned barrier.sync6,384; 3 collective epochs0/6/0",
      "compiled_gate": "Independent fresh CAN_LAUNCH38 actualsource d25/full3role/newobjectbenchtest"
    },
    "36": {
      "actual_name": "megakernel_wgmma_psplit_64x32",
      "producer_quota": 64,
      "consumer_quotas": [
        96,
        96
      ],
      "launch_bounds_min_ctas": 1,
      "enabled": true,
      "BM": 64,
      "BN": 32,
      "BK": 32,
      "threads": 384,
      "dynamic_shared_bytes": 86144,
      "static_shared_bytes": 1024
    },
    "10": {
      "algorithm": "hopper",
      "label": "fused10",
      "variant": 10,
      "enabled": true,
      "complete_run_kernel_launches": 2,
      "scope": "Retained immutable control, actual algorithm name from measured CSV; inputtransform/packing and combine inside each complete timed run.",
      "actual_name": "winograd_fused_input_32x64",
      "launch_count": 2
    },
    "12": {
      "algorithm": "hopper",
      "label": "matched12",
      "variant": 12,
      "enabled": true,
      "complete_run_kernel_launches": 3,
      "scope": "Retained immutable control, actual algorithm name from measured CSV; inputtransform/packing and combine inside each complete timed run.",
      "actual_name": "winograd_packed_input_32x64",
      "launch_count": 3
    }
  }
}
```

初始pending字段保持其初始身份，不因后续final compiled PASS而追溯认证；最终配置只取冻结plan。

CPU DAG实际归档exit=0，tests=3，[psplit_rsa_cpu_dag_iter36.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_cpu_dag_iter36.txt)。

CPU stage实际归档exit=0，tests=6，[psplit_rsa_cpu_stage_iter36.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_cpu_stage_iter36.txt)。

CPU几何与抽象模型实际归档exit=0，status=PASS_CPU_MODEL_ONLY_UNCOMPILED_RS_A_FUTURE；[psplit_rsa_cpu_state_model_iter36.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_cpu_state_model_iter36.txt)。只检查相应抽象/源码模型范围，不认证GPU memory-order、forward-progress或最终修复配额的硬件ownership。原RS-A九件保护与old main是否调用按实际runner/stdout证据核查，不能由JSON status推断真实进程exit。

default：exit=0，2410 CASE /1627 PASS；[correctness_psplit_rsa_iter36.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_rsa_iter36.txt)。
selected39：exit=0，56 CASE /135 PASS；[correctness_psplit_rsa39_targeted_iter36.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_rsa39_targeted_iter36.txt)。
disabled37_cli：exit=1，0 CASE /0 PASS；[disabled_psplit_rsa37_cli_iter36.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_rsa37_cli_iter36.txt)。

新Graph configs/replays=36/108；全部Graph=258/774。Mega37继续disabled；结构PASS不能替代原始FP64数值认证。

## 原始精度与sanitizer

本轮strict实际30 gates：18 PASS /12 FAIL /nonfinite 0；24 native、6 Dense fallback。BF16/FP16原floor保持0.001/0.0001，门限保持3×Dense_eF+floor；回退PASS不认证native。

历史strict按各自source/bin身份保留；本轮执行计数只来自实际终止的stdout/exit，源码域计数不能作为执行结果。原始数值FAIL保留，Auto保持Dense，目标未宣布完成。

本轮实际执行作用域：

```json
"Only archived current-source runs are actual; blocked or absent scopes stay null. Full1620 is code scope, not an executed rerun. Historical source34 new36-only30 and source35 new38-only30 remain separately bound."
```

实际归档3项CUDA Compute Sanitizer，其中0项已测作用域的exit/errors或race hazards/warnings非零或未记录；memcheck leak bytes=[0]。race/synccheck未测泄漏字段保持null，实际summary见profile；没有替Mega37执行sanitizer。

旧CUDA保留32个，历史证据947个绑定，onsite-only 4个按现场SHA保留，不声称成为Git对象。本生成器填报实际原件；统计、来源、compiled gate及最终Git绑定由独立审计核验。
