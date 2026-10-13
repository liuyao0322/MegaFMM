# 1024 RS-A BK32 megakernel实测（iteration37）

新Mega40相对同binary保留的RS-A BK64参照 Mega39（上轮已测最快）：普通launch下BF16耗时变化+9.935%、FP16耗时变化+5.403%；Graph下BF16耗时变化+9.583%、FP16耗时变化+5.607%。

新Mega40相对同binary matched BK32 SS Mega36：普通launch下BF16耗时变化+8.299%、FP16耗时变化+2.926%；Graph下BF16耗时变化+8.598%、FP16耗时变化+3.926%。

BF16新Mega40完整调用25.497279µs，同行cuBLASLt 12.267520µs，Dense/SW 0.481131×；相对同binary保留RS-A BK64参照 Mega39自身耗时变化+9.935%。FP16新Mega40完整调用24.322560µs，同行cuBLASLt 9.944960µs，Dense/SW 0.408878×；相对同binary保留RS-A BK64参照 Mega39自身耗时变化+5.403%。

本轮mandatory16；实际16行、160条TIME、48个fresh CHECK；完整矩阵=是。原始门限0 PASS /48 FAIL，nonfinite=0，实际记录行0/16通过。

实际原始数值门限存在FAIL或nonfinite；性能与结构检查不构成原始精度认证。

完整调用性能、fresh gate和strict执行范围分别报告；本报告不宣布目标完成或提升Auto策略。

测量源码checkpoint `5360e2500a3a0e7a2d1402950a072cf4f172f0d4`；bench SHA256 `49e5f8c790402db85af6d8c2fb6989d92e478d3c32e9c66e4c9b82ce3aea3980`。七源、双binary与原始证据见[small_psplit_rsa_bk32_profile_iter37.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk32_profile_iter37.json)，展开测量见[small_psplit_rsa_bk32_measurements_iter37.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk32_measurements_iter37.json)。

## 测量协议

完整目标为新40 RS-A BK32/保留39 RS-A BK64（上轮已测最快）/匹配36 SS BK32/Lt1 × BF16/FP16 × ordinary/Graph。若新40编译安全门限阻止，则只有显式绑定最新gate的四个缺失key才允许partial12；旧结果不能替代本轮测量。每行20warmup、100calls/group、5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates和共同8GiB workspace cap。BF16/FP16输入、FP32输出。median与线性p10/p90来自五个组均值，CV使用总体标准差/均值。每行cuBLASLt独立调优，调优发生在capture前，没有Graph专用调优；Dense和SW按同行相同模式计时。Dense/SW使用同行Dense event median / 完整SW event median。quick仅作筛选：20warmup、20calls/group、5groups、1fresh，不能替代这里的mandatory full测量。

新40采用RS-A路线，两个独立consumer WG按whole-K计算P3+4。最终几何、名称、role配额、launch bounds、barrier与shared配置见下列冻结plan记录；初始CPU模型和候选预算只证明各自归档的抽象模型范围，不能充当最终编译fit或硬件执行证据。

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
  "40": {
    "actual_name": "megakernel_wgmma_psplit_rsa_bk32_64x32",
    "producer_quota": 64,
    "consumer_quotas": [
      192,
      192
    ],
    "launch_bounds_min_ctas": 1,
    "enabled": true,
    "BM": 64,
    "BN": 32,
    "BK": 32,
    "threads": 384,
    "dynamic_shared_bytes": 61568,
    "predicted_dynamic_shared_bytes": 61568,
    "static_shared_bytes": 1024,
    "drain_barrier": "nonaligned barrier.sync6,384; collective epochs0/6/0",
    "compiled_gate": {
      "path": "results/psplit_rsa_bk32_static_review_initial_iter37.json",
      "sha256": "842e07f61015d2ef811c183f24843894e722a2f2dfc0bd92e937b43ce049578c",
      "status": "PASS_COMPILED_CAN_LAUNCH_40_ONLY_37_REMAINS_BLOCKED"
    },
    "initial_registers_per_thread": 168,
    "initial_owned_CTA_pool_registers": 64512,
    "requested_role_pool_registers": 57344,
    "producer_gpr_highest_including_implicit": 39,
    "wg0_gpr_highest_including_implicit_by_dtype": {
      "fp16": 124,
      "bf16": 150
    },
    "wg1_gpr_highest_including_implicit": 141
  }
}
```

RS-A BK32 40对SS BK32 36，以及RS-A BK32 40对RS-A BK64 39，是两个预声明比较轴。仅共享BK或BN配置不能自动证明受控因果关系；是否只改变A操作数路径由独立源码与compiled scope审核。所有性能数字只取本轮实际完整调用。native资格、partial-K回退与sameatom reference范围见本轮实际测试coverage/profile；fallback PASS不认证native。

本轮实际测得的native Mega36/Mega39/Mega40 每次读取当前原始A/B，重新构造输入变换并完成七个P、固定U和epilogue，完整调用为一个kernel、零native矩阵workspace、无host descriptor encode。Lt1是pack→七batch leaf GEMM→combine的完整多launch控制，不能替代单launch目标。

Graph捕获固定地址，replay读取当前A/B内容并重做变换；三次fresh检查更换内容。setup/tuning/capture不计入稳态。event覆盖完整GPU工作，可能包含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，再取五组median；不是独立同步单调用CPU wall。clock设置按本轮环境记录，环境快照不认证组内clock。

环境记录：GPU=NVIDIA H100 80GB HBM3，SMs=132，host=g08，CUDA=12.4.99，CUTLASS=3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee；记录作用域见profile。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega40 RS-A BK32 BK32 64/192/192 min1 | bf16 | 12.2675 [12.2397,12.2764] | 25.4973 [25.4543,25.5156] | 0.481131 | 0.1489/0.1143 | 12.34169/25.56948 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter37_full1024.csv:2) |
| Mega40 RS-A BK32 BK32 64/192/192 min1 | fp16 | 9.9450 [9.9249,9.9629] | 24.3226 [24.2964,24.3297] | 0.408878 | 0.1720/0.0624 | 10.01722/24.39948 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter37_full1024.csv:3) |
| Mega39 RS-A BK64 reference, prior best BK64 64/192/192 min1 | bf16 | 12.2701 [12.2604,12.2839] | 23.1930 [23.1822,23.2137] | 0.529043 | 0.0861/0.0604 | 12.34153/23.28219 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter37_full1024.csv:2) |
| Mega39 RS-A BK64 reference, prior best BK64 64/192/192 min1 | fp16 | 9.8858 [9.8783,9.9013] | 23.0758 [23.0700,23.1010] | 0.428403 | 0.1071/0.0679 | 9.96044/23.15705 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter37_full1024.csv:3) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | bf16 | 12.2694 [12.2546,12.2757] | 23.5434 [23.5216,23.5942] | 0.521142 | 0.0774/0.1421 | 12.34278/23.61453 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter37_full1024.csv:2) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | fp16 | 9.9398 [9.9089,9.9437] | 23.6310 [23.6061,23.6569] | 0.420626 | 0.1650/0.0970 | 10.01160/23.70421 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter37_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2973 [12.2494,12.3219] | 19.3686 [19.3321,19.4216] | 0.634907 | 0.2635/0.2114 | 12.37569/19.44699 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter37_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.8829 [9.8556,9.8938] | 19.4803 [19.4674,19.5314] | 0.507326 | 0.1742/0.1660 | 9.96088/19.55701 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter37_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega40 RS-A BK32 BK32 64/192/192 min1 | bf16 | 11.2973 [11.2845,11.2980] | 25.1309 [25.1252,25.1752] | 0.449538 | 0.0603/0.0945 | 11.36982/25.20839 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter37_full1024.csv:2) |
| Mega40 RS-A BK32 BK32 64/192/192 min1 | fp16 | 9.5610 [9.5502,9.5661] | 24.1578 [24.1510,24.1644] | 0.395772 | 0.0771/0.0248 | 9.63546/24.23680 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter37_full1024.csv:3) |
| Mega39 RS-A BK64 reference, prior best BK64 64/192/192 min1 | bf16 | 11.3709 [11.3638,11.3795] | 22.9331 [22.9247,22.9448] | 0.495828 | 0.0676/0.0390 | 11.44336/23.01074 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter37_full1024.csv:2) |
| Mega39 RS-A BK64 reference, prior best BK64 64/192/192 min1 | fp16 | 9.4166 [9.4081,9.4303] | 22.8752 [22.8586,22.8879] | 0.411653 | 0.1075/0.0599 | 9.48911/22.95898 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter37_full1024.csv:3) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | bf16 | 11.2918 [11.2696,11.2949] | 23.1411 [23.1267,23.1558] | 0.487956 | 0.1190/0.0590 | 11.37813/23.23364 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter37_full1024.csv:2) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | fp16 | 9.5101 [9.5039,9.5135] | 23.2451 [23.2315,23.2657] | 0.409122 | 0.0527/0.0668 | 9.59671/23.34444 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter37_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.3600 [11.3541,11.3805] | 17.1622 [17.1560,17.1711] | 0.661918 | 0.1059/0.0396 | 11.43809/17.24580 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter37_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.5331 [9.5010,9.5926] | 17.9280 [17.8900,17.9396] | 0.531745 | 0.4351/0.1253 | 9.61270/18.00536 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter37_full1024.csv:3) |

## 新40相对保留RS-A BK64参照39及匹配SS BK32对照36的自身耗时

| 模式 | dtype | 对照 | 对照 µs | 新40 µs | 对照/40自身延迟比 | 40自身耗时变化 | 对照/40 batchwall µs/call |
|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega39 | 23.192959 | 25.497279 | 0.909625× | +9.935% | 23.28219/25.56948 |
| launch | bf16 | Mega36 | 23.543360 | 25.497279 | 0.923368× | +8.299% | 23.61453/25.56948 |
| launch | fp16 | Mega39 | 23.075840 | 24.322560 | 0.948742× | +5.403% | 23.15705/24.39948 |
| launch | fp16 | Mega36 | 23.631041 | 24.322560 | 0.971569× | +2.926% | 23.70421/24.39948 |
| graph | bf16 | Mega39 | 22.933121 | 25.130880 | 0.912547× | +9.583% | 23.01074/25.20839 |
| graph | bf16 | Mega36 | 23.141119 | 25.130880 | 0.920824× | +8.598% | 23.23364/25.20839 |
| graph | fp16 | Mega39 | 22.875199 | 24.157760 | 0.946909× | +5.607% | 22.95898/24.23680 |
| graph | fp16 | Mega36 | 23.245120 | 24.157760 | 0.962222× | +3.926% | 23.34444/24.23680 |

40对39比较保留的RS-A BK64参照（上轮已测最快）；40对36比较匹配的SS BK32路线。RS_A_vs_SS_BK32_axis_comparison实际保留4个轴metadata记录，RS_A_BK_axis_comparison实际保留4个轴metadata记录；两个轴的controlled_axis_claimed默认false，不重复添加性能表行。本生成器不单独归因动态occupancy、stall或traffic；收益判断使用SW自身耗时。缺失的新40不能由历史结果或其他binary补齐。

## 编译资源与实际检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 40 | FP16 | 64×32 /32 | 384 | 61568 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |
| 40 | BF16 | 64×32 /32 | 384 | 61568 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

动态shared、ELF资源和spill只引用当前实际compiled metadata。候选预算或CPU预测不是actualfit；静态资源不代表实测occupancy、traffic或物理floor。最终对象和linked binary关系见独立编译审查。

本轮measured identity绑定的fresh compiled gate：status=PASS_COMPILED_CAN_LAUNCH_40_ONLY_37_REMAINS_BLOCKED，[psplit_rsa_bk32_static_review_initial_iter37.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_bk32_static_review_initial_iter37.json)，SHA256 `842e07f61015d2ef811c183f24843894e722a2f2dfc0bd92e937b43ce049578c`。生成器仅检查绑定；CAN_LAUNCH内容及source/object/linked-bin时序由独立审计核验。

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
    "40": {
      "actual_name": "megakernel_wgmma_psplit_rsa_bk32_64x32",
      "producer_quota": 64,
      "consumer_quotas": [
        192,
        192
      ],
      "launch_bounds_min_ctas": 1,
      "enabled": true,
      "BM": 64,
      "BN": 32,
      "BK": 32,
      "threads": 384,
      "dynamic_shared_bytes": 61568,
      "predicted_dynamic_shared_bytes": 61568,
      "static_shared_bytes": 1024,
      "drain_barrier": "nonaligned barrier.sync6,384; collective epochs0/6/0",
      "compiled_gate": {
        "path": "results/psplit_rsa_bk32_static_review_initial_iter37.json",
        "sha256": "842e07f61015d2ef811c183f24843894e722a2f2dfc0bd92e937b43ce049578c",
        "status": "PASS_COMPILED_CAN_LAUNCH_40_ONLY_37_REMAINS_BLOCKED"
      },
      "initial_registers_per_thread": 168,
      "initial_owned_CTA_pool_registers": 64512,
      "requested_role_pool_registers": 57344,
      "producer_gpr_highest_including_implicit": 39,
      "wg0_gpr_highest_including_implicit_by_dtype": {
        "fp16": 124,
        "bf16": 150
      },
      "wg1_gpr_highest_including_implicit": 141
    }
  }
}
```

初始pending字段保持其初始身份，不因后续final compiled PASS而追溯认证；最终配置只取冻结plan。

CPU DAG实际归档exit=0，tests=3，[psplit_rsa_bk32_cpu_dag_iter37.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_bk32_cpu_dag_iter37.txt)。

CPU stage实际归档exit=0，tests=6，[psplit_rsa_bk32_cpu_stage_iter37.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_bk32_cpu_stage_iter37.txt)。

CPU几何与抽象模型实际归档exit=0，status=PASS_CPU_ONLY_RS_A_BK32_GEOMETRY_VIRTUAL_LIFETIME，predicted_shared_storage_bytes_only=61568；[psplit_rsa_bk32_cpu_state_model_iter37.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_rsa_bk32_cpu_state_model_iter37.txt)。只检查相应抽象/源码模型范围，不认证GPU memory-order、forward-progress或最终修复配额的硬件ownership。原RS-A九件保护与old main是否调用按实际runner/stdout证据核查，不能由JSON status推断真实进程exit。

default：exit=0，2466 CASE /1760 PASS；[correctness_psplit_rsa_bk32_iter37.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_rsa_bk32_iter37.txt)。
selected40：exit=0，56 CASE /135 PASS；[correctness_psplit_rsa_bk3240_targeted_iter37.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_rsa_bk3240_targeted_iter37.txt)。
disabled37_cli：exit=1，0 CASE /0 PASS；[disabled_psplit_rsa_bk3237_cli_iter37.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_rsa_bk3237_cli_iter37.txt)。

新Graph configs/replays=36/108；全部Graph=294/882。Mega37继续disabled；结构PASS不能替代原始FP64数值认证。

## 原始精度与sanitizer

本轮strict实际30 gates：18 PASS /12 FAIL /nonfinite 0；24 native、6 Dense fallback。BF16/FP16原floor保持0.001/0.0001，门限保持3×Dense_eF+floor；回退PASS不认证native。

历史strict按各自source/bin身份保留；本轮执行计数只来自实际终止的stdout/exit，源码域计数不能作为执行结果。原始数值FAIL保留，Auto保持Dense，目标未宣布完成。

本轮实际执行作用域：

```json
"Only archived current-source runs are actual; blocked or absent scopes stay null. Full1650 is code scope, not an executed rerun. Historical source34 new36-only30, source35 new38-only30 and source36 new39-only30 remain separately bound."
```

实际归档3项CUDA Compute Sanitizer，其中0项已测作用域的exit/errors或race hazards/warnings非零或未记录；memcheck leak bytes=[0]。race/synccheck未测泄漏字段保持null，实际summary见profile；没有替Mega37执行sanitizer。

旧CUDA保留33个，历史证据1119个绑定，onsite-only 4个按现场SHA保留，不声称成为Git对象。本生成器填报实际原件；统计、来源、compiled gate及最终Git绑定由独立审计核验。
