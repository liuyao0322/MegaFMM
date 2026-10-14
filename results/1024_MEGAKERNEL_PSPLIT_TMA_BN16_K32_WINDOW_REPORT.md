# 1024 BN16 K32-window选择性TMA rawA实测（iteration46）

Mega50对同binary Mega49自身完整调用：launch bf16 25.792961µs对25.807040µs，减少0.055%；launch fp16 25.290239µs对25.263360µs，延长0.106%；graph bf16 25.419841µs对25.414400µs，延长0.021%；graph fp16 24.968319µs对24.995201µs，减少0.108%。

Mega50对同binary Mega45自身完整调用：launch bf16 25.792961µs对17.776320µs，延长45.097%；launch fp16 25.290239µs对17.894400µs，延长41.330%；graph bf16 25.419841µs对16.967360µs，延长49.816%；graph fp16 24.968319µs对17.350720µs，延长43.904%。

Mega45对同binary Mega39自身完整调用：launch bf16 17.776320µs对23.218241µs，减少23.438%；launch fp16 17.894400µs对23.046081µs，减少22.354%；graph bf16 16.967360µs对22.966080µs，减少26.120%；graph fp16 17.350720µs对22.843201µs，减少24.044%。

Mega50对同binary Mega39自身完整调用：launch bf16 25.792961µs对23.218241µs，延长11.089%；launch fp16 25.290239µs对23.046081µs，延长9.738%；graph bf16 25.419841µs对22.966080µs，延长10.684%；graph fp16 24.968319µs对22.843201µs，延长9.303%。

本轮launch bf16已测native中最快为Mega45，17.776320µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.894400µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，16.967360µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.350720µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit 9988cce85a02fd3e469802194f82dfb6b7975abf；[small_psplit_tma_bn16_k32_window_profile_iter46.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_profile_iter46.json)，[small_psplit_tma_bn16_k32_window_measurements_iter46.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_measurements_iter46.json)。

## 测量协议

ONLY50是新BN16 K32-window选择性四路TMA rawA路线；49为上一轮BN16 atom-local min2 TMA，45为BN32 TMA，39为原RS-A BK64，Lt1为完整多launch对照。51未注册、未运行。fresh compiled blocked50必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替；45对39四个同binary配对仍可实际测得。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

50、49和45的源码入口在普通启动时每次从当前A指针/lda编码一个A tensor map；实际enabled路线的编码与单个kernel均包含在完整调用计时内，map实际被四路rawA TMA消费，B仍SIMT；39无host descriptor encode。50最终几何/配额/lifetime由fresh gate与冻结参数读取；49为BN16/BK64，45为BN32/BK64。min2是launch-bounds配置，API预测也不等于实际两CTA驻留；所有controlled_axis_claimed=false，不认证纯BN或纯搬运因果。全部实际运行native完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，为一个kernel、零矩阵workspace；Lt1完整多launch计时。Graph50/49/45在capture时编码，稳态replay使用固定capture地址读取当前内容，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时。rawU的reference variant与sameatom/crossatom关系只按本轮实际coverage记录；严格bitwise fixture和独立host RN alpha/FMA不构成通用FP64精度认证，任何实际不一致/FAIL必须保留。

巨大stride记录只按本轮实际host metadata检查与Plan setup结果表述，不能称作已分配巨大GPU输入并执行fallback；setup不可用时保留原message。此处不认证巨大strideGPU运行。

batchwall是每组100调用含批末同步的host wall除以调用数，再取组median，不是独立同步单调用CPU wall。setup/tuning/capture不计稳态；event可能含提交空隙，环境快照不认证组内clock。所有组、原FAIL/nonfinite/退出码均保留。

冻结参数：

```json
{
  "49": {
    "actual_name": "megakernel_wgmma_psplit_tma_bn16_min2_64x16",
    "enabled": true,
    "producer_quota": 40,
    "consumer_quotas": [
      88,
      112
    ],
    "launch_bounds_min_ctas": 2,
    "BM": 64,
    "BN": 16,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 95232,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "initial_registers_per_thread": 80,
    "initial_owned_CTA_pool_registers": 30720,
    "requested_role_pool_registers": 30720,
    "physical_windows": {
      "windows_per_stage_per_WG": 4,
      "order": "ascending K16 atom0,1,2,3",
      "WG0_A_packed_words_per_window": 12,
      "WG1_A_packed_words_per_window": 16,
      "WG0_whole_P_words": 24,
      "WG1_whole_P_words": 32,
      "pending_generic_overlap_access_count": 0,
      "postwait_A_and_P_fences_before_next_reuse": true,
      "single_stage_release_after_all_four_windows": true,
      "whole_stage_A48_64_is_not_current_physical_backing": true
    },
    "all_P_clear_after_successful_INC": false,
    "legal_initial_owned_preINC_zero_present": true,
    "occupancy_API_predicted_active_CTAs_per_SM": 2,
    "actual_measured_CTA_residency": null,
    "function_probe_path": "results/psplit_tma_bn16_min2_function_occupancy_probe_iter45.txt",
    "resource_metadata_scope": "Retained legacy attributes; not a fresh iteration46 function or residency query."
  },
  "45": {
    "actual_name": "megakernel_wgmma_psplit_selective_tma_a_64x32",
    "enabled": true,
    "producer_quota": 64,
    "consumer_quotas": [
      192,
      192
    ],
    "launch_bounds_min_ctas": 1,
    "BM": 64,
    "BN": 32,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 123904,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "device_reserved_shared_bytes_per_CTA": 1024,
    "initial_registers_per_thread": 168,
    "initial_owned_CTA_pool_registers": 64512,
    "requested_role_pool_registers": 57344,
    "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained",
    "raw_A_transaction_bytes_per_stage": 32768,
    "raw_A_TMA_boxes_per_stage": 4,
    "A_native_lda_upper_exclusive": 549755813888,
    "B_native_stride_extra_bound": null,
    "A_shared_absolute_alignment_bytes": 1024,
    "occupancy_API_predicted_active_CTAs_per_SM": 1,
    "actual_measured_CTA_residency": null,
    "function_probe_path": "results/psplit_selective_tma_a_function_occupancy_probe_iter41.txt",
    "resource_metadata_scope": "Retained legacy attributes; not a fresh iteration46 function or residency query."
  },
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
    "wg1_gpr_highest_including_implicit": 186,
    "resource_metadata_scope": "Retained legacy attributes; not a fresh iteration46 function or residency query."
  },
  "50": {
    "actual_name": "megakernel_wgmma_psplit_tma_bn16_k32_window_64x16",
    "enabled": true,
    "producer_quota": 40,
    "consumer_quotas": [
      96,
      96
    ],
    "launch_bounds_min_ctas": 2,
    "BM": 64,
    "BN": 16,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 95232,
    "static_shared_bytes": 0,
    "function_probe_path": "results/psplit_tma_bn16_k32_window_function_occupancy_probe_iter46.txt",
    "resource_metadata_scope": "Fresh50 original gate/probe; occupancy API is static prediction and measured residency remains null."
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | bf16 | 12.247040 / 12.245248 / 12.276672 | 25.792961 / 25.787584 / 25.802945 | 0.474821 | 0.1380/0.0318 | 12.32324/25.86728 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_50_launch_iter46_full1024.csv:2) |
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | fp16 | 9.932480 / 9.916416 / 9.952128 | 25.290239 / 25.282048 / 25.341632 | 0.392740 | 0.1584/0.1394 | 10.00574/25.36486 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_50_launch_iter46_full1024.csv:3) |
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | bf16 | 12.280000 / 12.254656 / 12.282112 | 25.807040 / 25.800448 / 25.818496 | 0.475839 | 0.1413/0.0311 | 12.35352/25.88974 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_launch_iter46_full1024.csv:2) |
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | fp16 | 9.971200 / 9.944768 / 9.979712 | 25.263360 / 25.247808 / 25.273216 | 0.394690 | 0.1581/0.0448 | 10.04228/25.34790 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_launch_iter46_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.287040 / 12.266496 / 12.296320 | 17.776320 / 17.743936 / 17.783360 | 0.691203 | 0.1151/0.1019 | 12.35957/17.84974 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter46_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 10.003520 / 9.979776 / 10.024640 | 17.894400 / 17.866240 / 17.904832 | 0.559031 | 0.2103/0.1122 | 10.07857/17.96762 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter46_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.385280 / 12.355712 / 12.412096 | 23.218241 / 23.208576 / 23.221440 | 0.533429 | 0.2052/0.0253 | 12.46842/23.30565 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter46_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.922240 / 9.916224 / 9.968768 | 23.046081 / 23.030784 / 23.064257 | 0.430539 | 0.2645/0.0640 | 10.00702/23.13042 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter46_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.211519 / 12.203776 / 12.249728 | 19.336320 / 19.319680 / 19.340225 | 0.631533 | 0.1789/0.0570 | 12.29158/19.41420 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter46_full1024.csv:2) |
| Lt1完整多launch | fp16 | 10.016960 / 9.996480 / 10.023040 | 19.633280 / 19.594624 / 19.641600 | 0.510203 | 0.1220/0.1127 | 10.09417/19.71239 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter46_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | bf16 | 11.364800 / 11.354304 / 11.370368 | 25.419841 / 25.412928 / 25.424384 | 0.447084 | 0.0652/0.0200 | 11.43572/25.49508 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_50_graph_iter46_full1024.csv:2) |
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | fp16 | 9.534400 / 9.522816 / 9.550336 | 24.968319 / 24.956352 / 24.976512 | 0.381860 | 0.1315/0.0388 | 9.60743/25.04062 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_50_graph_iter46_full1024.csv:3) |
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | bf16 | 11.341760 / 11.332864 / 11.354048 | 25.414400 / 25.392641 / 25.419776 | 0.446273 | 0.0899/0.0491 | 11.41334/25.48574 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_graph_iter46_full1024.csv:2) |
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | fp16 | 9.413120 / 9.408896 / 9.517952 | 24.995201 / 24.985664 / 25.206208 | 0.376597 | 0.5557/0.4218 | 9.48630/25.07046 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_graph_iter46_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.382720 / 11.379840 / 11.387136 | 16.967360 / 16.961216 / 16.973504 | 0.670860 | 0.0307/0.0320 | 11.46799/17.05596 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter46_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.599680 / 9.588544 / 9.603456 | 17.350720 / 17.346880 / 17.357120 | 0.553273 | 0.0762/0.0296 | 9.68393/17.43476 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter46_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.353600 / 11.336960 / 11.358912 | 22.966080 / 22.953471 / 22.977345 | 0.494364 | 0.1005/0.0479 | 11.42656/23.04376 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter46_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.559040 / 9.557760 / 9.568384 | 22.843201 / 22.826369 / 22.851393 | 0.418463 | 0.0519/0.0526 | 9.63421/22.91552 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter46_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.352000 / 11.346624 / 11.365504 | 17.002240 / 16.996992 / 17.012928 | 0.667677 | 0.0747/0.0434 | 11.42553/17.08241 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter46_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.570240 / 9.564800 / 9.575360 | 17.680960 / 17.677440 / 17.704832 | 0.541274 | 0.0544/0.0724 | 9.64249/17.76103 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter46_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega49 | Mega50 | 25.807040 | 25.792961 | 1.000546× | -0.055% | 25.88974/25.86728 |
| launch | bf16 | Mega45 | Mega50 | 17.776320 | 25.792961 | 0.689193× | +45.097% | 17.84974/25.86728 |
| launch | bf16 | Mega39 | Mega45 | 23.218241 | 17.776320 | 1.306133× | -23.438% | 23.30565/17.84974 |
| launch | bf16 | Mega39 | Mega50 | 23.218241 | 25.792961 | 0.900177× | +11.089% | 23.30565/25.86728 |
| launch | fp16 | Mega49 | Mega50 | 25.263360 | 25.290239 | 0.998937× | +0.106% | 25.34790/25.36486 |
| launch | fp16 | Mega45 | Mega50 | 17.894400 | 25.290239 | 0.707562× | +41.330% | 17.96762/25.36486 |
| launch | fp16 | Mega39 | Mega45 | 23.046081 | 17.894400 | 1.287893× | -22.354% | 23.13042/17.96762 |
| launch | fp16 | Mega39 | Mega50 | 23.046081 | 25.290239 | 0.911264× | +9.738% | 23.13042/25.36486 |
| graph | bf16 | Mega49 | Mega50 | 25.414400 | 25.419841 | 0.999786× | +0.021% | 25.48574/25.49508 |
| graph | bf16 | Mega45 | Mega50 | 16.967360 | 25.419841 | 0.667485× | +49.816% | 17.05596/25.49508 |
| graph | bf16 | Mega39 | Mega45 | 22.966080 | 16.967360 | 1.353545× | -26.120% | 23.04376/17.05596 |
| graph | bf16 | Mega39 | Mega50 | 22.966080 | 25.419841 | 0.903471× | +10.684% | 23.04376/25.49508 |
| graph | fp16 | Mega49 | Mega50 | 24.995201 | 24.968319 | 1.001077× | -0.108% | 25.07046/25.04062 |
| graph | fp16 | Mega45 | Mega50 | 17.350720 | 24.968319 | 0.694909× | +43.904% | 17.43476/25.04062 |
| graph | fp16 | Mega39 | Mega45 | 22.843201 | 17.350720 | 1.316556× | -24.044% | 22.91552/17.43476 |
| graph | fp16 | Mega39 | Mega50 | 22.843201 | 24.968319 | 0.914887× | +9.303% | 22.91552/25.04062 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新50实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 50 | fp16 | 64×16 /64 | 384 | 95232 | 80 | 80 | 30720 | 40/96/96 | 29696 | 0/0/0 |
| 50 | bf16 | 64×16 /64 | 384 | 95232 | 80 | 80 | 30720 | 40/96/96 | 29696 | 0/0/0 |

Mega50 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2；measured_residency=None。
Mega50 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2；measured_residency=None。

原编译门限与本轮资源修复时序：[psplit_tma_bn16_k32_window_static_review_initial_iter46.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_k32_window_static_review_initial_iter46.json)；这是本轮编译资源放行记录，不认证数值或性能。

最终fp16配额 40/96/96，actual I=80、owned/requested pool=30720/29696；producer/WG0/WG1最大物理GPR=37/91/88。实例初始化原标志 all_P_clear_after_successful_INC=false、legal_initial_owned_preINC_zero_present=true；原字段按本轮门限逐字记录，初始化或grown标志只按当轮原件记录，不从旧控制继承。

本轮fp16实际每stage每WG为2个窗口，每窗2个K16 atom；A packed words WG0/WG1=24/32，whole P words=24/32，每窗HGMMA=6/8。当前为K32窗口生命周期，原计数按fresh physical_windows逐项读取；不继承49的四个K16单atom窗口或A12/16。

最终bf16配额 40/96/96，actual I=80、owned/requested pool=30720/29696；producer/WG0/WG1最大物理GPR=37/91/88。实例初始化原标志 all_P_clear_after_successful_INC=false、legal_initial_owned_preINC_zero_present=true；原字段按本轮门限逐字记录，初始化或grown标志只按当轮原件记录，不从旧控制继承。

本轮bf16实际每stage每WG为2个窗口，每窗2个K16 atom；A packed words WG0/WG1=24/32，whole P words=24/32，每窗HGMMA=6/8。当前为K32窗口生命周期，原计数按fresh physical_windows逐项读取；不继承49的四个K16单atom窗口或A12/16。


Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=3042/3112/5692；[correctness_psplit_tma_bn16_k32_window_iter46.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn16_k32_window_iter46.txt)。
selected50：actualexit=0，CASE/PASS/BEGIN=60/140/204；[correctness_psplit_tma_bn16_k32_window50_targeted_iter46.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn16_k32_window50_targeted_iter46.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_tma_bn16_k32_window37_cli_iter46.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_tma_bn16_k32_window37_cli_iter46.txt)。

新50 rawU groups=84，reference=45（CROSSATOM_N16_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=40/120。

新50实际完成10项主机端A-only stride边界检查（1组PASS），其中native helper guard有4项。Plan设置成功0项、不可用10项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
原始不可用信息：query Lt heuristics: cuBLAS status 15。这些记录仅验证helper predicate与Plan setup，setup_unavailable不能被计为成功Plan路由或已执行GPU fallback。

```json
{
  "PASS_groups": 1,
  "cases": 10,
  "native_guard_cases": 4,
  "packed_guard_cases": 4,
  "dense_guard_cases": 2,
  "setup_success_cases": 0,
  "setup_unavailable_cases": 10,
  "B_bound_not_applied": true,
  "metadata_only": true,
  "allocated_input_bytes": 0,
  "tensor_map_encodes": 0,
  "native_runs": 0,
  "successful_Plan_setup_actual_routes": {},
  "setup_unavailable_not_claimed_as_Plan_route_PASS": true,
  "complete": true,
  "case_records": [
    {
      "dtype": "bf16",
      "scenario": 0,
      "lda": 549755813880,
      "ldb": 32,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_k32_window_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725776,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 1,
      "lda": 549755813888,
      "ldb": 32,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727808,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 2,
      "lda": 549755813896,
      "ldb": 32,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976729840,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 3,
      "lda": 549755813889,
      "ldb": 32,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976728062,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 4,
      "lda": 128,
      "ldb": 549755813888,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_k32_window_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 32768,
      "safe_b_span_bytes": 139637976727616,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 0,
      "lda": 549755813880,
      "ldb": 32,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_k32_window_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725776,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 1,
      "lda": 549755813888,
      "ldb": 32,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727808,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 2,
      "lda": 549755813896,
      "ldb": 32,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976729840,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 3,
      "lda": 549755813889,
      "ldb": 32,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976728062,
      "safe_b_span_bytes": 8192,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 4,
      "lda": 128,
      "ldb": 549755813888,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_k32_window_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 32768,
      "safe_b_span_bytes": 139637976727616,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    }
  ]
}
```

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1950、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_tma_bn16_k32_window_enabled_only",
    "path": "results/strict_psplit_tma_bn16_k32_window_iter46.txt",
    "exit": 2,
    "CASE": 60,
    "PASS": 0,
    "BEGIN": 0,
    "source": "Actual closed stdout and archived process exit; never expected counts.",
    "gates": 30,
    "pass": 18,
    "fail": 12,
    "nonfinite": 0,
    "native_gates": 24,
    "Dense_range_fallback_gates": 6,
    "shared_Dense_reference_CASE": 30
  }
]
```

SAN50 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_tma_bn16_k32_window50_memcheck_iter46.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn16_k32_window50_memcheck_iter46.txt)。非mem泄漏未测null。

SAN50 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_tma_bn16_k32_window50_racecheck_iter46.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn16_k32_window50_racecheck_iter46.txt)。非mem泄漏未测null。

SAN50 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_tma_bn16_k32_window50_synccheck_iter46.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn16_k32_window50_synccheck_iter46.txt)。非mem泄漏未测null。

环境：

```json
{
  "path": "results/small_psplit_tma_bn16_k32_window_environment_iter46.txt",
  "exit_path": "results/small_psplit_tma_bn16_k32_window_environment_iter46.exit",
  "sha256": "a2379c2a4bc43081d07e39824eb2439b7b2017c3e884542b56b34c3ea6d7695d",
  "exit": 0,
  "allocation": 107488,
  "host": "g08",
  "GPU": "NVIDIA H100 80GB HBM3",
  "gpu_uuid": "GPU-78def847-b6c1-9dd4-ff30-269a9ae1a73f",
  "driver": "575.51.03",
  "SMs": 132,
  "SMs_evidence": "Actual archived DriverAPI cuDeviceGetAttribute",
  "application_cuda": "12.4.99",
  "CUTLASS": "3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee",
  "sanitizer_path": "/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer",
  "sanitizer_version": "2025.1.0.0 build35583870",
  "clocks_locked": false,
  "snapshot_scope": "Actual readonly postvalidation snapshot; no kernel launch or within-group clock stability/residency proof.",
  "function_attributes_and_occupancy_API": {
    "path": "results/psplit_tma_bn16_k32_window_function_occupancy_probe_iter46.txt",
    "sha256": "1772e6e3b6d87ec3f019208e42f50a2ce4421ac0d4e4505aed29579c7d8aaf7e",
    "exit": 0,
    "records": [
      {
        "variant": 50,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm79_GLOBAL__N__6ed7d2d8_46_megakernel_wgmma_psplit_tma_bn16_k32_window_cu_536ac81941winograd_wgmma_psplit_tma_bn16_k32_windowIN7cutlass6half_tELi16EEEvNS_7ProblemENS0_17TmaBN16DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn16_k32_window_final_megakernel_wgmma_psplit_tma_bn16_k32_window_iter46.cubin",
        "cubin_sha256": "bb4b1a07e9c057f985b095f86f16ad0b5b1cbe2c8ccd7fe0ee2cd217c984dffc",
        "queried_dynamic_shared_bytes": 95232,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 95232
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      },
      {
        "variant": 50,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm79_GLOBAL__N__6ed7d2d8_46_megakernel_wgmma_psplit_tma_bn16_k32_window_cu_536ac81941winograd_wgmma_psplit_tma_bn16_k32_windowIN7cutlass10bfloat16_tELi16EEEvNS_7ProblemENS0_17TmaBN16DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn16_k32_window_final_megakernel_wgmma_psplit_tma_bn16_k32_window_iter46.cubin",
        "cubin_sha256": "bb4b1a07e9c057f985b095f86f16ad0b5b1cbe2c8ccd7fe0ee2cd217c984dffc",
        "queried_dynamic_shared_bytes": 95232,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 95232
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      }
    ]
  }
}
```

唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。


## 原始两条路线的同binary补充实测

主计划实际20行与补充8行分别计数，总观察28行。H10为单P输入融合、每完整调用两个kernel；H12为S/T只构造一次、每完整调用三个kernel。它们与native Mega50/49/45/39及Lt1有独立身份。原始FP64 gates/floors、五组均值与全部FAIL/NF保留。

| 原始路线 / 模式 | dtype | Dense event /p10/p90 µs | SW event /p10/p90 µs | Dense/SW | CV Dense/SW % | batch wall Dense/SW µs | 原数值 | WS bytes | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| H10 / launch | bf16 | 12.279680 / 12.254016 / 12.303680 | 38.329599 / 38.280192 / 38.337792 | 0.320371 | 0.1896/0.0844 | 12.35666/38.40505 | FAIL finite | 7340032 | [small_hopper_fused10_launch_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_fused10_launch_iter46_full1024.csv:2) |
| H10 / launch | fp16 | 9.863360 / 9.852928 / 9.886656 | 38.427839 / 38.379841 / 38.447040 | 0.256672 | 0.1714/0.0867 | 9.94003/38.51364 | FAIL finite | 7340032 | [small_hopper_fused10_launch_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_fused10_launch_iter46_full1024.csv:3) |
| H10 / graph | bf16 | 11.275520 / 11.269248 / 11.291008 | 36.274559 / 36.226496 / 36.318464 | 0.310838 | 0.0904/0.1124 | 11.35156/36.34720 | FAIL finite | 7340032 | [small_hopper_fused10_graph_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_fused10_graph_iter46_full1024.csv:2) |
| H10 / graph | fp16 | 9.448640 / 9.407104 / 9.574464 | 36.746240 / 36.719680 / 36.763712 | 0.257132 | 0.7985/0.0540 | 9.52272/36.82559 | FAIL finite | 7340032 | [small_hopper_fused10_graph_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_fused10_graph_iter46_full1024.csv:3) |
| H12 / launch | bf16 | 12.277120 / 12.206784 / 12.297408 | 34.260161 / 34.220416 / 34.321088 | 0.358350 | 0.3347/0.1462 | 12.35522/34.33879 | FAIL finite | 48234496 | [small_hopper_st_once12_launch_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_st_once12_launch_iter46_full1024.csv:2) |
| H12 / launch | fp16 | 9.934400 / 9.912704 / 9.940224 | 34.536321 / 34.513856 / 34.574656 | 0.287651 | 0.1293/0.0858 | 10.01107/34.61430 | FAIL finite | 48234496 | [small_hopper_st_once12_launch_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_st_once12_launch_iter46_full1024.csv:3) |
| H12 / graph | bf16 | 11.334720 / 11.318144 / 11.377728 | 31.418879 / 31.412096 / 31.484993 | 0.360761 | 0.2357/0.1108 | 11.40447/31.53661 | FAIL finite | 48234496 | [small_hopper_st_once12_graph_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_st_once12_graph_iter46_full1024.csv:2) |
| H12 / graph | fp16 | 9.580800 / 9.477312 / 9.591360 | 31.959040 / 31.907583 / 32.042817 | 0.299784 | 0.5609/0.1938 | 9.66303/32.03500 | FAIL finite | 48234496 | [small_hopper_st_once12_graph_iter46_full1024.csv](/share/home/liuyao/workspace/MegaFMM/results/small_hopper_st_once12_graph_iter46_full1024.csv:3) |

H12相对H10的自身耗时比与下降百分比仅来自本轮补充同行SW event。比率分母变化不代表代码收益；batch wall是100call同步批次/call，非每次独立同步。

| 模式 | dtype | H10 µs | H12 µs | H10/H12自身耗时比 | H12自身耗时下降 % | H10/H12 batchwall µs |
|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | 38.329599 | 34.260161 | 1.118780 | 10.616959 | 38.40505/34.33879 |
| launch | fp16 | 38.427839 | 34.536321 | 1.112679 | 10.126820 | 38.51364/34.61430 |
| graph | bf16 | 36.274559 | 31.418879 | 1.154547 | 13.385912 | 36.34720/31.53661 |
| graph | fp16 | 36.746240 | 31.959040 | 1.149792 | 13.027726 | 36.82559/32.03500 |

补充TIME=80，freshCHECK=24，PASS/FAIL=0/24，nonfinite=0；[small_psplit_tma_bn16_k32_window_input_placement_supplement_measurements_iter46.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_input_placement_supplement_measurements_iter46.json)。编译资源CAN50不转作H10/H12数值认证。


## 独立下一方向 CPU 设计证据

ROOT 冻结7文件 CPU几何设计包，actual CPU exit0。SS原位BM64BN32BK32/q40-88-104/I80及BN16 sibling I64均未编译；128CTAs对132SM不能推出两CTA驻留收益。没有CuTe、寄存器fit、ABI、CAN、GPU、数值或性能认证；不改变本轮50/49/45或原精度策略。

{
  "manifest": {
    "path": "results/wide_tile_feasibility_future46_manifest.json",
    "sha256": "e7db30a58206f3b69da2296b4afd4a31d59293c9555db0d97cc56d7c5554debd"
  },
  "frozen_future_sha256": {
    "results/wide_tile_feasibility_future46_design.md": "4a9b1f63f2221eb9c4676c3ab32ec9971791a7500f0052f126d032dfa9d03ceb",
    "results/wide_tile_feasibility_future46_geometry.exit": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "results/wide_tile_feasibility_future46_geometry.json": "6a1158de7085d9f48b19cd7c7ec4c0d8f3789851c2f3cf70b6d6b6de97d67fa2",
    "results/wide_tile_feasibility_future46_geometry.py": "2cf4289885ecf35426c94e71be1410634adae007f2a80c837d25c98f5b57895f",
    "results/wide_tile_feasibility_future46_geometry.stderr": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/wide_tile_feasibility_future46_geometry.stdout": "b66e4c050750b1990227ee77bd18bb23ff3114c043a59d956bc346b571e4276f",
    "results/wide_tile_feasibility_future46_manifest.json": "e7db30a58206f3b69da2296b4afd4a31d59293c9555db0d97cc56d7c5554debd"
  },
  "actual_CPU_exit": 0,
  "CPU_only": true,
  "CUDA_compilation_performed": false,
  "GPU_execution_performed": false,
  "numerical_model_performed": false,
  "precision_certified": false,
  "performance_claim": false,
  "scope": "Separate ROOT-frozen CPU geometry/design feasibility only; not CuTe/compiler/register/ABI/CAN, measured residency or speed/numerical evidence. Original historical future files remain separate. readonly_inputs_sha256 HINTS binding describes preserved47994B input prefix, not current appended HINTS file."
}
