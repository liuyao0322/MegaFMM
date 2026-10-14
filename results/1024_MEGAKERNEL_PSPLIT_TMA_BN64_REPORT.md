# 1024 BN64选择性TMA rawA实测（iteration43）

Mega47对同binary Mega45自身完整调用：launch bf16 25.914559µs对17.650239µs，延长46.823%；launch fp16 26.044481µs对17.826880µs，延长46.097%；graph bf16 25.493121µs对16.940160µs，延长50.489%；graph fp16 25.657599µs对17.283200µs，延长48.454%。

Mega45对同binary Mega39自身完整调用：launch bf16 17.650239µs对23.166080µs，减少23.810%；launch fp16 17.826880µs对23.111041µs，减少22.864%；graph bf16 16.940160µs对22.928319µs，减少26.117%；graph fp16 17.283200µs对22.907200µs，减少24.551%。

Mega47对同binary Mega39自身完整调用：launch bf16 25.914559µs对23.166080µs，延长11.864%；launch fp16 26.044481µs对23.111041µs，延长12.693%；graph bf16 25.493121µs对22.928319µs，延长11.186%；graph fp16 25.657599µs对22.907200µs，延长12.007%。

本轮launch bf16已测native中最快为Mega45，17.650239µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.826880µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，16.940160µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.283200µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit 7161777e2d66f4bba8fb24eb69a687b8698fda3e；[small_psplit_tma_bn64_profile_iter43.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn64_profile_iter43.json)，[small_psplit_tma_bn64_measurements_iter43.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn64_measurements_iter43.json)。

## 测量协议

ONLY47是新BN64选择性四路TMA rawA路线；45为同binary选择性四路TMA rawA，39为原RS-A BK64，36为SS BK32，Lt1为完整多launch对照。48未注册、未运行。fresh compiled blocked47必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替；45对39四个同binary配对仍可实际测得。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

47和45的源码入口在普通启动时每次从当前A指针/lda编码一个A tensor map；实际enabled路线的编码与单个kernel均包含在完整调用计时内，map实际被四路rawA TMA消费，B仍SIMT；39/36无host descriptor encode。47使用BM64/BN64/BK64，45为BN32/BK64，WGMMA N64/N32 atom不同；源码预算、shared及尾部也不同，所有controlled_axis_claimed=false，不认证纯BN或纯搬运因果。全部实际运行native完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，为一个kernel、零矩阵workspace；Lt1完整多launch计时。Graph45/47在capture时编码，稳态replay使用固定capture地址读取当前内容，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时。rawU对45为CROSSATOM_N64_N32的源限定严格bitwise fixture与独立host RN alpha/FMA检查，绝不是sameatom或通用FP64精度认证；任何实际不一致/FAIL必须保留。

巨大stride记录只按本轮实际host metadata检查与Plan setup结果表述，不能称作已分配巨大GPU输入并执行fallback；setup不可用时保留原message。此处不认证巨大strideGPU运行。

batchwall是每组100调用含批末同步的host wall除以调用数，再取组median，不是独立同步单调用CPU wall。setup/tuning/capture不计稳态；event可能含提交空隙，环境快照不认证组内clock。所有组、原FAIL/nonfinite/退出码均保留。

冻结参数：

```json
{
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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration43 resource/residency query."
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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration43 resource/residency query."
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
    "static_shared_bytes": 1024,
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration43 resource/residency query."
  },
  "47": {
    "actual_name": "megakernel_wgmma_psplit_tma_bn64_64x64",
    "enabled": true,
    "producer_quota": 48,
    "consumer_quotas": [
      216,
      240
    ],
    "launch_bounds_min_ctas": 1,
    "BM": 64,
    "BN": 64,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 181248,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "device_reserved_shared_bytes_per_CTA": 1024,
    "initial_registers_per_thread": 168,
    "initial_owned_CTA_pool_registers": 64512,
    "requested_role_pool_registers": 64512,
    "producer_gpr_highest_including_implicit": 37,
    "wg0_gpr_highest_including_implicit": 209,
    "wg1_gpr_highest_including_implicit": 215,
    "raw_A_transaction_bytes_per_stage": 32768,
    "raw_A_TMA_boxes_per_stage": 4,
    "A_native_lda_upper_exclusive": 549755813888,
    "B_native_stride_extra_bound": null,
    "A_shared_absolute_alignment_bytes": 1024,
    "occupancy_API_predicted_active_CTAs_per_SM": 1,
    "actual_measured_CTA_residency": null,
    "all_P_clear_after_INC": false,
    "legal_initial_owned_preINC_zero": true,
    "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained before tail crosses bank1",
    "function_probe_path": "results/psplit_tma_bn64_function_occupancy_probe_iter43.txt",
    "resource_metadata_scope": "Actual fresh iteration43 feeding function attributes/hostABI plus independent fullrole/lifetime gate; API occupancy prediction only."
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | bf16 | 12.244480 / 12.229888 / 23.975039 | 25.914559 / 25.894593 / 25.929280 | 0.472494 | 48.4429/0.0609 | 12.32076/25.99406 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_launch_iter43_full1024.csv:2) |
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | fp16 | 9.964480 / 9.955328 / 9.985728 | 26.044481 / 26.024832 / 26.061184 | 0.382595 | 0.1390/0.0664 | 10.04237/26.12422 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_launch_iter43_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.137600 / 12.131584 / 12.142848 | 17.650239 / 17.644416 / 17.666240 | 0.687673 | 0.0433/0.0575 | 12.21571/17.72935 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter43_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 10.000000 / 9.974912 / 10.078016 | 17.826880 / 17.797312 / 17.845440 | 0.560951 | 0.5341/0.1274 | 10.09483/17.90495 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter43_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.272960 / 12.252032 / 12.280064 | 23.166080 / 23.163008 / 23.183296 | 0.529781 | 0.1058/0.0459 | 12.34982/23.25070 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter43_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.921280 / 9.908160 / 9.931840 | 23.111041 / 23.109439 / 23.113599 | 0.429287 | 0.1062/0.0087 | 9.99676/23.18863 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter43_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 12.253439 / 12.229056 / 12.274944 | 23.518400 / 23.497921 / 23.553408 | 0.521015 | 0.1657/0.1068 | 12.32808/23.59073 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter43_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.786240 / 9.780352 / 9.823424 | 23.384960 / 23.364224 / 23.451713 | 0.418484 | 0.2093/0.1979 | 9.85873/23.45988 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter43_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.127039 / 12.116928 / 12.136192 | 19.284480 / 19.269056 / 19.285952 | 0.628850 | 0.0718/0.0413 | 12.20857/19.36422 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter43_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.980160 / 9.968704 / 9.987136 | 19.687040 / 19.669760 / 19.726336 | 0.506941 | 0.0842/0.1287 | 10.05837/19.76603 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter43_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | bf16 | 11.360960 / 11.357184 / 11.361856 | 25.493121 / 25.471936 / 25.514560 | 0.445648 | 0.0187/0.0780 | 11.44041/25.57228 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_graph_iter43_full1024.csv:2) |
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | fp16 | 9.470400 / 9.460544 / 9.484864 | 25.657599 / 25.641472 / 25.672320 | 0.369107 | 0.1134/0.0536 | 9.55490/25.73865 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_graph_iter43_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.313280 / 11.299072 / 11.323136 | 16.940160 / 16.927744 / 16.944896 | 0.667838 | 0.0973/0.0479 | 11.39047/17.01749 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter43_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.494080 / 9.485888 / 9.505152 | 17.283200 / 17.265856 / 17.298176 | 0.549324 | 0.0949/0.0898 | 9.57770/17.36129 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter43_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.345280 / 11.337729 / 11.352896 | 22.928319 / 22.923136 / 22.934017 | 0.494815 | 0.0590/0.0229 | 11.42527/23.00889 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter43_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.592320 / 9.582592 / 19.117760 | 22.907200 / 22.893440 / 22.939200 | 0.418747 | 49.7601/0.0909 | 9.67004/22.98787 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter43_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 11.329600 / 11.323136 / 11.341504 | 23.095040 / 23.074240 / 23.110847 | 0.490564 | 0.0726/0.0726 | 11.41945/23.18454 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter43_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.563520 / 9.561216 / 9.565376 | 23.146241 / 23.132927 / 23.155200 | 0.413178 | 0.0200/0.0471 | 9.64312/23.22551 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter43_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.330240 / 11.317312 / 11.338496 | 17.086080 / 17.073920 / 17.092096 | 0.663127 | 0.0840/0.0502 | 11.40249/17.15956 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter43_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.445440 / 9.436096 / 9.469056 | 17.889920 / 17.887040 / 17.901120 | 0.527976 | 0.1616/0.0361 | 9.51780/17.97103 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter43_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega45 | Mega47 | 17.650239 | 25.914559 | 0.681094× | +46.823% | 17.72935/25.99406 |
| launch | bf16 | Mega39 | Mega45 | 23.166080 | 17.650239 | 1.312508× | -23.810% | 23.25070/17.72935 |
| launch | bf16 | Mega39 | Mega47 | 23.166080 | 25.914559 | 0.893941× | +11.864% | 23.25070/25.99406 |
| launch | fp16 | Mega45 | Mega47 | 17.826880 | 26.044481 | 0.684478× | +46.097% | 17.90495/26.12422 |
| launch | fp16 | Mega39 | Mega45 | 23.111041 | 17.826880 | 1.296415× | -22.864% | 23.18863/17.90495 |
| launch | fp16 | Mega39 | Mega47 | 23.111041 | 26.044481 | 0.887368× | +12.693% | 23.18863/26.12422 |
| graph | bf16 | Mega45 | Mega47 | 16.940160 | 25.493121 | 0.664499× | +50.489% | 17.01749/25.57228 |
| graph | bf16 | Mega39 | Mega45 | 22.928319 | 16.940160 | 1.353489× | -26.117% | 23.00889/17.01749 |
| graph | bf16 | Mega39 | Mega47 | 22.928319 | 25.493121 | 0.899392× | +11.186% | 23.00889/25.57228 |
| graph | fp16 | Mega45 | Mega47 | 17.283200 | 25.657599 | 0.673609× | +48.454% | 17.36129/25.73865 |
| graph | fp16 | Mega39 | Mega45 | 22.907200 | 17.283200 | 1.325403× | -24.551% | 22.98787/17.36129 |
| graph | fp16 | Mega39 | Mega47 | 22.907200 | 25.657599 | 0.892804× | +12.007% | 22.98787/25.73865 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。

Mega/Lt47 launch bf16：Dense CV=48.442920%，SW CV=0.060922%；stdout全部Dense event组均值µs=[12.233, 12.2278, 12.2445, 12.2448, 31.7952]；[CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_launch_iter43_full1024.csv:2)。没有剔除较慢组；分母波动不能单独证明代码收益。
Mega/Lt39 graph fp16：Dense CV=49.760102%，SW CV=0.090901%；stdout全部Dense event组均值µs=[9.59104, 9.57696, 9.59232, 9.5936, 25.4672]；[CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter43_full1024.csv:3)。没有剔除较慢组；分母波动不能单独证明代码收益。

## 新47实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 47 | fp16 | 64×64 /64 | 384 | 181248 | 168 | 168 | 64512 | 48/216/240 | 64512 | 0/0/0 |
| 47 | bf16 | 64×64 /64 | 384 | 181248 | 168 | 168 | 64512 | 48/216/240 | 64512 | 0/0/0 |

Mega47 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。
Mega47 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。

原编译门限与资源修复时序：[psplit_tma_bn64_static_review_initial_iter43.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn64_static_review_initial_iter43.json)；这是编译资源放行记录，不认证数值或性能。

初始源码 [psplit_tma_bn64_initial_source_iter43.cu](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn64_initial_source_iter43.cu)、SHA 1b75138ff32255c6f08e259f183c1294f86390ed05e8460590e9f876a0c5404b、配额 40/224/240：STACK=128、spill loads/stores=168/168，status=FAIL_ZERO_STACK_SPILL_NO_NATIVE_LAUNCH，native launches=0。

revision1源码 [psplit_tma_bn64_revision1_source_iter43.cu](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn64_revision1_source_iter43.cu)、SHA bd3ff2014754b1395a7da8aa6a2c1a05bfd6f978fe3a94eedb9293eb43d27d11：STACK=8、spill loads/stores=8/8，native launches=0；失败原件保持独立身份。

最终门限明确记录 all_P_clear_after_INC=false、legal_initial_owned_preINC_zero=true；accepted_by=/root explicit iteration43 message before CAN。common P zero寄存器范围 56..151 在实际initial-owned范围内，允许这部分清零发生在consumer INC前；不能写成全部P在INC后清零。All 96 common P registers are owned under actual initial I168 and remain owned after INC. No grown register access occurs before successful INC; all P words are zero at first MMA on every examined CFG path. Literal after-INC scheduling requirement revised explicitly; no source/compiler repair performed.

最终fp16配额 48/216/240，actual I=168、owned/requested pool=64512/64512；producer/WG0/WG1最大物理GPR=37/209/215。实例初始化标志 all_P_clear_after_INC=false、legal_initial_owned_preINC_zero=true；所有grown访问/首MMA初始化证明按本次原门限记录，不继承旧配置。

最终bf16配额 48/216/240，actual I=168、owned/requested pool=64512/64512；producer/WG0/WG1最大物理GPR=37/209/215。实例初始化标志 all_P_clear_after_INC=false、legal_initial_owned_preINC_zero=true；所有grown访问/首MMA初始化证明按本次原门限记录，不继承旧配置。


Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=2862/2698/5080；[correctness_psplit_tma_bn64_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn64_iter43.txt)。
selected47：actualexit=0，CASE/PASS/BEGIN=60/140/204；[correctness_psplit_tma_bn6447_targeted_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn6447_targeted_iter43.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_tma_bn6437_cli_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_tma_bn6437_cli_iter43.txt)。

新47 rawU groups=84，reference=45（CROSSATOM_N64_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=40/120。

新47实际完成10项主机端A-only stride边界检查（1组PASS），其中native helper guard有4项。Plan设置成功0项、不可用10项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
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
      "ldb": 128,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_bn64_64x64",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725776,
      "safe_b_span_bytes": 32768,
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
      "ldb": 128,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727808,
      "safe_b_span_bytes": 32768,
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
      "ldb": 128,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976729840,
      "safe_b_span_bytes": 32768,
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
      "ldb": 128,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976728062,
      "safe_b_span_bytes": 32768,
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn64_64x64",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 32768,
      "safe_b_span_bytes": 139637976727808,
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
      "ldb": 128,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_bn64_64x64",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725776,
      "safe_b_span_bytes": 32768,
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
      "ldb": 128,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727808,
      "safe_b_span_bytes": 32768,
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
      "ldb": 128,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976729840,
      "safe_b_span_bytes": 32768,
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
      "ldb": 128,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976728062,
      "safe_b_span_bytes": 32768,
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn64_64x64",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 32768,
      "safe_b_span_bytes": 139637976727808,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    }
  ]
}
```

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1860、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_tma_bn64_enabled_only",
    "path": "results/strict_psplit_tma_bn64_iter43.txt",
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

SAN47 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_tma_bn6447_memcheck_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn6447_memcheck_iter43.txt)。非mem泄漏未测null。

SAN47 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_tma_bn6447_racecheck_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn6447_racecheck_iter43.txt)。非mem泄漏未测null。

SAN47 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_tma_bn6447_synccheck_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn6447_synccheck_iter43.txt)。非mem泄漏未测null。

CPUdag actualexit=0，tests=3；[psplit_tma_bn64_cpu_dag_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn64_cpu_dag_iter43.txt)。

CPUstage actualexit=0，tests=6；[psplit_tma_bn64_cpu_stage_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn64_cpu_stage_iter43.txt)。

CPU模型actualexit=0，originalstatus=PASS_CPU_BN64_GEOMETRY_AND_PROTOCOL；[psplit_tma_bn64_cpu_state_model_iter43.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn64_cpu_state_model_iter43.txt)。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。

环境：

```json
{
  "path": "results/small_psplit_tma_bn64_environment_iter43.txt",
  "exit_path": "results/small_psplit_tma_bn64_environment_iter43.exit",
  "sha256": "511104d11bf4d0dae6110c7546c4185081cc2c4acf6946b0cca8d6b074aa03b2",
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
    "path": "results/psplit_tma_bn64_function_occupancy_probe_iter43.txt",
    "sha256": "46d3a0ec578aab1fd79ff263bc9636e6771646ecd7c42d92d62a201ed8e6ad49",
    "exit": 0,
    "records": [
      {
        "variant": 47,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm68_GLOBAL__N__affad8eb_35_megakernel_wgmma_psplit_tma_bn64_cu_6ac820b730winograd_wgmma_psplit_tma_bn64IN7cutlass6half_tELi64EEEvNS_7ProblemENS0_17TmaBN64DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn64_final_megakernel_wgmma_psplit_tma_bn64_iter43.cubin",
        "cubin_sha256": "78cf68b34b563b04dbde75d8a37ac7b421044a41477eddaff7ffde23e4c3f947",
        "queried_dynamic_shared_bytes": 181248,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 168,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 181248
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      },
      {
        "variant": 47,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm68_GLOBAL__N__affad8eb_35_megakernel_wgmma_psplit_tma_bn64_cu_6ac820b730winograd_wgmma_psplit_tma_bn64IN7cutlass10bfloat16_tELi64EEEvNS_7ProblemENS0_17TmaBN64DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn64_final_megakernel_wgmma_psplit_tma_bn64_iter43.cubin",
        "cubin_sha256": "78cf68b34b563b04dbde75d8a37ac7b421044a41477eddaff7ffde23e4c3f947",
        "queried_dynamic_shared_bytes": 181248,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 168,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 181248
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      }
    ]
  }
}
```

唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。
