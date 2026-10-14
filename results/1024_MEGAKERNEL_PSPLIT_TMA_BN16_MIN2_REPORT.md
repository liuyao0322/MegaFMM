# 1024 BN16 min2选择性TMA rawA实测（iteration45）

Mega49对同binary Mega48自身完整调用：launch bf16 25.678401µs对27.596159µs，减少6.949%；launch fp16 25.307200µs对28.145919µs，减少10.086%；graph bf16 25.517759µs对27.116799µs，减少5.897%；graph fp16 24.939840µs对27.740481µs，减少10.096%。

Mega49对同binary Mega45自身完整调用：launch bf16 25.678401µs对17.748160µs，延长44.682%；launch fp16 25.307200µs对17.865280µs，延长41.656%；graph bf16 25.517759µs对17.063360µs，延长49.547%；graph fp16 24.939840µs对17.331520µs，延长43.899%。

Mega45对同binary Mega39自身完整调用：launch bf16 17.748160µs对23.184640µs，减少23.449%；launch fp16 17.865280µs对23.109119µs，减少22.692%；graph bf16 17.063360µs对22.890561µs，减少25.457%；graph fp16 17.331520µs对22.799680µs，减少23.983%。

Mega49对同binary Mega39自身完整调用：launch bf16 25.678401µs对23.184640µs，延长10.756%；launch fp16 25.307200µs对23.109119µs，延长9.512%；graph bf16 25.517759µs对22.890561µs，延长11.477%；graph fp16 24.939840µs对22.799680µs，延长9.387%。

本轮launch bf16已测native中最快为Mega45，17.748160µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.865280µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，17.063360µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.331520µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit 65a22921f232a74fbdfe6dcd50e0fe3dc0dfef99；[small_psplit_tma_bn16_min2_profile_iter45.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_profile_iter45.json)，[small_psplit_tma_bn16_min2_measurements_iter45.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_measurements_iter45.json)。

## 测量协议

ONLY49是新BN16 min2选择性四路TMA rawA路线；48为上一轮BN16 TMA，45为BN32 TMA，39为原RS-A BK64，Lt1为完整多launch对照。50未注册、未运行。fresh compiled blocked49必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替；45对39四个同binary配对仍可实际测得。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

49、48和45的源码入口在普通启动时每次从当前A指针/lda编码一个A tensor map；实际enabled路线的编码与单个kernel均包含在完整调用计时内，map实际被四路rawA TMA消费，B仍SIMT；39无host descriptor encode。49最终几何/配额/lifetime由fresh gate与冻结参数读取；48为BN16/BK64，45为BN32/BK64。min2是launch-bounds配置，API预测也不等于实际两CTA驻留；所有controlled_axis_claimed=false，不认证纯BN或纯搬运因果。全部实际运行native完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，为一个kernel、零矩阵workspace；Lt1完整多launch计时。Graph49/48/45在capture时编码，稳态replay使用固定capture地址读取当前内容，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时。rawU的reference variant与sameatom/crossatom关系只按本轮实际coverage记录；严格bitwise fixture和独立host RN alpha/FMA不构成通用FP64精度认证，任何实际不一致/FAIL必须保留。

巨大stride记录只按本轮实际host metadata检查与Plan setup结果表述，不能称作已分配巨大GPU输入并执行fallback；setup不可用时保留原message。此处不认证巨大strideGPU运行。

batchwall是每组100调用含批末同步的host wall除以调用数，再取组median，不是独立同步单调用CPU wall。setup/tuning/capture不计稳态；event可能含提交空隙，环境快照不认证组内clock。所有组、原FAIL/nonfinite/退出码均保留。

冻结参数：

```json
{
  "48": {
    "actual_name": "megakernel_wgmma_psplit_tma_bn16_64x16",
    "enabled": true,
    "producer_quota": 40,
    "consumer_quotas": [
      128,
      128
    ],
    "launch_bounds_min_ctas": 1,
    "BM": 64,
    "BN": 16,
    "BK": 64,
    "threads": 384,
    "dynamic_shared_bytes": 95232,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "initial_registers_per_thread": 128,
    "initial_owned_CTA_pool_registers": 49152,
    "requested_role_pool_registers": 37888,
    "producer_gpr_highest_including_implicit": 37,
    "wg0_gpr_highest_including_implicit": 125,
    "wg1_gpr_highest_including_implicit": 124,
    "raw_A_transaction_bytes_per_stage": 32768,
    "raw_A_TMA_boxes_per_stage": 4,
    "A_native_lda_upper_exclusive": 549755813888,
    "B_native_stride_extra_bound": null,
    "A_shared_absolute_alignment_bytes": 1024,
    "occupancy_API_predicted_active_CTAs_per_SM": 1,
    "actual_measured_CTA_residency": null,
    "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained before tail reuse",
    "function_probe_path": "results/psplit_tma_bn16_function_occupancy_probe_iter44.txt",
    "resource_metadata_scope": "Retained legacy attributes; not a fresh iteration45 function/residency query."
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
    "resource_metadata_scope": "Retained legacy attributes; not a fresh iteration45 function/residency query."
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
    "resource_metadata_scope": "Retained legacy attributes; not a fresh iteration45 function/residency query."
  },
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
    "resource_metadata_scope": "Fresh49 gate original actual attributes and four-atom physical lifetimes; API prediction2 is not measured residency."
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | bf16 | 12.452800 / 12.304896 / 12.619776 | 25.678401 / 25.675840 / 25.686592 | 0.484952 | 1.1296/0.0197 | 12.53212/25.76457 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_launch_iter45_full1024.csv:2) |
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | fp16 | 10.031360 / 9.999296 / 10.035136 | 25.307200 / 25.295104 / 25.312256 | 0.396384 | 0.1682/0.0317 | 10.11452/25.39271 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_launch_iter45_full1024.csv:3) |
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | bf16 | 12.276480 / 12.241664 / 12.313472 | 27.596159 / 27.587264 / 27.630465 | 0.444862 | 0.2600/0.0713 | 12.35371/27.67932 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_48_launch_iter45_full1024.csv:2) |
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | fp16 | 9.852480 / 9.843776 / 9.925376 | 28.145919 / 28.140224 / 28.193472 | 0.350050 | 0.4430/0.0928 | 9.92826/28.23670 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_48_launch_iter45_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.265919 / 12.253312 / 12.285376 | 17.748160 / 17.731712 / 17.767616 | 0.691109 | 0.1163/0.0923 | 12.33695/17.82163 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter45_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.929920 / 9.918464 / 9.942912 | 17.865280 / 17.848960 / 17.874176 | 0.555822 | 0.1086/0.0649 | 10.00400/17.93745 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter45_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.181760 / 12.167232 / 12.209600 | 23.184640 / 23.167295 / 23.196992 | 0.525424 | 0.1557/0.0626 | 12.25693/23.25682 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter45_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.917760 / 9.906368 / 9.929088 | 23.109119 / 23.095296 / 23.115840 | 0.429171 | 0.1035/0.0410 | 9.98946/23.18324 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter45_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.233280 / 12.207040 / 12.262720 | 19.355520 / 19.326464 / 19.370816 | 0.632031 | 0.2051/0.1019 | 12.31039/19.42665 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter45_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.958400 / 9.946176 / 9.960512 | 19.712321 / 19.690048 / 19.721792 | 0.505187 | 0.0794/0.0736 | 10.03030/19.78911 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter45_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | bf16 | 11.363519 / 11.359424 / 11.367296 | 25.517759 / 25.505023 / 25.534464 | 0.445318 | 0.0314/0.0530 | 11.43813/25.59445 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_graph_iter45_full1024.csv:2) |
| Mega49 megakernel_wgmma_psplit_tma_bn16_min2_64x16 | fp16 | 9.425600 / 9.416896 / 9.441024 | 24.939840 / 24.937984 / 24.943807 | 0.377933 | 0.1148/0.0121 | 9.50216/25.01422 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_min2_49_graph_iter45_full1024.csv:3) |
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | bf16 | 11.336319 / 11.285504 / 11.349248 | 27.116799 / 27.107840 / 27.135039 | 0.418055 | 0.2569/0.0480 | 11.41702/27.20307 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_48_graph_iter45_full1024.csv:2) |
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | fp16 | 9.410560 / 9.383040 / 9.429696 | 27.740481 / 27.689472 / 27.756735 | 0.339236 | 0.2191/0.1133 | 9.49226/27.81932 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_48_graph_iter45_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.356800 / 11.352191 / 11.361536 | 17.063360 / 17.047872 / 17.067840 | 0.665566 | 0.0363/0.0537 | 11.43527/17.14329 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter45_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.566400 / 9.562496 / 9.571392 | 17.331520 / 17.329856 / 17.334400 | 0.551965 | 0.0410/0.0136 | 9.64395/17.41461 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter45_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.315200 / 11.307968 / 11.328768 | 22.890561 / 22.882240 / 22.902912 | 0.494317 | 0.0845/0.0403 | 11.40264/22.97786 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter45_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.403520 / 9.396544 / 9.410496 | 22.799680 / 22.796479 / 22.807552 | 0.412441 | 0.0655/0.0226 | 9.48150/22.88594 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter45_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.344000 / 11.337408 / 11.363136 | 16.941760 / 16.936960 / 16.947072 | 0.669588 | 0.1019/0.0264 | 11.42235/17.01819 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter45_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.493440 / 9.447680 / 9.522368 | 17.375680 / 17.349440 / 17.404224 | 0.546364 | 0.3598/0.1426 | 9.56763/17.44853 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter45_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega48 | Mega49 | 27.596159 | 25.678401 | 1.074684× | -6.949% | 27.67932/25.76457 |
| launch | bf16 | Mega45 | Mega49 | 17.748160 | 25.678401 | 0.691171× | +44.682% | 17.82163/25.76457 |
| launch | bf16 | Mega39 | Mega45 | 23.184640 | 17.748160 | 1.306312× | -23.449% | 23.25682/17.82163 |
| launch | bf16 | Mega39 | Mega49 | 23.184640 | 25.678401 | 0.902885× | +10.756% | 23.25682/25.76457 |
| launch | fp16 | Mega48 | Mega49 | 28.145919 | 25.307200 | 1.112170× | -10.086% | 28.23670/25.39271 |
| launch | fp16 | Mega45 | Mega49 | 17.865280 | 25.307200 | 0.705937× | +41.656% | 17.93745/25.39271 |
| launch | fp16 | Mega39 | Mega45 | 23.109119 | 17.865280 | 1.293521× | -22.692% | 23.18324/17.93745 |
| launch | fp16 | Mega39 | Mega49 | 23.109119 | 25.307200 | 0.913144× | +9.512% | 23.18324/25.39271 |
| graph | bf16 | Mega48 | Mega49 | 27.116799 | 25.517759 | 1.062664× | -5.897% | 27.20307/25.59445 |
| graph | bf16 | Mega45 | Mega49 | 17.063360 | 25.517759 | 0.668686× | +49.547% | 17.14329/25.59445 |
| graph | bf16 | Mega39 | Mega45 | 22.890561 | 17.063360 | 1.341504× | -25.457% | 22.97786/17.14329 |
| graph | bf16 | Mega39 | Mega49 | 22.890561 | 25.517759 | 0.897044× | +11.477% | 22.97786/25.59445 |
| graph | fp16 | Mega48 | Mega49 | 27.740481 | 24.939840 | 1.112296× | -10.096% | 27.81932/25.01422 |
| graph | fp16 | Mega45 | Mega49 | 17.331520 | 24.939840 | 0.694933× | +43.899% | 17.41461/25.01422 |
| graph | fp16 | Mega39 | Mega45 | 22.799680 | 17.331520 | 1.315504× | -23.983% | 22.88594/17.41461 |
| graph | fp16 | Mega39 | Mega49 | 22.799680 | 24.939840 | 0.914187× | +9.387% | 22.88594/25.01422 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新49实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 49 | fp16 | 64×16 /64 | 384 | 95232 | 80 | 80 | 30720 | 40/88/112 | 30720 | 0/0/0 |
| 49 | bf16 | 64×16 /64 | 384 | 95232 | 80 | 80 | 30720 | 40/88/112 | 30720 | 0/0/0 |

Mega49 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2；measured_residency=None。
Mega49 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2；measured_residency=None。

原编译门限与本轮资源修复时序：[psplit_tma_bn16_min2_static_review_initial_iter45.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_min2_static_review_initial_iter45.json)；这是本轮编译资源放行记录，不认证数值或性能。

initial_failed_revision：[psplit_tma_bn16_min2_initial_source_iter45.cu](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_min2_initial_source_iter45.cu)；失败身份与当前放行配置分开保留。

```json
{
  "source_path": "results/psplit_tma_bn16_min2_initial_source_iter45.cu",
  "source_sha256": "4a0be150b0ade3b892d0828c084ebf2f0570a1780b60a7c8dbd1a9d6d4f60b01",
  "diagnostic_path": ".scratch/psplit_tma_bn16_min2_static_review_failed_initial_iter45.json",
  "diagnostic_sha256": "899fc3a192f4ca7ba06fd832fb7c5d21dc66051bebd93b25940a35a0dc48dca7",
  "status": "FAIL_STACK24_FP16_SPILL40_BF16_SPILL36_NEW_C7512_NO_NATIVE_LAUNCH",
  "test_linked": false
}
```

原门限记录的源码修复：{"delta_path": "results/psplit_tma_bn16_min2_atom_lifetime_repair_delta_iter45.json", "physical_A_words_per_window": [12, 16], "source_math_API_independent_review": "PASS; inverse whole bytes to initial4a0, original RN pairs/onecast/ascending perP K16 chronology retained; pinned rank2A/B with rank3P dispatch valid.", "source_sha256": "7a1b609d4ae3094b93438c672e96c3c63f8a0cc7e5d1037ebe9b14d14b565388"}。这不构成性能收益的因果证明。

最终fp16配额 40/88/112，actual I=80、owned/requested pool=30720/30720；producer/WG0/WG1最大物理GPR=37/77/71。实例初始化原标志 all_P_clear_after_successful_INC=false、legal_initial_owned_preINC_zero_present=true；原字段按本轮门限逐字记录，初始化或grown标志只按当轮原件记录，不从旧控制继承。

最终bf16配额 40/88/112，actual I=80、owned/requested pool=30720/30720；producer/WG0/WG1最大物理GPR=37/77/71。实例初始化原标志 all_P_clear_after_successful_INC=false、legal_initial_owned_preINC_zero_present=true；原字段按本轮门限逐字记录，初始化或grown标志只按当轮原件记录，不从旧控制继承。


Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=2982/2974/5488；[correctness_psplit_tma_bn16_min2_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn16_min2_iter45.txt)。
selected49：actualexit=0，CASE/PASS/BEGIN=60/140/204；[correctness_psplit_tma_bn16_min249_targeted_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn16_min249_targeted_iter45.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_tma_bn16_min237_cli_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_tma_bn16_min237_cli_iter45.txt)。

新49 rawU groups=84，reference=45（CROSSATOM_N16_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=40/120。

新49实际完成10项主机端A-only stride边界检查（1组PASS），其中native helper guard有4项。Plan设置成功0项、不可用10项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_min2_64x16",
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_min2_64x16",
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_min2_64x16",
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_min2_64x16",
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

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1920、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_tma_bn16_min2_enabled_only",
    "path": "results/strict_psplit_tma_bn16_min2_iter45.txt",
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

SAN49 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_tma_bn16_min249_memcheck_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn16_min249_memcheck_iter45.txt)。非mem泄漏未测null。

SAN49 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_tma_bn16_min249_racecheck_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn16_min249_racecheck_iter45.txt)。非mem泄漏未测null。

SAN49 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_tma_bn16_min249_synccheck_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn16_min249_synccheck_iter45.txt)。非mem泄漏未测null。

CPUdag actualexit=0，tests=3；[psplit_tma_bn16_min2_cpu_dag_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_min2_cpu_dag_iter45.txt)。

CPUstage actualexit=0，tests=6；[psplit_tma_bn16_min2_cpu_stage_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_min2_cpu_stage_iter45.txt)。

CPU模型actualexit=0，originalstatus=PASS_CPU_SOURCE_AND_REGISTER_ARITHMETIC_ONLY；[psplit_tma_bn16_min2_cpu_state_model_iter45.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_min2_cpu_state_model_iter45.txt)。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。

环境：

```json
{
  "path": "results/small_psplit_tma_bn16_min2_environment_iter45.txt",
  "exit_path": "results/small_psplit_tma_bn16_min2_environment_iter45.exit",
  "sha256": "274fb417233060ccd6467da0294834b148e83fb275564a509eeed98327e22925",
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
    "path": "results/psplit_tma_bn16_min2_function_occupancy_probe_iter45.txt",
    "sha256": "f400bce1e13295de7d0179651ec664e7ba2793b8ebbadc2b905d8c4ae9b983c2",
    "exit": 0,
    "records": [
      {
        "variant": 49,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm73_GLOBAL__N__43bdf2f2_40_megakernel_wgmma_psplit_tma_bn16_min2_cu_fb92dea735winograd_wgmma_psplit_tma_bn16_min2IN7cutlass6half_tELi16EEEvNS_7ProblemENS0_17TmaBN16DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn16_min2_final_megakernel_wgmma_psplit_tma_bn16_min2_iter45.cubin",
        "cubin_sha256": "5b468eaa2aaae4cd1622f508d7aad176c6fe9048147253dc98e19e42f55a3105",
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
        "variant": 49,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm73_GLOBAL__N__43bdf2f2_40_megakernel_wgmma_psplit_tma_bn16_min2_cu_fb92dea735winograd_wgmma_psplit_tma_bn16_min2IN7cutlass10bfloat16_tELi16EEEvNS_7ProblemENS0_17TmaBN16DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn16_min2_final_megakernel_wgmma_psplit_tma_bn16_min2_iter45.cubin",
        "cubin_sha256": "5b468eaa2aaae4cd1622f508d7aad176c6fe9048147253dc98e19e42f55a3105",
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
