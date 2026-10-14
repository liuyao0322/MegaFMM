# 1024 TMA原地S/T BN32与BN16实测（iteration47）

Mega51对同binary Mega45自身完整调用：launch bf16 24.739201µs对17.721280µs，延长39.602%；launch fp16 24.906881µs对17.759360µs，延长40.246%；graph bf16 24.419200µs对17.019520µs，延长43.478%；graph fp16 24.438720µs对17.394561µs，延长40.496%。

Mega52对同binary Mega45自身完整调用：launch bf16 31.077120µs对17.721280µs，延长75.366%；launch fp16 30.458241µs对17.759360µs，延长71.505%；graph bf16 30.775361µs对17.019520µs，延长80.824%；graph fp16 30.010879µs对17.394561µs，延长72.530%。

Mega51对同binary Mega52自身完整调用：launch bf16 24.739201µs对31.077120µs，减少20.394%；launch fp16 24.906881µs对30.458241µs，减少18.226%；graph bf16 24.419200µs对30.775361µs，减少20.653%；graph fp16 24.438720µs对30.010879µs，减少18.567%。

Mega51对同binary Mega50自身完整调用：launch bf16 24.739201µs对25.714879µs，减少3.794%；launch fp16 24.906881µs对25.208001µs，减少1.195%；graph bf16 24.419200µs对25.464320µs，减少4.104%；graph fp16 24.438720µs对24.968960µs，减少2.124%。

Mega52对同binary Mega50自身完整调用：launch bf16 31.077120µs对25.714879µs，延长20.853%；launch fp16 30.458241µs对25.208001µs，延长20.828%；graph bf16 30.775361µs对25.464320µs，延长20.857%；graph fp16 30.010879µs对24.968960µs，延长20.193%。

Mega45对同binary Mega39自身完整调用：launch bf16 17.721280µs对23.224959µs，减少23.697%；launch fp16 17.759360µs对23.072641µs，减少23.028%；graph bf16 17.019520µs对22.967680µs，减少25.898%；graph fp16 17.394561µs对22.829759µs，减少23.808%。

本轮launch bf16已测native中最快为Mega45，17.721280µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.759360µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，17.019520µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.394561µs，原数值门限FAIL。

实际行数24/预声明24，complete=True；fresh PASS/FAIL=0/72，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit 5bc3207d52b59d91c16246775fcfec7cd44853e6；[small_psplit_tma_inplace_ss_profile_iter47.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss_profile_iter47.json)，[small_psplit_tma_inplace_ss_measurements_iter47.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss_measurements_iter47.json)。

## 测量协议

51是新BN32原地S/T，52是新BN16原地S/T，均BM64/BK32；45为BN32/BK64选择性TMA rawA，50为BN16/BK64 K32窗口，39为RS-A BK64，Lt1为完整多launch。两条新路线各自fresh compiled blocked时准确保留四个缺失键，24行预声明相应剩20或16行，任何缺失不使用历史数据补齐。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

本轮实际enabled新51/52在每次普通完整调用中从当前A/B指针及lda/ldb编码两个tensor map，A/B原始输入由TMA搬入shared，并在原地构造S/T；完整编码与单个kernel均计时。Graph在capture时编码两个map，固定capture地址读取fresh内容。45/50普通调用编码一个被消费的A map，B仍SIMT；39没有host descriptor encode。各路线几何、quota、shared及生命周期按本轮实际源和fresh gate读取。所有实际native包含原RN变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，零矩阵workspace；Lt1计完整多launch。Graph差值不能单独当作encode成本或纯TMA收益。51 rawU参考36、52 rawU参考43的atom与共享native域只按实际source-bound coverage表述，reference runs与candidate runs分别计数；bit fixture与独立host epilogue不构成通用FP64精度认证。minCTA及API预测均不认证实际residency，六个比较的controlled_axis_claimed=false。

本轮原地S/T的一次构造范围是每个CTA、当前bank stage：每个pair task对其负责坐标的最终S/T plane各写一次；同一输入区域在不同CTA中仍可能重复搬入和变换。这是CTA内stage复用，不能称作H12的全矩阵global S/T一次构造。

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
  "51": {
    "actual_name": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
    "enabled": true,
    "BM": 64,
    "BN": 32,
    "BK": 32,
    "threads": 384,
    "producer_quota": 40,
    "consumer_quotas": [
      128,
      128
    ],
    "launch_bounds_min_ctas": 1,
    "dynamic_shared_bytes": 87040,
    "static_shared_bytes": 0,
    "resource_metadata_scope": "Fresh actual51 four-instance certificate/Driver/link, occupancyAPIpredictionnotresidency"
  },
  "52": {
    "actual_name": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
    "enabled": true,
    "BM": 64,
    "BN": 16,
    "BK": 32,
    "threads": 384,
    "producer_quota": 48,
    "consumer_quotas": [
      80,
      80
    ],
    "launch_bounds_min_ctas": 2,
    "dynamic_shared_bytes": 72704,
    "static_shared_bytes": 0,
    "resource_metadata_scope": "Fresh actual52 four-instance certificate/Driver/link, occupancyAPIpredictionnotresidency"
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega51 megakernel_wgmma_psplit_tma_inplace_ss_64x32 | bf16 | 12.341440 / 12.336128 / 12.347136 | 24.739201 / 24.702080 / 24.766528 | 0.498862 | 0.0402/0.1158 | 12.42024/24.81872 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss51_launch_iter47_full1024.csv:2) |
| Mega51 megakernel_wgmma_psplit_tma_inplace_ss_64x32 | fp16 | 9.917120 / 9.915648 / 9.930944 | 24.906881 / 24.895745 / 24.912384 | 0.398168 | 0.0729/0.0305 | 9.99864/24.98819 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss51_launch_iter47_full1024.csv:3) |
| Mega52 megakernel_wgmma_psplit_tma_inplace_ss_64x16 | bf16 | 12.263680 / 12.249984 / 12.282240 | 31.077120 / 31.063808 / 31.102144 | 0.394621 | 0.1238/0.0629 | 12.34462/31.15526 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss52_launch_iter47_full1024.csv:2) |
| Mega52 megakernel_wgmma_psplit_tma_inplace_ss_64x16 | fp16 | 9.897920 / 9.892096 / 9.965952 | 30.458241 / 30.424831 / 30.484672 | 0.324967 | 0.4469/0.0872 | 9.97634/30.54554 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss52_launch_iter47_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.212800 / 12.191936 / 12.231360 | 17.721280 / 17.708288 / 17.750400 | 0.689160 | 0.1420/0.1066 | 12.29127/17.80754 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter47_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.934080 / 9.932416 / 10.033792 | 17.759360 / 17.732608 / 17.768384 | 0.559372 | 0.6406/0.0974 | 10.01247/17.83782 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter47_full1024.csv:3) |
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | bf16 | 12.258559 / 12.246592 / 12.266560 | 25.714879 / 25.707649 / 25.739968 | 0.476711 | 0.0733/0.0575 | 12.33777/25.80659 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window50_launch_iter47_full1024.csv:2) |
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | fp16 | 9.965120 / 9.952768 / 9.975872 | 25.208001 / 25.204736 / 25.225856 | 0.395316 | 0.1077/0.0389 | 10.04371/25.28970 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window50_launch_iter47_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.317120 / 12.310144 / 12.346304 | 23.224959 / 23.220928 / 23.233984 | 0.530340 | 0.1342/0.0251 | 12.39819/23.31178 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter47_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.939520 / 9.929600 / 10.017408 | 23.072641 / 23.060864 / 23.077697 | 0.430792 | 0.5343/0.0349 | 10.01764/23.15265 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter47_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.282881 / 12.267520 / 12.292160 | 19.401920 / 19.391680 / 19.409088 | 0.633076 | 0.0896/0.0398 | 12.36029/19.48398 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter47_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.963840 / 9.941312 / 9.978496 | 19.682879 / 19.657217 / 19.694528 | 0.506219 | 0.1698/0.0915 | 10.04011/19.76240 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter47_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega51 megakernel_wgmma_psplit_tma_inplace_ss_64x32 | bf16 | 11.360641 / 11.347520 / 11.375936 | 24.419200 / 24.411904 / 24.484992 | 0.465234 | 0.1099/0.1631 | 11.43897/24.51450 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss51_graph_iter47_full1024.csv:2) |
| Mega51 megakernel_wgmma_psplit_tma_inplace_ss_64x32 | fp16 | 9.530560 / 9.516352 / 9.532800 | 24.438720 / 24.436928 / 24.446528 | 0.389978 | 0.0809/0.0180 | 9.60940/24.52193 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss51_graph_iter47_full1024.csv:3) |
| Mega52 megakernel_wgmma_psplit_tma_inplace_ss_64x16 | bf16 | 11.407360 / 11.399104 / 11.408768 | 30.775361 / 30.760705 / 30.809984 | 0.370665 | 0.0417/0.0775 | 11.48724/30.85475 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss52_graph_iter47_full1024.csv:2) |
| Mega52 megakernel_wgmma_psplit_tma_inplace_ss_64x16 | fp16 | 9.483520 / 9.471808 / 9.502528 | 30.010879 / 30.000448 / 30.026816 | 0.316003 | 0.1491/0.0396 | 9.56902/30.09045 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_inplace_ss52_graph_iter47_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.423039 / 11.413184 / 11.428480 | 17.019520 / 17.014336 / 17.035008 | 0.671173 | 0.0615/0.0568 | 11.50020/17.09660 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter47_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.528640 / 9.514368 / 9.541632 | 17.394561 / 17.391040 / 17.404992 | 0.547794 | 0.1345/0.0361 | 9.60896/17.47714 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter47_full1024.csv:3) |
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | bf16 | 11.379520 / 11.369216 / 11.384576 | 25.464320 / 25.457025 / 25.500224 | 0.446881 | 0.0633/0.0864 | 11.45742/25.55411 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window50_graph_iter47_full1024.csv:2) |
| Mega50 megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 | fp16 | 9.440320 / 9.433088 / 9.442240 | 24.968960 / 24.965313 / 24.992192 | 0.378082 | 0.0471/0.0498 | 9.52178/25.04891 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window50_graph_iter47_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.271040 / 11.264320 / 11.280000 | 22.967680 / 22.953024 / 22.980032 | 0.490735 | 0.0615/0.0524 | 11.34872/23.05545 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter47_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.420160 / 9.415168 / 9.476608 | 22.829759 / 22.823999 / 22.837120 | 0.412626 | 0.3187/0.0282 | 9.49929/22.91899 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter47_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.305920 / 11.292800 / 11.322944 | 17.088640 / 17.087168 / 17.096448 | 0.661604 | 0.1171/0.0285 | 11.38428/17.16928 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter47_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.551040 / 9.543680 / 9.590656 | 17.564800 / 17.555328 / 17.575552 | 0.543760 | 0.2480/0.0554 | 9.62905/17.64687 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter47_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega45 | Mega51 | 17.721280 | 24.739201 | 0.716324× | +39.602% | 17.80754/24.81872 |
| launch | bf16 | Mega45 | Mega52 | 17.721280 | 31.077120 | 0.570236× | +75.366% | 17.80754/31.15526 |
| launch | bf16 | Mega52 | Mega51 | 31.077120 | 24.739201 | 1.256189× | -20.394% | 31.15526/24.81872 |
| launch | bf16 | Mega50 | Mega51 | 25.714879 | 24.739201 | 1.039439× | -3.794% | 25.80659/24.81872 |
| launch | bf16 | Mega50 | Mega52 | 25.714879 | 31.077120 | 0.827454× | +20.853% | 25.80659/31.15526 |
| launch | bf16 | Mega39 | Mega45 | 23.224959 | 17.721280 | 1.310569× | -23.697% | 23.31178/17.80754 |
| launch | fp16 | Mega45 | Mega51 | 17.759360 | 24.906881 | 0.713030× | +40.246% | 17.83782/24.98819 |
| launch | fp16 | Mega45 | Mega52 | 17.759360 | 30.458241 | 0.583072× | +71.505% | 17.83782/30.54554 |
| launch | fp16 | Mega52 | Mega51 | 30.458241 | 24.906881 | 1.222885× | -18.226% | 30.54554/24.98819 |
| launch | fp16 | Mega50 | Mega51 | 25.208001 | 24.906881 | 1.012090× | -1.195% | 25.28970/24.98819 |
| launch | fp16 | Mega50 | Mega52 | 25.208001 | 30.458241 | 0.827625× | +20.828% | 25.28970/30.54554 |
| launch | fp16 | Mega39 | Mega45 | 23.072641 | 17.759360 | 1.299182× | -23.028% | 23.15265/17.83782 |
| graph | bf16 | Mega45 | Mega51 | 17.019520 | 24.419200 | 0.696973× | +43.478% | 17.09660/24.51450 |
| graph | bf16 | Mega45 | Mega52 | 17.019520 | 30.775361 | 0.553024× | +80.824% | 17.09660/30.85475 |
| graph | bf16 | Mega52 | Mega51 | 30.775361 | 24.419200 | 1.260294× | -20.653% | 30.85475/24.51450 |
| graph | bf16 | Mega50 | Mega51 | 25.464320 | 24.419200 | 1.042799× | -4.104% | 25.55411/24.51450 |
| graph | bf16 | Mega50 | Mega52 | 25.464320 | 30.775361 | 0.827426× | +20.857% | 25.55411/30.85475 |
| graph | bf16 | Mega39 | Mega45 | 22.967680 | 17.019520 | 1.349490× | -25.898% | 23.05545/17.09660 |
| graph | fp16 | Mega45 | Mega51 | 17.394561 | 24.438720 | 0.711762× | +40.496% | 17.47714/24.52193 |
| graph | fp16 | Mega45 | Mega52 | 17.394561 | 30.010879 | 0.579608× | +72.530% | 17.47714/30.09045 |
| graph | fp16 | Mega52 | Mega51 | 30.010879 | 24.438720 | 1.228005× | -18.567% | 30.09045/24.52193 |
| graph | fp16 | Mega50 | Mega51 | 24.968960 | 24.438720 | 1.021697× | -2.124% | 25.04891/24.52193 |
| graph | fp16 | Mega50 | Mega52 | 24.968960 | 30.010879 | 0.831997× | +20.193% | 25.04891/30.09045 |
| graph | fp16 | Mega39 | Mega45 | 22.829759 | 17.394561 | 1.312465× | -23.808% | 22.91899/17.47714 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新51/52实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | reported/effective initial REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 52 | fp16 | 64×16 /32 | 384 | 72704 | 80 | 80 | 30720 | 48/80/80 | 26624 | 0/0/0 |
| 52 | bf16 | 64×16 /32 | 384 | 72704 | 80 | 80 | 30720 | 48/80/80 | 26624 | 0/0/0 |
| 51 | fp16 | 64×32 /32 | 384 | 87040 | 128 | 128 | 49152 | 40/128/128 | 37888 | 0/0/0 |
| 51 | bf16 | 64×32 /32 | 384 | 87040 | 128 | 128 | 49152 | 40/128/128 | 37888 | 0/0/0 |

Mega52 fp16：launch_bounds_min_ctas=2，quota=48/80/80；Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2；measured_residency=None。
Mega52 bf16：launch_bounds_min_ctas=2，quota=48/80/80；Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2；measured_residency=None。
Mega51 fp16：launch_bounds_min_ctas=1，quota=40/128/128；Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。
Mega51 bf16：launch_bounds_min_ctas=1，quota=40/128/128；Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。

本轮原编译门限：[psplit_tma_inplace_ss_static_review_initial_iter47.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_inplace_ss_static_review_initial_iter47.json)；原字段与编译诊断按实际身份分别保留，资源放行不认证数值或性能。

chronological_failures：

```json
[
  {
    "source_path": "results/psplit_tma_inplace_ss_initial_source_iter47.cu",
    "scope": "First source460352 frontend compile exit2: absent API / eager unselected layout instantiation. No new machine proof. Earlier mathematical/protocol read is not a compiled API PASS."
  },
  {
    "source_path": "results/psplit_tma_inplace_ss_revision1_source_iter47.cu",
    "scope": "Rev1 BN32 REG80 STACK64 P16 locals/C7512; BN16 REG72 yet INC64 direction illegal. No launch."
  },
  {
    "source_path": "results/psplit_tma_inplace_ss_revision2_source_iter47.cu",
    "scope": "Rev2 BN32 still P16 spill;52 conservative quotas legal. No launch."
  },
  {
    "source_path": "solution/megakernel_wgmma_psplit_tma_inplace_ss.cu",
    "sha256": "733709ff8c820ac708cdb4977946d3312950df864e1e736a22ea437bc992e818",
    "scope": "Final rev3:51min1 40/128/128;52min2 48/80/80; all actual fresh four-instance gate PASS."
  }
]
```

新Mega52 fp16原role最高GPR={"WG0": 55, "WG1": 57, "producer": 37}，原初始化标志={"initialization_gate": {"actual_CFG_successful_INC_exit_PCs": {"WG0": "0x1780", "WG1": "0x3e0", "producer": "-0x1"}, "all_P_clear_after_INC": false, "first_MMA_accumulator_zero_definition_missing_count": 0, "grown_register_read_before_definition_count": 0, "grown_registers_actually_accessed": [], "high_register_access_before_successful_INC_count": 0, "legal_initial_owned_preINC_zero": true, "scope": "Actual current initial-owned zero definitions may precede INC. Current INC equals initial I; no expanded-register access exists."}}。这些字段只按本轮fresh门限解释，不从49/50或旧SS控制推断实际lifetime。

```json
{
  "physical_windows": {
    "operand_mode": "SS",
    "windows_per_K32_stage_per_WG": 1,
    "k16_atoms_per_product": 2,
    "WG0_product_count": 3,
    "WG1_product_count": 4,
    "WG0_MMA_count": 6,
    "WG1_MMA_count": 8,
    "P_words_per_product": 8,
    "RS_A_register_backing": false,
    "generic_P_access_before_matching_wait0": 0,
    "shared_or_local_store_before_matching_wait0": 0,
    "all_pending_operand_shared_banks_retained_until_both_WG_wait0_release": true
  },
  "raw_TMA": {
    "copies_per_stage": 8,
    "raw_expected_TX_bytes": 20480,
    "raw_phase_is_pipeline_phase_XOR_1": true,
    "all128_producer_proxy_fence_then_full_arrival": true
  }
}
```

新Mega52 bf16原role最高GPR={"WG0": 55, "WG1": 57, "producer": 37}，原初始化标志={"initialization_gate": {"actual_CFG_successful_INC_exit_PCs": {"WG0": "0x1780", "WG1": "0x3e0", "producer": "-0x1"}, "all_P_clear_after_INC": false, "first_MMA_accumulator_zero_definition_missing_count": 0, "grown_register_read_before_definition_count": 0, "grown_registers_actually_accessed": [], "high_register_access_before_successful_INC_count": 0, "legal_initial_owned_preINC_zero": true, "scope": "Actual current initial-owned zero definitions may precede INC. Current INC equals initial I; no expanded-register access exists."}}。这些字段只按本轮fresh门限解释，不从49/50或旧SS控制推断实际lifetime。

```json
{
  "physical_windows": {
    "operand_mode": "SS",
    "windows_per_K32_stage_per_WG": 1,
    "k16_atoms_per_product": 2,
    "WG0_product_count": 3,
    "WG1_product_count": 4,
    "WG0_MMA_count": 6,
    "WG1_MMA_count": 8,
    "P_words_per_product": 8,
    "RS_A_register_backing": false,
    "generic_P_access_before_matching_wait0": 0,
    "shared_or_local_store_before_matching_wait0": 0,
    "all_pending_operand_shared_banks_retained_until_both_WG_wait0_release": true
  },
  "raw_TMA": {
    "copies_per_stage": 8,
    "raw_expected_TX_bytes": 20480,
    "raw_phase_is_pipeline_phase_XOR_1": true,
    "all128_producer_proxy_fence_then_full_arrival": true
  }
}
```

新Mega51 fp16原role最高GPR={"WG0": 105, "WG1": 105, "producer": 37}，原初始化标志={"initialization_gate": {"actual_CFG_successful_INC_exit_PCs": {"WG0": "0x1960", "WG1": "0x4a0", "producer": "-0x1"}, "all_P_clear_after_INC": false, "first_MMA_accumulator_zero_definition_missing_count": 0, "grown_register_read_before_definition_count": 0, "grown_registers_actually_accessed": [], "high_register_access_before_successful_INC_count": 0, "legal_initial_owned_preINC_zero": true, "scope": "Actual current initial-owned zero definitions may precede INC. Current INC equals initial I; no expanded-register access exists."}}。这些字段只按本轮fresh门限解释，不从49/50或旧SS控制推断实际lifetime。

```json
{
  "physical_windows": {
    "operand_mode": "SS",
    "windows_per_K32_stage_per_WG": 1,
    "k16_atoms_per_product": 2,
    "WG0_product_count": 3,
    "WG1_product_count": 4,
    "WG0_MMA_count": 6,
    "WG1_MMA_count": 8,
    "P_words_per_product": 16,
    "RS_A_register_backing": false,
    "generic_P_access_before_matching_wait0": 0,
    "shared_or_local_store_before_matching_wait0": 0,
    "all_pending_operand_shared_banks_retained_until_both_WG_wait0_release": true
  },
  "raw_TMA": {
    "copies_per_stage": 8,
    "raw_expected_TX_bytes": 24576,
    "raw_phase_is_pipeline_phase_XOR_1": true,
    "all128_producer_proxy_fence_then_full_arrival": true
  }
}
```

新Mega51 bf16原role最高GPR={"WG0": 105, "WG1": 105, "producer": 37}，原初始化标志={"initialization_gate": {"actual_CFG_successful_INC_exit_PCs": {"WG0": "0x1960", "WG1": "0x4a0", "producer": "-0x1"}, "all_P_clear_after_INC": false, "first_MMA_accumulator_zero_definition_missing_count": 0, "grown_register_read_before_definition_count": 0, "grown_registers_actually_accessed": [], "high_register_access_before_successful_INC_count": 0, "legal_initial_owned_preINC_zero": true, "scope": "Actual current initial-owned zero definitions may precede INC. Current INC equals initial I; no expanded-register access exists."}}。这些字段只按本轮fresh门限解释，不从49/50或旧SS控制推断实际lifetime。

```json
{
  "physical_windows": {
    "operand_mode": "SS",
    "windows_per_K32_stage_per_WG": 1,
    "k16_atoms_per_product": 2,
    "WG0_product_count": 3,
    "WG1_product_count": 4,
    "WG0_MMA_count": 6,
    "WG1_MMA_count": 8,
    "P_words_per_product": 16,
    "RS_A_register_backing": false,
    "generic_P_access_before_matching_wait0": 0,
    "shared_or_local_store_before_matching_wait0": 0,
    "all_pending_operand_shared_banks_retained_until_both_WG_wait0_release": true
  },
  "raw_TMA": {
    "copies_per_stage": 8,
    "raw_expected_TX_bytes": 24576,
    "raw_phase_is_pipeline_phase_XOR_1": true,
    "all128_producer_proxy_fence_then_full_arrival": true
  }
}
```


reported/effective initial REG在原gate无独立reported字段时取其effective_initial_registers_per_thread，不能虚构另一个ELF计数。Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=3162/3388/6120；[correctness_psplit_tma_inplace_ss_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_inplace_ss_iter47.txt)。
selected51：actualexit=0，CASE/PASS/BEGIN=60/140/214；[correctness_psplit_tma_inplace_ss51_targeted_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_inplace_ss51_targeted_iter47.txt)。
selected52：actualexit=0，CASE/PASS/BEGIN=60/140/214；[correctness_psplit_tma_inplace_ss52_targeted_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_inplace_ss52_targeted_iter47.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_tma_inplace_ss37_cli_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_tma_inplace_ss37_cli_iter47.txt)。

新51 rawU groups=84，reference=36（SAMEATOM_N32_BK32）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=40/120。

新51实际完成20项主机端A/B stride边界检查（1组PASS），其中native helper guard有6项。Plan设置成功0项、不可用20项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
原始不可用信息：query Lt heuristics: cuBLAS status 15。这些记录仅验证helper predicate与Plan setup，setup_unavailable不能被计为成功Plan路由或已执行GPU fallback。

```json
{
  "PASS_groups": 1,
  "cases": 20,
  "native_guard_cases": 6,
  "packed_guard_cases": 10,
  "dense_guard_cases": 4,
  "setup_success_cases": 0,
  "setup_unavailable_cases": 20,
  "both_bounds_applied": true,
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
      "ldb": 64,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
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
      "ldb": 64,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727680,
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
      "ldb": 64,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976729712,
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
      "ldb": 64,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727934,
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
      "lda": 64,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232549008,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 5,
      "lda": 64,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550016,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 6,
      "lda": 64,
      "ldb": 549755813896,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232551024,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 7,
      "lda": 64,
      "ldb": 549755813889,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550142,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 8,
      "lda": 549755813880,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
      "safe_b_span_bytes": 69269232549008,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 9,
      "lda": 549755813888,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727680,
      "safe_b_span_bytes": 69269232550016,
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
      "ldb": 64,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
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
      "ldb": 64,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727680,
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
      "ldb": 64,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976729712,
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
      "ldb": 64,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727934,
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
      "lda": 64,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232549008,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 5,
      "lda": 64,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550016,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 6,
      "lda": 64,
      "ldb": 549755813896,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232551024,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 7,
      "lda": 64,
      "ldb": 549755813889,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550142,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 8,
      "lda": 549755813880,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
      "safe_b_span_bytes": 69269232549008,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 9,
      "lda": 549755813888,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727680,
      "safe_b_span_bytes": 69269232550016,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    }
  ]
}
```

新52 rawU groups=84，reference=43（SAMEATOM_N16_BK32）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=40/120。

新52实际完成20项主机端A/B stride边界检查（1组PASS），其中native helper guard有6项。Plan设置成功0项、不可用20项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
原始不可用信息：query Lt heuristics: cuBLAS status 15。这些记录仅验证helper predicate与Plan setup，setup_unavailable不能被计为成功Plan路由或已执行GPU fallback。

```json
{
  "PASS_groups": 1,
  "cases": 20,
  "native_guard_cases": 6,
  "packed_guard_cases": 10,
  "dense_guard_cases": 4,
  "setup_success_cases": 0,
  "setup_unavailable_cases": 20,
  "both_bounds_applied": true,
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
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
      "safe_b_span_bytes": 4096,
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
      "safe_a_span_bytes": 139637976727680,
      "safe_b_span_bytes": 4096,
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
      "safe_a_span_bytes": 139637976729712,
      "safe_b_span_bytes": 4096,
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
      "safe_a_span_bytes": 139637976727934,
      "safe_b_span_bytes": 4096,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 4,
      "lda": 64,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232548944,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 5,
      "lda": 64,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232549952,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 6,
      "lda": 64,
      "ldb": 549755813896,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550960,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 7,
      "lda": 64,
      "ldb": 549755813889,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550078,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 8,
      "lda": 549755813880,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
      "safe_b_span_bytes": 69269232548944,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "bf16",
      "scenario": 9,
      "lda": 549755813888,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727680,
      "safe_b_span_bytes": 69269232549952,
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
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
      "safe_b_span_bytes": 4096,
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
      "safe_a_span_bytes": 139637976727680,
      "safe_b_span_bytes": 4096,
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
      "safe_a_span_bytes": 139637976729712,
      "safe_b_span_bytes": 4096,
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
      "safe_a_span_bytes": 139637976727934,
      "safe_b_span_bytes": 4096,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 4,
      "lda": 64,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232548944,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 5,
      "lda": 64,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232549952,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 6,
      "lda": 64,
      "ldb": 549755813896,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550960,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 7,
      "lda": 64,
      "ldb": 549755813889,
      "native_guard": false,
      "expected_route": "dense_cublaslt",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 16384,
      "safe_b_span_bytes": 69269232550078,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 8,
      "lda": 549755813880,
      "ldb": 549755813880,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_tma_inplace_ss_64x16",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725648,
      "safe_b_span_bytes": 69269232548944,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    },
    {
      "dtype": "fp16",
      "scenario": 9,
      "lda": 549755813888,
      "ldb": 549755813888,
      "native_guard": false,
      "expected_route": "megakernel_warp_mma_packed",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976727680,
      "safe_b_span_bytes": 69269232549952,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    }
  ]
}
```

当前strict candidate gates=60，PASS/FAIL=36/24，nonfinite=0；native=48，Dense范围回退=12。full code domain=2010、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_inplace_ss_enabled_only",
    "path": "results/strict_psplit_tma_inplace_ss_iter47.txt",
    "exit": 2,
    "CASE": 90,
    "PASS": 0,
    "BEGIN": 0,
    "source": "Actual closed stdout and archived process exit; never expected counts.",
    "gates": 60,
    "pass": 36,
    "fail": 24,
    "nonfinite": 0,
    "native_gates": 48,
    "Dense_range_fallback_gates": 12,
    "shared_Dense_reference_CASE": 30
  }
]
```

SAN51 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_tma_inplace_ss51_memcheck_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_inplace_ss51_memcheck_iter47.txt)。非mem泄漏未测null。

SAN51 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_tma_inplace_ss51_racecheck_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_inplace_ss51_racecheck_iter47.txt)。非mem泄漏未测null。

SAN51 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_tma_inplace_ss51_synccheck_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_inplace_ss51_synccheck_iter47.txt)。非mem泄漏未测null。

SAN52 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_tma_inplace_ss52_memcheck_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_inplace_ss52_memcheck_iter47.txt)。非mem泄漏未测null。

SAN52 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_tma_inplace_ss52_racecheck_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_inplace_ss52_racecheck_iter47.txt)。非mem泄漏未测null。

SAN52 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_tma_inplace_ss52_synccheck_iter47.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_inplace_ss52_synccheck_iter47.txt)。非mem泄漏未测null。

环境：

```json
{
  "path": "results/small_psplit_tma_inplace_ss_environment_iter47.txt",
  "exit_path": "results/small_psplit_tma_inplace_ss_environment_iter47.exit",
  "sha256": "6098140257f27964054cf138ecff650e720e737c9c64c099f7455759400bd9c0",
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
    "path": "results/psplit_tma_inplace_ss_function_occupancy_probe_iter47.txt",
    "sha256": "acf6675bb4f7f3059bed5a15825972cf8b75902afcfe6a9efe5d80797f36aada",
    "exit": 0,
    "records": [
      {
        "variant": 52,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm74_GLOBAL__N__3c75913d_41_megakernel_wgmma_psplit_tma_inplace_ss_cu_62d0210a36winograd_wgmma_psplit_tma_inplace_ssIN7cutlass6half_tELi16EEEvNS_7ProblemENS0_23TmaInplaceSSDescriptorsEPKfPf",
        "cubin_path": "results/psplit_tma_inplace_ss_final_megakernel_wgmma_psplit_tma_inplace_ss_iter47.cubin",
        "cubin_sha256": "43a26889fce5a3992c05d57e1e2c24b9369f7494bac64dd69cdda47464e24c14",
        "queried_dynamic_shared_bytes": 72704,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 72704
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      },
      {
        "variant": 52,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm74_GLOBAL__N__3c75913d_41_megakernel_wgmma_psplit_tma_inplace_ss_cu_62d0210a36winograd_wgmma_psplit_tma_inplace_ssIN7cutlass10bfloat16_tELi16EEEvNS_7ProblemENS0_23TmaInplaceSSDescriptorsEPKfPf",
        "cubin_path": "results/psplit_tma_inplace_ss_final_megakernel_wgmma_psplit_tma_inplace_ss_iter47.cubin",
        "cubin_sha256": "43a26889fce5a3992c05d57e1e2c24b9369f7494bac64dd69cdda47464e24c14",
        "queried_dynamic_shared_bytes": 72704,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 72704
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      },
      {
        "variant": 51,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm74_GLOBAL__N__3c75913d_41_megakernel_wgmma_psplit_tma_inplace_ss_cu_62d0210a36winograd_wgmma_psplit_tma_inplace_ssIN7cutlass6half_tELi32EEEvNS_7ProblemENS0_23TmaInplaceSSDescriptorsEPKfPf",
        "cubin_path": "results/psplit_tma_inplace_ss_final_megakernel_wgmma_psplit_tma_inplace_ss_iter47.cubin",
        "cubin_sha256": "43a26889fce5a3992c05d57e1e2c24b9369f7494bac64dd69cdda47464e24c14",
        "queried_dynamic_shared_bytes": 87040,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 128,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 87040
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      },
      {
        "variant": 51,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm74_GLOBAL__N__3c75913d_41_megakernel_wgmma_psplit_tma_inplace_ss_cu_62d0210a36winograd_wgmma_psplit_tma_inplace_ssIN7cutlass10bfloat16_tELi32EEEvNS_7ProblemENS0_23TmaInplaceSSDescriptorsEPKfPf",
        "cubin_path": "results/psplit_tma_inplace_ss_final_megakernel_wgmma_psplit_tma_inplace_ss_iter47.cubin",
        "cubin_sha256": "43a26889fce5a3992c05d57e1e2c24b9369f7494bac64dd69cdda47464e24c14",
        "queried_dynamic_shared_bytes": 87040,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 128,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 87040
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      }
    ]
  }
}
```

唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。


## 原始单P融合与S/T一次构造的历史实测

用户要求的两条原始Hopper路线已在iteration46用同binary完整补充8行测量：H10单P输入融合为两launch，H12 S/T只构造一次为三launch。本轮不重复这8行，也不把native45或Lt1当作H10/H12。原件与原FP64 FAIL保持。
[small_psplit_tma_bn16_k32_window_input_placement_supplement_measurements_iter46.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_k32_window_input_placement_supplement_measurements_iter46.json)
[1024_MEGAKERNEL_PSPLIT_TMA_BN16_K32_WINDOW_REPORT.md](/share/home/liuyao/workspace/MegaFMM/results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_K32_WINDOW_REPORT.md)
