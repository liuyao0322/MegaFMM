# 1024 SW128 rawA SIMT对照实测（iteration42）

Mega46对同binary Mega45自身完整调用：launch bf16 23.485119µs对17.651200µs，延长33.051%；launch fp16 23.366721µs对17.754880µs，延长31.607%；graph bf16 22.959361µs对16.921600µs，延长35.681%；graph fp16 22.853439µs对17.347200µs，延长31.741%。

Mega45对同binary Mega39自身完整调用：launch bf16 17.651200µs对23.233919µs，减少24.028%；launch fp16 17.754880µs对23.033600µs，减少22.917%；graph bf16 16.921600µs对22.877440µs，减少26.034%；graph fp16 17.347200µs对22.858241µs，减少24.110%。

Mega46对同binary Mega39自身完整调用：launch bf16 23.485119µs对23.233919µs，延长1.081%；launch fp16 23.366721µs对23.033600µs，延长1.446%；graph bf16 22.959361µs对22.877440µs，延长0.358%；graph fp16 22.853439µs对22.858241µs，减少0.021%。

本轮launch bf16已测native中最快为Mega45，17.651200µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.754880µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，16.921600µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.347200µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit d372045901ed1ef97ed4a933109d409835ed1d83；[small_psplit_sw128_simt_control_profile_iter42.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_sw128_simt_control_profile_iter42.json)，[small_psplit_sw128_simt_control_measurements_iter42.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_sw128_simt_control_measurements_iter42.json)。

## 测量协议

ONLY46是新SW128 rawA vector8 SIMT对照；45为同binary选择性四路TMA rawA，39为原RS-A BK64，36为SS BK32，Lt1为完整多launch对照。47未注册、未运行。fresh compiled blocked46必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替；45对39四个同binary配对仍可实际测得。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

46的源码入口在普通启动时仍每次从当前A指针/lda编码一个A tensor map，编码与单个kernel均包含在完整调用计时内，用作host cost对照；该map明确未被46的SIMT kernel消费。46保留SW128 rawA布局、共享内存逻辑安排及1024对齐，A和B均用SIMT搬运，没有rawA TMA issue/rawTX等待，raw_done存储和初始化保留但unused。45编码并实际用map完成四路rawA TMA，B仍SIMT；39/36无host descriptor encode。全部native完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，为一个kernel、零矩阵workspace；Lt1的pack/GEMM/combine完整多launch均计入。Graph45/46在capture时编码，稳态replay使用固定capture地址读取当前内容，46在capture和replay都不消费map，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时，或把46对45差异认证为纯搬运因果。

46虽不消费tensor map，仍保留lda<2^39条件以控制同样的host encode适用域。巨大stride记录只按本轮实际host metadata检查与Plan setup结果表述，不能被称为已分配巨大GPU输入并执行了fallback；若setup不可用则保留原message。此处不认证巨大stride的GPU运行。

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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration42 resource/residency query."
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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration42 resource/residency query."
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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration42 resource/residency query."
  },
  "46": {
    "actual_name": "megakernel_wgmma_psplit_sw128_simt_control_64x32",
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
    "raw_A_transaction_bytes_per_stage": 0,
    "raw_A_TMA_boxes_per_stage": 0,
    "A_native_lda_upper_exclusive": 549755813888,
    "B_native_stride_extra_bound": null,
    "A_shared_absolute_alignment_bytes": 1024,
    "occupancy_API_predicted_active_CTAs_per_SM": 1,
    "actual_measured_CTA_residency": null,
    "function_probe_path": "results/psplit_sw128_simt_control_function_occupancy_probe_iter42.txt",
    "resource_metadata_scope": "Fresh independent CAN46 and current DriverAPI actual probe; no measured residency.",
    "raw_A_SIMT_vector8_tasks_per_stage": 512,
    "map_encode_per_ordinary_call": 1,
    "encoded_map_consumed": false,
    "raw_done_scope": "Storage/init retained unused; no rawTX arm/issue/wait"
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega46 megakernel_wgmma_psplit_sw128_simt_control_64x32 | bf16 | 12.370560 / 12.337472 / 12.736640 | 23.485119 / 23.478079 / 23.497729 | 0.526740 | 1.7927/0.0385 | 12.45051/23.57428 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_sw128_simt_control46_launch_iter42_full1024.csv:2) |
| Mega46 megakernel_wgmma_psplit_sw128_simt_control_64x32 | fp16 | 9.903360 / 9.884288 / 9.924928 | 23.366721 / 23.355585 / 23.380288 | 0.423823 | 0.1867/0.0505 | 9.98182/23.45114 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_sw128_simt_control46_launch_iter42_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.287360 / 12.277376 / 12.309952 | 17.651200 / 17.645760 / 17.669183 | 0.696120 | 0.1398/0.0601 | 12.36204/17.73544 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter42_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.909760 / 9.883648 / 9.919232 | 17.754880 / 17.742592 / 17.766144 | 0.558143 | 0.1623/0.0610 | 9.98037/17.82847 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter42_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.272640 / 12.267520 / 12.283136 | 23.233919 / 23.218368 / 23.251968 | 0.528221 | 0.0583/0.0640 | 12.35010/23.32220 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter42_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.842240 / 9.824576 / 9.844928 | 23.033600 / 23.005312 / 23.042304 | 0.427299 | 0.1017/0.0736 | 9.92179/23.11160 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter42_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 12.260799 / 12.253568 / 12.264704 | 23.471999 / 23.448768 / 23.500735 | 0.522359 | 0.0404/0.1054 | 12.33726/23.56147 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter42_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.917760 / 9.911488 / 9.932544 | 23.546560 / 23.531136 / 23.555264 | 0.421198 | 0.1090/0.0526 | 9.99728/23.62402 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter42_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.312320 / 12.296576 / 12.328192 | 19.400640 / 19.381888 / 19.412800 | 0.634635 | 0.1184/0.0703 | 12.38918/19.47812 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter42_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.945600 / 9.937920 / 9.960704 | 19.498560 / 19.482048 / 19.522753 | 0.510068 | 0.1069/0.1062 | 10.02206/19.57492 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter42_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega46 megakernel_wgmma_psplit_sw128_simt_control_64x32 | bf16 | 11.331199 / 11.322304 / 11.349888 | 22.959361 / 22.957121 / 22.969024 | 0.493533 | 0.1084/0.0251 | 11.40456/23.03862 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_sw128_simt_control46_graph_iter42_full1024.csv:2) |
| Mega46 megakernel_wgmma_psplit_sw128_simt_control_64x32 | fp16 | 9.361280 / 9.355136 / 9.377472 | 22.853439 / 22.846400 / 22.898048 | 0.409622 | 0.1187/0.1302 | 9.43328/22.92970 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_sw128_simt_control46_graph_iter42_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.318400 / 11.306688 / 11.329984 | 16.921600 / 16.909120 / 16.924928 | 0.668873 | 0.0908/0.0425 | 11.40942/17.00542 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter42_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.604480 / 9.595456 / 9.607232 | 17.347200 / 17.343808 / 17.349760 | 0.553662 | 0.0590/0.0172 | 9.69390/17.44284 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter42_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.351360 / 11.348416 / 11.358336 | 22.877440 / 22.872320 / 22.883712 | 0.496181 | 0.0394/0.0219 | 11.43125/22.96048 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter42_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.404480 / 9.400384 / 9.415808 | 22.858241 / 22.854784 / 22.868096 | 0.411426 | 0.0735/0.0293 | 9.48622/22.94048 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter42_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 11.224639 / 11.223488 / 11.227328 | 23.038399 / 23.027009 / 23.049152 | 0.487214 | 0.0157/0.0477 | 11.30795/23.12029 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter42_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.465600 / 9.421312 / 9.499712 | 23.144960 / 23.132096 / 23.170433 | 0.408970 | 0.3642/0.0757 | 9.54636/23.22169 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter42_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.348799 / 11.336384 / 11.379712 | 17.021120 / 17.019648 / 17.022464 | 0.666748 | 0.1718/0.0084 | 11.42307/17.09618 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter42_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.594560 / 9.588224 / 9.600960 | 17.353600 / 17.333888 / 17.361216 | 0.552886 | 0.0586/0.0860 | 9.67330/17.42801 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter42_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega45 | Mega46 | 17.651200 | 23.485119 | 0.751591× | +33.051% | 17.73544/23.57428 |
| launch | bf16 | Mega39 | Mega45 | 23.233919 | 17.651200 | 1.316280× | -24.028% | 23.32220/17.73544 |
| launch | bf16 | Mega39 | Mega46 | 23.233919 | 23.485119 | 0.989304× | +1.081% | 23.32220/23.57428 |
| launch | fp16 | Mega45 | Mega46 | 17.754880 | 23.366721 | 0.759836× | +31.607% | 17.82847/23.45114 |
| launch | fp16 | Mega39 | Mega45 | 23.033600 | 17.754880 | 1.297311× | -22.917% | 23.11160/17.82847 |
| launch | fp16 | Mega39 | Mega46 | 23.033600 | 23.366721 | 0.985744× | +1.446% | 23.11160/23.45114 |
| graph | bf16 | Mega45 | Mega46 | 16.921600 | 22.959361 | 0.737024× | +35.681% | 17.00542/23.03862 |
| graph | bf16 | Mega39 | Mega45 | 22.877440 | 16.921600 | 1.351967× | -26.034% | 22.96048/17.00542 |
| graph | bf16 | Mega39 | Mega46 | 22.877440 | 22.959361 | 0.996432× | +0.358% | 22.96048/23.03862 |
| graph | fp16 | Mega45 | Mega46 | 17.347200 | 22.853439 | 0.759063× | +31.741% | 17.44284/22.92970 |
| graph | fp16 | Mega39 | Mega45 | 22.858241 | 17.347200 | 1.317691× | -24.110% | 22.94048/17.44284 |
| graph | fp16 | Mega39 | Mega46 | 22.858241 | 22.853439 | 1.000210× | -0.021% | 22.94048/22.92970 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新46实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 46 | fp16 | 64×32 /64 | 384 | 123904 | 168 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |
| 46 | bf16 | 64×32 /64 | 384 | 123904 | 168 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

Mega46 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。
Mega46 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。


Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=2802/2560/4876；[correctness_psplit_sw128_simt_control_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_sw128_simt_control_iter42.txt)。
selected46：actualexit=0，CASE/PASS/BEGIN=56/136/196；[correctness_psplit_sw128_simt_control46_targeted_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_sw128_simt_control46_targeted_iter42.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_sw128_simt_control37_cli_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_sw128_simt_control37_cli_iter42.txt)。

新46 rawU groups=84，reference=39（sameatom_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=36/108。

新46实际完成10项主机端A-only stride边界检查（1组PASS），其中native helper guard有4项。Plan设置成功0项、不可用10项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
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
      "ldb": 64,
      "native_guard": true,
      "expected_route": "megakernel_wgmma_psplit_sw128_simt_control_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725776,
      "safe_b_span_bytes": 16384,
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
      "safe_a_span_bytes": 139637976727808,
      "safe_b_span_bytes": 16384,
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
      "safe_a_span_bytes": 139637976729840,
      "safe_b_span_bytes": 16384,
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
      "safe_a_span_bytes": 139637976728062,
      "safe_b_span_bytes": 16384,
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
      "expected_route": "megakernel_wgmma_psplit_sw128_simt_control_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 32768,
      "safe_b_span_bytes": 139637976727680,
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
      "expected_route": "megakernel_wgmma_psplit_sw128_simt_control_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 139637976725776,
      "safe_b_span_bytes": 16384,
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
      "safe_a_span_bytes": 139637976727808,
      "safe_b_span_bytes": 16384,
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
      "safe_a_span_bytes": 139637976729840,
      "safe_b_span_bytes": 16384,
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
      "safe_a_span_bytes": 139637976728062,
      "safe_b_span_bytes": 16384,
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
      "expected_route": "megakernel_wgmma_psplit_sw128_simt_control_64x32",
      "actual_route": "setup_unavailable",
      "setup_supported": false,
      "safe_a_span_bytes": 32768,
      "safe_b_span_bytes": 139637976727680,
      "setup_unavailable_message": "query Lt heuristics: cuBLAS status 15",
      "metadata_only": true,
      "allocated_input_bytes": 0,
      "tensor_map_encodes": 0,
      "native_runs": 0
    }
  ]
}
```

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1830、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_sw128_simt_control_enabled_only",
    "path": "results/strict_psplit_sw128_simt_control_iter42.txt",
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

SAN46 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_sw128_simt_control46_memcheck_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_sw128_simt_control46_memcheck_iter42.txt)。非mem泄漏未测null。

SAN46 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_sw128_simt_control46_racecheck_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_sw128_simt_control46_racecheck_iter42.txt)。非mem泄漏未测null。

SAN46 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_sw128_simt_control46_synccheck_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_sw128_simt_control46_synccheck_iter42.txt)。非mem泄漏未测null。

CPUdag actualexit=0，tests=3；[psplit_sw128_simt_control_cpu_dag_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_sw128_simt_control_cpu_dag_iter42.txt)。

CPUstage actualexit=0，tests=6；[psplit_sw128_simt_control_cpu_stage_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_sw128_simt_control_cpu_stage_iter42.txt)。

CPU模型actualexit=0，originalstatus=PASS_CPU_ONLY；[psplit_sw128_simt_control_cpu_state_model_iter42.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_sw128_simt_control_cpu_state_model_iter42.txt)。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。

环境：

```json
{
  "path": "results/small_psplit_sw128_simt_control_environment_iter42.txt",
  "exit_path": "results/small_psplit_sw128_simt_control_environment_iter42.exit",
  "sha256": "1fbfda14481d725a309dcae84954db436f58481c24d1d01255e6116d242a5665",
  "exit": 0,
  "allocation": 107488,
  "host": "g08",
  "GPU": "NVIDIA H100 80GB HBM3",
  "gpu_uuid": "GPU-78def847-b6c1-9dd4-ff30-269a9ae1a73f",
  "SMs": 132,
  "SMs_evidence": "actual archived DriverAPI cuDeviceGetAttribute",
  "application_cuda": "12.4.99",
  "driver": "575.51.03",
  "CUTLASS": "3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee",
  "sanitizer_path": "/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer",
  "sanitizer_version": "2025.1.0.0 build35583870",
  "clocks_locked": false,
  "snapshot_scope": "Actual readonly postvalidation snapshot; no kernel launch or within-group clock stability/residency proof.",
  "function_attributes_and_occupancy_API": {
    "path": "results/psplit_sw128_simt_control_function_occupancy_probe_iter42.txt",
    "sha256": "378c4e84ae3fbdc793ad432eb17f14fce3866f9f4975ee27935ec9b8fae34908",
    "exit": 0,
    "records": [
      {
        "variant": 46,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm78_GLOBAL__N__8e6b49be_45_megakernel_wgmma_psplit_sw128_simt_control_cu_7c2d4f1440winograd_wgmma_psplit_sw128_simt_controlIN7cutlass6half_tELi32EEEvNS_7ProblemENS0_26SW128SimtControlDescriptorEPKT_S8_PKfPf",
        "cubin_path": "results/psplit_sw128_simt_control_final_megakernel_wgmma_psplit_sw128_simt_control_iter42.cubin",
        "cubin_sha256": "dec13dafd0952cc9571f606e1108d79bcd20c15e44efe848851215d77a051d66",
        "queried_dynamic_shared_bytes": 123904,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 168,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 123904
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      },
      {
        "variant": 46,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm78_GLOBAL__N__8e6b49be_45_megakernel_wgmma_psplit_sw128_simt_control_cu_7c2d4f1440winograd_wgmma_psplit_sw128_simt_controlIN7cutlass10bfloat16_tELi32EEEvNS_7ProblemENS0_26SW128SimtControlDescriptorEPKT_S8_PKfPf",
        "cubin_path": "results/psplit_sw128_simt_control_final_megakernel_wgmma_psplit_sw128_simt_control_iter42.cubin",
        "cubin_sha256": "dec13dafd0952cc9571f606e1108d79bcd20c15e44efe848851215d77a051d66",
        "queried_dynamic_shared_bytes": 123904,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 168,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 123904
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      }
    ]
  }
}
```

唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。
