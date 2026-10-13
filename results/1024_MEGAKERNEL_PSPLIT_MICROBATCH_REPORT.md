# 1024 RS-A逐P A microbatch实测（iteration40）

新44对同binary Mega39自身完整调用：launch bf16 23.607681µs对23.114560µs，延长2.133%；launch fp16 23.268161µs对23.110399µs，延长0.683%；graph bf16 23.354881µs对22.922239µs，延长1.887%；graph fp16 23.147521µs对22.904639µs，延长1.060%。

新44对同binary Mega40自身完整调用：launch bf16 23.607681µs对25.518401µs，减少7.488%；launch fp16 23.268161µs对24.247041µs，减少4.037%；graph bf16 23.354881µs对25.075519µs，减少6.862%；graph fp16 23.147521µs对24.086080µs，减少3.897%。

新44对同binary Mega36自身完整调用：launch bf16 23.607681µs对23.401921µs，延长0.879%；launch fp16 23.268161µs对23.594241µs，减少1.382%；graph bf16 23.354881µs对23.059199µs，延长1.282%；graph fp16 23.147521µs对23.132160µs，延长0.066%。

本轮launch bf16已测native中最快为Mega39，23.114560µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega39，23.110399µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega39，22.922239µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega39，22.904639µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit b7fc1ebb725affeb2fd1044b98e097ae1d0cbcd9；[small_psplit_microbatch_profile_iter40.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch_profile_iter40.json)，[small_psplit_microbatch_measurements_iter40.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch_measurements_iter40.json)。

## 测量协议

ONLY44是逐P构造A的microbatch，39为RS-A BK64参照，40为RS-A BK32，36为SS BK32，Lt1为完整多launch对照。45未注册、未运行。fresh compiled blocked44必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

实测native读取当前原始A/B，完成变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue；一个kernel、零native矩阵workspace、无host descriptor encode。Lt1的pack/GEMM/combine完整多launch均计入。Graph固定capture地址读取当前内容，fresh检查更换内容。

batchwall是每组100调用含批末同步的host wall除以调用数，再取组median，不是独立同步单调用CPU wall。setup/tuning/capture不计稳态；event可能含提交空隙，环境快照不认证组内clock。所有组、原FAIL/nonfinite/退出码均保留。

冻结参数：

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
    "wg1_gpr_highest_including_implicit": 186,
    "resource_metadata_scope": "retained legacy compiled metadata; not a new function-attribute query this iteration"
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
    "wg1_gpr_highest_including_implicit": 141,
    "resource_metadata_scope": "retained legacy compiled metadata; not a new function-attribute query this iteration"
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
    "resource_metadata_scope": "retained legacy compiled metadata; not a new function-attribute query this iteration"
  },
  "44": {
    "actual_name": "megakernel_wgmma_psplit_rsa_microbatch_64x32",
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
    "dynamic_shared_bytes": 123008,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "device_reserved_shared_bytes_per_CTA": 1024,
    "initial_registers_per_thread": 168,
    "initial_owned_CTA_pool_registers": 64512,
    "requested_role_pool_registers": 57344,
    "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained",
    "occupancy_API_predicted_active_CTAs_per_SM": 1,
    "actual_measured_CTA_residency": null,
    "function_probe_path": "results/psplit_microbatch_function_occupancy_probe_iter40.txt"
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | bf16 | 12.236800 / 12.221824 / 12.267392 | 23.607681 / 23.600896 / 23.621248 | 0.518340 | 0.1669/0.0384 | 12.31263/23.68687 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_launch_iter40_full1024.csv:2) |
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | fp16 | 9.878080 / 9.871808 / 9.886272 | 23.268161 / 23.246913 / 23.277312 | 0.424532 | 0.0648/0.0624 | 9.95499/23.34604 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_launch_iter40_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.231680 / 12.211391 / 12.257408 | 23.114560 / 23.089408 / 23.117504 | 0.529176 | 0.1660/0.0649 | 12.31346/23.19279 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter40_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.977280 / 9.970624 / 9.982208 | 23.110399 / 23.102399 / 23.119552 | 0.431723 | 0.0511/0.0355 | 10.05476/23.19176 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter40_full1024.csv:3) |
| Mega40 megakernel_wgmma_psplit_rsa_bk32_64x32 | bf16 | 12.294720 / 12.286592 / 12.298304 | 25.518401 / 25.482495 / 25.521280 | 0.481798 | 0.0446/0.0734 | 12.36957/25.59046 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter40_full1024.csv:2) |
| Mega40 megakernel_wgmma_psplit_rsa_bk32_64x32 | fp16 | 9.878720 / 9.857280 / 9.886592 | 24.247041 / 24.230848 / 24.260415 | 0.407420 | 0.1361/0.0544 | 9.95133/24.31879 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter40_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 12.171839 / 12.158016 / 12.188544 | 23.401921 / 23.345664 / 23.408064 | 0.520121 | 0.1119/0.1254 | 12.24364/23.47617 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter40_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.932480 / 9.924864 / 9.950976 | 23.594241 / 23.574976 / 23.644544 | 0.420971 | 0.1200/0.1565 | 10.00938/23.66525 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter40_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.241920 / 12.233408 / 12.249600 | 19.510400 / 19.504192 / 19.524224 | 0.627456 | 0.0671/0.0492 | 12.31497/19.58417 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter40_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.905920 / 9.878720 / 9.921536 | 19.483200 / 19.479616 / 19.498304 | 0.508434 | 0.1912/0.0465 | 9.97880/19.56266 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter40_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | bf16 | 11.196800 / 11.189632 / 11.200192 | 23.354881 / 23.350080 / 23.365505 | 0.479420 | 0.0422/0.0314 | 11.27066/23.43059 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_graph_iter40_full1024.csv:2) |
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | fp16 | 9.575360 / 9.571776 / 9.581440 | 23.147521 / 23.129279 / 23.158208 | 0.413667 | 0.0452/0.0589 | 9.64742/23.21846 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_graph_iter40_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.563520 / 11.548608 / 11.567168 | 22.922239 / 22.921343 / 22.928192 | 0.504467 | 0.0734/0.0147 | 11.63665/23.00233 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter40_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.479040 / 9.471616 / 9.502528 | 22.904639 / 22.887297 / 22.931200 | 0.413848 | 0.1543/0.0907 | 9.55378/22.98202 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter40_full1024.csv:3) |
| Mega40 megakernel_wgmma_psplit_rsa_bk32_64x32 | bf16 | 11.300480 / 11.294720 / 11.310592 | 25.075519 / 25.065216 / 25.083200 | 0.450658 | 0.0682/0.0337 | 11.37611/25.15731 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter40_full1024.csv:2) |
| Mega40 megakernel_wgmma_psplit_rsa_bk32_64x32 | fp16 | 9.435200 / 9.420992 / 9.442304 | 24.086080 / 24.070464 / 24.094976 | 0.391728 | 0.1032/0.0512 | 9.51182/24.16203 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter40_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 11.356800 / 11.349184 / 11.364352 | 23.059199 / 23.039297 / 23.063488 | 0.492506 | 0.0641/0.0490 | 11.43202/23.13073 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter40_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.432000 / 9.408320 / 9.471360 | 23.132160 / 23.125185 / 23.163455 | 0.407744 | 0.3270/0.0791 | 9.51561/23.20770 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter40_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.218240 / 11.202944 / 11.226048 | 16.904960 / 16.898176 / 16.919936 | 0.663606 | 0.0997/0.0677 | 11.29210/16.98093 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter40_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.586240 / 9.579328 / 9.597312 | 17.365119 / 17.351040 / 17.374528 | 0.552040 | 0.0839/0.0603 | 9.65946/17.44132 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter40_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega39 | Mega44 | 23.114560 | 23.607681 | 0.979112× | +2.133% | 23.19279/23.68687 |
| launch | bf16 | Mega40 | Mega44 | 25.518401 | 23.607681 | 1.080936× | -7.488% | 25.59046/23.68687 |
| launch | bf16 | Mega36 | Mega44 | 23.401921 | 23.607681 | 0.991284× | +0.879% | 23.47617/23.68687 |
| launch | fp16 | Mega39 | Mega44 | 23.110399 | 23.268161 | 0.993220× | +0.683% | 23.19176/23.34604 |
| launch | fp16 | Mega40 | Mega44 | 24.247041 | 23.268161 | 1.042070× | -4.037% | 24.31879/23.34604 |
| launch | fp16 | Mega36 | Mega44 | 23.594241 | 23.268161 | 1.014014× | -1.382% | 23.66525/23.34604 |
| graph | bf16 | Mega39 | Mega44 | 22.922239 | 23.354881 | 0.981475× | +1.887% | 23.00233/23.43059 |
| graph | bf16 | Mega40 | Mega44 | 25.075519 | 23.354881 | 1.073674× | -6.862% | 25.15731/23.43059 |
| graph | bf16 | Mega36 | Mega44 | 23.059199 | 23.354881 | 0.987340× | +1.282% | 23.13073/23.43059 |
| graph | fp16 | Mega39 | Mega44 | 22.904639 | 23.147521 | 0.989507× | +1.060% | 22.98202/23.21846 |
| graph | fp16 | Mega40 | Mega44 | 24.086080 | 23.147521 | 1.040547× | -3.897% | 24.16203/23.21846 |
| graph | fp16 | Mega36 | Mega44 | 23.132160 | 23.147521 | 0.999336× | +0.066% | 23.20770/23.21846 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新44实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 44 | fp16 | 64×32 /64 | 384 | 123008 | 168 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

Mega44 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。
| 44 | bf16 | 64×32 /64 | 384 | 123008 | 168 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

Mega44 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。

Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=2690/2292/4484；[correctness_psplit_microbatch_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_microbatch_iter40.txt)。
selected44：actualexit=0，CASE/PASS/BEGIN=56/135/186；[correctness_psplit_microbatch44_targeted_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_microbatch44_targeted_iter40.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_microbatch37_cli_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_microbatch37_cli_iter40.txt)。

新44 rawU groups=84，reference=39（sameatom_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=36/108。

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1770、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_microbatch_enabled_only",
    "path": "results/strict_psplit_microbatch_iter40.txt",
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

SAN44 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_microbatch44_memcheck_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_microbatch44_memcheck_iter40.txt)。非mem泄漏未测null。

SAN44 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_microbatch44_racecheck_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_microbatch44_racecheck_iter40.txt)。非mem泄漏未测null。

SAN44 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_microbatch44_synccheck_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_microbatch44_synccheck_iter40.txt)。非mem泄漏未测null。

CPUdag actualexit=0，tests=3；[psplit_microbatch_cpu_dag_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_microbatch_cpu_dag_iter40.txt)。

CPUstage actualexit=0，tests=6；[psplit_microbatch_cpu_stage_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_microbatch_cpu_stage_iter40.txt)。

CPU模型actualexit=0，originalstatus=PASS_CPU_BOUNDED_RN_DAG_AND_MICROBATCH_REUSE_ONLY；[psplit_microbatch_cpu_state_model_iter40.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_microbatch_cpu_state_model_iter40.txt)。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。

环境：

```json
{
  "path": "results/small_psplit_microbatch_environment_iter40.txt",
  "exit_path": "results/small_psplit_microbatch_environment_iter40.exit",
  "sha256": "0e50b6f36c1234c28b849996d8011515366a692a541aec88462bd5c3288900ae",
  "exit": 0,
  "allocation": 107488,
  "host": "g08",
  "GPU": "NVIDIA H100 80GB HBM3",
  "gpu_uuid": "GPU-78def847-b6c1-9dd4-ff30-269a9ae1a73f",
  "SMs": 132,
  "SMs_evidence": "actual archived driver API cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
  "application_cuda": "12.4.99",
  "driver": "575.51.03",
  "CUTLASS": "3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee",
  "sanitizer_path": "/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer",
  "sanitizer_version": "2025.1.0.0 build35583870",
  "clocks_locked": false,
  "snapshot_scope": "Actual read-only snapshot after all iteration40 performance and three serial SAN runs. No device kernel launched; no within-group clock stability or measured residency proof.",
  "function_attributes_and_occupancy_API": {
    "path": "results/psplit_microbatch_function_occupancy_probe_iter40.txt",
    "sha256": "e096d74878a6c285c23ca94e3e47247833e99c83d0127741661d2382eeaec4c5",
    "exit": 0,
    "records": [
      {
        "variant": 44,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm74_GLOBAL__N__9d02337e_41_megakernel_wgmma_psplit_rsa_microbatch_cu_743570c836winograd_wgmma_psplit_rsa_microbatchIN7cutlass6half_tELi32EEEvNS_7ProblemEPKT_S7_PKfPf",
        "cubin_path": "results/psplit_microbatch_final_megakernel_wgmma_psplit_rsa_microbatch_iter40.cubin",
        "cubin_sha256": "5c47e365fbf261826958f1819f2f4fd77dda1e3ea4933d38035ae8055c61912f",
        "queried_dynamic_shared_bytes": 123008,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 168,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 123008
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      },
      {
        "variant": 44,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm74_GLOBAL__N__9d02337e_41_megakernel_wgmma_psplit_rsa_microbatch_cu_743570c836winograd_wgmma_psplit_rsa_microbatchIN7cutlass10bfloat16_tELi32EEEvNS_7ProblemEPKT_S7_PKfPf",
        "cubin_path": "results/psplit_microbatch_final_megakernel_wgmma_psplit_rsa_microbatch_iter40.cubin",
        "cubin_sha256": "5c47e365fbf261826958f1819f2f4fd77dda1e3ea4933d38035ae8055c61912f",
        "queried_dynamic_shared_bytes": 123008,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 168,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 123008
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      }
    ],
    "scope": "DriverAPI actual static0, resource-tool SHARED1024 and occupancyAPI predicted1 are distinct; measured residency null."
  }
}
```

唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。
