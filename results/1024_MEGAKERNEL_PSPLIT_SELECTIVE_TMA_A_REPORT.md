# 1024 RS-A选择性四路TMA rawA实测（iteration41）

新45对同binary Mega39自身完整调用：launch bf16 17.632639µs对23.107519µs，减少23.693%；launch fp16 17.750080µs对23.118720µs，减少23.222%；graph bf16 16.913280µs对22.933440µs，减少26.251%；graph fp16 17.351360µs对22.911360µs，减少24.267%。

新45对同binary Mega44自身完整调用：launch bf16 17.632639µs对23.667519µs，减少25.499%；launch fp16 17.750080µs对23.280640µs，减少23.756%；graph bf16 16.913280µs对23.356481µs，减少27.586%；graph fp16 17.351360µs对23.090241µs，减少24.854%。

新45对同binary Mega36自身完整调用：launch bf16 17.632639µs对23.399680µs，减少24.646%；launch fp16 17.750080µs对23.582721µs，减少24.733%；graph bf16 16.913280µs对23.083839µs，减少26.731%；graph fp16 17.351360µs对23.204479µs，减少25.224%。

本轮launch bf16已测native中最快为Mega45，17.632639µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.750080µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，16.913280µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.351360µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit 61b1c1f5255326894e96e63c33254e481c89ce84；[small_psplit_selective_tma_a_profile_iter41.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a_profile_iter41.json)，[small_psplit_selective_tma_a_measurements_iter41.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a_measurements_iter41.json)。

## 测量协议

ONLY45是选择性四路TMA rawA、B仍SIMT，39为RS-A BK64参照，44为逐P A microbatch BK64，36为SS BK32，Lt1为完整多launch对照。46未注册、未运行。fresh compiled blocked45必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

新45普通调用每次从当前A指针/lda构造一个A tensor map，编码与单个kernel均包含在完整调用计时内；kernel用四路TMA搬运rawA，B保持SIMT，完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue。39/44/36对照无host descriptor encode。全部native为一个kernel、零矩阵workspace；Lt1的pack/GEMM/combine完整多launch均计入。Graph在capture时完成描述符编码，稳态replay使用固定capture地址读取当前内容，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时。

新增lda<2^39的TMA native predicate是源码/host metadata适用域检查；巨大stride边界不能被表述为已分配巨大GPU输入并执行了fallback。此处不认证巨大stride的GPU运行。

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
    "function_probe_path": "results/psplit_microbatch_function_occupancy_probe_iter40.txt",
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
    "function_probe_path": "results/psplit_selective_tma_a_function_occupancy_probe_iter41.txt"
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.136320 / 12.119808 / 12.153600 | 17.632639 / 17.621632 / 17.658496 | 0.688287 | 0.1228/0.0933 | 12.21245/17.71261 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter41_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.880000 / 9.873088 / 9.904896 | 17.750080 / 17.742720 / 17.763008 | 0.556617 | 0.1470/0.0564 | 9.96583/17.83149 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter41_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.236480 / 12.231168 / 12.251264 | 23.107519 / 23.103425 / 23.109312 | 0.529545 | 0.0763/0.0115 | 12.31287/23.18624 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter41_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.944320 / 9.936320 / 10.020992 | 23.118720 / 23.105087 / 23.131329 | 0.430141 | 0.5037/0.0528 | 10.02457/23.20469 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter41_full1024.csv:3) |
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | bf16 | 12.249280 / 12.205376 / 12.261056 | 23.667519 / 23.595008 / 23.688704 | 0.517557 | 0.2093/0.1802 | 12.32744/23.75516 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_launch_iter41_full1024.csv:2) |
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | fp16 | 9.928960 / 9.910464 / 9.983680 | 23.280640 / 23.236864 / 23.320896 | 0.426490 | 0.4087/0.1669 | 10.00692/23.37506 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_launch_iter41_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 12.177600 / 12.158656 / 12.191936 | 23.399680 / 23.372800 / 23.415296 | 0.520417 | 0.1309/0.0949 | 12.25041/23.48192 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter41_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.952960 / 9.937152 / 9.957568 | 23.582721 / 23.566080 / 23.631807 | 0.422045 | 0.0928/0.1270 | 10.02725/23.66110 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter41_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.271360 / 12.255680 / 12.285312 | 19.352320 / 19.332032 / 19.364033 | 0.634103 | 0.1079/0.0855 | 12.34863/19.43040 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter41_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.862720 / 9.856768 / 9.894656 | 19.751680 / 19.739904 / 19.769984 | 0.499336 | 0.2170/0.0673 | 9.94373/19.83327 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter41_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.299200 / 11.276800 / 11.306688 | 16.913280 / 16.893952 / 16.920768 | 0.668067 | 0.1191/0.0725 | 11.37990/16.99304 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter41_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.600000 / 9.594240 / 9.613888 | 17.351360 / 17.346048 / 17.359104 | 0.553271 | 0.1065/0.0339 | 9.67950/17.43271 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter41_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.338880 / 11.329216 / 11.349632 | 22.933440 / 22.922816 / 22.942721 | 0.494426 | 0.0793/0.0404 | 11.42783/23.01986 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter41_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.436160 / 9.421632 / 9.439040 | 22.911360 / 22.908160 / 22.926143 | 0.411855 | 0.0851/0.0366 | 9.51457/22.99870 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter41_full1024.csv:3) |
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | bf16 | 11.256640 / 11.245184 / 11.263232 | 23.356481 / 23.344064 / 23.363968 | 0.481949 | 0.0749/0.0396 | 11.33347/23.44341 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_graph_iter41_full1024.csv:2) |
| Mega44 megakernel_wgmma_psplit_rsa_microbatch_64x32 | fp16 | 9.555840 / 9.551232 / 9.561664 | 23.090241 / 23.088705 / 23.108864 | 0.413848 | 0.0499/0.0410 | 9.63431/23.17439 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_microbatch44_graph_iter41_full1024.csv:3) |
| Mega36 megakernel_wgmma_psplit_64x32 | bf16 | 11.303680 / 11.298240 / 11.310016 | 23.083839 / 23.054657 / 23.090751 | 0.489679 | 0.0485/0.0739 | 11.37498/23.15854 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter41_full1024.csv:2) |
| Mega36 megakernel_wgmma_psplit_64x32 | fp16 | 9.583680 / 9.574016 / 9.611392 | 23.204479 / 23.175295 / 23.228735 | 0.413010 | 0.2146/0.1104 | 9.66215/23.28329 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter41_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.315520 / 11.306368 / 11.335232 | 16.990080 / 16.988992 / 16.995520 | 0.666007 | 0.1145/0.0187 | 11.38669/17.06487 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter41_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.448000 / 9.439872 / 9.461312 | 17.559040 / 17.546048 / 17.562176 | 0.538070 | 0.1034/0.0465 | 9.52087/17.63141 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter41_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega39 | Mega45 | 23.107519 | 17.632639 | 1.310497× | -23.693% | 23.18624/17.71261 |
| launch | bf16 | Mega44 | Mega45 | 23.667519 | 17.632639 | 1.342256× | -25.499% | 23.75516/17.71261 |
| launch | bf16 | Mega36 | Mega45 | 23.399680 | 17.632639 | 1.327066× | -24.646% | 23.48192/17.71261 |
| launch | fp16 | Mega39 | Mega45 | 23.118720 | 17.750080 | 1.302457× | -23.222% | 23.20469/17.83149 |
| launch | fp16 | Mega44 | Mega45 | 23.280640 | 17.750080 | 1.311579× | -23.756% | 23.37506/17.83149 |
| launch | fp16 | Mega36 | Mega45 | 23.582721 | 17.750080 | 1.328598× | -24.733% | 23.66110/17.83149 |
| graph | bf16 | Mega39 | Mega45 | 22.933440 | 16.913280 | 1.355943× | -26.251% | 23.01986/16.99304 |
| graph | bf16 | Mega44 | Mega45 | 23.356481 | 16.913280 | 1.380955× | -27.586% | 23.44341/16.99304 |
| graph | bf16 | Mega36 | Mega45 | 23.083839 | 16.913280 | 1.364835× | -26.731% | 23.15854/16.99304 |
| graph | fp16 | Mega39 | Mega45 | 22.911360 | 17.351360 | 1.320436× | -24.267% | 22.99870/17.43271 |
| graph | fp16 | Mega44 | Mega45 | 23.090241 | 17.351360 | 1.330745× | -24.854% | 23.17439/17.43271 |
| graph | fp16 | Mega36 | Mega45 | 23.204479 | 17.351360 | 1.337329× | -25.224% | 23.28329/17.43271 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新45实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 45 | fp16 | 64×32 /64 | 384 | 123904 | 168 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |
| 45 | bf16 | 64×32 /64 | 384 | 123904 | 168 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

Mega45 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。
Mega45 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。


Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=2746/2426/4680；[correctness_psplit_selective_tma_a_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_selective_tma_a_iter41.txt)。
selected45：actualexit=0，CASE/PASS/BEGIN=56/136/196；[correctness_psplit_selective_tma_a45_targeted_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_selective_tma_a45_targeted_iter41.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_selective_tma_a37_cli_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_selective_tma_a37_cli_iter41.txt)。

新45 rawU groups=84，reference=39（sameatom_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=36/108。

新45实际完成10项主机端A-only stride边界检查（1组PASS），其中native helper guard有4项。Plan设置成功0项、不可用10项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
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
      "expected_route": "megakernel_wgmma_psplit_selective_tma_a_64x32",
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
      "expected_route": "megakernel_wgmma_psplit_selective_tma_a_64x32",
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
      "expected_route": "megakernel_wgmma_psplit_selective_tma_a_64x32",
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
      "expected_route": "megakernel_wgmma_psplit_selective_tma_a_64x32",
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

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1800、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_selective_tma_a_enabled_only",
    "path": "results/strict_psplit_selective_tma_a_iter41.txt",
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

SAN45 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_selective_tma_a45_memcheck_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_selective_tma_a45_memcheck_iter41.txt)。非mem泄漏未测null。

SAN45 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_selective_tma_a45_racecheck_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_selective_tma_a45_racecheck_iter41.txt)。非mem泄漏未测null。

SAN45 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_selective_tma_a45_synccheck_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_selective_tma_a45_synccheck_iter41.txt)。非mem泄漏未测null。

CPUdag actualexit=0，tests=3；[psplit_selective_tma_a_cpu_dag_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_selective_tma_a_cpu_dag_iter41.txt)。

CPUstage actualexit=0，tests=6；[psplit_selective_tma_a_cpu_stage_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_selective_tma_a_cpu_stage_iter41.txt)。

CPU模型actualexit=0，originalstatus=PASS_CPU_SELECTIVE_TMA_A_GEOMETRY_STAGE_SOURCE_ONLY；[psplit_selective_tma_a_cpu_state_model_iter41.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_selective_tma_a_cpu_state_model_iter41.txt)。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。

环境：

```json
{
  "path": "results/small_psplit_selective_tma_a_environment_iter41.txt",
  "exit_path": "results/small_psplit_selective_tma_a_environment_iter41.exit",
  "sha256": "5ac9c538154edfbbb8411832396db5a79ef6a8a8433d1fd821703e6304187a21",
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
    "path": "results/psplit_selective_tma_a_function_occupancy_probe_iter41.txt",
    "sha256": "fafbd3388c5ca1affae34355dd1b65222a88090206d4eef733e58e24d8164fe2",
    "exit": 0,
    "records": [
      {
        "variant": 45,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm75_GLOBAL__N__2de34d57_42_megakernel_wgmma_psplit_selective_tma_a_cu_7cb8385e37winograd_wgmma_psplit_selective_tma_aIN7cutlass6half_tELi32EEEvNS_7ProblemENS0_23SelectiveTmaADescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_selective_tma_a_final_megakernel_wgmma_psplit_selective_tma_a_iter41.cubin",
        "cubin_sha256": "36d6c68168566a6972484b2825782882f6213ae074e80c21806f7a51ce89b7b2",
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
        "variant": 45,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm75_GLOBAL__N__2de34d57_42_megakernel_wgmma_psplit_selective_tma_a_cu_7cb8385e37winograd_wgmma_psplit_selective_tma_aIN7cutlass10bfloat16_tELi32EEEvNS_7ProblemENS0_23SelectiveTmaADescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_selective_tma_a_final_megakernel_wgmma_psplit_selective_tma_a_iter41.cubin",
        "cubin_sha256": "36d6c68168566a6972484b2825782882f6213ae074e80c21806f7a51ce89b7b2",
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
