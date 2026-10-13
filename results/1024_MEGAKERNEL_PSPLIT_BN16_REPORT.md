# 1024 BN16 raw RS-A / SS megakernel实测（iteration39）

Mega42相对同binary Mega39完整SW自身延迟：普通launch：BF16耗时变化+35.973%、FP16耗时变化+38.467%；Graph：BF16耗时变化+37.010%、FP16耗时变化+38.543%。

Mega43相对同binary Mega39完整SW自身延迟：普通launch：BF16耗时变化+27.313%、FP16耗时变化+29.667%；Graph：BF16耗时变化+29.015%、FP16耗时变化+29.272%。

Mega42相对同binary Mega40完整SW自身延迟：普通launch：BF16耗时变化+23.953%、FP16耗时变化+31.720%；Graph：BF16耗时变化+24.434%、FP16耗时变化+31.571%。

Mega43相对同binary Mega36完整SW自身延迟：普通launch：BF16耗时变化+25.134%、FP16耗时变化+26.550%；Graph：BF16耗时变化+28.236%、FP16耗时变化+27.912%。

Mega42相对同binary Mega43完整SW自身延迟：普通launch：BF16耗时变化+6.802%、FP16耗时变化+6.786%；Graph：BF16耗时变化+6.197%、FP16耗时变化+7.172%。

本轮mandatory24；实际24行、240条TIME、72次fresh CHECK；完整矩阵=是。原始门限0 PASS /72 FAIL，nonfinite=0。

本轮同binary已测native路线中，按完整SW自身延迟排序的最快结果：launch/bf16为Mega39（23.135359µs）；launch/fp16为Mega39（23.067520µs）；graph/bf16为Mega39（22.852800µs）；graph/fp16为Mega39（22.873919µs）。该排名不表示通过原始数值门限；完整多launch Lt1另列。

本报告分别记录完整调用性能、fresh gate、strict及结构测试。原始数值FAIL保留，Auto保持Dense；不宣布精度认证、策略提升或目标完成。

测量源码checkpoint `17b3e19d6a285e7cfc0a83af702a91134005f4ba`；bench SHA256 `149bf628e3201be18e591c6f72d6ea00da58c051bcacbc2487f401954e74bb6b`。八源、双binary及证据见[small_psplit_bn16_profile_iter39.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bn16_profile_iter39.json)，展开数据见[small_psplit_bn16_measurements_iter39.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_bn16_measurements_iter39.json)。

## 测量协议

完整目标为新42 raw RS-A BN16、新43 SS BN16、保留39 RS-A BK64、匹配40 raw RS-A BN32、匹配36 SS BN32及Lt1 × BF16/FP16 × ordinary/Graph。每个新路线被fresh compiled门限阻止时，必须绑定该路线四个准确缺失键：单条受阻partial20、两条受阻partial16。旧结果不能代替本轮测量。
每行20warmup、100calls/group、5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates、共同8GiB workspace cap。BF16/FP16输入、FP32输出。median和线性p10/p90来自五个组均值，CV为总体标准差/均值；quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优，没有Graph专用调优；Dense/SW按同行相同模式计时，比例为同行Dense event median / 完整SW event median。quick为20warmup、20calls/group、5groups、1fresh，只作筛选。

42采用raw RS-A BN16，43采用shared-A SS BN16。两者沿用原FP32 RN S/T DAG、最终onecast、whole-K七个FP32产品、固定U和RNalpha/activebetaFMA epilogue；实际source/geometry/quota/minCTAs/shared/drain/barrier/tail配置只取冻结plan与fresh compiled证据。CPU预测或静态min2 annotation不表示已测two-CTA residency。

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
  "42": {
    "actual_name": "megakernel_wgmma_psplit_rsa_bn16_64x16",
    "enabled": true,
    "producer_quota": 48,
    "consumer_quotas": [
      96,
      96
    ],
    "launch_bounds_min_ctas": 2,
    "BM": 64,
    "BN": 16,
    "BK": 32,
    "threads": 384,
    "dynamic_shared_bytes": 47232,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "device_reserved_shared_bytes_per_CTA": 1024,
    "bank_bytes": 23552,
    "tail_product_bytes": 28672,
    "tail_bank1_overlap_bytes": 5120,
    "initial_registers_per_thread": 80,
    "initial_owned_CTA_pool_registers": 30720,
    "requested_role_pool_registers": 30720,
    "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained",
    "occupancy_API_predicted_active_CTAs_per_SM": 2,
    "actual_measured_CTA_residency": null,
    "function_probe_path": "results/psplit_bn16_function_occupancy_probe_iter39.txt"
  },
  "43": {
    "actual_name": "megakernel_wgmma_psplit_ss_bn16_64x16",
    "enabled": true,
    "producer_quota": 64,
    "consumer_quotas": [
      80,
      80
    ],
    "launch_bounds_min_ctas": 2,
    "BM": 64,
    "BN": 16,
    "BK": 32,
    "threads": 384,
    "dynamic_shared_bytes": 71808,
    "static_shared_bytes": 0,
    "tool_resource_shared_bytes": 1024,
    "device_reserved_shared_bytes_per_CTA": 1024,
    "bank_bytes": 35840,
    "tail_product_bytes": 28672,
    "tail_bank1_overlap_bytes": 0,
    "initial_registers_per_thread": 80,
    "initial_owned_CTA_pool_registers": 30720,
    "requested_role_pool_registers": 28672,
    "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained",
    "occupancy_API_predicted_active_CTAs_per_SM": 2,
    "actual_measured_CTA_residency": null,
    "function_probe_path": "results/psplit_bn16_function_occupancy_probe_iter39.txt"
  }
}
```

五个唯一自身延迟比较为42对39、42对40、42对43、43对39、43对36。BN、role配额、launch bounds及tail映射可同时改变，所以所有controlled_axis_claimed=false；CPU scalar bank一词/每bank不证明实际transaction grouping、bank stalls、occupancy或GPU速度。

本轮实际测得的native Mega36/Mega39/Mega40/Mega42/Mega43每次读取当前原始A/B，重新构造变换，完成七个P、固定U及epilogue；完整调用为一个kernel，零native矩阵workspace，无host descriptor encode。Lt1为pack→七batch leaf GEMM→combine的完整多launch对照，不能代替单launch目标。
Graph固定capture地址，replay读取当前A/B内容并重新变换；fresh检查更换内容。setup/tuning/capture不计稳态；event覆盖完整GPU工作，可能含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，再取五组median，不是独立同步单调用CPU wall。环境快照不认证组内clock。

环境及记录作用域：

```json
{
  "path": "results/small_psplit_bn16_environment_iter39.txt",
  "exit_path": "results/small_psplit_bn16_environment_iter39.exit",
  "sha256": "8b7737b8f2b94d939ad043f2798b10326d9de9884c221e2e1cdacaaef3432a18",
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
  "snapshot_scope": "Actual read-only snapshot after all iteration39 performance and six serial SAN runs. No device kernel launched; no within-group clock stability or measured residency proof.",
  "function_attributes_and_occupancy_API": {
    "path": "results/psplit_bn16_function_occupancy_probe_iter39.txt",
    "sha256": "ad065ebaca84b7a920ebb17ba50a225fb50a35d56d995daa742203635051614b",
    "exit": 0,
    "records": [
      {
        "variant": 42,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm68_GLOBAL__N__6601b768_35_megakernel_wgmma_psplit_rsa_bn16_cu_5e1a6c4d30winograd_wgmma_psplit_rsa_bn16IN7cutlass6half_tELi16EEEvNS_7ProblemEPKT_S7_PKfPf",
        "cubin_path": "results/psplit_bn16_final_megakernel_wgmma_psplit_rsa_bn16_iter39.cubin",
        "cubin_sha256": "5ac8d00ac81aa3edee71e27709b80c14308d679a5dfda302d9ebd6f9adaf8933",
        "queried_dynamic_shared_bytes": 47232,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 47232
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      },
      {
        "variant": 42,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm68_GLOBAL__N__6601b768_35_megakernel_wgmma_psplit_rsa_bn16_cu_5e1a6c4d30winograd_wgmma_psplit_rsa_bn16IN7cutlass10bfloat16_tELi16EEEvNS_7ProblemEPKT_S7_PKfPf",
        "cubin_path": "results/psplit_bn16_final_megakernel_wgmma_psplit_rsa_bn16_iter39.cubin",
        "cubin_sha256": "5ac8d00ac81aa3edee71e27709b80c14308d679a5dfda302d9ebd6f9adaf8933",
        "queried_dynamic_shared_bytes": 47232,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 47232
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      },
      {
        "variant": 43,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm67_GLOBAL__N__4db7ea63_34_megakernel_wgmma_psplit_ss_bn16_cu_06492c2229winograd_wgmma_psplit_ss_bn16IN7cutlass6half_tELi16EEEvNS_7ProblemEPKT_S7_PKfPf",
        "cubin_path": "results/psplit_bn16_final_megakernel_wgmma_psplit_ss_bn16_iter39.cubin",
        "cubin_sha256": "5b4ab84b14e2d3254ba404598e249b3bfbcd90c82425280f18d6c1b0b38c438e",
        "queried_dynamic_shared_bytes": 71808,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 71808
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      },
      {
        "variant": 43,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm67_GLOBAL__N__4db7ea63_34_megakernel_wgmma_psplit_ss_bn16_cu_06492c2229winograd_wgmma_psplit_ss_bn16IN7cutlass10bfloat16_tELi16EEEvNS_7ProblemEPKT_S7_PKfPf",
        "cubin_path": "results/psplit_bn16_final_megakernel_wgmma_psplit_ss_bn16_iter39.cubin",
        "cubin_sha256": "5b4ab84b14e2d3254ba404598e249b3bfbcd90c82425280f18d6c1b0b38c438e",
        "queried_dynamic_shared_bytes": 71808,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 80,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 71808
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 2,
        "measured_residency": null
      }
    ],
    "scope": "Only current new42/new43 final exact feeding cubins queried without a kernel launch. Driver static shared attribute0; resource/ELF SHARED1024 and device reserve1024 are separate accounting. API2CTA/SM is a prediction, not measured residency or causal performance evidence. Retained control config shared fields preserve earlier tool-accounting declarations, not these new Driver function queries."
  },
  "timing_noise_observations": [
    {
      "csv_path": "results/small_psplit_rsa39_launch_iter39_full1024.csv",
      "csv_sha256": "d7444b40d23a23f391c5d58e3de0b43f9668f352d2e4128b913953380e799932",
      "variant": "39",
      "dtype": "fp16",
      "mode": "launch",
      "dense_population_CV": 0.539882776039,
      "solution_population_CV": 0.00127346747953,
      "dense_event_ms": 0.00994112014771,
      "dense_p90_group_mean_ms": 0.0209621763229,
      "all_raw_groups": [
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "0",
          "case": "solution",
          "event_ms": "0.0230634",
          "wall_ms": "0.0231441"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "0",
          "case": "dense",
          "event_ms": "0.00994112",
          "wall_ms": "0.0100134"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "1",
          "case": "dense",
          "event_ms": "0.00990496",
          "wall_ms": "0.00997754"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "1",
          "case": "solution",
          "event_ms": "0.0230688",
          "wall_ms": "0.0231402"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "2",
          "case": "solution",
          "event_ms": "0.0230589",
          "wall_ms": "0.0231322"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "2",
          "case": "dense",
          "event_ms": "0.00995232",
          "wall_ms": "0.0100267"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "3",
          "case": "solution",
          "event_ms": "0.0230675",
          "wall_ms": "0.0231411"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "3",
          "case": "dense",
          "event_ms": "0.00994112",
          "wall_ms": "0.0100131"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "4",
          "case": "dense",
          "event_ms": "0.0283021",
          "wall_ms": "0.0283834"
        },
        {
          "shape": "1024,1024,1024",
          "dtype": "fp16",
          "group": "4",
          "case": "solution",
          "event_ms": "0.0231376",
          "wall_ms": "0.0232149"
        }
      ],
      "scope": "All five original groups retained, authority CSV unchanged. Dense denominator fluctuation is not own SW code benefit; this ratio is not a stable speedup certification."
    }
  ]
}
```

## 同行Dense组均值的波动

以下Dense总体CV达到2%；全部五组保留，没有剔除较慢组。分母波动不能单独证明代码收益。

Mega39 RS-A BK64参照（上轮已测最快） BM64 BN32 BK64 64/192/192 min1 launch fp16：Dense CV=53.988278%，SW CV=0.127347%；stdout五个Dense event组均值µs=[9.94112, 9.90496, 9.95232, 9.94112, 28.3021]；[CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter39_full1024.csv:3)。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega42 raw RS-A BN16 BM64 BN16 BK32 48/96/96 min2 | bf16 | 12.5677 [12.3225,12.6847] | 31.4579 [31.4420,31.4629] | 0.399508 | 1.2959/0.0351 | 12.64615/31.53610 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bn1642_launch_iter39_full1024.csv:2) |
| Mega42 raw RS-A BN16 BM64 BN16 BK32 48/96/96 min2 | fp16 | 9.9542 [9.9491,9.9592] | 31.9408 [31.9237,31.9707] | 0.311647 | 0.0481/0.0714 | 10.02746/32.01315 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bn1642_launch_iter39_full1024.csv:3) |
| Mega43 SS BN16 BM64 BN16 BK32 64/80/80 min2 | bf16 | 12.2317 [12.2142,12.2484] | 29.4544 [29.4414,29.4940] | 0.415275 | 0.1362/0.0838 | 12.30240/29.54373 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_ss_bn1643_launch_iter39_full1024.csv:2) |
| Mega43 SS BN16 BM64 BN16 BK32 64/80/80 min2 | fp16 | 9.9683 [9.9554,9.9705] | 29.9110 [29.9007,29.9221] | 0.333266 | 0.0704/0.0345 | 10.04127/29.99069 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_ss_bn1643_launch_iter39_full1024.csv:3) |
| Mega39 RS-A BK64参照（上轮已测最快） BM64 BN32 BK64 64/192/192 min1 | bf16 | 12.2288 [12.2093,12.2540] | 23.1354 [23.1283,23.1462] | 0.528576 | 0.1609/0.0393 | 12.30135/23.21203 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter39_full1024.csv:2) |
| Mega39 RS-A BK64参照（上轮已测最快） BM64 BN32 BK64 64/192/192 min1 | fp16 | 9.9411 [9.9194,20.9622] | 23.0675 [23.0607,23.1101] | 0.430957 | 53.9883/0.1273 | 10.01341/23.14112 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter39_full1024.csv:3) |
| Mega40 raw RS-A BN32对照 BM64 BN32 BK32 64/192/192 min1 | bf16 | 12.2294 [12.2183,12.2408] | 25.3789 [25.3451,25.3948] | 0.481875 | 0.0902/0.0874 | 12.30143/25.46227 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter39_full1024.csv:2) |
| Mega40 raw RS-A BN32对照 BM64 BN32 BK32 64/192/192 min1 | fp16 | 9.8458 [9.8282,9.8722] | 24.2490 [24.2390,24.2664] | 0.406028 | 0.2140/0.0584 | 9.91762/24.32071 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter39_full1024.csv:3) |
| Mega36 SS BN32对照 BM64 BN32 BK32 64/96/96 min1 | bf16 | 12.2650 [12.2522,12.2949] | 23.5382 [23.5251,23.5636] | 0.521065 | 0.1570/0.0729 | 12.33766/23.61015 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter39_full1024.csv:2) |
| Mega36 SS BN32对照 BM64 BN32 BK32 64/96/96 min1 | fp16 | 9.9434 [9.9302,9.9635] | 23.6358 [23.6095,23.6485] | 0.420690 | 0.1532/0.0744 | 10.01707/23.70641 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter39_full1024.csv:3) |
| Lt1 打包一次（完整多launch对照） | bf16 | 12.3078 [12.2888,12.3144] | 19.2246 [19.2031,19.2699] | 0.640212 | 0.1005/0.1804 | 12.38167/19.29562 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter39_full1024.csv:2) |
| Lt1 打包一次（完整多launch对照） | fp16 | 9.8701 [9.8425,9.8964] | 19.5994 [19.5788,19.6490] | 0.503592 | 0.2512/0.1663 | 9.93924/19.67312 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter39_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega42 raw RS-A BN16 BM64 BN16 BK32 48/96/96 min2 | bf16 | 11.3107 [11.3039,11.3193] | 31.3107 [31.2858,31.3282] | 0.361241 | 0.0650/0.0648 | 11.38353/31.38265 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bn1642_graph_iter39_full1024.csv:2) |
| Mega42 raw RS-A BN16 BM64 BN16 BK32 48/96/96 min2 | fp16 | 9.4902 [9.4851,9.4935] | 31.6902 [31.6753,31.7085] | 0.299469 | 0.0394/0.0466 | 9.56523/31.76282 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bn1642_graph_iter39_full1024.csv:3) |
| Mega43 SS BN16 BM64 BN16 BK32 64/80/80 min2 | bf16 | 11.3632 [11.3491,11.3738] | 29.4835 [29.4284,29.5269] | 0.385409 | 0.0955/0.1531 | 11.43886/29.55560 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_ss_bn1643_graph_iter39_full1024.csv:2) |
| Mega43 SS BN16 BM64 BN16 BK32 64/80/80 min2 | fp16 | 9.3507 [9.3465,9.3579] | 29.5696 [29.5481,29.5967] | 0.316227 | 0.0555/0.0818 | 9.42814/29.64494 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_ss_bn1643_graph_iter39_full1024.csv:3) |
| Mega39 RS-A BK64参照（上轮已测最快） BM64 BN32 BK64 64/192/192 min1 | bf16 | 11.3059 [11.3020,11.3172] | 22.8528 [22.8313,22.8649] | 0.494728 | 0.0631/0.0666 | 11.38360/22.94297 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter39_full1024.csv:2) |
| Mega39 RS-A BK64参照（上轮已测最快） BM64 BN32 BK64 64/192/192 min1 | fp16 | 9.4781 [9.4557,9.4900] | 22.8739 [22.8648,22.8796] | 0.414362 | 0.1714/0.0287 | 9.55583/22.95394 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter39_full1024.csv:3) |
| Mega40 raw RS-A BN32对照 BM64 BN32 BK32 64/192/192 min1 | bf16 | 11.3517 [11.3458,11.3552] | 25.1626 [25.1258,25.1850] | 0.451134 | 0.0423/0.1195 | 11.42848/25.25697 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter39_full1024.csv:2) |
| Mega40 raw RS-A BN32对照 BM64 BN32 BK32 64/192/192 min1 | fp16 | 9.5386 [9.5205,9.5610] | 24.0861 [24.0592,24.1039] | 0.396020 | 0.1956/0.0843 | 9.62804/24.16410 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter39_full1024.csv:3) |
| Mega36 SS BN32对照 BM64 BN32 BK32 64/96/96 min1 | bf16 | 11.2810 [11.2766,11.2874] | 22.9917 [22.9700,23.0202] | 0.490654 | 0.0446/0.0974 | 11.35532/23.06450 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter39_full1024.csv:2) |
| Mega36 SS BN32对照 BM64 BN32 BK32 64/96/96 min1 | fp16 | 9.4122 [9.3898,9.4602] | 23.1171 [23.1047,23.1620] | 0.407151 | 0.3369/0.1266 | 9.48617/23.20043 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter39_full1024.csv:3) |
| Lt1 打包一次（完整多launch对照） | bf16 | 11.5654 [11.5503,11.5700] | 17.0941 [17.0908,17.1040] | 0.676576 | 0.0770/0.0351 | 11.63964/17.16823 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter39_full1024.csv:2) |
| Lt1 打包一次（完整多launch对照） | fp16 | 9.5254 [9.5224,9.5418] | 17.4998 [17.4810,17.5125] | 0.544316 | 0.1024/0.0809 | 9.60140/17.57537 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter39_full1024.csv:3) |

## 新42/43相对同binary对照的自身耗时

| 模式 | dtype | 对照 | 新路线 | 对照 µs | 新路线 µs | 对照/新路线自身延迟比 | 新路线自身耗时变化 | 对照/新路线 batchwall µs/call |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega39 | Mega42 | 23.135359 | 31.457920 | 0.735438× | +35.973% | 23.21203/31.53610 |
| launch | bf16 | Mega40 | Mega42 | 25.378881 | 31.457920 | 0.806756× | +23.953% | 25.46227/31.53610 |
| launch | bf16 | Mega43 | Mega42 | 29.454401 | 31.457920 | 0.936311× | +6.802% | 29.54373/31.53610 |
| launch | bf16 | Mega39 | Mega43 | 23.135359 | 29.454401 | 0.785464× | +27.313% | 23.21203/29.54373 |
| launch | bf16 | Mega36 | Mega43 | 23.538239 | 29.454401 | 0.799142× | +25.134% | 23.61015/29.54373 |
| launch | fp16 | Mega39 | Mega42 | 23.067520 | 31.940801 | 0.722196× | +38.467% | 23.14112/32.01315 |
| launch | fp16 | Mega40 | Mega42 | 24.248960 | 31.940801 | 0.759184× | +31.720% | 24.32071/32.01315 |
| launch | fp16 | Mega43 | Mega42 | 29.911039 | 31.940801 | 0.936452× | +6.786% | 29.99069/32.01315 |
| launch | fp16 | Mega39 | Mega43 | 23.067520 | 29.911039 | 0.771204× | +29.667% | 23.14112/29.99069 |
| launch | fp16 | Mega36 | Mega43 | 23.635840 | 29.911039 | 0.790205× | +26.550% | 23.70641/29.99069 |
| graph | bf16 | Mega39 | Mega42 | 22.852800 | 31.310720 | 0.729871× | +37.010% | 22.94297/31.38265 |
| graph | bf16 | Mega40 | Mega42 | 25.162561 | 31.310720 | 0.803640× | +24.434% | 25.25697/31.38265 |
| graph | bf16 | Mega43 | Mega42 | 29.483521 | 31.310720 | 0.941643× | +6.197% | 29.55560/31.38265 |
| graph | bf16 | Mega39 | Mega43 | 22.852800 | 29.483521 | 0.775104× | +29.015% | 22.94297/29.55560 |
| graph | bf16 | Mega36 | Mega43 | 22.991681 | 29.483521 | 0.779815× | +28.236% | 23.06450/29.55560 |
| graph | fp16 | Mega39 | Mega42 | 22.873919 | 31.690240 | 0.721797× | +38.543% | 22.95394/31.76282 |
| graph | fp16 | Mega40 | Mega42 | 24.086080 | 31.690240 | 0.760047× | +31.571% | 24.16410/31.76282 |
| graph | fp16 | Mega43 | Mega42 | 29.569600 | 31.690240 | 0.933082× | +7.172% | 29.64494/31.76282 |
| graph | fp16 | Mega39 | Mega43 | 22.873919 | 29.569600 | 0.773562× | +29.272% | 22.95394/29.64494 |
| graph | fp16 | Mega36 | Mega43 | 23.117120 | 29.569600 | 0.781787× | +27.912% | 23.20043/29.64494 |

实际自身延迟配对20个，未测配对0个；各axis配置、准确missing keys及pair contract见measurements。不重复反向添加42对43。缺失新路线不能用其他binary或历史结果补齐；收益判断使用SW自身延迟。

## 编译资源与实际检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 42 | FP16 | 64×16 /32 | 384 | 47232 | 80 | 30720 | 48/96/96 | 30720 | 0/0/0 |
| 42 | BF16 | 64×16 /32 | 384 | 47232 | 80 | 30720 | 48/96/96 | 30720 | 0/0/0 |
| 43 | FP16 | 64×16 /32 | 384 | 71808 | 80 | 30720 | 64/80/80 | 28672 | 0/0/0 |
| 43 | BF16 | 64×16 /32 | 384 | 71808 | 80 | 30720 | 64/80/80 | 28672 | 0/0/0 |

Driver function attribute的静态shared和resource tool的SHARED accounting分别记录，不互相代换：

Mega42 FP16：Driver static_shared_bytes=0，resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2，measured_residency=None。
Mega42 BF16：Driver static_shared_bytes=0，resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2，measured_residency=None。
Mega43 FP16：Driver static_shared_bytes=0，resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2，measured_residency=None。
Mega43 BF16：Driver static_shared_bytes=0，resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=2，measured_residency=None。
这是实际function/cubin资源查询的静态预测，没有启动kernel测量residency；不作为实际two-CTA驻留、transactions、bank stalls或性能因果证明。


本轮identity绑定fresh compiled gate：status=PASS_COMPILED_CAN_LAUNCH_42_43，[psplit_bn16_static_review_initial_iter39.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_bn16_static_review_initial_iter39.json)，SHA256 `ac1d8d71ffee3d0a8f59b5c01043ce54596d17feaa62a1993daea8e48a5276d5`。映射compiled input和原审文件分别绑定；原instances字段保持。源、object、bench/test全stream与实际resource合法性由独立review核验。

初始与最终时序/模型作用域：

```json
{
  "scope": "Frozen BN16 raw/SS candidates, tail-layout derivatives and three manifests remain CPU-only inputs. Each newroute requires its own fresh source/binary compiled safety proof; final budgets/name/shared/initial ownership come from the frozen plan and actual gate. Chronological build attempts preserve actual exits; no pending scope is retrospectively certified and retained unsafe37 belongs to iteration34.",
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
    "42": {
      "actual_name": "megakernel_wgmma_psplit_rsa_bn16_64x16",
      "enabled": true,
      "producer_quota": 48,
      "consumer_quotas": [
        96,
        96
      ],
      "launch_bounds_min_ctas": 2,
      "BM": 64,
      "BN": 16,
      "BK": 32,
      "threads": 384,
      "dynamic_shared_bytes": 47232,
      "static_shared_bytes": 0,
      "tool_resource_shared_bytes": 1024,
      "device_reserved_shared_bytes_per_CTA": 1024,
      "bank_bytes": 23552,
      "tail_product_bytes": 28672,
      "tail_bank1_overlap_bytes": 5120,
      "initial_registers_per_thread": 80,
      "initial_owned_CTA_pool_registers": 30720,
      "requested_role_pool_registers": 30720,
      "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained",
      "occupancy_API_predicted_active_CTAs_per_SM": 2,
      "actual_measured_CTA_residency": null,
      "function_probe_path": "results/psplit_bn16_function_occupancy_probe_iter39.txt"
    },
    "43": {
      "actual_name": "megakernel_wgmma_psplit_ss_bn16_64x16",
      "enabled": true,
      "producer_quota": 64,
      "consumer_quotas": [
        80,
        80
      ],
      "launch_bounds_min_ctas": 2,
      "BM": 64,
      "BN": 16,
      "BK": 32,
      "threads": 384,
      "dynamic_shared_bytes": 71808,
      "static_shared_bytes": 0,
      "tool_resource_shared_bytes": 1024,
      "device_reserved_shared_bytes_per_CTA": 1024,
      "bank_bytes": 35840,
      "tail_product_bytes": 28672,
      "tail_bank1_overlap_bytes": 0,
      "initial_registers_per_thread": 80,
      "initial_owned_CTA_pool_registers": 30720,
      "requested_role_pool_registers": 28672,
      "drain_barrier": "nonaligned barrier.sync6,384; epochs0/6/0; BOTH banks drained",
      "occupancy_API_predicted_active_CTAs_per_SM": 2,
      "actual_measured_CTA_residency": null,
      "function_probe_path": "results/psplit_bn16_function_occupancy_probe_iter39.txt"
    }
  }
}
```

初始pending/失败字段保留当时身份，最终配置只取fresh gate和冻结plan，不能追溯认证旧候选。

CPU DAG实际exit=0，tests=3，[psplit_bn16_cpu_dag_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bn16_cpu_dag_iter39.txt)。

CPU stage实际exit=0，tests=6，[psplit_bn16_cpu_stage_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bn16_cpu_stage_iter39.txt)。

CPU抽象/几何模型实际exit=0，status=PASS_CPU_LITERAL_GEOMETRY_ONLY_UNCOMPILED_UNREGISTERED；[psplit_bn16_cpu_raw_model_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bn16_cpu_raw_model_iter39.txt)。冻结原件保护及writer剥离按真实runner核验，JSON PASS不是进程退出证明；该模型不认证CUDA transaction、liveness、配额fit、GPU精度或性能。

CPU抽象/几何模型实际exit=0，status=PASS_CPU_LITERAL_SS_BN16_GEOMETRY_ONLY_UNCOMPILED_UNREGISTERED；[psplit_bn16_cpu_ss_model_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bn16_cpu_ss_model_iter39.txt)。冻结原件保护及writer剥离按真实runner核验，JSON PASS不是进程退出证明；该模型不认证CUDA transaction、liveness、配额fit、GPU精度或性能。

CPU抽象/几何模型实际exit=0，status=PASS_CPU_LITERAL_SCALAR_TAIL_BANK_GEOMETRY_ONLY；[psplit_bn16_cpu_tail_model_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_bn16_cpu_tail_model_iter39.txt)。冻结原件保护及writer剥离按真实runner核验，JSON PASS不是进程退出证明；该模型不认证CUDA transaction、liveness、配额fit、GPU精度或性能。

default：exit=0，2634 CASE /2159 PASS；[correctness_psplit_bn16_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_bn16_iter39.txt)。

selected42：exit=0，56 CASE /135 PASS；[correctness_psplit_bn1642_targeted_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_bn1642_targeted_iter39.txt)。

selected43：exit=0，56 CASE /135 PASS；[correctness_psplit_bn1643_targeted_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_bn1643_targeted_iter39.txt)。

disabled37_cli：exit=1，0 CASE /0 PASS；[disabled_psplit_bn1637_cli_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_bn1637_cli_iter39.txt)。

每个新variant的实际Graph/rebind/bitwise范围：

```json
{
  "42": {
    "new_graph_configurations": 36,
    "new_graph_fresh_replays": 108,
    "sameatom_rawU_host_epilogue_groups": 84,
    "crossatom_rawU_host_epilogue_groups": 0,
    "bitwise_rawU_host_epilogue_groups": 84,
    "current_pointer_binding_checks": 20,
    "enabled": true,
    "reference_scope": "42 uses43 N16 sameatom reference only when43 is enabled, otherwise42 uses36 as crossatom N16-vs-N32;43 always uses36 as explicit crossatom N16-vs-N32 check. Reference identity, actual fixture multiplicity and any FAIL come from frozen test source/coverage; no N32 oracle relabeled N16."
  },
  "43": {
    "new_graph_configurations": 36,
    "new_graph_fresh_replays": 108,
    "sameatom_rawU_host_epilogue_groups": 0,
    "crossatom_rawU_host_epilogue_groups": 84,
    "bitwise_rawU_host_epilogue_groups": 84,
    "current_pointer_binding_checks": 20,
    "enabled": true,
    "reference_scope": "42 uses43 N16 sameatom reference only when43 is enabled, otherwise42 uses36 as crossatom N16-vs-N32;43 always uses36 as explicit crossatom N16-vs-N32 check. Reference identity, actual fixture multiplicity and any FAIL come from frozen test source/coverage; no N32 oracle relabeled N16."
  }
}
```

42对43只在43 enabled且实际执行时属于N16 sameatom参照；43 blocked时42改对36，43始终对36，均明确属于N16-vs-N32 crossatom检查。严格bitwise/hostepilogue的失败如实保留，不改标为sameatom。原base K32与新增partial fixture的重复按冻结source计数，不能去重删样本。Mega37继续disabled；fallback PASS不认证native。

## 原始精度与sanitizer

本轮strict实际60 gates：36 PASS /24 FAIL /nonfinite 0；48 native、12 Dense fallback。BF16/FP16原floor保持0.001/0.0001，门限3×Dense_eF+floor。

新Mega42实际30 gates、18 PASS /12 FAIL，nonfinite=0，exit=2；[strict_psplit_bn16_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/strict_psplit_bn16_iter39.txt)。

新Mega43实际30 gates、18 PASS /12 FAIL，nonfinite=0，exit=2；[strict_psplit_bn16_iter39.txt](/share/home/liuyao/workspace/MegaFMM/results/strict_psplit_bn16_iter39.txt)。

历史strict按各source/bin身份保留；本轮full代码域为1680+每enabled新variant30（both1740/one1710/none1680），不表示该整个域已运行。原数值FAIL保留，AutoDense，目标仍active。

实际执行作用域：

```json
"Only archived current-source runs are actual; blocked or absent scopes stay null. Current full code domain is 1680 plus30 for each enabled new variant (both1740/one1710/none1680); it is not an executed full rerun. Historical scopes remain separately bound."
```

实际归档6项CUDA Compute Sanitizer，0项实测错误字段非零或未记录；memcheck leak bytes=[0, 0]。race/sync的leak字段为null；每项selected fixture multiset及variant归属由独立审计核验，不替被blocked路线或Mega37执行。

旧CUDA 35个，历史证据1476绑定；历史onsite-only 4件按现场SHA保留，不冒称Git对象。受保护scratch原件与Git归档正式副本分别分类。统计、来源与最后Git绑定由独立数据审计核验。
