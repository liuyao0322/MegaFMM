# 1024 BN16选择性TMA rawA实测（iteration44）

Mega48对同binary Mega45自身完整调用：launch bf16 27.426879µs对17.664959µs，延长55.261%；launch fp16 28.274879µs对17.799680µs，延长58.850%；graph bf16 26.989441µs对16.950080µs，延长59.229%；graph fp16 27.788479µs对17.348480µs，延长60.178%。

Mega48对同binary Mega47自身完整调用：launch bf16 27.426879µs对26.040959µs，延长5.322%；launch fp16 28.274879µs对26.115520µs，延长8.268%；graph bf16 26.989441µs对25.579519µs，延长5.512%；graph fp16 27.788479µs对25.565441µs，延长8.695%。

Mega45对同binary Mega39自身完整调用：launch bf16 17.664959µs对23.144000µs，减少23.674%；launch fp16 17.799680µs对23.068161µs，减少22.839%；graph bf16 16.950080µs对22.894399µs，减少25.964%；graph fp16 17.348480µs对22.825279µs，减少23.994%。

Mega48对同binary Mega39自身完整调用：launch bf16 27.426879µs对23.144000µs，延长18.505%；launch fp16 28.274879µs对23.068161µs，延长22.571%；graph bf16 26.989441µs对22.894399µs，延长17.887%；graph fp16 27.788479µs对22.825279µs，延长21.744%。

本轮launch bf16已测native中最快为Mega45，17.664959µs，原数值门限FAIL。
本轮launch fp16已测native中最快为Mega45，17.799680µs，原数值门限FAIL。
本轮graph bf16已测native中最快为Mega45，16.950080µs，原数值门限FAIL。
本轮graph fp16已测native中最快为Mega45，17.348480µs，原数值门限FAIL。

实际行数20/预声明20，complete=True；fresh PASS/FAIL=0/60，nonfinite=0。

结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。

测量源commit 987ef70b735617dc6dcffa867947f96ee76935ef；[small_psplit_tma_bn16_profile_iter44.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_profile_iter44.json)，[small_psplit_tma_bn16_measurements_iter44.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn16_measurements_iter44.json)。

## 测量协议

ONLY48是新BN16选择性四路TMA rawA路线；45为同binary选择性四路TMA rawA，39为原RS-A BK64，47为BN64 TMA BK64，Lt1为完整多launch对照。49未注册、未运行。fresh compiled blocked48必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替；45对39四个同binary配对仍可实际测得。

warmup=20，calls/group=100，groups=5，fresh/row=3，rotate_inputs=1，heuristic candidates=32，共同workspace cap=8589934592bytes。

event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。

48、47和45的源码入口在普通启动时每次从当前A指针/lda编码一个A tensor map；实际enabled路线的编码与单个kernel均包含在完整调用计时内，map实际被四路rawA TMA消费，B仍SIMT；39无host descriptor encode。48使用BM64/BN16/BK64，45为BN32/BK64，WGMMA N16/N32 atom不同；源码预算、shared及尾部也不同，所有controlled_axis_claimed=false，不认证纯BN或纯搬运因果。全部实际运行native完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，为一个kernel、零矩阵workspace；Lt1完整多launch计时。Graph45/47/48在capture时编码，稳态replay使用固定capture地址读取当前内容，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时。rawU对45为CROSSATOM_N16_N32的源限定严格bitwise fixture与独立host RN alpha/FMA检查，绝不是sameatom或通用FP64精度认证；任何实际不一致/FAIL必须保留。

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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration44 resource/residency query."
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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration44 resource/residency query."
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
    "resource_metadata_scope": "Retained legacy compiled metadata, not a fresh iteration44 resource/residency query."
  },
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
    "resource_metadata_scope": "Actual fresh iteration44 feeding function attributes/hostABI plus independent fullrole/init/lifetime gate; API prediction1 is not measured residency."
  }
}
```

## 普通启动完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | bf16 | 12.436481 / 12.314432 / 12.559872 | 27.426879 / 27.419712 / 27.438848 | 0.453441 | 0.9157/0.0309 | 12.52394/27.52253 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn1648_launch_iter44_full1024.csv:2) |
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | fp16 | 9.944960 / 9.936448 / 9.966336 | 28.274879 / 28.253823 / 28.296897 | 0.351724 | 0.1493/0.0732 | 10.03115/28.37374 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn1648_launch_iter44_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 12.265600 / 12.231936 / 12.273728 | 17.664959 / 17.652160 / 17.683264 | 0.694346 | 0.1622/0.0779 | 12.33685/17.75230 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter44_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.952960 / 9.942720 / 9.969664 | 17.799680 / 17.788864 / 17.812672 | 0.559165 | 0.1203/0.0589 | 10.02434/17.87373 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_launch_iter44_full1024.csv:3) |
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | bf16 | 12.266240 / 12.253568 / 12.277056 | 26.040959 / 26.032257 / 26.055039 | 0.471036 | 0.0941/0.0388 | 12.33781/26.12667 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_launch_iter44_full1024.csv:2) |
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | fp16 | 9.937920 / 9.921856 / 9.948928 | 26.115520 / 26.105280 / 26.130047 | 0.380537 | 0.1327/0.0479 | 10.01032/26.18695 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_launch_iter44_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 12.262720 / 12.253376 / 12.278272 | 23.144000 / 23.138945 / 23.154752 | 0.529844 | 0.0902/0.0314 | 12.33912/23.23289 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter44_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.936320 / 9.923200 / 9.948352 | 23.068161 / 23.052224 / 23.076416 | 0.430737 | 0.1112/0.0558 | 10.01443/23.14606 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter44_full1024.csv:3) |
| Lt1完整多launch | bf16 | 12.205760 / 12.177664 / 12.208064 | 19.451520 / 19.425984 / 19.460928 | 0.627496 | 0.1174/0.0822 | 12.27712/19.52390 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter44_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.932800 / 9.924288 / 9.935552 | 19.637440 / 19.618112 / 19.645440 | 0.505809 | 0.0511/0.0666 | 10.00465/19.71007 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter44_full1024.csv:3) |

## CUDA Graph完整端到端耗时

| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | bf16 | 11.277120 / 11.243328 / 11.312000 | 26.989441 / 26.977408 / 27.022208 | 0.417835 | 0.2955/0.0745 | 11.35048/27.06362 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn1648_graph_iter44_full1024.csv:2) |
| Mega48 megakernel_wgmma_psplit_tma_bn16_64x16 | fp16 | 9.460160 / 9.438656 / 9.465344 | 27.788479 / 27.783873 / 27.819456 | 0.340435 | 0.1291/0.0669 | 9.53664/27.87162 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn1648_graph_iter44_full1024.csv:3) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | bf16 | 11.517440 / 11.510976 / 11.531328 | 16.950080 / 16.940160 / 16.960000 | 0.679492 | 0.0789/0.0514 | 11.59250/17.02364 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter44_full1024.csv:2) |
| Mega45 megakernel_wgmma_psplit_selective_tma_a_64x32 | fp16 | 9.525760 / 9.519040 / 9.587712 | 17.348480 / 17.346240 / 17.357312 | 0.549083 | 0.3682/0.0323 | 9.59846/17.43183 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_selective_tma_a45_graph_iter44_full1024.csv:3) |
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | bf16 | 11.365120 / 11.362816 / 11.371264 | 25.579519 / 25.570687 / 25.595904 | 0.444305 | 0.0339/0.0440 | 11.44197/25.66454 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_graph_iter44_full1024.csv:2) |
| Mega47 megakernel_wgmma_psplit_tma_bn64_64x64 | fp16 | 9.535040 / 9.519744 / 9.554560 | 25.565441 / 25.563073 / 25.591040 | 0.372966 | 0.1613/0.0515 | 9.62138/25.64001 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_tma_bn6447_graph_iter44_full1024.csv:3) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | bf16 | 11.287360 / 11.275456 / 11.289024 | 22.894399 / 22.879681 / 22.900097 | 0.493018 | 0.0631/0.0423 | 11.36175/22.96878 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter44_full1024.csv:2) |
| Mega39 megakernel_wgmma_psplit_rsa_64x32 | fp16 | 9.565440 / 9.561088 / 9.570752 | 22.825279 / 22.820608 / 22.838144 | 0.419072 | 0.0473/0.0344 | 9.64167/22.91012 | FAIL finite | 0 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter44_full1024.csv:3) |
| Lt1完整多launch | bf16 | 11.488000 / 11.459200 / 11.491712 | 16.980160 / 16.976384 / 16.982976 | 0.676554 | 0.1358/0.0179 | 11.55848/17.05174 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter44_full1024.csv:2) |
| Lt1完整多launch | fp16 | 9.424960 / 9.418176 / 9.443136 | 17.215680 / 17.202368 / 17.246208 | 0.547464 | 0.1208/0.1231 | 9.49655/17.28915 | FAIL finite | 46 | [CSV](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter44_full1024.csv:3) |

## 同binary自身延迟比较

| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |
|---|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega45 | Mega48 | 17.664959 | 27.426879 | 0.644075× | +55.261% | 17.75230/27.52253 |
| launch | bf16 | Mega47 | Mega48 | 26.040959 | 27.426879 | 0.949469× | +5.322% | 26.12667/27.52253 |
| launch | bf16 | Mega39 | Mega45 | 23.144000 | 17.664959 | 1.310164× | -23.674% | 23.23289/17.75230 |
| launch | bf16 | Mega39 | Mega48 | 23.144000 | 27.426879 | 0.843844× | +18.505% | 23.23289/27.52253 |
| launch | fp16 | Mega45 | Mega48 | 17.799680 | 28.274879 | 0.629523× | +58.850% | 17.87373/28.37374 |
| launch | fp16 | Mega47 | Mega48 | 26.115520 | 28.274879 | 0.923630× | +8.268% | 26.18695/28.37374 |
| launch | fp16 | Mega39 | Mega45 | 23.068161 | 17.799680 | 1.295987× | -22.839% | 23.14606/17.87373 |
| launch | fp16 | Mega39 | Mega48 | 23.068161 | 28.274879 | 0.815854× | +22.571% | 23.14606/28.37374 |
| graph | bf16 | Mega45 | Mega48 | 16.950080 | 26.989441 | 0.628026× | +59.229% | 17.02364/27.06362 |
| graph | bf16 | Mega47 | Mega48 | 25.579519 | 26.989441 | 0.947760× | +5.512% | 25.66454/27.06362 |
| graph | bf16 | Mega39 | Mega45 | 22.894399 | 16.950080 | 1.350696× | -25.964% | 22.96878/17.02364 |
| graph | bf16 | Mega39 | Mega48 | 22.894399 | 26.989441 | 0.848272× | +17.887% | 22.96878/27.06362 |
| graph | fp16 | Mega45 | Mega48 | 17.348480 | 27.788479 | 0.624305× | +60.178% | 17.43183/27.87162 |
| graph | fp16 | Mega47 | Mega48 | 25.565441 | 27.788479 | 0.920001× | +8.695% | 25.64001/27.87162 |
| graph | fp16 | Mega39 | Mega45 | 22.825279 | 17.348480 | 1.315693× | -23.994% | 22.91012/17.43183 |
| graph | fp16 | Mega39 | Mega48 | 22.825279 | 27.788479 | 0.821394× | +21.744% | 22.91012/27.87162 |

所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。


## 新48实际编译资源

| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 48 | fp16 | 64×16 /64 | 384 | 95232 | 128 | 128 | 49152 | 40/128/128 | 37888 | 0/0/0 |
| 48 | bf16 | 64×16 /64 | 384 | 95232 | 128 | 128 | 49152 | 40/128/128 | 37888 | 0/0/0 |

Mega48 fp16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。
Mega48 bf16：Driver static_shared_bytes=0；resource_tool_reported_SHARED_bytes=1024；occupancy API predicted CTAs/SM=1；measured_residency=None。

原编译门限与本轮资源修复时序：[psplit_tma_bn16_static_review_initial_iter44.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_static_review_initial_iter44.json)；这是本轮编译资源放行记录，不认证数值或性能。

initial_failed_revision：[psplit_tma_bn16_initial_source_iter44.cu](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_initial_source_iter44.cu)；失败身份与当前放行配置分开保留。

```json
{
  "archived_source_path": "results/psplit_tma_bn16_initial_source_iter44.cu",
  "source_sha256": "c48dee6016ed68e1cf9a7c0837c2a018fb3164de4ae7ff99d04e1c3db4dc3df8",
  "status": "FAIL_STACK168_SPILL224_NO_LAUNCH",
  "logical_source_path": "solution/megakernel_wgmma_psplit_tma_bn16.cu",
  "current_source_distinct": true,
  "fresh_role_maxima": [
    34,
    125,
    125
  ],
  "producer_local_spills": 0,
  "consumer_spill_attribution": "31WG0 raw A addresses;24WG1 raw A addresses+1 barrier CTA-rank/sharedbase word; not P or packed finalA values inside pending window"
}
```

原门限记录的源码修复：Stage-local fresh volatile tid reconstructs same lane/coordinates; independent B pair finalcasts shortened lifetime retaining unrounded f2 for T4. All RN DAGs, geometry, quotas, typed layout, guards and output unchanged.。这不构成性能收益的因果证明。

最终fp16配额 40/128/128，actual I=128、owned/requested pool=49152/37888；producer/WG0/WG1最大物理GPR=37/125/124。实例初始化原标志 all_P_clear_after_successful_INC=true、legal_initial_owned_preINC_zero_present=false；原字段按本轮门限逐字记录，不继承旧47的preINC例外。

本实例两consumer配额等于actual initial I，没有新增grown寄存器；首MMA accumulator zero与pending A/P证明仍来自本轮原门限。

最终bf16配额 40/128/128，actual I=128、owned/requested pool=49152/37888；producer/WG0/WG1最大物理GPR=37/125/124。实例初始化原标志 all_P_clear_after_successful_INC=true、legal_initial_owned_preINC_zero_present=false；原字段按本轮门限逐字记录，不继承旧47的preINC例外。

本实例两consumer配额等于actual initial I，没有新增grown寄存器；首MMA accumulator zero与pending A/P证明仍来自本轮原门限。


Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。

## 实际检查范围

default：actualexit=0，CASE/PASS/BEGIN=2922/2836/5284；[correctness_psplit_tma_bn16_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn16_iter44.txt)。
selected48：actualexit=0，CASE/PASS/BEGIN=60/140/204；[correctness_psplit_tma_bn1648_targeted_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_tma_bn1648_targeted_iter44.txt)。
disabled37_cli：actualexit=1，CASE/PASS/BEGIN=0/0/0；[disabled_psplit_tma_bn1637_cli_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_tma_bn1637_cli_iter44.txt)。

新48 rawU groups=84，reference=45（CROSSATOM_N16_N32_BK64）；bound total kernel runs=246，reference=84，candidate=162。AB=20，epi groups/runs=10/30，Graph configs/fresh=40/120。

新48实际完成10项主机端A-only stride边界检查（1组PASS），其中native helper guard有4项。Plan设置成功0项、不可用10项；input allocation bytes=0、tensor map encode=0、native GPU run=0。
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_64x16",
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_64x16",
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_64x16",
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
      "expected_route": "megakernel_wgmma_psplit_tma_bn16_64x16",
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

当前strict candidate gates=30，PASS/FAIL=18/12，nonfinite=0；native=24，Dense范围回退=6。full code domain=1890、actually executed=False；Dense reference CASE单独计数，不能加到candidate gate。

```json
[
  {
    "record_label": "strict_tma_bn16_enabled_only",
    "path": "results/strict_psplit_tma_bn16_iter44.txt",
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

SAN48 memcheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=0；[sanitizer_psplit_tma_bn1648_memcheck_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn1648_memcheck_iter44.txt)。非mem泄漏未测null。

SAN48 racecheck：actualexit=0，errors=0，race hazards/warnings=0/0，leaked bytes=None；[sanitizer_psplit_tma_bn1648_racecheck_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn1648_racecheck_iter44.txt)。非mem泄漏未测null。

SAN48 synccheck：actualexit=0，errors=0，race hazards/warnings=None/None，leaked bytes=None；[sanitizer_psplit_tma_bn1648_synccheck_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/sanitizer_psplit_tma_bn1648_synccheck_iter44.txt)。非mem泄漏未测null。

CPUdag actualexit=0，tests=3；[psplit_tma_bn16_cpu_dag_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_cpu_dag_iter44.txt)。

CPUstage actualexit=0，tests=6；[psplit_tma_bn16_cpu_stage_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_cpu_stage_iter44.txt)。

CPU模型actualexit=0，originalstatus=PASS_CPU_BN16_GEOMETRY_AND_PROTOCOL；[psplit_tma_bn16_cpu_state_model_iter44.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_tma_bn16_cpu_state_model_iter44.txt)。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。

环境：

```json
{
  "path": "results/small_psplit_tma_bn16_environment_iter44.txt",
  "exit_path": "results/small_psplit_tma_bn16_environment_iter44.exit",
  "sha256": "f015ba3cd1e81b61590c11d2f3b44f29527805924e851d3df4d802cfa1f16204",
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
    "path": "results/psplit_tma_bn16_function_occupancy_probe_iter44.txt",
    "sha256": "5b015b257562b6409c9ea159ac1b3296f84c137b026b72a26c40c00148d21644",
    "exit": 0,
    "records": [
      {
        "variant": 48,
        "dtype": "fp16",
        "symbol": "_ZN7megafmm68_GLOBAL__N__b7d3cc70_35_megakernel_wgmma_psplit_tma_bn16_cu_cf92e38030winograd_wgmma_psplit_tma_bn16IN7cutlass6half_tELi16EEEvNS_7ProblemENS0_17TmaBN16DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn16_final_megakernel_wgmma_psplit_tma_bn16_iter44.cubin",
        "cubin_sha256": "163f8377b13b91e4e66e6470fd8d466212bf4af1e22de3219e6079778ee3eef7",
        "queried_dynamic_shared_bytes": 95232,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 128,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 95232
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      },
      {
        "variant": 48,
        "dtype": "bf16",
        "symbol": "_ZN7megafmm68_GLOBAL__N__b7d3cc70_35_megakernel_wgmma_psplit_tma_bn16_cu_cf92e38030winograd_wgmma_psplit_tma_bn16IN7cutlass10bfloat16_tELi16EEEvNS_7ProblemENS0_17TmaBN16DescriptorEPKT_PKfPf",
        "cubin_path": "results/psplit_tma_bn16_final_megakernel_wgmma_psplit_tma_bn16_iter44.cubin",
        "cubin_sha256": "163f8377b13b91e4e66e6470fd8d466212bf4af1e22de3219e6079778ee3eef7",
        "queried_dynamic_shared_bytes": 95232,
        "queried_threads": 384,
        "function_attributes": {
          "max_threads_per_block": 384,
          "static_shared_bytes": 0,
          "local_bytes": 0,
          "registers_per_thread": 128,
          "max_dynamic_shared_bytes_before_optin": 49152,
          "max_dynamic_shared_bytes_after_optin": 95232
        },
        "occupancy_API_predicted_active_CTAs_per_SM": 1,
        "measured_residency": null
      }
    ]
  }
}
```

唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。
