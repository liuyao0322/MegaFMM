# 1024 Hybrid S BK32 megakernel实测（iteration38）

新Mega41相对同binary保留的RS-A BK64参照 Mega39（上轮已测最快）：普通launch下BF16耗时变化+26.445%、FP16耗时变化+26.490%；Graph下BF16耗时变化+26.152%、FP16耗时变化+26.932%。

新Mega41相对同binary匹配raw RS-A BK32 Mega40：普通launch下BF16耗时变化+15.176%、FP16耗时变化+20.302%；Graph下BF16耗时变化+14.435%、FP16耗时变化+19.966%。

新Mega41相对同binary matched BK32 SS Mega36：普通launch下BF16耗时变化+24.708%、FP16耗时变化+23.069%；Graph下BF16耗时变化+24.881%、FP16耗时变化+24.180%。

BF16新Mega41完整调用29.221120µs，同行cuBLASLt 12.576640µs，Dense/SW 0.430396×；相对同binary保留RS-A BK64参照 Mega39自身耗时变化+26.445%。FP16新Mega41完整调用29.146240µs，同行cuBLASLt 9.880000µs，Dense/SW 0.338980×；相对同binary保留RS-A BK64参照 Mega39自身耗时变化+26.490%。

本轮mandatory20；实际20行、200条TIME、60个fresh CHECK；完整矩阵=是。原始门限0 PASS /60 FAIL，nonfinite=0，实际记录行0/20通过。

本轮同binary已测native路线中，按完整SW调用耗时排序的最快结果：launch/bf16为Mega39（23.109760µs）；launch/fp16为Mega39（23.042240µs）；graph/bf16为Mega39（22.815361µs）；graph/fp16为Mega39（22.777600µs）。该排名不表示通过原始数值门限；完整多launch Lt1另列。

实际原始数值门限存在FAIL或nonfinite；性能与结构检查不构成原始精度认证。

完整调用性能、fresh gate和strict执行范围分别报告；本报告不宣布目标完成或提升Auto策略。

测量源码checkpoint `dec5e4b55f838f77fda3ec31a9869931cd9829f4`；bench SHA256 `6f81bf43e39d7d6bc089f4877cb8cdceb16c2e7f1dbdc47a47e4a6f02b0e97d8`。七源、双binary与原始证据见[small_psplit_hybrid_s_profile_iter38.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_hybrid_s_profile_iter38.json)，展开测量见[small_psplit_hybrid_s_measurements_iter38.json](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_hybrid_s_measurements_iter38.json)。

## 测量协议

完整目标为新41 Hybrid S BK32/保留39 RS-A BK64（上轮已测最快）/匹配40 raw RS-A BK32/匹配36 SS BK32/Lt1 × BF16/FP16 × ordinary/Graph。若新41编译安全门限阻止，则只有显式绑定最新gate的四个缺失key才允许partial16；旧结果不能替代本轮测量。每行20warmup、100calls/group、5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates和共同8GiB workspace cap。BF16/FP16输入、FP32输出。median与线性p10/p90来自五个组均值，CV使用总体标准差/均值。每行cuBLASLt独立调优，调优发生在capture前，没有Graph专用调优；Dense和SW按同行相同模式计时。Dense/SW使用同行Dense event median / 完整SW event median。quick仅作筛选：20warmup、20calls/group、5groups、1fresh，不能替代这里的mandatory full测量。

新41采用Hybrid S路线：producer计算未舍入FP32 s1与s2，roundedS1只供P5；S2保持FP32，even/odd分平面。两个独立consumer WG各构造一个A表达式并按whole-K计算P3+4。最终几何、名称、role配额、launch bounds、barrier与shared配置见下列冻结plan记录；初始CPU模型和候选预算只证明各自归档的抽象模型范围，不能充当最终编译fit或硬件执行证据。

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
  "41": {
    "actual_name": "megakernel_wgmma_psplit_hybrid_s_64x32",
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
    "dynamic_shared_bytes": 86144,
    "predicted_dynamic_shared_bytes": 86144,
    "static_shared_bytes": 1024,
    "bank_bytes": 43008,
    "tail_product_bytes": 57344,
    "tail_bank1_overlap_bytes": 14336,
    "drain_barrier": "nonaligned barrier.sync6,384; collective epochs0/6/0",
    "compiled_gate": {
      "path": "results/psplit_hybrid_s_static_review_initial_iter38.json",
      "sha256": "bdd8ae67d1206ecf870a5b17eb255fc8f5d76789fa996a61f650b0719dd86932",
      "status": "PASS_COMPILED_CAN_LAUNCH_41_ONLY_37_REMAINS_BLOCKED"
    },
    "initial_registers_per_thread": 168,
    "initial_owned_CTA_pool_registers": 64512,
    "requested_role_pool_registers": 57344,
    "producer_gpr_highest_including_implicit_by_dtype": {
      "fp16": 41,
      "bf16": 39
    },
    "wg0_gpr_highest_including_implicit": 124,
    "wg1_gpr_highest_including_implicit": 164
  }
}
```

Hybrid S BK32 41对SS BK32 36，以及Hybrid S BK32 41对RS-A BK64 39，是两个预声明比较轴。仅共享BK或BN配置不能自动证明受控因果关系；是否只改变A操作数路径由独立源码与compiled scope审核。所有性能数字只取本轮实际完整调用。native资格、partial-K回退与sameatom reference范围见本轮实际测试coverage/profile；fallback PASS不认证native。

本轮实际测得的native Mega36/Mega39/Mega40/Mega41 每次读取当前原始A/B，重新构造输入变换并完成七个P、固定U和epilogue，完整调用为一个kernel、零native矩阵workspace、无host descriptor encode。Lt1是pack→七batch leaf GEMM→combine的完整多launch控制，不能替代单launch目标。

Graph捕获固定地址，replay读取当前A/B内容并重做变换；三次fresh检查更换内容。setup/tuning/capture不计入稳态。event覆盖完整GPU工作，可能包含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，再取五组median；不是独立同步单调用CPU wall。clock设置按本轮环境记录，环境快照不认证组内clock。

环境记录：GPU=NVIDIA H100 80GB HBM3，SMs=132，host=g08，CUDA=12.4.99，CUTLASS=3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee；记录作用域见profile。

## 普通启动

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega41 Hybrid S BK32 BK32 64/192/192 min1 | bf16 | 12.5766 [12.3731,12.7216] | 29.2211 [29.2122,29.2269] | 0.430396 | 1.2252/0.0243 | 12.66345/29.30479 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_hybrid_s41_launch_iter38_full1024.csv:2) |
| Mega41 Hybrid S BK32 BK32 64/192/192 min1 | fp16 | 9.8800 [9.8543,9.8989] | 29.1462 [29.1340,29.1523] | 0.338980 | 0.2100/0.0282 | 9.96672/29.22517 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_hybrid_s41_launch_iter38_full1024.csv:3) |
| Mega39 RS-A BK64 reference, prior fastest BK64 64/192/192 min1 | bf16 | 12.2205 [12.2046,12.2308] | 23.1098 [23.1075,23.1155] | 0.528802 | 0.0980/0.0191 | 12.29847/23.18702 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter38_full1024.csv:2) |
| Mega39 RS-A BK64 reference, prior fastest BK64 64/192/192 min1 | fp16 | 9.9357 [9.9011,9.9521] | 23.0422 [23.0368,23.0482] | 0.431194 | 0.2394/0.0226 | 10.01235/23.12110 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_launch_iter38_full1024.csv:3) |
| Mega40 raw RS-A BK32 matched BK32 64/192/192 min1 | bf16 | 12.1347 [12.1039,12.1519] | 25.3709 [25.3579,25.4135] | 0.478293 | 0.2047/0.1069 | 12.20726/25.44874 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter38_full1024.csv:2) |
| Mega40 raw RS-A BK32 matched BK32 64/192/192 min1 | fp16 | 9.9094 [9.8992,9.9219] | 24.2275 [24.2170,24.2496] | 0.409016 | 0.1022/0.0609 | 9.98116/24.30026 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_launch_iter38_full1024.csv:3) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | bf16 | 12.1939 [12.1818,12.2065] | 23.4317 [23.4188,23.4549] | 0.520403 | 0.0910/0.0691 | 12.26766/23.52420 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter38_full1024.csv:2) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | fp16 | 9.9718 [9.9620,9.9816] | 23.6829 [23.5571,23.6884] | 0.421057 | 0.0874/0.3510 | 10.04415/23.75636 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_launch_iter38_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 12.2659 [12.2559,12.2916] | 19.3168 [19.2740,19.3494] | 0.634987 | 0.1314/0.1920 | 12.33720/19.39382 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter38_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.8826 [9.8689,9.9455] | 19.5030 [19.4997,19.5319] | 0.506719 | 0.3563/0.0775 | 9.95430/19.58261 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_launch_iter38_full1024.csv:3) |

## CUDA Graph

| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Mega41 Hybrid S BK32 BK32 64/192/192 min1 | bf16 | 11.2096 [11.2043,11.2278] | 28.7821 [28.7761,28.7972] | 0.389465 | 0.0960/0.0330 | 11.28791/28.87307 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_hybrid_s41_graph_iter38_full1024.csv:2) |
| Mega41 Hybrid S BK32 BK32 64/192/192 min1 | fp16 | 9.4058 [9.4004,9.4108] | 28.9120 [28.9024,28.9213] | 0.325324 | 0.0495/0.0292 | 9.48357/28.99667 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_hybrid_s41_graph_iter38_full1024.csv:3) |
| Mega39 RS-A BK64 reference, prior fastest BK64 64/192/192 min1 | bf16 | 11.2074 [11.2043,11.2180] | 22.8154 [22.7979,22.8205] | 0.491220 | 0.0608/0.0448 | 11.28562/22.89392 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter38_full1024.csv:2) |
| Mega39 RS-A BK64 reference, prior fastest BK64 64/192/192 min1 | fp16 | 9.5139 [9.4831,9.5560] | 22.7776 [22.7708,22.8321] | 0.417688 | 0.3385/0.1558 | 9.59229/22.85587 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa39_graph_iter38_full1024.csv:3) |
| Mega40 raw RS-A BK32 matched BK32 64/192/192 min1 | bf16 | 11.3552 [11.3458,11.3638] | 25.1514 [25.1389,25.1747] | 0.451475 | 0.0701/0.0710 | 11.43018/25.23290 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter38_full1024.csv:2) |
| Mega40 raw RS-A BK32 matched BK32 64/192/192 min1 | fp16 | 9.5290 [9.5178,9.5406] | 24.1002 [24.0829,24.1030] | 0.395390 | 0.1062/0.0416 | 9.61112/24.17402 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit_rsa_bk3240_graph_iter38_full1024.csv:3) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | bf16 | 11.2893 [11.2751,11.3341] | 23.0477 [23.0141,23.0712] | 0.489823 | 0.2352/0.1147 | 11.36066/23.12051 | FAIL finite | 0 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter38_full1024.csv:2) |
| Mega36 SS BK32 matched BK32 64/96/96 min1 | fp16 | 9.5610 [9.4498,9.5702] | 23.2822 [23.1861,23.2980] | 0.410655 | 0.5973/0.2243 | 9.63217/23.36328 | FAIL finite | 0 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_psplit36_graph_iter38_full1024.csv:3) |
| Lt1 打包一次（多launch对照） | bf16 | 11.2688 [11.2647,11.2858] | 17.0118 [17.0060,17.0132] | 0.662409 | 0.0855/0.0234 | 11.34201/17.08397 | FAIL finite | 46 | [CSV行2](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter38_full1024.csv:2) |
| Lt1 打包一次（多launch对照） | fp16 | 9.5712 [9.5639,9.5748] | 17.6339 [17.6307,17.6385] | 0.542772 | 0.0513/0.0200 | 9.64764/17.70874 | FAIL finite | 46 | [CSV行3](/share/home/liuyao/workspace/MegaFMM/results/small_packed_lt1_graph_iter38_full1024.csv:3) |

## 新41相对保留RS-A BK64参照39及匹配BK32对照40、36的自身耗时

| 模式 | dtype | 对照 | 对照 µs | 新41 µs | 对照/41自身延迟比 | 41自身耗时变化 | 对照/41 batchwall µs/call |
|---|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | Mega39 | 23.109760 | 29.221120 | 0.790858× | +26.445% | 23.18702/29.30479 |
| launch | bf16 | Mega40 | 25.370879 | 29.221120 | 0.868238× | +15.176% | 25.44874/29.30479 |
| launch | bf16 | Mega36 | 23.431680 | 29.221120 | 0.801875× | +24.708% | 23.52420/29.30479 |
| launch | fp16 | Mega39 | 23.042240 | 29.146240 | 0.790573× | +26.490% | 23.12110/29.22517 |
| launch | fp16 | Mega40 | 24.227519 | 29.146240 | 0.831240× | +20.302% | 24.30026/29.22517 |
| launch | fp16 | Mega36 | 23.682880 | 29.146240 | 0.812554× | +23.069% | 23.75636/29.22517 |
| graph | bf16 | Mega39 | 22.815361 | 28.782079 | 0.792693× | +26.152% | 22.89392/28.87307 |
| graph | bf16 | Mega40 | 25.151360 | 28.782079 | 0.873855× | +14.435% | 25.23290/28.87307 |
| graph | bf16 | Mega36 | 23.047681 | 28.782079 | 0.800765× | +24.881% | 23.12051/28.87307 |
| graph | fp16 | Mega39 | 22.777600 | 28.912001 | 0.787825× | +26.932% | 22.85587/28.99667 |
| graph | fp16 | Mega40 | 24.100161 | 28.912001 | 0.833569× | +19.966% | 24.17402/28.99667 |
| graph | fp16 | Mega36 | 23.282239 | 28.912001 | 0.805279× | +24.180% | 23.36328/28.99667 |

41对39比较保留的RS-A BK64参照（上轮已测最快）；41对40比较raw RS-A BK32；41对36比较SS BK32。hybrid_S_vs_raw_RS_BK32_axis_comparison实际保留4个轴metadata记录，hybrid_S_vs_SS_BK32_axis_comparison实际保留4个轴metadata记录；hybrid_S_vs_RS_A_BK64_axis_comparison实际保留4个轴metadata记录；三个比较的controlled_axis_claimed默认false，不重复添加性能表行。本生成器不单独归因动态occupancy、stall或traffic；收益判断使用SW自身耗时。缺失的新41不能由历史结果或其他binary补齐。

## 编译资源与实际检查

| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 41 | FP16 | 64×32 /32 | 384 | 86144 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |
| 41 | BF16 | 64×32 /32 | 384 | 86144 | 168 | 64512 | 64/192/192 | 57344 | 0/0/0 |

动态shared、ELF资源和spill只引用当前实际compiled metadata。候选预算或CPU预测不是actualfit；静态资源不代表实测occupancy、traffic或物理floor。最终对象和linked binary关系见独立编译审查。

本轮measured identity绑定的fresh compiled gate：status=PASS_COMPILED_CAN_LAUNCH_41_ONLY_37_REMAINS_BLOCKED，[psplit_hybrid_s_static_review_initial_iter38.json](/share/home/liuyao/workspace/MegaFMM/results/psplit_hybrid_s_static_review_initial_iter38.json)，SHA256 `bdd8ae67d1206ecf870a5b17eb255fc8f5d76789fa996a61f650b0719dd86932`。生成器仅检查绑定；CAN_LAUNCH内容及source/object/linked-bin时序由独立审计核验。

初始与最终证据时序/模型作用域按本轮实际metadata保留：

```json
{
  "scope": "Original CPU-only Hybrid S prototype and models are frozen inputs, with all candidate initial quotas hypothetical until a fresh compiled source/binary gate. Final budgets/name/initial allocation come from the frozen plan and actual gate. Chronological archived build snapshots preserve their actual exit/status without retrospectively certifying pending scopes; retained unsafe37 belongs to iteration34.",
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
    "41": {
      "actual_name": "megakernel_wgmma_psplit_hybrid_s_64x32",
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
      "dynamic_shared_bytes": 86144,
      "predicted_dynamic_shared_bytes": 86144,
      "static_shared_bytes": 1024,
      "bank_bytes": 43008,
      "tail_product_bytes": 57344,
      "tail_bank1_overlap_bytes": 14336,
      "drain_barrier": "nonaligned barrier.sync6,384; collective epochs0/6/0",
      "compiled_gate": {
        "path": "results/psplit_hybrid_s_static_review_initial_iter38.json",
        "sha256": "bdd8ae67d1206ecf870a5b17eb255fc8f5d76789fa996a61f650b0719dd86932",
        "status": "PASS_COMPILED_CAN_LAUNCH_41_ONLY_37_REMAINS_BLOCKED"
      },
      "initial_registers_per_thread": 168,
      "initial_owned_CTA_pool_registers": 64512,
      "requested_role_pool_registers": 57344,
      "producer_gpr_highest_including_implicit_by_dtype": {
        "fp16": 41,
        "bf16": 39
      },
      "wg0_gpr_highest_including_implicit": 124,
      "wg1_gpr_highest_including_implicit": 164
    }
  }
}
```

初始pending字段保持其初始身份，不因后续final compiled PASS而追溯认证；最终配置只取冻结plan。

CPU DAG实际归档exit=0，tests=3，[psplit_hybrid_s_cpu_dag_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_hybrid_s_cpu_dag_iter38.txt)。

CPU stage实际归档exit=0，tests=6，[psplit_hybrid_s_cpu_stage_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_hybrid_s_cpu_stage_iter38.txt)。

CPU几何与抽象模型实际归档exit=0，status=PASS_CPU_ONLY_HYBRID_S_DAG_AND_LOGICAL_SLOT_MODEL；[psplit_hybrid_s_cpu_state_model_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_hybrid_s_cpu_state_model_iter38.txt)。只检查相应抽象/源码模型范围，不认证GPU memory-order、forward-progress或最终配额的硬件ownership。冻结Hybrid S、planar和候选原件保护与旧的artifact-writing main是否调用按实际runner/stdout证据核查，不能由JSON status推断真实进程exit。

CPU几何与抽象模型实际归档exit=0，status=PASS_CPU_ONLY_LITERAL_PLANAR_GEOMETRY；[psplit_hybrid_s_cpu_planar_model_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/psplit_hybrid_s_cpu_planar_model_iter38.txt)。只检查相应抽象/源码模型范围，不认证GPU memory-order、forward-progress或最终配额的硬件ownership。冻结Hybrid S、planar和候选原件保护与旧的artifact-writing main是否调用按实际runner/stdout证据核查，不能由JSON status推断真实进程exit。

default：exit=0，2522 CASE /1893 PASS；[correctness_psplit_hybrid_s_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_hybrid_s_iter38.txt)。
selected41：exit=0，56 CASE /135 PASS；[correctness_psplit_hybrid_s41_targeted_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/correctness_psplit_hybrid_s41_targeted_iter38.txt)。
disabled37_cli：exit=1，0 CASE /0 PASS；[disabled_psplit_hybrid_s37_cli_iter38.txt](/share/home/liuyao/workspace/MegaFMM/results/disabled_psplit_hybrid_s37_cli_iter38.txt)。

新Graph configs/replays=36/108；全部Graph=330/990。Mega37继续disabled；结构PASS不能替代原始FP64数值认证。

## 原始精度与sanitizer

本轮strict实际30 gates：18 PASS /12 FAIL /nonfinite 0；24 native、6 Dense fallback。BF16/FP16原floor保持0.001/0.0001，门限保持3×Dense_eF+floor；回退PASS不认证native。

历史strict按各自source/bin身份保留；本轮执行计数只来自实际终止的stdout/exit，源码域计数不能作为执行结果。原始数值FAIL保留，Auto保持Dense，目标未宣布完成。

本轮实际执行作用域：

```json
"Only archived current-source runs are actual; blocked or absent scopes stay null. Full1680 is code scope, not an executed rerun. Historical source34 new36-only30, source35 new38-only30 and source36 new39-only30 and source37 new40-only30 remain separately bound."
```

实际归档3项CUDA Compute Sanitizer，其中0项已测作用域的exit/errors或race hazards/warnings非零或未记录；memcheck leak bytes=[0]。race/synccheck未测泄漏字段保持null，实际summary见profile；没有替Mega37执行sanitizer。

旧CUDA保留34个，历史证据1294个绑定，onsite-only 4个按现场SHA保留，不声称成为Git对象。本生成器填报实际原件；统计、来源、compiled gate及最终Git绑定由独立审计核验。
