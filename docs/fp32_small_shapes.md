# FP32 256³ / 768³ 配置与验收结果

默认注册表按原顺序保留旧11项配置，追加两项带任务波次上限的模板配置。Iteration188已通过固定验收门槛。

| M×N×K | 旧公开入口 μs | 新公开入口 μs | 旧/新 | 新 cuBLASLt/SW |
|---|---:|---:|---:|---:|
| 256×256×256 | 9.196 | 8.150 | 1.1283 | 1.0040 |
| 768×768×768 | 33.462 | 32.179 | 1.0399 | 1.0210 |

直接配置复测（三个独立进程；每进程每配置5×100普通完整调用）：

| Shape | 旧配置范围 μs | 新配置范围 μs | 同进程旧/新范围 | 胜过旧配置 / Lt的组数 |
|---|---:|---:|---:|---:|
| 256³ | 9.184–9.236 | 7.991–8.033 | 1.1485–1.1498 | 15/15 ; 15/15 |
| 768³ | 33.742–33.905 | 32.004–32.168 | 1.0540–1.0548 | 15/15 ; 15/15 |

表中为调优后的普通完整调用。保留原20warmup、5×100计时、3次fresh原输入FP64检查、2组轮换输入和8GiB预算。首次setup和tune成本在完整审计中单列；公开入口数据与直接配置复测分别报告。

| Configuration | Tile / BK | Thread tile | Warp grid | ComputeK / GroupM | PDL / Zigzag / Persistent | Max task waves |
|---|---|---|---|---|---|---:|
|256 candidate11|32×8 /32|2×4|1×1|32 /1|0 /true /true|1|
|768 candidate12|128×32 /32|8×8|2×1|16 /4|2 /false /false|2|

256 uses cooperative packing and fused seven-product output. Its task grid increases32→64CTA on132SM; shared memory86144→71808bytes, registers72→70. The only device helper extension is float2 A shared loads for fusedTM2. Other fragment widths retain the original float4 implementation.
768 retains DirectB materialized leaves and8×8thread reuse. Its leaf grid increases126→252CTA, shared memory49280→41088bytes and registers remain117. PDLmask2 lets combine start early and wait for leaves before reading products. Reported kernel intervals overlap and include waiting; do not sum their durations or describe reconstruction arithmetic as occurring before the dependency completes.

TuningConfig<Kernel,MaxTaskWaves> controls candidate enumeration only. Task count is ceil(ceil(M/2)/BM)×ceil(ceil(N/2)/BN), multiplied by7 for materialized leaves. The limit compares tasks with SMcount×MaxTaskWaves. Kernel legality, cooperative capacity checks, per-call address binding, packed fallback and caller workspace/stream remain intact. The existing tuner makes the final measured selection.
On this132SMdevice and78JSONshapes, new candidates participate on8shapes (four valid plus four known precision rejects). Other70shapes retain exactly the old candidate sets.

Three independent packed5shape processes:195rows /585fresh FP64 checks, all three frozenSWbitwise controls passed. Target repeat gates: {'256x256x256': {'candidate_id': 11, 'old_over_new': [1.148477448996797, 1.1498227603006586, 1.1492871939179319], 'paired_wins': 15, 'lt_group_wins': 15, 'stable_gain': True}, '768x768x768': {'candidate_id': 12, 'old_over_new': [1.0548105488087978, 1.0540064009022323, 1.0542824541149172], 'paired_wins': 15, 'lt_group_wins': 15, 'stable_gain': True}}
Full78paired public sweep:74valid per pool and same4precision rejects; Lt wins={'old': 23, 'new': 24}; geometric mean old/new=1.001943; >3%regressions=[].
全78公开入口为单轮配对结果，不代表74项均通过独立重复性能认证。256/768及指定邻近和对照shape另有三个独立packed进程复测。
Two complete clean builds each passed7CTest;29oldleafbodies unchanged and all33current bodies instruction-identical187, with noLDL/STL. Six complete packed-path sanitizer suites and four actual-route NSYS captures passed.
Iteration187 unbounded default13 was rejected after513×8191×1025 switched old9→old8 and regressed3.9%. Its evidence is preserved separately. The bounds are a conservative tuning scope restriction, not a universal repair for timing noise.

Sources: [configuration registry](../solution/sw/fp32/sw_fp32_batch.h), [FP32 kernel](../solution/sw/fp32/sw_fp32_batch.cu), [public candidate enumeration](../solution/core/plan.cpp), [full comparison](../results/fp32_small188/report.md), [audit](../results/fp32_small188/audit.json).
