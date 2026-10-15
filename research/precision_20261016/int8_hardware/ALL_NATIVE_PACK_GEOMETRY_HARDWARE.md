# P50 all-plane spatial subdivision: own precision and complete-call evidence

P49 raw-tuple reuse alone regresses complete-call time and is rejected. P50
halves A tile rows64->32 and B128->64, keeping Cols128/32, all seven high/four
low planes, exact per-column inside, p11FP16/p8BF16, explicit RN transforms,
all seven main/eight cross dots, the P47 consumer, workspace and descriptors.
The unchanged cooperative grid288 has192 active packing CTAs versus P49's96.
B factor precomputation uses lane%8 to match eight loop iterations and remain
within1024inside entries; normal/fallback guard scope follows the smaller tile.
No input-derived caching, allocation or host synchronization in run.

Before any P50 MMA, own54fresh configurations with freshly poisoned workspace
compare all11nativeplanes bothsides and1024inside values against frozenP47:
311427072uint16 plus55296exponents, zero differences. A separate18fresh finite
edge configurations add103809024native comparisons plus18432exponents, also
zero. BF16 complementary large/subnormal, reversed and alternating64Kslab
patterns reach actual inside absmax130, with1024/1024/512 outside-normal
exponents. FP16 controls stay representable (maxabs19). These diagnostics
specifically exercise the changed B guard scope; they do not prove arbitrary
range equivalence or themselves certify complete matrix precision.

Own standard78 original rounded-input FP64 AB gates and79872 exact per-column
exponents pass. Max BF16/FP16 eF1.61320431617e-5 /2.16394614392e-5, printed
per-case errors matchP47. Independent four-process A/B/B/A measurements:

| mode | BF16 P47 us | BF16 P50 us | reduction | FP16 P47 us | FP16 P50 us | reduction |
|---|---:|---:|---:|---:|---:|---:|
| ordinary |28.897120475875|28.143240213375|2.60884%|28.852760314875|28.051479578125|2.77714%|
| Graph |28.8728404045|28.184719801|2.38328%|28.852360010125|28.3417203425|1.76984%|

Each mode312checks and319488exactexponents, all pass. Unlockedclocks/noCI;
positive small reductions are measured, not statistical significance or a
3percent breakthrough. Complete-call costs include current input preprocessing
and compute; ordinary includes descriptor setup, Graph descriptors capture-only.
Profiler-process timing arms are excluded.

Own fresh stress uses seed50957+401trial+211stream+101pattern,12patterns,
144configurations/1152interleaved ordinary+Graph calls, two independent
streams/workspaces,128float guards before/after output. Original FP64AB and
Dense-dependent gate unchanged, plus a stricter fixed floor BF16<=1e-3,
FP16<=1e-4. All pass with guards intact/noNF; maxBF16/FP16 eF
1.53744945035e-5 /2.09392400621e-5. The three added extreme patterns also pass
complete matrices (max7.18287135131e-6 /6.59769867479e-7). Stress Dense control
is historical cuBLASGemmEx; ordinary/Graph compare use production tunedDense.

Own memcheck/racecheck/synccheck terminal exit0, each48configurations/96calls,
all gates/fixed floors/guards pass, zero errors; racecheck zero warnings/hazards.
Scope kns=int8_research, CUDA12.8. These are its own results, not P47 transfers.
p50_validation.json/manifest freeze source, binary and terminal-log identity.
Retain P50 as the ordinary/Graph research reference; P47 historical fallback.

Own firstbalancedBF16 NCU17passes/kernel: prep18.464us,54regs/65536B, shared
limit3CTAs/SM and registerlimit4, issue14.653062%; consumer17.440us,
143regs/196736B, shared/registerlimit1, issue31.567722%. Both0localRWsectors.
BF16/FP16 build prep54/47regs,consumer143,zero stack/spill, noC7515/C7510;
actual runtime init3CTA/SM supports288 on132SMs. P50 prep issue is lower than
P49 despite faster complete calls: aggregate issue cannot establish a unique
causal explanation, phase cost, useful occupancy or physical floor.

After N1/N2/N3 rejects, P49 reject and P50 gains below3percent, reassessment
reviewed their frozen history, own P50 profile and the official CUDA12.4
BestPractices sections10.1/10.3/10.4:
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
More block-level parallelism can hide latency; higher occupancy does not by
itself guarantee speed. Actual compiled occupancy and cooperative capacity
must be queried. No physical floor or permission to drop exact scaling/dots.
This supports another controlled geometry experiment only after P50 acceptance.

Isolated CPU regeneration reproduces all four P49/P50 frozen files. Sources,
binaries, bit diagnostics, separate edges, ordinary/Graph ABBA and profile are
frozen in own manifests. Scope contiguous1024^3 nontranspose alpha1 beta0,
FP16/BF16 in/FP32 out, H100132SM. Production/library/Auto remain unchanged.
At about28us, the complete accuracy-corrected path still costs roughly2.8x
productionSW10us. The goal remains active; no production performance winner.
