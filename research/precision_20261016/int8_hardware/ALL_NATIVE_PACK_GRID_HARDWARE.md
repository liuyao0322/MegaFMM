# P51: smaller all-native packing tiles and a complete384-CTA grid

P51 applies ALL_NATIVE_PACK_GRID_DESIGN.md after P50 own complete acceptance.
A32x128->16x128/B64x32->32x32, PT16/8->8/4, actual packingCTAs192->384.
Cooperative/joint grid384 covers B0..255,A256..383. First128CTAs still compute
all1024exactinside exponents, all384CTAs participate in the unconditional
barrier. B factor lane%4 matches four loop rows; standalone component grids
128A/256B are necessary for coverage. Compare's diagnostic grids are adapted;
originalFP64reference, gates and full-call timing methods unchanged.
Consumer andpow2 byte-exactP50, all RN transforms/nativep11FP16/p8BF16 and
all seven main/eight cross dots remain. No production sources/library edits.

Own pre-MMA54fresh poisoned-workspace configurations compare311427072native
uint16 values plus55296inside exponents against frozenP47: zero differences.
Additional18fresh finite edge configurations compare103809024nativevalues+
18432exponents, zero errors. BF16 actualabsinside130, outside-normal1024,
1024,512 for high/subnormal, reversed and alternating32Kslabs respectively;
FP16 representablecontrols maxabs19/outside0. Packing diagnostics are separate
from matrix precision/tools and do not prove arbitrary range equivalence.

Build BF16/FP16prep52/48regs, consumer143regs, zero stack/spill, noC7515/C7510.
Actual runtime queries report3CTA/SM on132SMs, capacity396>=newgrid384.
Own ordinary78 original rounded-input FP64AB gates plus79872 exactinside
comparisons pass; printed per-case eF matchesP50. MaxBF16/FP16 eF
1.61320431617e-5 /2.16394614392e-5. Independent four fresh process A/B/B/A:

| mode | BF16 P50 us | BF16 P51 us | reduction | FP16 P50 us | FP16 P51 us | reduction |
|---|---:|---:|---:|---:|---:|---:|
| ordinary |28.12879967675|27.3898398875|2.62706%|27.967040062|27.339400053|2.24421%|
| Graph |27.973719596875|27.405320406|2.03190%|28.18163967125|27.650880813625|1.88335%|

Eachmode312 originalchecks/319488exactinside comparisons allpass. Unlocked
clocks/noCI; no statistical significance or3percent breakthrough claimed.
Complete calls include preprocessing/compute and ordinary descriptor setup;
Graph descriptors capture-only. Profiler-process timing arms excluded.

Own fresh12-pattern stress seed51971+431trial+223stream+109pattern:
144configurations/1152interleaved ordinary+Graph calls,2independentstreams/
workspaces,128float guards before/after output, allpass/noNF. Includes actual
fullFP64matrix complementary high/subnormal, reversed and alternating32Kslab
cases. Original Dense-dependent gate plus stricterBF16<=1e-3/FP16<=1e-4
fixedfloors enforced. MaxBF16/FP16eF1.53057462997e-5 /2.08831273132e-5.
Stress Densecontrol historicalcuBLASGemmEx; ordinary/Graph productiontunedDense.

Own memcheck/racecheck/synccheck terminalexit0, each48configurations/96calls,
all gates/fixedfloors/guards pass, zeroerrors; racecheck zerohazards/warnings.
CUDA12.8 scopekns=int8_research. Ownp51_validation audit passes with source,
binary and log hashes; no certificate transfer. Retain P51 ordinary/Graph
research reference; P50 remains a fully validated historical fallback.

Own firstbalancedBF16 NCU17passes/kernel: prep17.056us/52regs/65536B,
sharedlimit3/registerlimit4/issue15.787778%; consumer17.472us/143regs/196736B,
sharedlimit1/registerlimit1/issue31.610582%; both0localRWsectors. Profiletime
excluded; aggregate issue/wave/stall metrics do not prove a unique bottleneck
or physical floor. Source and binary identities are frozen in own manifests;
isolated CPU generator reproduces all four frozen files.

Stall reassessment after P49/P50/P51 (<3percent) reviewed their own profiles,
N1–N3/P48 and consumer history, fresh primary CUDA12.4BestPractices10.1/10.3/10.4,
PTX8.7 WGMMA fence/commit/wait ordering and local official CUTLASS64x16 shapes:
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations
History has no measured native-correction64x16 consumer; earlier F13N16 refers
only to INT8 packing subtiles. A later consumer tile/ring resource control is
separate from merely reducing unused producer scratch (P48), and must retain
hardware fences, safe slot retirement, own operand layout/matrix/tool checks.
No next candidate applied at this checkpoint. Current accuracy-corrected path
~27.4us remains~2.7x productionSW~10us; overall goal remains active.
