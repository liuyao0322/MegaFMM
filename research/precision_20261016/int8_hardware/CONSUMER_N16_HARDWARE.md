# P52 native64x16/two-slot consumer: accuracy passes, performance rejected

P52 follows CONSUMER_N16_DESIGN.md, from fully validated P51. Consumer N32->N16,
256CTAs, eight-value FP32 fragments, two40960B actual TMA slots, Shared82048B.
pack/compare/pow2 byte-exactP51. Both original dtypes, exact inside, all seven
main/eight cross dots, fullK128/fourchunks per leaf, nativep11/8 and explicit
RN recombination/fences/retirement retained. No production/library edits.

Initial pre-MMA probe compile failed2 at the new host capacity-print helper's
unspaced variable-template ternary. No binary/device/MMA execution. Archived
initial sources/probe/builddriver/log/manifest in preflight_p52_01; original
frozen proposal/generator preserved. Retry changes whitespace only in the host
ternary, via generate_consumer_n16_preflight_fix.py, and compiles normally.

Own pre-MMA6 synthetic configurations (two dtypes, low16index/upperindex/fresh
hashed planes) cover256CTAs*28sequences*two slots. All required high/low operands,
shared/workspace guards and complete output-coordinate coverage pass.
692060160uint16comparisons include CTA reuse, not692million independent values;
each workspace has5767168unique native values. 6291456output coordinate checks.
Diagnostic producer copied verbatim from candidate, generic reads retire each
current slot; no MMA, so it is not a WGMMA wait-order or sanitizer certificate.

Own ordinary78 original rounded-input FP64AB gates, noNF and79872exactinside
exponents pass. MaximumBF16/FP16ef1.61320431617e-5 /2.16394614392e-5; printed
per-case errors matchP51, without proving full-output bit equality. Build
prep52/48regs, consumer95regs bothtypes, zero stack/spill/noC7515/C7510.
Actual runtime consumer2CTA/SM on132SMs,82048B; cooperativeprep3/grid384.

Independent four-process A/B/B/A complete calls (us), eachmode312passinggates/
319488exactexponents, same deterministic78 configurations repeated inprocesses:

| mode | BF16 P51 | BF16 P52 | FP16 P51 | FP16 P52 |
|---|---:|---:|---:|---:|
| ordinary |27.3965599535|38.779400348625|27.492360591875|39.038880825125|
| Graph |27.612559795375|39.06247997275|27.6344804765|39.00627994525|

Ordinary regresses41.55%/42.00%, Graph41.47%/41.15%. Unlocked clocks/noCI.
Reject P52; no own fresh-stream/guard or sanitizer acceptance, no P51 transfers.
All profiler-process timing arms excluded from these measurements.

Own firstbalancedBF16 NCU17passes/kernel: prep16.832us/52regs/65536B,
sharedlimit3/registerlimit4/issue14.638310%; consumer29.376us/95regs/82048B,
shared+registerlimit2/issue32.668724%. Both0localRWsectors. These profiletimes
are diagnostic. Capacity/aggregate issue increase does not establish useful
throughput or guarantee lower full-call time. N16 doubles CTA/MMA-instruction
count relative to N32 while keeping mathematical FLOPs, and whole-grid nominal
TMA payload grows138412032->230686720B (5/3); these are source counts, not physical
traffic measurements. Tile shape, ring depth and reuse change together, so no
unique cause or physical-floor claim follows.

Restored four working sources from committedP51 (2a1ae63) and compare binary
byte-exactP51. New process ordinary78gates passes. P52 evidence frozen separately.

After P50/P51/P52 fail to exceed3percent, reassessment read ownP52profile,
P48/N1–N3/P49–P52 history, fresh primary CUDA12.4BestPractices10.1/10.3/10.4
and ProgrammingGuide3.2.8.5 streams:
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#streams
Same-stream command order permits an explicit leaf-result/recombination boundary
without host synchronization; it does not predict a speedup. Next controlled
proposal returns to N32, exposes seven leaf CTAs per outputtile and includes all
extra global intermediates/recombination/launches in complete-call timing.
The goal remains active; P51~27.4us versus productionSW~10us, no performance winner.
