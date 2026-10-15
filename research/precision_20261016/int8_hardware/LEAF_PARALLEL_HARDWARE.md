# P53 hardware: full-K512 leaves in parallel, rejected performance

P53 is rejected; current research reference remains fully validated P51. Sources
and binary are restored byte-exact to P51 and a new process passed all78 original
FP64 gates and79872 exact per-column exponents. Production/library/Auto unchanged;
the precision-without-performance-loss goal remains active. Scope is contiguous
non-transposed1024^3,alpha1,beta0,original FP16/BF16,FP32output.

## Controlled change and own evidence

Frozen LEAF_PARALLEL_DESIGN.md/proposal/generator preceded application. Generated
four sources matched isolated CPU hashes exactly. Relative to P51, retain native
m64n32k16/fullK512 accumulation and restoration per leaf, exact original inside,
p11FP16/p8BF16 lows and all seven main/eight cross dots. Each896CTA computes one
leaf with two49152B slots, Shared98432B, then writes seven FP32 leaf planes into
an additional7MiB caller workspace. The same-stream FP32 output kernel preserves
P51's RN dependency graph. Whole run includes preparation, ordinary descriptors,
leaf writes and the extra recombination kernel/reads. No allocation/hostsync/cache
in run. compare.cu/pow2.cuh unchanged; pack change only workspace constants.

Before first P53 MMA, own probe executed six newly seeded synthetic configurations
covering both dtypes/all896CTAs/four K128 sequences/two-slot reuse. Verbatim P53
TMA producer, generic position-independent swizzle reads checked415236096 native
uint16 values including CTA reuse (5767168 unique workspace values per case),
11010048 seven-plane output coordinates and shared/workspace/output guards with
zero errors. Generic reads retire the current slot without executing MMA, so this
is not a WGMMA wait-order or sanitizer certificate. Runtime capacity was two
CTAs/SM for both original dtypes, SMcount132.

Own separate GPU recombination probe used three fresh host-generated signed,
scaled, canceling leaf arrays and signed zeros, seeds118097/118546/118995. An
independent CPU oracle directly updates four outputs in P51's original leaf order.
All3145728 output bit comparisons, guards and5505792 unchanged leaf/input-guard
values pass, no nonfinite. Changed C12 addition order differs22524/22407/22554
outputs. Earlier frozen CPU model seed117089 also passed1048576 output bit checks.
These scalar tests do not establish bit equality of complete matrix outputs.

Own standard78 original-input FP64 gates and79872 exactinside checks pass. MaxeF
BF16/FP16=1.61320431617e-5/2.16394614392e-5; each printed per-case eF matches P51.
No complete matrix-output bitwise equality claimed. Build uses consumer74regs,
recombine18regs, prep52BF16/48FP16, zero stack/spill and noC7515/C7510 warnings.

## Independent complete calls

Each mode has four fresh ABBA processes (P51/P53/P53/P51),312 original gates and
319488 exactinside exponents, all pass. These repeat78 deterministic numerical
configurations; no newly seeded stress or sanitizer acceptance is transferred.
All complete timing arms are retained, unlocked clocks, no confidence interval.

| Mode | dtype | P51 us | P53 us | P53 slower |
|---|---|---:|---:|---:|
| ordinary | BF16 |27.676800251|41.052640915|48.3287%|
| ordinary | FP16 |27.467400074|41.1387996675|49.7732%|
| Graph | BF16 |27.5353598595|37.3694400785|35.7144%|
| Graph | FP16 |27.3442800045|37.463359356|37.0062%|

No own fresh dual-stream/guard full-matrix stress or memcheck/racecheck/synccheck
acceptance for this rejected candidate. Probe guards have the narrower scope above.

## Own profile and interpretation

Own first balancedBF16 full run, all three kernels profiled with17passes each:

| Kernel | profiler us | registers | dynamic shared B | resource limit | issueactive |
|---|---:|---:|---:|---|---:|
| cooperative preparation |16.992|52|65536|shared3,register4 CTA/SM|14.578428%|
| corrected native leaves |23.296|74|98432|shared2,register3 CTA/SM|25.172080%|
| FP32 recombine |6.240|18|0|shared32,register10 CTA/SM|18.315580%|

All three measured zero local read/write sectors. Profiler process times and its
other timing arms are excluded from ordinary performance; their sum is not a
decomposition of the ABBA call time. Source nominal TMA payload138412032B and
11264commands over the whole grid match P51, but scheduling, stage depth and
temporary outputs change together. This does not identify a unique slowdown cause
or establish a physical floor. It rejects the hypothesis that exposing one full
leaf per small N32 CTA with fewer persistent fragments pays for these added costs.

After three sub3percent controls P51/P52/P53, reviewed their own profiles plus
P48/N1–N3/frozen history and newly searched/read primary NVIDIA CUTLASS Efficient
GEMM (threadblock tiles, register/reuse tradeoff, software pipelines, Hopper warp
specialization) and CUDA12.4 Hopper Tuning Guide. Larger tiles can improve reuse
while reducing concurrency; no speed guarantee follows. The documented Hopper
resource capacities are bounds, actual compiled capacity remains mandatory.

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html

Frozen sources/results/manifests preserve identity; p53_hardware.json records exact
scope. Future shapes/schedules need their own pre-MMA layout and original gates,
independent ordinary/Graph comparisons, and fresh streams/tools before retention.
