# P53 premeasurement: independent full-K512 native leaves, followed by FP32 recombination

Written before P53 application, compilation or device measurement. Base is frozen,
fully validated P51. P52 is rejected and working sources/binary are restored to
P51 with a new 78-case process passing. P52's own six pre-MMA layouts and original
78 gates passed, but ordinary calls regressed BF16/FP16 41.548/41.999 percent and
Graph regressed 41.466/41.151 percent in independent ABBA. Its actual two-CTA/SM
capacity and 95 registers did not make it faster. P52 also changed MMA shape,
ring depth and B reuse; no single cause follows from that control.

The three recent controls P50/P51/P52 stayed below a 3 percent improvement. Before
this proposal, reviewed their own profiles and frozen P48/N1–N3 failure history,
and re-read primary CUDA12.4 Best Practices occupancy/thread-block guidance,
CUDA12.4 Programming Guide stream ordering/capture guidance, and PTX8.7 WGMMA
fence/commit/wait requirements:

- https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
- https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#streams
- https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations

P52's own first balanced BF16 NCU: prep16.832us/52regs/65536B/sharedlimit3;
consumer29.376us/95regs/82048B/register+sharedlimit2/issue32.668724percent;
zero local read/write sectors. P51's own consumer was17.472us/143regs/196736B,
limit1, issue31.610582percent. These are profiler-only measurements, excluded
from ordinary/Graph timing; higher occupancy/issue is not a speed guarantee or
a physical floor. Same-stream kernel ordering permits a leaf-to-recombination
boundary without host synchronization, but says nothing about its performance.

## Independent variable and costs

Retain P51 native m64n32k16 shapes, all seven main and eight cross dots, original
FP16/BF16, exact per-column inside, original FP64 AB and gates, p11/p8 low planes,
and full K512 accumulation/restoration per leaf. Launch one corrected leaf per
CTA: grid(16,8,7)=896 CTAs instead of 128 CTAs doing seven leaves sequentially.
Original grid128 is smaller than132 SMs: reducing its resource use alone cannot
create two simultaneously active CTAs per SM. This proposal exposes independent
leaves while retaining N32 rather than repeating P52's smaller-N operand loads.

Each CTA keeps only d[16]/r[16], four K128 chunks and two actual 49152B operand
slots. Shared98432B includes ready/free and alignment; verify static_assert and
actual compiled resources/capacity outside run/capture, after setting attributes.
Expected capacity two is a hypothesis. The first128 threads retain producer/idle
roles, last128 form one consumer warpgroup. No extra consumer groups or raw-input
fusion. The four persistent c[4][16] output fragments move out of this kernel.

Per-slot offsets stay Bhigh0, Blow16384, Ahigh32768, Alow40960. Original N32 TMA
boxes/swizzles/descriptor construction remain. p is blockIdx.z; each CTA has
sequence k=0..3 rather than a global28-sequence loop. Expected bytes/chunk are
24576 for p0/1,32768 for p2,40960 for p3,49152 for p4/5/6. Across all896CTAs,
nominal TMA payload remains138412032B and commands11264, the same as P51;
these are source box counts, not physical memory traffic measurements. More
CTAs, redundant setup, shorter lifetimes and two instead of four slots may help
or hurt. Four chunks are still accumulated in one FP32 fragment per dot; no K
split or reordering of a leaf's cross-dot accumulation is allowed.

Append seven Plane=512*512 FP32 leaf planes, 7340032B (7MiB), to the caller's
workspace after BaseWorkBytes=11804032B; total19144064B. Existing Work pointers,
all preparation geometry/math and historical allocations remain. compare.cu
and pow2.cuh remain byte-exact P51. pack.cuh changes only workspace constants;
corrected.cuh adds the separate leaf consumer and output kernel. No allocation,
hidden stream, input cache or host synchronization in run. Original descriptor
handling during Graph capture remains; descriptors are not reused across new
ordinary input pointers. Complete run now has prep, leaf and recombine launches.
Both ordinary and Graph timing must include all three and all 7MiB temporary
writes/reads plus final output; never report leaf-only timing as full cost.

## Recombination and ordering

Leaf storage is leaf[p*Plane+(n0+mn/64)*512+m0+mn%64]. After all leaves complete,
recombine<<<1024,256,0,caller_stream>>> writes the four output quadrants, one
thread per leaf coordinate. Preserve P51's scalar RN dependency graph exactly:

```
C11 = RN(P0 + P1)
t   = RN(P0 + P5)
u   = RN(t + P6)
C12 = RN(RN(t + P4) + P2)
C21 = RN(u - P3)
C22 = RN(u + P4)
```

Explicit __fadd_rn/__fsub_rn prevent reassociation. No low-low term is added or
existing cross term omitted. CPU model seed117089 tests262144 newly generated
signed/scaled/canceling leaf tuples and signed zeros:1048576 output bit checks,
zero errors; changed-order negative control differs34042 times. This is scalar
CPU arithmetic evidence only, not original-input matrix/WGMMA/GPU acceptance.

Keep two-stage ready/free initialization, four consumer-warp free arrivals,
producer wait-before-reuse, and wait1 then release previous k only for k>0.
After final wait0 release slot for k3; k2 was released after k3's wait1. Compiler
operand fences cover both fragments before first MMA and after final wait0;
hardware warpgroup_arrive precedes the first MMA. Each CTA owns one leaf, so
there is no inter-leaf register reuse. Same caller stream orders the recombine
kernel after every leaf CTA, avoiding grid-wide spin barriers or atomics.

## Required measurements before retention

Before first P53 MMA, require an own TMA/layout/slot reuse probe for both dtypes,
all896CTAs and four sequences, with six newly seeded synthetic configurations
(index low16, upper bits, fresh hash), new poisoned workspace and shared/output
guards. Suggested seeds116081+443*trial+(FP16?233:0). Use position-independent
swizzles and copy the P53 TMA producer verbatim. Check all high/required-low
values, every seven-plane output CLayout_64x32 coordinate once and all guards.
Expected six-case415236096 uint16 comparisons include CTA reuse, not that many
independent source values (5767168 unique native values per case). Generic probe
reads release the current slot without MMA: this does not certify WGMMA waits.

Also require an own pre-MMA GPU recombination probe with newly generated signed,
scaled, canceling leaf values, independent P51-order CPU oracle, all1048576
output coordinates and guards. Then original78/noNF/79872inside gates and
independent complete ordinary/Graph ABBA against P51. If retained, own fresh
12-pattern dual-stream/independent workspace/guard ordinary+Graph stress and
scoped memcheck/racecheck/synccheck are mandatory. No P51 or P52 certificate
transfers. If unhelpful restore P51 and recheck it. Own profile must include all
three kernels, and profiler timings stay excluded. All GPU tasks serial;
production/library/Auto unchanged, no formal iter371 and no goal completion.
