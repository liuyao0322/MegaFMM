# P52 premeasurement: 64x16 native consumer with two actual TMA slots

Written before any P52 application/build/measurement. Base is frozen P51,
now accepted with its own12-pattern144/1152freshstress and scoped tools48/96
calls each (zeroerrors/racewarnings). Historical P50 remains fully validated. P51 ordinary
and Graph ABBA improve all four dtype/mode medians by1.88–2.63percent but
remain~27.4us versus productionSW~10us. All main terms, eight cross dots,
original dtypes, exact per-column inside and original FP64 reference/gates
are mandatory. Do not change numerical representation or producer geometry.

Reassessment reviewed P49/P50/P51 controls and profiles plus N1–N3/P48/history.
Own P51 firstbalancedBF16 profile: prep17.056us/52regs/65536B/sharedlimit3;
consumer17.472us/143regs/196736B/register+sharedlimit1/issue31.610582%; no
localRWsectors. Timings are profiler-only, not ordinary-call measurements.
P48 only shrank unused producer shared scratch; it did not test this consumer.
Earlier F13N16 was an INT8 packing mapping, not a native-correction consumer.
No unique bottleneck, elapsed-time decomposition or physical floor established.

Fresh primary sources reviewed before this proposal: CUDA12.4BestPractices
10.1/10.3/10.4 (resource/latency hiding tradeoff, no guarantee of higher occupancy
being faster) and PTX8.7 wgmma.fence/commit/wait ordering:
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations
Local official CUTLASS supports SM90_64x16x16_F32F16F16_SS and BF16 counterpart,
each eight FP32 accumulator registers, plus CLayout_64x16 in
third_party/cutlass/include/cute/{arch/mma_sm90_gmma.hpp,atom/mma_traits_sm90_gmma.hpp}.
These establish feasibility/ordering requirements, not expected speed or accuracy.

Change only corrected.cuh from P51. Keep pack.cuh, pow2.cuh and compare.cu
byte-exact. Introduce a consumer-local constexpr stage count2, leaving legacy
pack globals untouched. Native consumer N32->N16 reduces each four persistent
output fragments16->8, plus d/r16->8, preserving each scalar RN recombination,
fullK128 four chunks/leaf, step order[0,1,5,6,4,2,3], p11/8 restoration once.
Use m64n16k16 native WGMMA for both dtypes, eight explicit fragment arguments.
Consumer output tile physicalBcolumns64 x Arows16, grid32x8=256 CTAs versus128.
Keep256threads: tid0 TMA producer, tid128..255 one consumer warpgroup.

Keep shared B high[0,16384), B low[16384,32768), A high[32768,36864),
A low[36864,40960). Per-slot bytes40960, two slots81920 plus ready/free arrays,
aligned Shared expected82048B (verify static_assert). A TMA4D box height32->16,
B TMA3D box64x128 unchanged. CuTe lb K_SW128 shape16x128 and its N16descriptor
replace32x128; la MN_SW128 shape64x128 unchanged. Descriptor offset unit16B.
Every slot40960B aligned128; A low offset36864 aligned128. Actual request82048B.
Expected two resident CTAs/SM requires actual compiled regs/shared API query,
logged outside run/capture after setting dynamicshared/carveout attributes.
Do not infer two from source alone; compiler could exceed register allocation.

TMA transaction expected bytes: p0/1 20480; p2 24576; p3 36864; p>=4 40960.
Still88TMAcommands/CTA, but two smaller-N CTAs duplicate B loads versus oldtile.
More active consumer warpgroups and lower fragment footprint may hide latency;
duplicated B traffic, extra CTAs and fewer stages may erase benefits.
No speed assertion from static resource arithmetic.

Preserve ready/free initialization, producer wait-before-slot-reuse, count4free
arrivals, wait1 then retirement of the previous sequence, finalwait0 then last
retirement. With two slots, producer can fill0/1, waits for0; consumer issues0/1,
wait1 makes0 complete and releases0, enabling seq2. Finalwait0 completes both
fragments before scalar access. Preserve compiler operand fences and hardware
warpgroup_arrive before first MMA of each leaf, including register read->reuse;
CuTe fence_operand is not a hardware ordering substitute. No source simplification
that releases a slot while any asynchronous MMA may read it.

Before first new MMA require own synthetic-native operand/TMA layout diagnostic,
both dtypes, all256CTAs and28sequences including actual two-slot reuse. Use new
plane-distinguishing uint16 patterns, compare every loaded high/required-low
value to expected global indices from frozen workspace; use position-independent
CuTe swizzle layouts for generic shared reads (N1 preflight caught flagged
byte->element index misuse). Expected B operand a(m,k)=hb[p*Plane+(k0+k)*512+
(m0+m)]; A operand b(n,k)=ha[p*Plane+(n0+n)*512+(k0+k)]. Respect current operand
swap: shared a is high_b and shared b is high_a. Check all offsets/descriptor
boxes/output CLayout_64x16 bijection and guards. This certifies layout only,
not WGMMA numerical/synchronization acceptance. Do not count CTA-reused values
as independent source values. Archive preflight failures, no MMA before zeroerrors.

Then own original-input78/noNF/79872exactinside gates, complete ordinary+Graph
independent ABBA against frozenP51, own newly seeded12pattern fresh dual-stream/
workspaces/guards and scoped memcheck/racecheck/synccheck if retained. All current
preprocessing/ordinary descriptors/compute in timing; no cached inputs or hidden
streams/allocations/hostsync. Keep performance rejection and restore P51 when
unhelpful. All GPU tasks serial; production/library/Auto unchanged, no iter371.
