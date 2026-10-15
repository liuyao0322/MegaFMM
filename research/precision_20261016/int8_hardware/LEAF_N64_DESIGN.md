# P54 premeasurement: N64 native shape within the independent full-leaf pipeline

Prepared before any P54 application/build/device execution. P51 remains the fully
validated working reference; P52/P53 are rejected and P51 is restored byte-exact
with their own new-process78-gate checks. P53's all896CTA layout and scalar GPU
recombination probes pass, original78 gates/exactinside pass, but independent
ordinary calls regress48.329/49.773percent BF16/FP16 and Graph35.714/37.006percent.
Its own profile is prep16.992us/52regs,leaf23.296us/74regs/98432B/sharedlimit2,
recombine6.240us/18regs; zero local read/write sectors in allthree. Profiler time
is excluded and its sum is not an ABBA latency decomposition. No unique cause or
physical floor follows from stage/grid/output changes together.

After the three sub3percent controls P51/P52/P53, reviewed own P53 three-kernel
profile, P51/P52 and P48/N1–N3 failure history, and freshly searched/read primary
NVIDIA CUTLASS Efficient GEMM (2026-09-21) and CUDA12.4 Hopper Tuning Guide:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html

Larger tiles can reduce duplicate fetches while increasing register/shared use
and limiting concurrency; higher occupancy alone is not the target. These guides
establish tradeoffs, not an expected speedup. Local official CUTLASS provides
SM90_64x64x16_F32F16F16_SS and BF16 counterpart with32FP32registers, plus
CLayout_64x64. Historical native direct-dense64x64 results and earlier INT8/raw
fusion tiles do not test this corrected fullK512 seven-leaf pipeline. Search of
the current native hardware generators/design records found no such N64 control.

Only corrected.cuh changes from frozen P53; pack.cuh/compare.cu/pow2.cuh and the
scalar recombine kernel stay byte-exact P53. Retain its three complete launches,
7MiB additional leaf storage (total workspace19144064B), both original dtypes,
exact per-column inside, nativep11FP16/p8BF16lowplanes, all seven main/eightcross
dots, fullK512 per leaf/fourK128chunks and original FP64 AB/gates. Do not split K,
drop terms, change quantization or introduce stream/alloc/input caching.

Change native m64n32k16 to m64n64k16, d/r16->32 registers and output mapping to
CLayout_64x64. One full leaf per CTA, grid(8,8,7)=448 versus P53's896;256threads,
first128 producer/idle and last128 one consumerwarpgroup unchanged. Each slot is
65536B: Bhigh0,Blow16384,Ahigh32768,Alow49152; two actualslots and ready/free
give Shared131200B. la MN_SW128 shape64x128/B TMA box64x128 stays; lb K_SW128
shape64x128 replaces32x128 and A4D box height64 replaces32. Descriptor units16B;
Alow delta16384B replaces8192B. Request131200B means a one-CTA/SM shared bound,
but actual compiled query outside run/capture remains mandatory.

Expected transaction bytes per chunk: p0/1=32768,p2=49152,p3=49152,p4/5/6=65536.
Wholegrid nominal TMA payload92274688B (two thirds of P51/P53),5632commands and
30720 native WGMMA instructions (half P53). Each N64 instruction computes twice
as many outputs; mathematical seven-main/eightcross work is unchanged. These are
source counts, not measured physical traffic or latency. Fewer CTAs/commands and
more reuse may help, but larger fragments, one resident CTA, ring2 and the unchanged
7MiBleaf/recombine/extra-launch costs may lose again. N64 versus N32 is a geometry
control that necessarily changes reuse/resources/concurrency; do not uniquely
attribute any resulting change to Tensor Core throughput.

Preserve ready/free phases,4free arrivals, wait1-releaseprevious/finalwait0,
both operand/compiler fences and hardware warpgroup_arrive before first MMA.
Preserve leaf restore once after completeK512 and byte-identical explicit scalar
recombine dependency graph. No additional output-fragment lifetime is introduced.

Before any P54 MMA require own six new synthetic native plane/TMA layouts, all448
CTAs/foursequences/twoslotreuse, shared/workspace/seven-plane coordinate guards,
position-independent swizzle reads and verbatim P54 producer. Suggested fresh
seeds119101+457*trial+(FP16?239:0). Expected276824064uint16 comparisons acrosssix
cases include CTAreuse;5767168 unique native values per case,11010048leafcoordinate
checks. This generic no-MMA diagnostic does not certify WGMMA waits. Scalar output
kernel is source-identical but an own new seed GPU recombination/guard check is
still required before first new MMA; do not relabel P53's probe as P54 evidence.

Then own original78 gates/noNF/79872exactinside and complete independent ordinary
and Graph comparisons. Measure P53/P54 to isolate the geometry control; compare
P54 with retained P51 independently before considering retention. All preparation,
ordinary descriptors, leaf writes/reads and output kernel stay in full costs.
If useful versus P51, own fresh12pattern dualstream/workspaces/guards and scoped
memcheck/racecheck/synccheck before retention. Own three-kernel profile for diagnosis;
all GPU work serial. Restore P51 if unhelpful. Production/library/Auto unchanged,
no formal iter371, goal active. Current evidence is CPU proposal generation only.
