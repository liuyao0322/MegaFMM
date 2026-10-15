# N1 native raw-input fusion: premeasurement contract

Written before the first N1 CUDA build or GPU execution. This standalone
research candidate leaves P47 and production untouched. Frozen numerical
screen M14 and `../int8_hardware/FUSED_NATIVE_DESIGN.md` motivate K64 streaming
recombination, but neither predicts hardware correctness or latency.

Scope: contiguous original FP16 and BF16 1024-cubed inputs, FP32 output, exact
1024 per-column inside exponents, all seven main and eight cross products.
Residual native scales are p11/p8; restore once per K64 corrected leaf. Product
order is 0,1,5,6,4,2,3, updating four persistent output fragments over eight K64
chunks. Preserve explicit RN transforms and compare original rounded inputs to
FP64 AB with the unchanged dense-relative gate and no nonfinite outputs.

128 cooperative CTAs of 256 threads compute eight exact inside columns each,
then grid-sync. Require measured cooperative resident capacity >=128. A producer
warpgroup loads original quadrants through two current-pointer 2D TMA maps into
two 49152-byte raw slots. Each raw chunk generates seven native high/low leaf
slots, each 24576 bytes, double buffered. Structure including barriers/alignment
is 147584 bytes. Workspace is 4096 bytes for inside exponents. Calls create both
current-pointer descriptors and launch the complete kernel; no allocations,
input cache, or host synchronization. Graph replay includes the captured kernel;
descriptors are created during capture, as for P47.

Raw TMA totals per CTA: 8 chunks x 8 commands =64 commands and393216 nominal
bytes, compared to P47's88 commands and1081344 operand bytes. Native instruction
count remains480. These source counts omit physical traffic and cannot establish
speedup. Duplicate transforms across output CTAs, raw shared traffic, repeated
scale conversions and56 per-leaf wait-all handoffs may dominate. First N1 uses
wait0 after every corrected leaf; no overlapping accumulator reuse is attempted.

Barrier contract: raw-ready count1 + expect49152 for each TMA transaction phase;
all producer threads wait before generic reads. Producers synchronize before raw
slot refills. Each of128 writers issues shared async-proxy fence before arriving
leaf-ready(count128). Consumer waits ready, orders register access with WGMMA
fence and fences both fragments, executes all native MMAs, commits and wait0,
then fences operands before scalar reads. Each of four consumer warp leaders
arrives leaf-free(count4) after warp synchronization; all producers wait free
before reusing a leaf slot. Initial regular shared scratch is proxy-fenced before
TMA reuse. Separate diagnostic kernel instantiation consumes slots with ordinary
reads and the same lifetime barriers, without executing MMA.

Pre-MMA gate: compare all used high/low native uint16 values against frozen P47
planes for54 fresh cases: 9 patterns x3 trials x2 types, seed103001+359*trial+
67*pattern. Each case compares69206016 values (including CTA operand reuse) and
1024 exponents. All values must match bitwise. This diagnostic is distinct from
FP64 matrix correctness and the eventual MMA sanitizer evidence.

Then run independent ordinary and Graph standard78-case suites and full-call
measurements. Retention requires independent paired P47 comparison, fresh
stream/workspace/guard cases and scoped memcheck/racecheck/synccheck. Reject a
slower candidate rather than weakening original precision or excluding terms.

Primary synchronization reference: NVIDIA PTX ISA12.8 WGMMA fence/wait and
fence.proxy.async.shared::cta instructions:
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html
The source design and expected traffic above are our inferences, not claims from
that reference. P47 remains the accepted research reference until full acceptance.
