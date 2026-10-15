# N2: reuse all balanced raw values within one K64 chunk

Premeasurement contract, after N1's independent ordinary/Graph rejection and own
first-balanced-BF16 profile. Preserve N1's original types, exact inside stage,
all seven main/eight cross dots, native p11/p8 correction and K64 streaming FP32
recombination. P47 remains the accepted research reference and production stays
unchanged. This changes shared storage and producer/consumer granularity; it
cannot borrow N1/P47 validation.

N1 repeatedly loads four raw values and two exponents, scales all four, and
computes transforms for every one of seven products. Its measured complete
call is ~615–616us versus same-run P47~28.5–28.8us. Its own BF16 NCU capture
has5746787 shared-load wavefronts and16.230425percent issue active, zero measured
local sectors. These aggregate counters do not isolate causal time fractions.

N2 producer loads/balances each four-quadrant tuple once and emits all seven
native high and four native low planes with compile-time plane indices. Each
K64 pack is11*(2048+4096)*2=135168B. One49152B raw slot plus one pack and
barriers/alignment gives184448B shared. Both shared slots are independently
reused: after all producer raw reads finish, a128-thread producer barrier permits
loading the next raw chunk while TC consumes the current complete pack; the
next transform waits until all four consumer warps release the entire previous
pack after all seven leaf MMAs complete. No transform/TC overlap across packs
is promised. All writers proxy-fence before ready(count128), TC warp leaders
arrive free(count4) after warp sync. Raw ready count1 has expect49152 each phase.
Inside scratch reuse keeps the initial proxy fence/grid synchronization.

One ready/free exchange per chunk replaces seven; each leaf still commits/wait0
before CPU reads and uses both compiler operand fences and hardware WGMMA fence
before next reuse. Descriptors use packed plane offsets, original swizzled CuTe
layouts for MMA and element-domain swizzle conversion for generic shared reads
and writes. Raw input TMA remains64 commands/393216nominalbytes per CTA and native
MMA480instructions. Producer raw loads/scales can fall by up to seven at source
level, but the larger pack, unhidden transformation phase, shared stores and
instruction scheduling may erase gains. This is a hypothesis, not a speed claim.

Own pre-MMA layout diagnostic:54fresh cases,9patterns*3trials*2types,
seed104003+367*trial+71*pattern,69206016 used native uint16 comparisons per case
including CTA reuse and1024exponents. Require bitwise zero errors before any
N2 MMA. Then own original-inputFP64 gates/exactinside/full ordinary/Graph costs,
independent paired P47 comparison if promising, fresh streams/guards and scoped
tools before retention. Reject if slower rather than weaken precision/terms.

Primary references reread for N1: NVIDIA PTX12.8 async WGMMA ordering and NVIDIA
CUTLASS pointer_flagged.hpp. Local priorF13diagnostic explicitly documented the
flagged byte-vs-element indexing hazard; N1's initial missed conversion is
archived as a failed pre-MMA implementation, with no MMA execution or acceptance.
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html
https://github.com/NVIDIA/cutlass/blob/main/include/cute/pointer_flagged.hpp
