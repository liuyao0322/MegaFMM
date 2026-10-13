# Wait1 consumer scheduling: CPU / pinned API proposal

This is an independent scheduling candidate for a later iteration. It is not
part of iteration31, has not been compiled or run on a GPU, and has no measured
performance or numerical certification. The accompanying vector-producer
prototype is `.scratch/megakernel_wgmma_wait1.cu`; it deliberately does not combine
this scheduling change with the TMA alias / host descriptor axis.

## Pinned evidence and exact claim

The local CUTLASS3.5.1 clone is pinned to
`f7b19de32c5d1f3cedfc735c2849f12b537522ee`. Its
[SM90 SS warp-specialized collective](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)
sets `K_PIPE_MMAS = 1` (local line262), initializes a release state independently
from the read state (line475), commits its prologue without releasing that bank,
then issues another group, calls `warpgroup_wait<1>()`, and releases the oldest
bank (lines538–553). Its `mma_tail` uses `warpgroup_wait<0>()` before releasing
the remaining bank (lines561–574). The required relation
`K_PIPE_MMAS < K_PIPE_MAX` holds for the proposed two banks.

[CUDA12.4 PTX8.4](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)
defines waitN in terms of the older committed groups, leaving the newest N
potentially incomplete. Same-shape WGMMA instructions have default accumulator
ordering. Generic accumulator access requires a wait that covers the relevant
group. These semantics support the protocol below; they do not promise faster
execution or any particular physical group completion order.

The CuTe pinned helpers in
[mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch/mma_sm90_gmma.hpp)
and its [tensor operand wrapper](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/atom/mma_traits_sm90_gmma.hpp)
must be distinguished:

- `warpgroup_arrive()` emits `wgmma.fence.sync.aligned` with a memory clobber.
- `warpgroup_wait<N>()` emits `wgmma.wait_group.sync.aligned N`.
- `warpgroup_commit_batch()` emits `wgmma.commit_group.sync.aligned`.
- `warpgroup_fence_operand(float&)` is an empty compiler asm with `+f` and a
  memory clobber; the tensor overload applies it to every accumulator register.
  It neither reads the accumulator using a generic GPU instruction nor waits
  for WGMMA completion. Retain these constraints at the same issue boundaries
  as the pinned collective. Actual emitted instructions must still be audited.

## Group / bank protocol

Each stage reads one bank, issues all seven products' two K16 substeps, and
commits exactly one group. This bank-to-group relation is essential: a group
must not span two banks, and extra empty commits must not silently change which
bank wait1 covers. The chronological stage order and the two K16 substeps per P
are retained. All seven P fragments remain live across the complete K traversal.

Use independent `PipelineState read_state` and `retire_state`, both initialized
to the normal consumer state. `pending` counts committed groups whose banks have
not been released; it is a software lifetime count, not the actual number of
incomplete groups in hardware.

```text
clear p1..p7 before the first WGMMA
read = retire = consumer-start; pending = 0
for each k stage in ascending order:
    consumer_wait(read)
    bind descriptors to read.index()
    compiler-fence all seven P; warpgroup_arrive()
    original seven gemm calls, each with the original K16 chronology
    commit exactly one group
    ++read; ++pending
    if pending == 2:
        wait_group<1>
        compiler-fence all seven P
        consumer_release(retire)     # the oldest committed bank only
        ++retire; --pending
wait_group<0>
while pending != 0:
    consumer_release(retire)
    ++retire; --pending
compiler-fence all seven P
original fixed U DAG and alpha / beta epilogue
```

For stage0, no wait1 or release occurs. After stage1 commits, wait1 proves that
stage0's group is complete, so only bank0 can be released. Stage1's bank stays
owned even if its group has already completed. In steady state `read` stays one
stage ahead of `retire` after each retirement. Every later commit temporarily
makes `pending == 2`; wait1 retires the oldest, returning it to1. At loop exit,
the final group remains owned. Wait0 covers it before the drain and epilogue.
For a single stage, the only completion wait is that final wait0.

All WGMMA collectives are executed uniformly by the full128-thread consumer
warpgroup. Producer acquisition still blocks on the empty barrier for the exact
bank / phase. Every producer writer's `fence.proxy.async.shared::cta` and the
128-thread full arrival remain unchanged. Register fences cannot substitute for
that shared publication. `producer_tail` may start once production ends. Each
of its two acquires waits for the required empty phase, and both are satisfied
by the final consumer drain; it adds no CTA barrier.

## Accumulator dependency and numerical scope

Successive stages use the same P registers. For each P, the issued instruction
shape is consistently m64n16k16 or m64n32k16 within a kernel. The same-shape
ordering rule and the pinned SS collective support continued WGMMA accumulation
without waiting for all earlier stages at every issue. The pre-issue compiler
fences and `warpgroup_arrive()` are retained for every stage, including when the
most recent committed group may still be active.

Wait1 does **not** make any entire P fragment available for generic access: the
latest group may write every P. No U calculation, clear, scale, debug observation,
generic copy, spill/reload or epilogue access is allowed until the final wait0.
The intervening tensor fences are compiler constraints. If compilation emits
generic accumulator materialization while a group is active, the candidate is
rejected; source inspection alone cannot discharge this requirement.

For each P, the source issues the identical sequence of K16 operations in the
identical chronological order as source30. Inputs, one-time original-to-S/T
rounding per stage, seven product mappings, full-K P accumulation and the fixed
FP32 U / alpha / active-beta-FMA DAG are unchanged. This is a source-level
chronology proof, **not** a GPU bitwise equality claim. PTX does not fully specify
all floating-point internal accumulation details. Matching P/U/output bits and
the independent original numerical gates must be measured later.

## CPU state exploration

Run `python3 .scratch/wgmma_wait1_state_model.py`. The model uses aggregate
produce/full-publication, issue/commit, group-completion, wait and release events.
It explores all enabled producer/consumer/completion interleavings for its
finite stage counts. It deliberately permits group completion out of submission
order and only uses a wait-proven completed submission prefix for retirement.
This overapproximation avoids assuming a stronger physical completion order.

The state carries exact owners for the two banks, monotone produced / issued /
retired counts, separate read / retirement positions, and the producer's two
independent final empty acquires. Those tail acquires can interleave with the
consumer waits / releases as soon as the matching empty phase is available.
Assertions check:

- no bank is overwritten until its corresponding group was wait-proven complete
  and its consumer release published the empty phase;
- at most two banks and two unretired groups exist, including prologue / tail;
- full/empty phase parity matches both independent PipelineState positions;
- every nonterminal state has an enabled progress transition;
- the final wait0 precedes the last release / generic-access boundary, and both
  final producer-tail empty acquires observe their required phases.

The current CPU run passes **1,570 states / 2,302 transitions** over stage counts
0..17, with one drained terminal state for every count. Another **416 randomized
sequences / 114,464 actions** cover counts
`0,1,2,3,7,8,15,16,17,32,33,64,512` with32 seeds each. Count0 is a boundary model;
native kernel calls require positive supported dimensions and the common Plan
continues to own degenerate handling. At1024, BK32 over half-K gives16 stages.

The model also finds reachable counterexamples for three unsafe mutations:
releasing the first bank after wait1 while only one group exists; releasing the
latest bank when only the older group is proven complete; and reading U after a
tail wait1 that can leave the latest group incomplete. It models unchanged
128-thread arrivals as aggregate events and does not certify device scheduling,
compiler output, mbarrier implementation, numerical behavior or hardware liveness.

## Measurable hypothesis and risks

The hypothesis is that replacing each stage's wait0 with one deferred-group
wait1 reduces exposed consumer waiting and permits more WGMMA issue overlap.
The clean first comparison is this vector producer with wait1 against the exact
source30 vector producer with wait0, at both BN16 and BN32. Use the same complete
fresh-input ordinary call and fixed-address Graph modes; report each mode
separately, together with the independently tuned cuBLAS baseline.

The candidate can regress. Same-P dependencies still constrain execution; a
group contains fourteen WGMMA instructions; the group queue may backpressure.
Holding the oldest bank until the next stage commits delays its producer reuse
relative to the wait0 loop. Added state lifetimes may increase registers or
force forbidden accumulator spills. No speedup, attainable floor, overlap amount
or numerical improvement is claimed before measurement.

Before GPU work, compile both dtypes / tiles against the pinned toolchain and
audit actual register ownership, role-specific setmaxnreg, accumulator accesses
and spills. Then check sanitizer / tail and phase-wrap cases, matching raw P/U
fixtures and the original numerical gates, fresh AB / active C0 / C0==D / beta0
null-pointer behavior, and truthful one-launch / workspace0 behavior. A certified
speedup claim requires those gates. Experimental timings from a failing gate
must be labeled and failures reported explicitly, without silently widening
the tolerance or enabling Auto.
