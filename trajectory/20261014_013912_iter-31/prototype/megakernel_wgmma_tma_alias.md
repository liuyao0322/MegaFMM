# Single-launch M1 with TMA loading directly into raw aliases of final banks

Status: scratch CPU and pinned-header prototype, 2026-10-14. Neither CUDA
compilation nor GPU execution, Graph capture/replay, sanitizer checks, timing,
or numerical certification has occurred for this candidate. Only the new
`.scratch/megakernel_wgmma_tma_alias.cu` and this document are owned by
`/root/tma1024`. Formal source31 integration/build/GPU/git is not authorized by
this scratch task. The active objective remains a1024 single-launch megakernel
faster than independently tuned cuBLAS under the original contract; no result
here establishes that objective.

The frozen source30 vector reference is
`.scratch/megakernel_wgmma_pipeline.cu`, SHA256
`e5724b712fe16d244b812af0ec04bdb6dcae8a80512854e2b5894e312930f9e5`.
Its resource-review document SHA is
`d7563df2b83c59a759b6b11a4fbfea3a981e2b380353dd5c3af5953a4aab935d`.
The candidate preserves the reference's consumer/full-seven-P/U/epilogue
source block byte for byte except the explicitly authorized config-dependent
register budget: BN16 requests96 consumer registers, BN32 retains160.
Producer input transport, raw aliases, shared
layout swizzle, raw completion barriers and host descriptors are the new axes.
Any later measured difference includes these axes together; it must not be
attributed exclusively to removing SIMD loads without a further control.

## Native configurations and fixed algorithm

| Scratch config | BM / BN / BK | CTA / producer / consumer threads | Final banks | Global temporary workspace |
|---|---|---|---:|---:|
| 0 | 64 /16 /32 |256 /128 /128 |2 |0 |
| 1 | 64 /32 /32 |256 /128 /128 |2 |0 |

One ordinary non-cluster CTA owns four complete output tiles and one complete
ascending parent-K/2 reduction. Producer threads0..127 form one warpgroup;
consumer threads128..255 form another and use `get_thread_slice(tid%128)`.
The consumer retains all seven FP32 P fragments across every K32 stage. Each
P sees two m64nBNk16 WGMMA operations per stage, then the original fixed RN U
DAG and epilogue run once. A successful native invocation launches exactly
one kernel, without global S/T/P buffers, another raw shared bank, a copy
kernel, atomic updates or input precomputation.

All four dependent S/T edges remain FP32 RN through explicit `__fadd_rn` and
`__fsub_rn`; only each final transformed operand is converted once to the
original BF16/FP16 dtype with the pinned RN paired converter. The three raw
original operands on each side keep their input bits. S2/T2 never read
rounded S1/T1. Seven full-K FP32 P products feed U1=P1+P2, U2=P1+P6,
U3=U2+P7, U4=U2+P5, U5=U4+P3, U6=U3-P4, U7=U3+P5; outputs are
U1/U5/U6/U7. Alpha is an explicit RN multiply, active beta an explicit RN FMA.
All four active C0 values for a coordinate are loaded before any D store;
ownership is unique across coordinates/CTAs, preserving exact C0=D with
common legal strides. Beta0 does not read C0 and accepts null C0. The
original FP64 gate, floor and Auto Dense rule remain unchanged.

## Raw aliases and exactly eight TMA copies per stage

Each acquired bank is the existing A7+B7 final operand bank. Four raw
quadrants temporarily occupy four slots on each side:

| Tensor | Final seven slots0..6 | Four raw TMA destinations | Transformed generic stores |
|---|---|---|---|
| A |A11,A12,S4,A22,S1,S2,S3 |A11→0,A12→1,A21→6,A22→3 |S4→2,S1→4,S2→5,S3→6 |
| B |B11,B21,B22,T4,T1,T2,T3 |B11→0,B12→4,B21→1,B22→2 |T4→3,T1→4,T2→5,T3→6 |

Producer thread0 issues four complete A64x32 boxes and four complete BBNx32
boxes using pinned `cute::SM90_TMA_LOAD_2D::copy`. They use one original-A
tensor map and one original-B tensor map, with offsets selecting quadrants;
there are **eight real TMA copy instructions per CTA/K32 stage**, not eight
logical copies lowered into12/16 split boxes. No TMA multicast or auxiliary
cluster CTA is used.

After all eight copies complete, each producer vector task loads all four
raw shared vectors through `ld.shared.v4.u32` before it overwrites any alias.
For each coordinate it unpacks all four inputs to FP32, computes the
dependency chain, converts each final transformed operand once, then writes
the four transformed vectors through `st.shared.v4.b32`. A tasks cover256
row/K8 vectors across128 producer threads; B tasks cover64/128 K/N8 vectors.
Each task owns disjoint coordinates even after swizzling. Hence an A21/S3 or
B12/T1 overwrite cannot destroy another task's input. Raw A slots0/1/3 and B
slots0/1/2 receive **no generic re-store**: their TMA-copied input bits are
already final operands.

Eight copies transfer20KiB for BN16 or24KiB for BN32. The same bytes are
requested by generic raw shared reads, and four transformed generic stores
write20/24KiB. Compared with writing all fourteen final operand slots, this
removes15/18KiB of generic raw-original shared stores per CTA/stage. These
are logical requested bytes, not measured HBM/L2/shared traffic or stall
attribution. Source30's original global input requests remain80MiB/48MiB
for parent1024; the input FP32 DAG operation counts are unchanged.

## TMA boxes, SW64 and descriptor alignment

The original SW32 A K-major atom has a32B contiguous span, while BK32 of a
16-bit operand is64B. Likewise BN32's contiguous B span is64B. CUDA12.4
`cuTensorMapEncodeTiled` requires a non-interleaved swizzled box's inner byte
span to fit the selected swizzle. Therefore the original SW32 layouts cannot
describe all eight complete boxes. Splitting A K32 into K16 would require
eight A copies; BN16 plus four B copies would total12, while splitting BN32
N32 into N16 would total16. This prototype instead changes the actual atoms:

| Tensor | Global map dimensions / byte stride | TMA box inner→outer | Shared atom / descriptor swizzle |
|---|---|---|---|
| A, both configs |(parentK,parentM) /2*lda |(32,64) |`Layout_K_SW64_Atom<Element>` /64B |
| B, BN16 |(parentN,parentK) /2*ldb |(16,32) |`Layout_MN_SW32_Atom<Element>` /32B |
| B, BN32 |(parentN,parentK) /2*ldb |(32,32) |`Layout_MN_SW64_Atom<Element>` /64B |

Rank is2, element strides are(1,1), interleave is NONE, L2 promotion is128B,
cache hint is the pinned EVICT_NORMAL constant, and OOB fill is NONE. BF16
uses BFLOAT16 and FP16 uses FLOAT16; the copies preserve raw bits. Strict
full-tile guards ensure no native box relies on OOB padding.

The two immutable `CUtensorMap` objects are128B each,64B aligned and passed
by value as a `CUTE_GRID_CONSTANT const TmaAliasDescriptors` kernel parameter.
Their descriptor pointers are addresses in immutable kernel parameter space;
the prototype does not allocate or copy tensor maps to mutable global memory.
The host stack aggregate is256B with align64 and a trivial-copyability assert.

Shared operand storage and the actual extern-shared declaration use
**align512**, satisfying the required128B TMA destination alignment. This
stronger alignment deliberately starts SW64's repeating pattern on its512B
boundary and SW32's on a256B boundary. Pinned CuTe's `make_gmma_desc` sets
`base_offset=0`; starting at a merely128B pattern offset and assuming a
relative layout is insufficient for the new TMA copy. Every bank, A/B region
and seven-slot base is a512B multiple in both configurations. No manual
descriptor base-offset patch is used. Actual CUDA compilation must verify
the alignment and generated parameter/shared address lowering before GPU use.

The independent16-bit offset model is:

```text
swizzle(x,bits) = x XOR (((x >>6) AND ((1<<bits)-1)) <<3)
Aplain(row,k,p) =32*row+k+2048*p                  // SW64 bits=2
Bplain(n,k,p) =BN*k+n+(BN*32)*p                 // BN16 bits=1; BN32 bits=2
A byte base =2*bank*(7*64*32+7*BN*32)
B byte base =A byte base+2*(7*64*32)
```

The source also supplies bounded constexpr checks against the **actual**
pinned CuTe layout types: vector8 edge continuity/alignment, compact cosize,
512B bank/slot bases, and atom/pattern/edge coordinate equality with the dense
TMA swizzle model. Full coordinate geometry is independently enumerated on
CPU. The bounded sentinels incorporate source30's observed CUDA12.4 constexpr
work limit without changing device execution. Those
template assertions have not yet been instantiated by a CUDA compiler; the
CPU model below is independent geometry evidence, not template-compilation
evidence. WGMMA K16 slice descriptors/strides and `make_fragment_A/B` must
also instantiate successfully with these new atoms.

## Raw transaction completion and final transformed publication

Two raw `uint64_t` transaction barriers, one per bank, are separate from
`cutlass::PipelineAsync<2>`'s two full/two empty barriers. There is no extra
raw bank. All256 threads construct the final pipeline with arrival counts
128 producer/128 consumer and dst_blockid0. Thread0 initializes each raw
barrier with arrival count1 and calls `fence_barrier_init`. The one uniform
CTA `__syncthreads` publishes initialization before the roles diverge. There
are no CTA barriers inside divergent producer or consumer loops.

1. All producers acquire the empty final bank. Reuse is blocked until all128
   consumers have completed WGMMA wait0 and released the previous bank.
2. Producer thread0 performs `arrive_and_expect_tx` for the aggregate20/24KiB,
   then issues all eight tensor copies against that raw barrier. Complete-tx
   decrements the bytes; raw completion requires all eight copies to finish.
3. Every producer waits on raw completion **before** generic shared reads.
   The final producer state initially has phase1; the raw barrier initially
   has phase0, so its wait phase is `state.phase() ^1`. Both alternate only
   when the bank is reused. A raw completed phase is never treated as the
   final transformed-ready phase. A following empty `asm volatile` with a
   memory clobber explicitly orders compiler memory operations before raw
   shared reads. This is a compiler fence, not another hardware barrier or
   proxy fence; the pinned wait's own volatile PTX has no memory clobber.
   Review found no demonstrated hoist, and this addition records the intended
   compiler ordering rather than claiming to fix an observed GPU error.
4. Producers transform their uniquely owned coordinates. Every writer
   executes `fence.proxy.async.shared::cta` with compiler memory clobber
   before its final full-barrier commit; the128th arrival publishes all
   fourteen final operand slots. TMA completion alone cannot publish S/T.
5. Consumers wait for that full barrier, construct descriptors, fence all
   seven accumulators, arrive, issue all seven complete P calls, commit one
   WGMMA group, wait0 and fence all accumulators. Only then do all128
   consumers release the empty barrier. Both role states advance separately.
6. Producer tail waits for both final empty banks before its role exits.
   There is no extra launch or divergent CTA synchronization.

Raw barrier completion makes TMA writes available to their generic readers;
the final writer proxy fences publish generic S/T stores to WGMMA readers.
These orderings and accumulator-register fencing are separate protocols.
CPU phase/layout proofs do not certify instruction-level synchronization;
racecheck/synccheck and randomized fresh-input equality are mandatory before
accepting later native GPU evidence.

## Current bindings, preparation and timing

The scratch public helpers, all in namespace `megafmm`, are:

```cpp
bool wgmma_tma_alias_can_implement(const Problem&, int configuration);
bool wgmma_tma_alias_full_tiles(const Problem&, const void* A,
                              const void* B, int configuration);
void prepare_wgmma_tma_alias(const Problem&, int configuration);
void launch_wgmma_tma_alias(const Problem&, const void* A, const void* B,
                           const float* C0, float* D, cudaStream_t,
                           int configuration);
```

Metadata first delegates `megakernel_can_implement(p,0)` for BF16/FP16,
positive legal even dimensions<=32768, no transpose, finite coefficients,
legal leading dimensions and checked spans. Native requires M%128=0,
N%(2*BN)=0, K%64=0, lda%8=ldb%8=0 and positive lda/ldb<2^39 elements,
ensuring TMA byte strides are16B multiples less than2^40 before multiplication.
The current original A/B pointers must be nonnull and16B aligned. Common
Plan allocation/capacity, span-end/overflow, overlap, D/C0 alignment/alias,
coefficient/range/certification, workspace-cap, degenerate/scale and truthful
fallback routing checks remain required. These helpers do not certify live
allocation capacity or overlap. Nonnative helpers reject rather than silently
timing another route. Auto remains Dense until both original gates pass.

Preparation queries the current device's compute capability9.0 and opt-ins
the exact typed/config kernel's dynamic shared bytes. Device/query/attribute
operations occur only during preparation; formal Plan hardware selection
and preparing the correct later execution device remain its owner's duties.
The successful native run has no explicit allocation, device query, attribute
change, memory-copy setup kernel or host synchronization.

**Every ordinary native invocation reconstructs both descriptors from its
current original A/B bindings through two host `cuTensorMapEncodeTiled`
calls inside `launch_wgmma_tma_alias_typed`, then passes them to that one
launch.** There is no pointer-bound descriptor cache or prepare-time old-A/B
assumption. Descriptor encoding does not compute S/T or inspect input data.
An actual rebound A/B uses those new pointers, and all input transformations
still occur on device in this invocation. The successful helper does not
insert hidden streams or work outside the timed complete-call boundary.

Ordinary complete-call timing must retain this descriptor construction and
submission work. The existing CUDA-event interval around100 repeated full
`run` calls can include GPU idle gaps caused by host descriptor/submission
cost; it **is not equivalent to wall time of one synchronous host call** and
does not independently isolate host encode latency. Record host complete-call
wall time or descriptor-only host time separately if needed; do not move the
two encode calls outside steady-state ordinary timing to claim a win.

During Graph capture the current fixed A/B addresses are encoded and passed
as immutable captured parameters. Replay recomputes fresh values at those
same addresses with the same one device kernel; no host encode or host guard
runs again. Thus Graph capture/encoding/setup may be excluded from steady
replay, but must be stated separately. A new A/B address requires a new
capture/node update plus correctly rebound descriptors, outside or inside the
chosen timing boundary as explicitly reported. Fresh contents at fixed
addresses do not require map re-encoding and never justify cached S/T. The
driver encode call's legality during stream capture is still a required GPU
test, not established by CPU inspection.

## Resource hypotheses and compile gates

| Quantity | BN16 | BN32 |
|---|---:|---:|
| A elements per bank |14,336 |14,336 |
| B elements per bank |3,584 |7,168 |
| A+B final operand bytes per bank |35,840 |43,008 |
| Two-bank payload |71,680 B /70KiB |86,016 B /84KiB |
| Final full+empty barriers |32 B |32 B |
| Raw transaction barriers |16 B |16 B |
| Predicted align512 `sizeof(SharedStorage)` |72,192 B /70.5KiB |86,528 B /84.5KiB |
| P FP32 values per consumer thread |56 |112 |
| Parent1024 grid CTAs |256 |128 |
| K32 stages per CTA at1024 |16 |16 |
| Aggregate raw transfer bytes per stage |20,480 |24,576 |
| Real TMA copy instructions per entire1024 invocation |32,768 |16,384 |

The new48 barrier bytes fit within an added512B alignment round-up; no raw
operand buffer is added. The table is a pinned-layout/header derivation.
Compiled actual shared alignment/size, ELF/PTX/SASS resources and launch
attributes must be inspected before GPU execution. The many small tensor
copies can themselves impose overhead; no bandwidth/launch/occupancy floor
or predicted win is asserted.

The source requests `warpgroup_reg_dealloc<64>` for producers and
`warpgroup_reg_alloc<96/160>` for BN16/BN32 consumers. Root explicitly
authorized this minimum config-dependent resource repair during source30's
first compile audit; this scratch mirrors that budget decision without
claiming source30's compiled resource evidence applies to the new producer.
These remain **compile-gated budget hypotheses**. The per-CTA owned register
pool must be at least128*(64+96)=20,480 for BN16 or128*(64+160)=28,672 for
BN32. Absent an explicit extra reserve, the initial uniform quota must
therefore be at least80/112 registers/thread respectively. It must also allow
the producer decrease and consumer increase directions, giving inspected
initial quota ranges80..96 for BN16 and112..160 for BN32.
`__launch_bounds__(256,1)` does not prove these bounds.

Before the first native GPU run, inspect initial owned quota/CTA pool, role
SASS register use (producer<=64, consumer<=96/160 after redistribution), all
SETMAXNREG paths, initialization before newly allocated register reads,
REG/STACK/LOCAL/spills, actual barriers/dynamic shared and TMA/WGMMA lowering.
Failure requires a new documented source/budget or no-redistribution control
before running; do not silently assume source30's compiled resources apply
to this changed producer. Driver linkage to CUDA12.4's tensor-map API,
grid-constant parameter alignment and all four typed/config descriptor/MMA
template instantiations must succeed. CPU/header checks do not replace any
of those gates.

H100's64K registers and228KiB shared/SM only give resource ceilings. With
these hypothetical register pools the register ceilings are three/two CTAs/SM;
shared
payload alone would permit three/two CTAs for BN16/32 before other reserves.
BN32 offers128 CTAs on132 SMs, while BN16 offers256. Neither a theoretical
ceiling nor the static counts establish achieved occupancy or speed.

## CPU and source evidence completed before review

Inline Python/std-library models were run on the login CPU; they create no
additional artifacts or GPU context. Assertions completed without failures:

| Enumeration | Count | Property checked |
|---|---:|---|
| Both configs/banks, all A/B seven-slot offsets |78,848 |compact bijection, slot/bank disjointness, all bases512B aligned |
| All aligned A K8/B N8 vectors |9,856 |eight consecutive elements,16B aligned in every slot/bank |
| Absolute pointer offsets at eight align512 shared bases |630,784 |byte-address TMA swizzle equals relative CuTe model, physical offsets remain in every slot/bank |
| Raw complete TMA-box models in alias simulations |256 |four A/four B boxes per bank, exact raw placement |
| Raw copied16-bit elements in those simulations |360,448 |unchanged original input bits in all four raw quadrants |
| Random alias vector-task execution orders |11,264 |unique owners snapshot four raw operands before overwrite |
| BF16/FP16 final seven-slot bit checks |630,784 |original three slots retained and all four S/T final casts match FP32 RN reference |
| Consumer C coordinates |3,072 |128 lanes x8/16 outputs exactly cover64xBN tile |
| Native shape/stride/CTA/K-corner groups |164 |minimum/full512/full1024/rectangular/max32768, compact and padded strides |
| Actual eight global-box corner spans |1,312 |all quadrant boxes in parent extents,16B global origins, legal byte strides |
| Raw transaction-barrier phase transitions |1,132 |both configs, stage counts1,2,3,7,8,16,17,512, complemented producer/raw phases and complete bytes |

Alias models use both dtypes, four independent random seeds, two banks and
both BN configs. Inputs are rounded first and lie within[-16,16]. FP32 DAG
edges use explicit binary32 rounding; BF16/FP16 final cast models apply RN.
The source check verified eight copy callsites, exactly four A/four B generic
transformed-store callsites, one CTA barrier, no global SIMT input loads,
no explicit allocation/memcpy, fresh encode calls for both current bindings,
and the frozen source30 consumer/epilogue block byte equality after normalizing
only the explicitly changed consumer register-budget template argument.

Global-box corner groups include `(128,2*BN,64)`, `(256,128,128)`,
`(512,512,512)`, `(1024,1024,1024)`, `(1024,512,2048)` and
`(32768,32768,32768)`, with lda/ldb compact or padded by8/16. The maximum
shape checks only first/last CTA/K corners; no32768 GPU allocation or GEMM is
implied. Raw phase tests verify serial bank reuse transitions; they do not
exhaustively model instruction-level interleavings or imply sanitizer PASS.

Pinned headers were read directly from `third_party/cutlass` version3.5.1
and `/share/app/cuda/cuda-12.4/include/cuda.h` (CUDA_VERSION12040), including
SM90_TMA_LOAD_2D's exact descriptor/barrier/cache/smem/two-coordinate signature,
ClusterTransactionBarrier static init/arrive_and_expect_tx/wait APIs,
PipelineAsync state/arrival protocol, GMMA SW64/32 atom shapes and fixed0
descriptor base offset, the64B tensor-map host alignment and CUDA12.4 encode
constraints. Direct header inspection is not a compiled API certification.

## Required next formal evidence

Only after explicit formal authorization and frozen source/document review:
add a new TU and policy IDs without modifying old kernel TUs or frozen
performance evidence. Integrate truthful native/packed7/Dense routing and
current binding checks. Complete the hard compile/register/descriptor gates
above before any native launch. Then test BF16/FP16, both configs,1/3/16 K32
stages,512/1024/rectangular full tiles, metadata tails and rejected strides,
actual rebound A/B and misalignment, arbitrary legal output strides,
beta0/null C0, active beta/C0=D, and Graph fresh changed A/B at fixed addresses.
Compare all seven integer P fixtures and random final M1 bits against the
same-atom frozen reference, then memcheck/racecheck/synccheck both native
configs. Preserve the original strict FP64 gates and visible failures.

Benchmark complete native ordinary launch and complete Graph replay
separately under the same independently tuned cuBLASLt32-candidate/8GiB
protocol. Include descriptor construction in ordinary repeated `run`, use
20warm/100calls/5groups/3fresh for full1024 and preserve actual one-launch/
zero-global-temporary metadata. Rank own latency and report same-row Dense/SW,
p10/p90/CV, original gate and resources. Slower Dense denominators, failing
numerical candidates or Dense fallback cannot establish goal completion or
enable Auto SW. Do not retry the session's unavailable ncu probe.

Primary references:

- [CUDA12.4 tensor-map encoding constraints](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html).
- [CUDA12.4 immutable grid-constant tensor-map parameters](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays).
- [PTX8.4 tensor swizzling patterns](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#swizzling-modes).
- [PTX8.4 WGMMA descriptor base offset](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma-async-matrix-descriptor-format).
- [CUTLASS3.5.1 SM90 TMA instruction wrappers](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch/copy_sm90_tma.hpp).
- [CUTLASS3.5.1 GMMA layouts/descriptors](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/atom/mma_traits_sm90_gmma.hpp).
- [CUTLASS3.5.1 raw barrier APIs](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/arch/barrier.h).
- [CUTLASS3.5.1 PipelineAsync](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/pipeline/sm90_pipeline.hpp).
- [PTX8.4 register redistribution requirements](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).
