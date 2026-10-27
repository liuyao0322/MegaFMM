# Single-launch M1 with a dedicated WGMMA producer/consumer pipeline

Status: scratch CPU/pinned-header prototype, 2026-10-14. No CUDA compilation,
GPU execution, timing, sanitizer run, or numerical certification has occurred
for this file. `/root/pipeline_impl` may write only this document and
`.scratch/megakernel_wgmma_pipeline.cu`; formal integration belongs to
`/root/core1024`. Preserve the older CUDA TUs and frozen results.

The active user goal is a **1024 single-launch megakernel faster than tuned
cuBLAS**, retaining the original dtype/output/numerical/timing contract.
Packed-once Lt1 is a control, not a substitute. This experiment introduces
dedicated producer overlap to the original M1 mechanism; it does not cache
inputs, use global S/T/P buffers, or change the SW algorithm/precision policy.

## Implemented boundary

| Scratch config | Proposed formal policy | BM / BN / BK | Threads | Operand stages |
|---|---|---|---:|---:|
| 0 | Mega28, integrating owner assigns | 64 / 16 / 32 | 256 | 2 |
| 1 | Mega29, integrating owner assigns | 64 / 32 / 32 | 256 | 2 |

Physical threads 0..127 form the producer warpgroup, threads 128..255 the
consumer warpgroup. The consumer calls `get_thread_slice(tid % 128)`, owns
8/16 FP32 coordinates per P, and keeps all seven P fragments across the entire
ascending parent-K/2 reduction. Every CTA independently writes its four
quadrant output tiles using the fixed U DAG. One invocation recomputes every
input transform from the current original A/B and launches exactly one kernel.

The tensor layouts duplicate existing `MegaTraits` construction: rank-3
`tile_to_shape(GMMA::Layout_K_SW32_Atom<Element>, (64,32,7))` for A and
`tile_to_shape(GMMA::Layout_MN_SW32_Atom<Element>, (BN,32,7))` for B.
There are two independent bank bases; the GMMA tensor construction is rank-3
inside each bank, preserving the existing descriptor partition/slice structure.
All descriptors and full seven product calls use the pinned BF16/FP16
`ss_op_selector` with A Major::K, B Major::MN. Each P sees two m64nBNk16
operations per K32 stage in the same ascending K order as existing M1.

Producer A tasks are `task = tid, tid+128`, covering all 256 row/vector pairs
with `row=task/4`, `k=(task%4)*8`. B uses 64/128 vector tasks and
`k=task/(BN/8)`, `col=(task%(BN/8))*8`. BN16 producer threads 64..127 have no
B task; every producer still fences and arrives. Each task loads the four
original quadrant vectors with `ld.global.v4.u32`, computes all dependent
FP32 additions/subtractions with explicit RN intrinsics, converts each final
transformed operand once through the existing pinned RN paired converter, and
stores every final product slot with `st.shared.v4.b32`. Original operands
retain their 16-bit bits. A temporaries are completed before B temporaries.
There is no raw global/shared prefetch buffer, TMA descriptor, hidden
allocation, copy launch, stream injection, host synchronization, or setup work
inside the timed native call.

A product slot order is `(A11,A12,S4,A22,S1,S2,S3)` and B is
`(B11,B21,B22,T4,T1,T2,T3)`. The four S/T edges stay FP32 until the final
BF16/FP16 RN cast. Seven FP32 P products remain complete; the explicit RN U
order is U1=P1+P2, U2=P1+P6, U3=U2+P7, U4=U2+P5, U5=U4+P3,
U6=U3-P4, U7=U3+P5. Outputs are U1/U5/U6/U7, with alpha RN multiplication
and optional beta RN FMA. All four active C0 values are loaded before any of
the four D stores. beta==0 does not read C0 and permits null C0. Exact C0=D
with the same legal stride remains supported by common Plan checks. Padded
legal output strides are supported by scalar FP32 stores.

## Pipeline and publication proof

The pinned CUTLASS 3.5.1 `PipelineAsync<2>` adds two full and two empty
8-byte barriers. Every one of the 256 threads constructs the pipeline using
producer/consumer arrival counts 128/128. The constructor initializes the
barriers through an elected lane in CTA warp 0 and calls the pinned init
fence. All 256 threads then execute the one uniform `__syncthreads()` before
role divergence. Ordinary launch uses a one-CTA cluster, with `dst_blockid=0`.
There is no CTA barrier inside either divergent role loop.

The producer state starts at `(index=0,phase=1,count=0)` through
`make_producer_start_state`; consumer default state is `(0,0,0)`.
`PipelineState::operator++` flips the phase on every wrap of two banks.

1. Producer acquire waits for the chosen empty bank. Each producer completes
   its stores and executes `fence.proxy.async.shared::cta` with a compiler
   memory clobber before its full-barrier commit. The 128th arrival publishes
   the complete fourteen operand slots, including BN16's inactive B lanes.
2. Consumer waits for the full bank, constructs the bank's descriptors,
   fences all seven accumulator operands, and executes warpgroup arrival.
   All four consumer warps issue all seven WGMMA product calls on the same
   path, commit one group, wait0, and fence all seven accumulator operands.
3. Only after wait0 does each consumer release the empty barrier. Therefore
   the next producer overwrite cannot start until all 128 async readers have
   finished. Shared proxy ordering and accumulator-register ordering are
   separate; both protocols are present.
4. Each role advances its own phase/index. Producer tail waits for both final
   empty banks before its role exits, without a CTA barrier or another launch.

The minimal first experiment intentionally keeps one outstanding WGMMA group
per stage. Producer/consumer overlap comes from independent warpgroups and two
final banks, without an additional two-outstanding-group mechanism.

## Binding, setup, and formal integration contract

Exact scratch public signatures, all in namespace `megafmm`:

```cpp
bool wgmma_pipeline_can_implement(const Problem& p, int configuration);
bool wgmma_pipeline_full_tiles(const Problem& p, const void* A,
                               const void* B, int configuration);
void prepare_wgmma_pipeline(const Problem& p, int configuration);
void launch_wgmma_pipeline(const Problem& p, const void* A, const void* B,
                           const float* C0, float* D, cudaStream_t stream,
                           int configuration);
```

`can_implement` first delegates the current existing general
`megakernel_can_implement(p,0)` validity domain: no transpose, BF16/FP16,
positive even dimensions <=32768, legal active leading dimensions, checked
64-bit span arithmetic, and finite alpha/beta. It then requires native metadata:
parent M%128==0, N%(2*BN)==0, K%64==0, lda%8==0, ldb%8==0.
`full_tiles` additionally checks the **current** original A/B pointers are
nonnull and 16-byte aligned. It does not certify that pointer allocations have
the required capacities or prove overlap/alias legality.

The integrating Plan must retain common current span-end/overflow,
allocation/overlap, D/C0 alignment and alias, coefficient/range, certification,
workspace-cap, degenerate/scale, and actual-route reporting rules. It must
select packed7 for incompatible native metadata and Dense for a failed common
actual A/B16 binding, as agreed with the formal owner, before calling this
native-only helper. No metadata/binding guard failure is timed or mislabeled
as the native pipeline. The scratch helper rejects nonnative bindings rather
than silently launching another kernel. Auto remains Dense until the original
numerical and performance gates are both passed.

`prepare` opt-ins the actual kernel's dynamic shared allocation with
`cudaFuncSetAttribute` during Plan creation. There is no fallback preparation
or host setup in the native run. Preparation and native execution require the
existing SM90a compilation/device support; the formal owner retains current
hardware-route selection. Guards run on every ordinary call and during Graph
capture. Graph replay inherits the captured fixed bindings and recomputes
fresh A/B values on device; it does not rerun host guards during replay.

## Resource hypotheses and hard compile gates

| Quantity | Config0 BN16 | Config1 BN32 |
|---|---:|---:|
| A elements per bank | 14,336 | 14,336 |
| B elements per bank | 3,584 | 7,168 |
| Final operands per bank | 35,840 B | 43,008 B |
| Two-bank payload | 70 KiB | 84 KiB |
| Predicted `sizeof(SharedStorage)`, including 32 B barriers and align128 padding | 71,808 B | 86,144 B |
| Complete P values per consumer thread | 56 | 112 |
| Grid CTAs at parent1024 | 256 | 128 |
| K32 stages at parent1024 | 16 | 16 |
| Logical original quadrant requests per CTA/stage | 20 KiB | 24 KiB |
| Logical original requests for whole1024 | 80 MiB | 48 MiB |
| Input FP32 DAG additions/subtractions per CTA/stage | 10,240 | 12,288 |

The shared-byte figures are layout/header derivations; actual compiled dynamic
allocation is `sizeof(SharedStorage)` and must be inspected. Logical input
requests are instruction-requested data, not measured HBM/L2 traffic or stall
attribution. This design does not establish a performance floor or win.

The source uses pinned `warpgroup_reg_dealloc<64>()` in the producer and
`warpgroup_reg_alloc<160>()` in the consumer, before consumer accumulator
initialization. This is a **budget hypothesis**, not a compiled allocation:
128*64+128*160=28,672 registers/CTA. The first formal build must establish all
of the following before executing the kernel:

- The kernel's initial owned register quota is >=64 and <=160, making the
  producer decrease and consumer increase directions legal under PTX
  `setmaxnreg`. `__launch_bounds__(256,1)` alone does not prove this.
- The actual **per-CTA owned register pool** can hold the final role budgets:
  at least128*64+128*160=28,672 registers, with hardware allocation granularity
  accounted for. The consumer increase waits for registers from this CTA's
  pool; insufficient capacity can wait permanently even when both instruction
  directions are legal. Without a separately established extra reserve, a
  uniform initial quota must be at least112/thread to supply that total.
  For example, initial96 is direction-legal but insufficient for these role
  budgets. Inspect the actual launch/ELF register allocation and redistribution
  mechanism; the whole SM's64K capacity cannot substitute for the CTA's owned
  pool. If this condition fails, revise the budgets or establish a legal
  initial reserve before any GPU execution.
- The actual SASS paths use no registers beyond the role's current quota:
  producer <=64 and consumer <=160 after redistribution. Do not treat a
  global ptxas REG value as proof of both role bounds. Newly allocated
  consumer registers must be initialized before reads; accumulators are
  cleared after the alloc instruction in source.
- Inspect actual REG/STACK/LOCAL, spill load/store count, barrier storage,
  dynamic shared and WGMMA/reconfiguration code. Compiler lowering and
  barriers are not certified by the CPU model. If bounds fail, revise the
  source budget under the authorized iteration before running it; possible
  larger consumer budgets or a no-redistribution control must be measured
  and documented, not silently assumed equivalent.

The H100 has 64K registers and 228 KiB shared/SM; CUDA reserves shared space
per CTA and allocations above48KiB need opt-in. The two-bank payload suggests
at most three/two CTAs from shared capacity for BN16/BN32, while this register
budget suggests at most two from registers. These are ceilings, not compiled
occupancy or achieved throughput. BN32 exposes only128 CTAs on132 H100 SMs;
BN16 is a deliberate grid-size/register-pressure control. The full source is
sm90a-specific through the existing build configuration.

## CPU/pinned-header evidence before freeze

The prototype author's CPU/source model completed with zero assertion
failures on 2026-10-14. It checked the actual source's sole CTA barrier,
seven product calls, producer fence-before-commit, wait0-before-release,
native-only API, and the following pinned-layout geometry:

| Enumeration | Count | Property |
|---|---:|---|
| Both configs, both banks, all A/B seven-slot elements | 78,848 | compact bijection, in bank, no slot/bank overlap |
| A K8 and B N8 shared vectors | 9,856 | 16-byte aligned, eight consecutive element offsets |
| Producer task elements | 5,632 | exact unique raw A/B tile coverage |
| Consumer C coordinates | 3,072 | 128 lanes x8/16 coordinates cover each64xBN tile exactly once |
| Global vector corner spans | 230,912 | row/column boundaries and 16-byte addresses preserved |
| Parent-shape/stride/CTA/K-corner groups | 164 | minimum/full512/full1024/rectangular/32768, compact and padded strides |
| Exhaustive legal role-action states, Kstage counts1..8 | 108 | no overwrite/read-before-full or deadlock |
| Random complete pipeline sequences | 256 | Kstage counts1,2,3,7,8,16,17,512; both final tail waits succeed |
| Random producer/consumer actions | 36,224 | bank readiness, complemented phase and <=2 unconsumed banks |

Pinned 16-bit GMMA atom models used for the independent checks:

```text
swizzle(x) = x XOR (((x >> 6) & 1) << 3)
Aplain(r,k,p) = 16*r + k%16 + 1024*(k//16) + 2048*p
Bplain(n,k,p) = n%16 + 16*(k%8) + 128*(n//16)
                 + (BN*8)*(k//8) + (BN*32)*p
A bank base = bank*(7*64*32 + 7*BN*32)
B bank base = A bank base + 7*64*32
```

The element offset is the appropriate bank base plus `swizzle(plain)`.
These element swizzles preserve all aligned eight-element vectors. MN_SW32's
16-bit atom is16x8, so BN16 is exactly a valid minimum contiguous atom.
The source includes constexpr vector-continuity and compact cosize assertions
against the actual CuTe types; their compiler instantiation remains pending.

Pinned CLayout uses thread shape `(4,8,4)`/strides `(128,1,16)` and value
shape `(2,2,BN/8)`/strides `(64,8,512)`. In explicit coordinates:

```text
row = (tid//4)%8 + 16*(tid//32) + 8*((i//2)%2)
col = 2*(tid%4) + i%2 + 8*(i//4)
```

The kernel uses the actual CuTe identity `partition_C`, not this handwritten
formula. The formula is only independent CPU evidence of its ownership.
Parent vector checks cover shapes `(128,2*BN,64)`, `(256,128,128)`,
`(512,512,512)`, `(1024,1024,1024)`, `(1024,512,2048)`,
`(32768,32768,32768)` with compact and lda+8/ldb+16 strides. For the large
shape, only all local vectors at first/last CTA/K corners are enumerated;
the CPU model does not allocate or multiply a32768 matrix.

`/root/design1024` separately reviewed pinned API, atom geometry, pipeline
phases, source ordering, native-only metadata and producer/consumer ownership.
Its CPU/header PASS is design evidence only and does not certify dynamic
register instructions, sanitizer safety, numerical gates, or speed.

## Required formal/GPU evidence

After freezing both files, integrate a new CUDA TU and public Mega28/29 routes
additively, with the original25 kernel TUs byte-preserved. Verify compile gates
above, default API/guard tests, integer seven-product fixtures and randomized
same-atom M1 equality; test BF16/FP16 both configs including1/3/16K stages,
full512/1024 and rectangular full tiles, metadata tails/padded misalignment,
actual rebound AB16 failures, arbitrary legal output strides, beta0/null C0,
active beta/C0=D, and Graph replay with fresh changed A/B at fixed addresses.
Re-test original strict FP64 gates without altering floors or precision.
Run memcheck/racecheck/synccheck on both native configs after checking resources.

All measured native results must identify one actual launch, zero S/T/P global
workspace and no hidden input precomputation. Complete-call launch and Graph
remain separate, with independently tuned same-row Dense and the existing
warmup/group/fresh-input protocol. Rank own SW latency; an inflated cuBLAS
denominator is not a code benefit. Numerical failure remains visible and
cannot certify completion or enable Auto SW. All perf and validation claims
above this boundary remain pending.

Primary references (pinned sources were read directly in this workspace):

- [CUTLASS3.5.1 PipelineAsync and PipelineState](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/pipeline/sm90_pipeline.hpp).
- [CUTLASS3.5.1 GMMA layouts/descriptors/traits](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/atom/mma_traits_sm90_gmma.hpp).
- [CUTLASS3.5.1 register redistribution](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/arch/reg_reconfig.h).
- [NVIDIA PTX8.4 setmaxnreg requirements](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).
- [NVIDIA PTX8.4 asynchronous WGMMA](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions).
- [Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html#occupancy).
