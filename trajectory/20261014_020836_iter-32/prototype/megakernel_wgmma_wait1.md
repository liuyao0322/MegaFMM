# Vector-producer M1 wait1 prototype

Status: scratch CPU / pinned API candidate only. No template compile, SASS,
device register-ownership check, GPU correctness, Graph legality, sanitizer,
timing or speedup is certified. It is a proposed later axis independent from
the frozen TMA alias source used by iteration31. It has no formal policy ID and
does not modify source30, source31, any frozen scratch or public Plan dispatch.

## Source and invariant contract

`.scratch/megakernel_wgmma_wait1.cu` is cloned from the frozen formal source30
`solution/megakernel_wgmma_pipeline.cu`, SHA256
`3243483e1e84e1fc2785c38904eba6b565b0c8f72876d1664e4c88881b4f4fde`.
Local symbols are renamed to avoid collision. Apart from comments / names, the
layout and vector helpers, pipeline initialization / producer body, accumulator
declarations / initial clear, original seven gemm calls, U / alpha / beta / C0 /
D block and native host helpers remain unchanged. The consumer scheduling is
the sole device-path change: independent read and retire states, wait1 for an
older group after the next commit, and final wait0 / drain.

The supported prototype keeps these source30 properties:

- BM64, BN16 or BN32, BK32, one256-thread CTA containing128 producer and128
  consumer threads; two final-operand banks and `PipelineAsync<2>`.
- BF16 or FP16 original operands, FP32 product accumulators and output. The
  original quadrants and FP32 S/T expressions are read / computed once for each
  CTA / stage / coordinate. The RN S/T result is rounded once into the 16-bit
  final operand; no additional intermediate casts are introduced.
- A product slots `[A11,A12,S4,A22,S1,S2,S3]`, B slots
  `[B11,B21,B22,T4,T1,T2,T3]`, with all seven P fragments accumulated over the
  complete chronological half-K traversal.
- The original fixed FP32 RN U DAG, RN alpha multiplication, active-beta FMA,
  all four active C0 loads before stores, C0==D behavior, beta0 null-C0 support.
- No global S/T/P buffers, raw shared bank, workspace, extra device launch,
  run-time allocation or synchronization, run-time device-property query or
  TMA descriptor construction. Original AB are transformed afresh on device.
- A single CTA barrier after all256 threads initialize the pipeline; the
  producer's proxy fence /128 full arrivals and unchanged producer_tail.

The bounded edge-vector constexpr checks are inherited from frozen source30.
They are compile-time sentinels, not an exhaustive template compile or the new
schedule's resource validation. No TMA layout or descriptor change is included.

## Exact consumer schedule

See [the separate protocol note](wgmma_wait1_next.md) for pinned sources, detailed
dependency reasoning and the CPU exploration. The actual source does:

1. Clear all seven fragments before issuing the first WGMMA. Initialize separate
   default consumer `read_state` and `retire_state`, and `pending=0`.
2. Wait for the full phase of `read_state`, bind the bank descriptors, compiler
   fence all seven fragments, call `warpgroup_arrive`, and execute the identical
   seven gemm calls. Each call retains both K16 substeps in source30 order.
3. Commit one group for that one bank, advance read and increment pending. The
   first stage is not released. Once pending reaches2, `warpgroup_wait<1>()`
   proves the older group complete; compiler fences precede the release of
   `retire_state`, then retire advances and pending returns to1.
4. After the final stage, `warpgroup_wait<0>()` completes every group. Drain the
   last pending bank in retire order, compiler fence all seven fragments, then
   execute the unchanged U / epilogue block.

Pending counts unreleased groups, including a group that may have already
completed physically. It is not an incomplete-hardware-group estimate. At most
two software groups exist between a commit and its wait1. No empty extra commit
is added. Wait1 is never used to release the latest group or as permission to
read any accumulator generically. All collectives and the pending branch are
uniform for the128-thread consumer warpgroup. The source's final wait0 followed
by release / compiler fences does not read P between wait and the U epilogue.

The pinned SS collective's `K_PIPE_MMAS=1` and distinct read/release states
provide the primary precedent. Its same accumulator fragment is used across
groups. CuTe operand fences are compiler constraints; the WGMMA fence is emitted
by `warpgroup_arrive`. Each P has a fixed instruction shape for this kernel.
Keeping each stage's operand fences / arrival and each P's original K16 order
supports the source protocol. Hardware emission still needs inspection for
generic accesses, accumulator spills or materialization while a group is active.

## Numerical chronology proof and limits

For any Pj and integer stage s, the source issues the same two instruction
substeps at half-K offsets `32*s` and `32*s+16` as source30. All stages are issued
in ascending s, all operands use the same vector producer / final-bank layout,
and the seven product mapping and U expressions are unchanged. The schedule
changes when a completion wait occurs and when a bank becomes reusable, while
retaining the instruction sequence for each P. The first generic P access is
after final wait0 and compiler fences.

This proves the source instruction chronology, not actual GPU bitwise equality.
PTX does not fully specify the internal floating-point reduction order. GPU
raw-P/U fixtures and final outputs must be compared with source30, and the
independent original numerical gates remain in force. Existing finite numerical
failures from source30 are neither fixed nor excused by this proposal. This
prototype must not be enabled in Auto merely because a scheduling test passes.

## Interfaces and binding behavior

New scratch entry points, requiring additive declarations / build integration by
a later formal owner, are:

```cpp
bool wgmma_wait1_can_implement(const Problem&, int configuration);
bool wgmma_wait1_full_tiles(const Problem&, const void* A, const void* B,
                           int configuration);
void prepare_wgmma_wait1(const Problem&, int configuration);
void launch_wgmma_wait1(const Problem&, const void* A, const void* B,
                       const float* C0, float* D, cudaStream_t,
                       int configuration);
```

Configuration0 uses BN16; configuration1 uses BN32. No public formal policy
numbers are assigned. The common megakernel validity guard is retained. Native
tiles require M%128==0, N%(2*BN)==0, K%64==0, lda/ldb%8==0 and current A/B16-byte
alignment. Run / capture guards use the current AB binding. Unsupported spans,
alignment, overlap, hardware and degenerate cases require the integrating Plan's
truthful common validation / packed-seven or Dense fallback; this launch helper
only launches the native path and does not silently invent a fallback.

Prepare only performs dynamic shared-memory opt-in. It does not cache transformed
AB. Ordinary calls perform the fresh single native device launch with their
current pointer bindings; Graph captures fixed bindings and fresh contents are
recomputed on every replay. Any later performance run must time the complete
authorized call and report ordinary versus Graph semantics separately. CUDA
event timing of repeated submissions may include host-induced GPU idle gaps and
does not substitute for single-call host wall time. This vector candidate adds
no tensor-map encoding / host descriptor work to source30.

## Resource estimates and hard pending gates

Static source estimates are unchanged: A each bank7*64*32*2=28,672 bytes, B each
bank7*BN*32*2=7,168 or14,336 bytes; two payload banks total70 or84KiB. Add32 bytes
of full/empty barriers and128-byte aggregate rounding for predicted
SharedStorage sizes **71,808 bytes (BN16)** and **86,144 bytes (BN32)**. These
sizes are not sizeof measurements from a new template compile. At1024 the grid
has256 or128 CTAs respectively and16 half-K stages per CTA.

The producer retains collective deallocation to64 registers; the consumer
requests96 for BN16 or160 for BN32. Seven P fragments alone occupy56 or112 FP32
registers per consumer thread. Added read / retire states and their lifetimes
can change actual demand, code scheduling and spills. Source30's successful
compile or role audit cannot be inherited by this source.

Before any launch, a new resource audit must prove producer instructions remain
within64 registers, all consumer instructions within96/160, no unauthorized
generic access to in-flight P, and no spills of those fragments. The per-CTA
owned register pool must cover128*(64+96)=20,480 or128*(64+160)=28,672 registers;
without other reservation, initial owned quotas must therefore be at least80
or112 per thread, respectively. Verify actual initial reservation, role SASS
and safe setmaxnreg protocol rather than equating ptxas numRegs with owned quota.
If the new consumer cannot satisfy the requested budget, reject this draft or
authorize a separate resource change before GPU work. Both dtypes and tiles must
be instantiated against pinned CUDA12.4 / CUTLASS3.5.1.

## CPU evidence and future comparison

The separate pure-stdlib state model was run with explicit independent
producer-tail acquire events: **1,570 states / 2,302 transitions**, stage counts0..17, no
nonterminal deadlock, one drained terminal per count. Another416 random
interleavings execute114,464 actions, including512-stage wrap cases. Completion
can occur out of group order in the abstraction; only the wait-proven prefix
permits retirement. The model checks exact bank ownership, full/empty parity,
read/retire positions, prologue, latest-group tail, producer-tail empty phases
and three reachable unsafe-mutation counterexamples. It abstracts unchanged
128-thread arrivals as collective events and cannot certify hardware liveness.

The intended isolated control is source30 vector wait0 against this vector
wait1, with identical tiles, producer, data, alpha/beta/C0 conditions and complete
measurement protocol. Measure both BN16/BN32 and BF16/FP16, ordinary launch and
Graph independently, then report µs and relative speedup versus separately tuned
cuBLAS. Keep numerical failures visible. There is no evidence yet that this
source is faster: same-P dependencies, fourteen instructions per group, group
backpressure, delayed bank reuse, extra state registers and spill risk can remove
or reverse the expected scheduling benefit. No performance floor is claimed.
