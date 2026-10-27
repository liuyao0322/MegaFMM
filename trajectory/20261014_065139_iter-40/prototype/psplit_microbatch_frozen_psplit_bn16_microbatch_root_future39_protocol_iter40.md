# CPU-only root future39 BN16 RS-A microbatch hypothesis

This frozen prefix contains one uncompiled, unregistered future candidate.
It does not start formal iteration40 or change owner39's formal source. Every
write belongs to .scratch/psplit_bn16_microbatch_root_future39_. No compiler,
build, GPU, benchmark, Git mutation, or network tool was invoked. Another
agent's possibly partial microbatch prefix was neither read nor overwritten.

The parent is exactly:

    /share/home/liuyao/workspace/MegaFMM/.scratch/psplit_bn16_tailbanks_future38_raw_candidate.cu
    SHA256 015f6ca799af720a859ffdf73db2655de3f927bb31e3e0bf5751832ea4d8353d

Its tailbanks manifest is bound to SHA256
9d13013fb507d170366ec242776fb80c69701e446c740f45cb3609a75e5f20d1.
The derivative candidate is psplit_bn16_microbatch_root_future39_candidate.cu,
SHA256 f0406ab0422e3385c27ad27a83daeb43c07b412c237939ae0e57656902ae8c68.
The exact ordered byte replacements are in source_delta.json; reversing them
restores the parent byte-for-byte. Name normalization additionally leaves the
traits/shared geometry, producer rawA and finalB transforms, producer pipeline
and producer_tail, complete tail/U, host qualification and launch text exactly
equal to their parent sections.

## Source protocol and pinned CUTLASS evidence

No source-level protocol hard blocker was found. Pinned CUTLASS remains the
parent's f7b19de32c5d1f3cedfc735c2849f12b537522ee reference. This review binds
the local header bytes, without invoking Git to infer the checkout identity.
The manifest and review retain full SHA256 values for seven relevant headers.

The N16 FP16/BF16 RS traits in mma_traits_sm90_gmma.hpp use ALayout_64x16,
ThrID128, and CLayout_64x16. The architecture operations use four uint32 A
registers and eight FP32 C registers per K16 atom. With BK32 the existing owning
stage A layout contains 16 16-bit elements, or 8 packed words per thread. The
rank3 register GEMM in cute/algorithm/gemm.hpp loops k upward from 0 to K-1,
so retaining the parent GEMM call issues each product's K16 atoms 0 then1.
It does not change each product's wholeK accumulation order or ScaleOut::One.

The derivative keeps exactly one owning stage A backing array. Each P is:

    build 16 final A elements into the same owning stage fragment
    -> fence that P accumulator and A operand
    -> warpgroup_arrive
    -> two ascending K16 GMMAs for this same P
    -> warpgroup_commit_batch
    -> warpgroup_wait<0>
    -> postwait operand fence of A and this P
    -> only now build the next P into A

Pinned operand fences use read/write register operands and a memory clobber.
This source ordering ends each logical asynchronous A/P live window before the
next A construction. Actual register allocation, scalarization, emitted
instructions and device behavior remain unmeasured.

WG0 keeps three wholeK FP32 P accumulators, 24 words/thread, and constructs
zero-based P0,1,2 in order. WG1 keeps four wholeK P accumulators, 32 words/thread,
and constructs P3,4,5,6 in order. The source retains both consumer PipelineState
objects and advances each only after release. Each WG calls consumer_release
exactly once per stage, after all its 3/4 products have completed wait0 and both
postwait fences. PipelineAsync release is an empty-barrier arrival; the unchanged
count 256 still requires both complete 128-thread consumer WGs before bank reuse.

The producer remains 128 threads with q48, four original A quadrants plus seven
final B shared slots, two banks, BK32 and all producer stores/proxy fences.
Consumers retain q96/q96. BM64, BN16, 384 threads and launch_bounds(384,2) are
unchanged source hypotheses. Initial allocation 80 is still hypothetical;
an actual initial quota or physical register budget is not established here.

## Exact final A mathematics and costs

Here P numbering is zero-based:

| P | Final A operand | Raw shared quadrant references | Source FP32 RN operations |
|---|---|---:|---:|
| 0 | raw A11 bits | 1 | 0 |
| 1 | raw A12 bits | 1 | 0 |
| 2 | RN16(RN(A12 - RN(RN(A21 + A22) - A11))) | 4 | 3 |
| 3 | raw A22 bits | 1 | 0 |
| 4 | RN16(RN(A21 + A22)) | 2 | 1 |
| 5 | RN16(RN(RN(A21 + A22) - A11)) | 3 | 2 |
| 6 | RN16(RN(A11 - A21)) | 2 | 1 |

Raw P0/1/3 copy packed uint32 bits directly. Each derived final has exactly one
RN16 conversion after the complete FP32 RN DAG. P5 recomputes unrounded S1;
it never subtracts A11 from a typed/cast S1. Zero-based P4/P5 are the parent's
one-based P5/P6 where S1 is repeated.

The total source FP32 RN count per A cell rises 6 ->7. Raw shared quadrant
references rise 7 ->14, explicitly 1+1+4+1+2+3+2. Commit/wait0 pairs per stage
rise from 1 per WG to WG0=3, WG1=4. WholeK P storage remains24/32 words/thread.
Owning A storage falls24/32 ->8/8 packed words/thread, so P+A payload lower
bounds fall48/64 ->32/40 words/thread. These are storage counts, not compiler
register totals or q96-fit proofs. More shared reads and more waits may make
this schedule slower.

An optional future WG1 variant could cache unrounded S1, requiring 16 extra
FP32 live values/thread. This protocol only records the idea; it implements
no second candidate.

## Explicit operand/tail/U lifetime epochs

The producer's unchanged producer_tail waits on both empty banks. Both WGs
drain after their last complete stage. All 384 threads then rendezvous at
nonaligned barrier.sync6,384; only afterward do consumers activate/write the
typed FP32 seven-P alias. The existing subsequent CTA bar0 publishes all7168
tail words before unified U loads. The three lifetime epochs are:

1. Two operand banks: producer/consumer pipeline and wholeK P accumulation.
2. Typed FP32 seven-P tail writes after both-bank drain and bar6.
3. Unified U after all tail writes and CTA bar0.

The selected parent tail mask ((row&2)>>1) | ((row&4)<<1), all product/row/column
strides and all output geometry are unchanged. Literal operand payload is 47104
bytes and typed tail payload 28672 bytes; this logical capacity check does not
instantiate the compiler SharedStorage ABI.

Unified U still assigns each linear 0..1023 coordinate to tid+384*round. Each
owner loads all seven P once, retains the parent's fixed RN U DAG, performs alpha
RN multiplication, and when beta is nonzero loads all four C0 quadrants before
any D store, then applies beta FMA. The full U and host source sections are byte
identical after naming normalization. Native full guard n%64, current AB stride/
range/pointer requirements and one ordinary-kernel launch are preserved.
No global workspace is added.

## Actual CPU validation and its limits

The final main CPU process exited 0 with empty stderr and status
PASS_CPU_MICROBATCH_PROTOCOL_AND_BOUNDED_A_BITS_ONLY.
Its independent state model covers stage domains
1,2,3,4,6,7,8,9,10,16,17,18,33,34. Each domain uses six deterministic role priority
permutations plus 32 randomized ready-actor policies, 532 total valid schedules.
It tracks two literal bank phases/generations, producer publication, per-WG per-P
build/fences/two ascending K16 issues/commit/wait, wholeK P issue history, A rewrite
prohibition, one release per WG per stage, both-bank producer_tail acquires and
both consumer drains before all 384 bar6. These are bounded logical actor
interleavings, not exhaustive hardware timing.

Its 28 negative controls deliberately:

- Rebuild the next A before the current P wait0/postfences.
- Release a stage before the last owned P wait0/postfences.
- Omit a P wait0.
- Finish producer_tail before both banks have drained.

All are rejected with the expected invariant violation.

The separate tail_model.py process also exited 0 with empty stderr, status
PASS_CPU_EXPLICIT_TAIL_THREE_EPOCHS_FIXEDU_ONLY. It explicitly tracks final bank
retirement votes, two producer-tail bank acquires, each consumer drain, 384 bar6
arrival tokens, typed alias activation, all 7168 unique pinned C-layout tail
stores, 384 bar0 arrival tokens and 1024 fixed U owners/4096 output words.
All 14 domains pass beta0 and beta-nonzero branches, 28 cases. Its 8 negative
controls reject early producer-tail acquire, early alias before drain/bar6,
tail store before alias, U before bar0, consumer bar0 before its stores,
tail overwrite during U, and changed fixed U ownership.

Independent NumPy FP32 RN/ST algebra compares all seven final A bit patterns
for 50,000 finite bounded FP16 quads and 50,000 BF16 quads, 700,000 final A
comparisons. Each type includes 40,000 bounded raw-bit random quads and10,000
edge-grid quads with signed zero and small values. All finals match the parent's
independently expressed DAG. Deliberately casting S1 early changes 7044 FP16 and
3474 BF16 S2 results, confirming this negative control is effective.
This CPU evidence establishes bounded operand algebra only; it does not prove
CUDA conversion, GMMA precision, global output correctness or performance.

The first CPU generator attempt exited 1 with empty stdout and a real traceback
from a source-section delimiter using the wrong prepare template marker. The
generator's delimiter was corrected before emitting any candidate/review.
Its actual attempt1_stdout.json, attempt1_stderr.txt and attempt1_exit.txt are
retained as diagnostic evidence. They are not a passing validation run. Final
successful stdout/stderr/exit and tail stdout/stderr/exit are separate artifacts.

Every old raw 8, SS 8, tailbanks 10 artifact and all 3 manifests retains its frozen
SHA before/after. An additional 123 historical formal iteration35 files are
protected, including solution/megakernel_wgmma_psplit_bk64.cu and existing results
paths matching iter35. Current owner39 sources are not snapshot claims.

The parent/root independently reported actual read-only CPU validation of all7
exact replacements, all occurrence counts, forward parent-to-candidate equality
and reverse candidate-to-parent equality, as well as direct review of the
per-P builder and consume_product schedule. This is independent source-scope
evidence only; it is not compilation, device execution or register-fit evidence.

Actual initial quota/register allocation, q48/q96 fit, scalarization, stack,
spill/no-spill, actual shared ABI, two-CTA physical residency, performance,
ELF/host/SASS gate and all GPU/CUDA correctness or precision fields remain
null/pending. This source candidate is available for a separately chosen future
formal experiment; this work neither blocks owner39 nor starts iteration40.
