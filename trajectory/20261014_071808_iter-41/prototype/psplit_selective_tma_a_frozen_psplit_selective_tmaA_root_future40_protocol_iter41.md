# Selective rawA TMA: independent CPU future40 preparation

This prefix contains one uncompiled, unregistered source hypothesis for a future
formal experiment. Owner40's ordinary candidate44 is independent and does not
contain this TMA candidate. This task neither modifies owner40's work nor starts
formal iteration41. All writes belong to
.scratch/psplit_selective_tmaA_root_future40_. No compiler, GPU, build, benchmark,
Git mutation, network request or download was performed.

The parent is the existing BM64/BN32/BK64 full-stage RS-A source:

    /share/home/liuyao/workspace/MegaFMM/solution/megakernel_wgmma_psplit_rsa.cu
    SHA256 1f964339034c3e800455704b6d33482d072ea99a2b6eb9f08773df57f705f054

The existing TMA reference is solution/megakernel_wgmma_tma_alias.cu; there is
no solution/megakernel_wgmma_tma.cu in this checkout. The reference's host encode,
immutable grid-constant descriptor, raw transaction barrier, phase complement,
and explicit postwait compiler fence are reused as source/API patterns.

Candidate:

    .scratch/psplit_selective_tmaA_root_future40_candidate.cu
    SHA256 dc0a07bcbf238586313d05a4feee2830a06d4d6b0223c22f777d58134d100ff8

The source_delta.json contains every ordered exact replacement and full parent/
candidate hashes. Reverse replacement restores the parent byte-for-byte. The
A transformation/register builder, wholeK consumer GMMA/commit/wait/release
schedule, SIMT B transformation code, tail coordinate mask, unified U, alpha RN
and beta FMA are byte identical after names normalize.

## Why the original A SW32 geometry cannot carry four full boxes

Each A quadrant is a dense 64x64 box of two-byte FP16/BF16 elements. A 2D tensor
map orders dimensions as K then M, so boxDim is {64,64}; inner bytes are 128.
The local CUDA driver header explicitly limits a non-interleaved swizzled box's
inner byte extent to its swizzle span: SW32 permits at most 32 bytes and SW128
permits at most 128. A full BK64 quadrant therefore cannot use the original
SW32 box with only one TMA copy per quadrant.

The original SW32 tiled raw layout is also not a dense64-element row layout:
raw element offset for row 1,k0 is16, whereas the dense box offset is64. Choosing
TMA SW128 while retaining that original CuTe A layout would therefore read wrong
coordinates. This is an actual source/API geometry blocker for unchanged A.

The candidate changes only rawA to the pinned
GMMA::Layout_K_SW128_Atom<Element>, tiled to {64,64,4}. The bit atom is
Shape<8,1024>/Stride<1024,1> with byte Swizzle<3,4,3>. Flagged upcast produces
an 8x64 Element layout while retaining the byte swizzle on the complete shared
pointer, as defined in pinned pointer_flagged.hpp. Thus the raw index is:

    plain = 4096*quadrant + 64*row + k
    byte_address = absolute_bank_base + 2*plain
    physical = byte_address XOR ((byte_address AND 0x380) >> 3)

For a zero-pattern slot base this equals the independently modeled TMA dense
128-byte row map:

    physical = slot_base + 128*row + 2*(k XOR ((row AND7)<<3))

The descriptor uses SWIZZLE_128B, not one of the newer special atom variants.

## Stronger absolute alignment and shared capacity

SW128's full byte pattern repeats every 1024 bytes. Merely 512-byte-aligned base
512 maps CuTe slot0's first address to 576, a 64-byte displacement from the plain
TMA destination base. This fails the zero-pattern correspondence needed here.
The candidate therefore declares both SharedStorage and dynamic shared storage
with 1024-byte alignment, which also satisfies 512-byte alignment.

Every A slot is 8192 bytes, all four A slots total 32768 bytes, each operand bank
is 61440 bytes, and both banks total 122880 bytes. All these A slot/bank/B offset
bases are 1024-byte multiples. All absolute bases tested at
0,1024,2048,8192,65536 preserve exact TMA/CuTe A correspondence and original B
geometry. The512-only base512 case is explicitly a failing geometry control.

The operand union remains 122880 bytes; the typed seven-P tail is 57344 bytes and
fits entirely inside that union. Pipeline full/empty barrier members and the new
two 8-byte raw_done barriers follow the union and never alias it. Static source
assertions constrain raw_done to follow Pipeline::SharedStorage. Under the
literal natural-member estimate, pipeline barriers occupy 32 bytes, raw_done
starts at 122912 and total storage rounds to 123904 bytes with align1024.
The old align128 source estimate was 123008, so the alignment/storage increase is
896 bytes. Actual compiler sizeof/offsetof, host opt-in, linked resource limits,
resident CTA count and compiled CAN are all pending; logical capacity does not
inherit the parent's CAN.

## Per-bank producer publication protocol

The candidate adds exactly four TMA rawA copies per stage, from one current-A
descriptor. Slots0/1/2/3 receive A11/A12/A21/A22 respectively. Each copy is 8192
bytes; transaction expectation is strictly 32768 bytes for A only. B remains the
parent's 128-thread SIMT construction: original four B quadrants are read once
per vector task and final B7 is written with the original onecast RN DAG.
There is no B tensor map and no shared rawB workspace.

Thread0 initializes one raw transaction barrier per bank with arrival count 1,
then the pinned barrier-init fence and original all 384 CTA initialization
barrier publish initialization. Each stage:

1. All producer threads acquire the empty operand bank.
2. Thread0 performs arrive_and_expect_tx(raw_done,32768) then issues four TMA
   copies. Other producer threads may already execute their independent B tasks.
3. All 128 B writers finish their original B stores and each executes the existing
   fence.proxy.async.shared::cta.
4. Every producer thread waits on that bank's raw_done barrier until all four A
   transactions complete. It executes an explicit compiler memory fence because
   the pinned raw barrier wait uses volatile PTX without that clobber.
5. Only then does each producer call its PipelineAsync producer_commit. The128th
   arrival publishes both complete rawA and all finalB stores to the two WGs.
6. The producer state advances. Each bank can be reacquired only after both128
   thread consumer WGs release it following their complete GMMA waits/fences.

Pinned PipelineState defaults the consumer phase to 0; producer start phase is 1.
Each raw barrier initializes phase 0 and flips after its expected transactions
complete. Consequently the raw wait argument is producer_state.phase XOR1.
The two-bank state model derives and tests this complement through bank reuse
and phase wrap; it is not copied as a general SW64 assumption.

The producer performs one additional raw mbarrier wait per thread per stage.
TMA issue may overlap B work, but the producer waits for each stage's A copies
before final publication. No speed advantage is established. Consumer quotas
remain source q192/q192 and producer q64; actual role GPR/register fit is pending.

## Actual A consumption, tail and mathematics

The RS identity partition remains 128 threads, four ascending K16 atoms and four
packed pairs per atom. The CPU model enumerates every paired coordinate for all
four A slots, both banks and all five absolute bases: each pair is 4-byte aligned,
stays in its original logical row, maps to two consecutive halfwords and cannot
cross a 128-byte swizzle row. All 4096 A coordinates are owned once per slot.
The unchanged register builder therefore reads the same raw uint16 bits before
performing the same FP32 RN A DAG and single final casts. This is a coordinate
and unchanged-source argument, not a CUDA precision proof.

The parent B SW32 layout is retained and independently enumerated at the new
absolute aligned bases. All B vector8 accesses remain 16-byte aligned,
contiguous and nonoverlapping. A and B together cover every operand halfword
once in each bank; no slot, bank or B offset alias is introduced.

The seven wholeK P accumulators, original GMMA grouping, producer_tail's two
bank waits, two consumer drains, nonaligned bar6, typed FP32 tail, CTA bar0 and
fixed unified U remain unchanged. The CPU coordinate model also checks all 14336
tail words and 2048 U owners. Added padding lies beyond the union and the barrier
members, so it cannot enter any tail word. Actual struct member offsets still
require future compilation.

## Host descriptor, current bindings and narrowed native domain

One immutable align 64 descriptor containing the 128-byte A tensor map is passed
by value as a grid-constant kernel parameter. Each ordinary launch or capture
constructs it inside that invocation's timed host helper from the current A
pointer, current lda, dtype and original full matrix dimensions. There is no
pointer-bound descriptor cache and no descriptor creation hoisted outside the
ordinary host loop. Graph replay keeps captured addresses and reads their fresh
device contents on each replay. Actual driver encode/Graph behavior is pending.

Host fields are:

- Dimensions {p.k,p.m}; common metadata limits each to at most 32768.
- Byte stride {2*p.lda}; original lda%8 gives a 16-byte stride multiple.
- Box {64,64}, element steps {1,1}, no interleave, standard128B swizzle.
- Type FLOAT16 or BFLOAT16 according to the original dtype; global A 16B alignment
  and all quadrant offsets remain valid under m%128/k%128 and original lda%8.
- All four global coordinate boxes are in bounds; signed int32 TMA coordinates
  fit because the common dimensions are at most 32768.
- Original native m%128/n%64/k%128, B ldb%8 and current B 16B guards remain.

TMA additionally requires byte stride strictly below2^40. The new native query
therefore explicitly adds lda<2^39. This is a real narrowing of the old RSA
domain: some huge lda values were legal under the original common span bound.
The model records lda=2^39 as the excluded boundary, and tests minimum,
moderately padded and largest legal aligned TMA strides.

A future Plan integration must query this narrowed native predicate before
attempting encode and select the existing ordinary RSA path or the established
truthful packed7/Dense fallback for old-legal lda>=2^39. It must not force those
inputs through this TMA helper and turn an expected fallback into an encode
error. All common current span/end, overlap, exact C0==D, active pointer, scalar
output alignment, policy, certification and degenerate handling remain the
integrating Plan's contract. This unregistered file implements only its native
helper and predicate; no Plan dispatcher or runtime fallback registration is
added here. Actual Plan fallback testing/registration is pending.

## Actual CPU evidence and preservation

The model process completed once with exit 0, empty stderr and status
PASS_CPU_SELECTIVE_TMA_A_GEOMETRY_STAGE_SOURCE_ONLY. Its real stdout/stderr/exit
files are frozen separately; there is no earlier failed process to summarize.

The geometry model completed 163840 A element TMA/CuTe equality checks,
81920 RS packed-pair alignment/row checks and17920 unchanged B vector checks.
The stage model covers domains1,2,3,4,6,7,8,9,10,16,17,18,33,34 under six
deterministic priority schedules and 32 randomized ready-actor schedules per
domain, 532 valid schedules. Four independent completion actors account exactly
four 8192-byte TMA transfers per stage. WholeK seven-P issue history, both
consumer releases, producer tail and raw/pipeline phase transitions are checked.
These bounded atomic-role schedules are not exhaustive device timing.

All 23 stage negative controls reject early final-ready publication, wrong raw
wait phase, missing B writer proxy fences, counting B stores in expected TX
bytes, early producer-tail completion and bank0 reuse after only WG0 release.
The SW32 box/layout mismatch,512-only SW128 absolute base and oversized A stride
are additional explicit geometry/API/domain counterexamples.

Exact normalized source protection binds original A mathematics/register
builder, producer B mathematics/layout, wholeK consumers/release and complete
tail/U/alphaRN/betaFMA. Header/API preservation binds 11 exact local headers,
including /usr/local/cuda/include/cuda.h, all required CUTLASS TMA/swizzle/
barrier/pipeline headers and the existing TMA path. The parent RSA and common
metadata implementation are bound separately. All 17 frozen future39 prefix
files, including its manifest, retain their before/after hashes.

All compilation/registration, actual sizeof/offsetof, driver tensor-map encode,
host opt-in, compiled CAN, every role's actual GPR/quota/stack/spill, SASS,
physical residency, GPU precision/correctness, Graph tests, ordinary host cost
and performance remain null/pending. The current prefix does not start a
formal experiment or modify owner40; formal iteration41 is not started here.
