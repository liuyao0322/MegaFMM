# Future RS-A experiment: raw A4 shared, final B7 shared

This is a CPU-only, uncompiled candidate for a later iteration. It has not been
registered in Plan, built, launched, benchmarked, or used to replace an existing
formal implementation. The exclusive lease covers only the new
`.scratch/psplit_rsA_future35_*` files. No formal source, old frozen file, GPU
process, build process, public result, or Git state is changed by this work.

The proposed first compile candidate is
`psplit_rsA_future35_candidate.cu`, SHA256
`4e32c7cf4bd7b916ae5fad3cab8aba601fb629aaca41ffd310ce40014113fca1`.
Its mathematical baseline is the frozen iteration34 revision6 source
`results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu`, SHA256
`fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91`.
The frozen BK64 future source is read only as an additional stage-count context;
its SHA256 is `55f119259536c1c788f0d8508c8005f820463ef8fa6b316018ea2733caf0d8e1`.
The accompanying diff is against the revision6 source, not a live GPU iteration.

## Intended change and preserved arithmetic

For 1024, the proposed geometry remains BM64/BN32, 8 by 16 independent CTAs,
384 threads per CTA, one producer warpgroup and two consumer warpgroups. BK64
uses eight ascending K stages in the parent half-K512 reduction. Each stage has
four ascending K16 atoms per whole product. Consumer WG0 owns P1/P2/P3, and
consumer WG1 owns P4/P5/P6/P7. Every CTA still computes all seven full FP32
products and uniquely owns its four output quadrants; there is no global S/T/P
workspace, grid barrier, split reduction, atomics, or additional kernel launch.

The producer loads each original A quadrant as the same 16-byte aligned vector,
but writes four raw 16-bit A slots instead of producing seven final A slots. It
continues to construct all seven final B slots with the original ordered FP32
T DAG and the original final onecast. The consumers construct their register A
fragments from raw A. Raw A11/A12/A22 bit patterns are copied unchanged.

For WG0, the ordered scalar DAG is S1=RN(A21+A22), S2=RN(S1-A11),
S4=RN(A12-S2); only final S4 is converted to the input type. Its three products
use [A11,A12,S4]. For WG1, the ordered DAG is S1=RN(A21+A22),
S2=RN(S1-A11), S3=RN(A11-A21); each final S1/S2/S3 is converted once. Its
four products use [A22,S1,S2,S3]. Neither WG converts S1 or S2 before a
subsequent FP32 operation. S1/S2 are computed independently in both WGs, so
the total A DAG has six RN operations per logical A element versus four in the
shared/shared baseline. This is six, not a guessed aggregate of seventeen or
twenty-four operations per element.

The whole-P reduction order, fixed U graph, alpha RN multiplication and beta
FMA epilogue are unchanged. The CPU model compares the normalized U/epilogue
source bytes with the frozen baseline. This experiment preserves the original
precision contract; it does not resolve any existing FP64 reference failure.

## Pinned CuTe API and register fragment construction

The API audit uses the repository's CUTLASS 3.5.1 headers, whose exact SHA256
values are bound in the CPU report and independent review. The selected atom is
`GMMA::rs_op_selector<Element,Element,float,Shape<64,32,64>,Major::K,Major::MN>`.
For BF16 and FP16 the pinned implementation selects the m64n32k16 RS atom.
Each atom has four packed uint32 A registers per thread, one uint64 B descriptor,
and sixteen FP32 accumulator registers. A major K is required; B major MN is
retained. This is a register A fragment, not a shared A descriptor.

The pinned RS A layout is the literal `ALayout_64x16=CLayout_64x16`. Partitioning
an identity 64x64 A tensor gives rank-three coordinates `(8,1,4)` per thread.
`make_fragment_A` provides the register layout without constructing storage in
the candidate's `decltype`. Plain, uninitialized, `alignas(16)` Element arrays
back explicit register tensor views. This alignment is necessary: natural
16-bit alignment alone does not meet the uint32 recast precondition, and the
recast API does not validate pointer alignment. The owning ArrayEngine's tuple
initialization is avoided so that no implicit input or accumulator zeroing is
requested before the role quota transition.

Recasting each 32-half stage fragment to uint32 produces `(4,1,4)` packed words.
Packed coordinate `(pair,0,atom)` uses the raw shared pair at the identity
coordinates `(2*pair,0,atom)` and `(2*pair+1,0,atom)`. The literal layout maps
these to the same row and adjacent K values. The builder assigns every owned
word before any MMA. The pinned `gemm` register-A/descriptor-B overload visits
all four K16 atoms in ascending order and updates the same full-P accumulator.
The template instantiation and actual lowering remain pending a CUDA build.

## A register lifetime and WGMMA protocol

The required lifetime applies to A input registers as well as accumulators.
`wgmma.fence` orders prior generic register accesses before following MMA;
`wgmma.wait_group 0` completes all committed groups. A generic read, write, or
compiler operand fence of an A register still used by a pending group is unsafe.
These requirements come from PTX ISA 8.4 sections 9.7.14.7.1 and 9.7.14.7.3.
[NVIDIA PTX ISA 8.4](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html)

Each stage follows this sequence within each complete consumer K loop:

1. Wait for the current shared bank's full barrier.
2. Build all P-specific, all K16-specific A words in disjoint register fragments.
3. Apply compiler operand fences to the active A and accumulator fragments,
   then execute the collective WGMMA fence.
4. Issue the three or four whole-product GEMM calls, each containing four K16
   atoms, without touching any active A input through generic instructions.
5. Commit the group, wait for group0, then apply A and accumulator compiler
   operand fences and release the bank.

All128 threads of the participating WG execute the same collective WGMMA
instructions. The post-wait A operand fences deliberately keep every stage input
live until completion. They are compiler data dependencies, not hardware fence
instructions. Actual PTX/SASS must confirm that the compiler retains distinct
physical input registers and introduces no pending generic moves, local access,
spill, or hidden overwrite. A four-word scratch repeatedly overwritten after
each atom would not satisfy this protocol.

For BK64, WG0 holds 3x4x4=48 packed A registers per thread and 48 FP32 P
registers. WG1 holds 4x4x4=64 packed A and 64 FP32 P registers. These are
96/128 payload lower bounds, excluding raw pair words, conversion values,
addresses, descriptors, predicates and pipeline state. The consumers gain
whole-stage input residency; the baseline's shared/shared A did not require this
register input set. The CPU event model reaches pending A peaks48/64, checks
zero generic-A/pending intersections, and rejects deliberately premature
four-word reuse and premature generic read/fence controls. It does not establish
physical register allocation or GPU forward progress.

## Shared layout, stores and count comparison

All counts in this section are analytic source references or operations. They
are not measured DRAM traffic, shared transactions, cache misses, stalls, or
performance. The old BK32 and BK64 total transform element counts are identical
for a full parent half-K512; BK64 reduces stage and barrier events16 to8.

| Per CTA, per BK64 stage | Shared/shared baseline | Raw-A RS candidate |
|---|---:|---:|
| Original A global bytes referenced | 32,768 | 32,768 |
| Original B global bytes referenced | 16,384 | 16,384 |
| A shared store bytes | 57,344 | 32,768 |
| B shared store bytes | 28,672 | 28,672 |
| A+B shared store bytes | 86,016 | 61,440 |
| Producer A RN operations | 16,384 | 0 |
| Producer B RN operations | 8,192 | 8,192 |
| Consumer A RN operations | 0 | 24,576 |
| Total S/T RN operations | 24,576 | 32,768 |
| A final 16-bit onecasts | 16,384 | 16,384 |
| B final 16-bit onecasts | 8,192 | 8,192 |
| Generic raw A shared bytes referenced by both WGs | not applicable | 57,344 |

Thus the producer loses all A FP32 DAG work and all A final onecasts: its S/T RN
and cast counts each fall by two thirds. Total S/T RN work rises by one third,
and total final onecasts do not change. The raw A consumer read count is seven
quadrant-equivalents: WG0 reads four and WG1 reads three. It is not four shares
read once by the entire CTA. WG0 makes4096 final element conversions; WG1
makes12288. These convert final S values once, rather than casting intermediate
DAG values.

The producer A stage has512 vector tasks. Each reads four16B original vectors
and writes either seven16B final vectors in the baseline or four16B raw vectors
here. This removes1536 vector store instructions' logical payload per stage.
B has256 vector tasks; its four original reads and seven final writes retain
their original DAG. Generic raw A reads use8192 uint32 loads for WG0 and6144
for WG1 per stage in the source model. Instruction generation is unverified.

Across eight stages, each CTA retains393216 logical original A+B global bytes,
while operand shared stores fall688128 to491520 bytes, a196608-byte reduction.
Across the128 CTAs this is84MiB to60MiB of analytic operand shared writes,
saving24MiB. Full-grid original A+B source references remain48MiB; these cannot
be reported as physical DRAM bytes. Total S/T RN operations per CTA rise196608
to262144. These opposing costs do not imply a net speedup.

The physical raw A layout is four compact64x64 slots using the pinned K-SW32
atom. B is seven compact32x64 slots using the pinned MN-SW32 atom. One operand
bank has32768 raw A bytes plus28672 final B bytes=61440 bytes. The two-bank
payload is122880 bytes versus172032 in BK64 shared/shared. The typed seven-P
tail is57344 bytes and fits within bank0, but neither bank may be overwritten
until both consumer WGs have released all outstanding stage reads. Source-derived
SharedStorage size is expected123008 bytes after32 barrier bytes and128-byte
struct alignment; actual `sizeof`, attributes and occupancy remain build gates.

The CPU physical address model covers all slot halfwords without gaps or
duplicate ownership;30720 vector checks and65536 pair checks validate16B
producer alignment and4B packed consumer alignment. The2048 raw generic warp
load patterns have at most one distinct uint32 word per bank for each instruction
pattern in the literal model. This result applies only to the generic raw A
layout and does not describe WGMMA hardware bank access or actual compiled
instruction grouping.

## CTA drain, typed tail, U and bindings

Pipeline full arrivals remain128 and empty arrivals256. Initial pipeline setup
is followed by the original collective384-thread barrier. The producer completes
`producer_tail`; both consumer WGs execute group0 waits and bank releases. The
original non-aligned `barrier.sync 6,384` role-local call sites then collectively
drain the operand lifetime before the consumer-only typed tail stores. The
barrier storage is outside the operand/tail union. The original final collective
barrier publishes all14336 FP32 P words before any U read. No additional CTA
barrier or global synchronization is introduced.

The tail keeps the original compact XOR address permutation. WG0 uniquely writes
products0..2 and WG1 products3..6. The CPU model checks complete disjoint P
ownership,2048 unique U owners per CTA, one read of each P word, and1048576
uniquely owned 1024 output coordinates over the full grid. Tail coordinates are
rebuilt from a fresh volatile thread-id after the drain, retaining the baseline's
attempt to avoid carrying tail-only integer coordinates through MMA loops. Actual
placement of S2R, operand storage and tail stores must be checked in SASS.

All four C0 reads precede all four output stores for each unique output owner
when beta is active, so the original exact C0==D behavior is retained subject to
the Plan contract's scalar alignment, equal-leading-dimension and span rules.
The native helper preserves full-tile and current A/B16B guards and changes only
the BK64 K-divisibility guard to K%128. It is a future native helper, not complete
Plan integration. A later integration must retain the common current-span/end,
overlap and alias checks; current A/B on every run; fresh computation on replay;
one captured kernel node; invalid policy before empty/scale; Auto Dense behavior;
and truthful unsupported or numeric fallback. Any kernel attributes or resource
preparation remain outside timing, while the single ordinary launch stays in
the end-to-end timed path. No caller ST/P workspace is added.

## Register pool hypotheses and required actual gate

The draft uses source role requests64/192/192 and launch bounds384,min1. The
final role pool is128*(64+192+192)=57344 registers per CTA. This algebra alone
does not prove either consumer's192 limit sufficient. The conservative first
compile axes are explicit initial allocation152/160/168 registers per thread:

| Initial I | Initial owned384-thread pool | Producer release to64 | Both consumer increments to192 |
|---|---:|---:|---:|
| 152 | 58,368 | 11,264 | 10,240 |
| 160 | 61,440 | 12,288 | 8,192 |
| 168 | 64,512 | 13,312 | 6,144 |

These eight-register-granularity budgets fit the64Ki algebraic bound using only
this CTA's owned pool. Uniform initial192 would require73728 and is not a valid
64Ki hypothesis. No neighbor-CTA register borrowing is assumed. Actual resource
rounding, compiler initial allocation, SETMAXNREG directions, implicit HGMMA
operand width, per-role maximum live R, scalarization, STACK/LOCAL/LDL/STL,
spills and serialization diagnostics must all be established before launch.
The large shared allocation also needs actual device SM limits and occupancy;
128 CTAs on132 SMs is geometry, not a residency guarantee. No static launch
bound is treated as a dynamic residency guarantee.

If the first build fails these gates, freeze that source/PTX/cubin/SASS/ELF and
adjust an explicitly named resource axis. Do not launch a spilled or WGMMA
serialized object as a passing candidate. Do not weaken the input type, onecast
DAG, reduction order or FP64 gate to obtain a launchable object. Subsequent
validation must cover both data types, direct ordinary launch, current binding,
fresh graph replay, alias/unsupported cases, sanitizers and exact artifact
identity before reporting any end-to-end/cuBLAS measurement.

## Direct-global A alternative

Constructing the same RS fragments directly from original global A removes the
raw shared A copy and generic raw shared reads, but both consumers collectively
reference seven A quadrant-equivalents instead of the producer's four. Its
logical A global references are57344 versus32768 bytes per BK64 stage,75% more.
A B-only two-bank payload would be57344 bytes, exactly the typed tail size, and
would still require complete two-bank retirement before union reuse. Cache,
coalescing and register lifetimes could change the outcome; this is a separate
unimplemented future axis. Raw A4 shared is the concrete candidate because it
retains the original vector global references while exposing the RS tradeoff.

## CPU evidence and limits

The report checks all65536 raw bit patterns for representation copying. Ordered
FP32/RNE emulation compares seven product A inputs for65280 finite BF16 test
quadruples and63488 finite FP16 test quadruples:456960 and444416 comparisons,
respectively, with zero mismatches. These are deterministic samples with complete
finite A11 coverage, not exhaustive four-input combinations. The emulation does
not establish CUDA converter behavior, FTZ, NaN payloads, ABI, full GEMM accuracy,
FP64 reference success, physical liveness, GPU safety or speed. The model,
independent source/API review and manifest are ready for a future actual compile;
all GPU results remain pending.
