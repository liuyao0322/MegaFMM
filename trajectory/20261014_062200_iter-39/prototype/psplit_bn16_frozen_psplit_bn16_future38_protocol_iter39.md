# Future38 raw RS-A BN16 candidate: CPU preparation only

The frozen candidate is not registered, compiled, launched, or measured. It is
derived from `solution/megakernel_wgmma_psplit_rsa_bk32.cu`, SHA256
`548f27dc7a284a150423bd0162fa751b9dcf945baeccd4e7d571f044e0a6c0d3`.
`source_delta.json` records every exact replacement; applying the replacements
in reverse reproduces all original bytes. No iteration39 experiment has begun.
All generated files use `.scratch/psplit_bn16_future38_`; the model rejects
writes outside that prefix. The retained original34 source and the base/kernel
headers were hashed before and after the CPU run and matched exactly. No files
in results, formal source, build, GPU, or Git were written by this work.

The candidate SHA256 is
`ab97837744a071341f7c4b7f21f8009171e4bfc471180eca7c07c37b408c3b90`.
The final CPU run exited0, stderr is empty, and its result status is
`PASS_CPU_LITERAL_GEOMETRY_ONLY_UNCOMPILED_UNREGISTERED`.

## Static source decisions and pinned evidence

CUTLASS is pinned to `f7b19de32c5d1f3cedfc735c2849f12b537522ee` (v3.5.1).
The FP16 RS atom is explicitly defined in
`third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp:2405`; BF16 is at5218.
Both have four packed32-bit A registers, one B descriptor, and eight FP32 C
registers. Both emit m64n16k16 WGMMA with register A and require A-major K;
B-major MN is a supported template argument. The N16 selector paths at
`mma_sm90.hpp:2165` and2243 do not require the extended-shapes macro.

The N16 RS traits at `mma_traits_sm90_gmma.hpp:1288` and2055 have ThrID128,
ALayout64x16 and CLayout64x16. The C layout at389 is
`Shape<Shape<4,8,4>,Shape<2,2,2>>`, with strides
`Stride<Stride<128,1,16>,Stride<64,8,512>>`.
Thus each P has `64*16/128=8` FP32 values/thread. The initial task's12/16
whole-group P estimate was corrected: wholeP3+4 requires24/32 FP32 words/thread,
while BK32 full-stage A remains24/32 packed32-bit words/thread. The payload
lower bounds are48/64 words/thread. They are not physical register fits.

The candidate directly selects the typed FP16/BF16 N16 RS atoms at lines70-72,
fixes Cols16, and uses BM*BN/128 for the per-P fragment assertion at355.
The rawA4 and finalB7 SW32 layout atom constructors remain the original ones.
N16 equals the minimum16-element contiguous extent of the 16-bit MN SW32 atom,
so it introduces no partial-layout-padding requirement in the literal model.
Actual CuTe instantiation/cosize/descriptor canonicalization is still pending.

BN32-only issues corrected in the candidate are the Cols32 trait assertion,
the per-P16 assertion, all four hardcoded typed instantiations32, preparation's
shared>48KiB assertion, and the tail address formula and every store/load call.
The tail at283 uses `product*Rows*Cols+row*Cols+(col^(row&(Cols/4-1)))`;
fixed BN16 gives a row XOR mask0..3, keeping all columns inside0..15.
The old tail formula still reached word14335 when iterating N16 coordinates.
Its incorrect32-element row/product strides and bit4 XOR are rejected by the
CPU negative controls; merely changing strides with the unbounded old mask
also produces collisions, for example row7/col0 and row8/col9 both at137.

The A final-building helper and the B final-building helper, including all
ordered FP32 RN operations, original bit copies and one final cast per derived
operand, match the parent byte-for-byte after helper renaming and the B-task
comment correction. U, alpha RN multiply, beta FMA, beta-zero C0 suppression,
and four-quadrant stores also match after helper/tail-template normalization.

## Independent literal CPU results

The model executed11,776 aligned contiguous vector-eight checks over two banks
at four128-byte-aligned base offsets, plus32,768 aligned rawA packed-pair checks.
It proved all2,048 full-stage A coordinates and1,024 coordinates/P unique.
All7,168 typed tail words are covered exactly once and read once by U.
The1,024 U owners use modulo384 assignment:256 threads own3 coordinates and128
threads own2. The1024 parent grid8x32 has256 CTAs; enumeration produced all
1,048,576 D locations once. A separate read-only agent checked the minimum
retained M128/N64 domain:2 CTAs, all8,192 D locations once.

These are independent integer interpretations of the pinned layouts, not
C++/CUDA template instantiation, physical memory transactions or hardware
synchronization evidence. The full-stage virtual A lifetime model was checked
for P3 andP4 over stage counts1,2,3,4,6,7,8,9,10,16,17,18,33,34. Each P owns
two distinct ascending K16 A fragments; no generic A rewrite/fence is allowed
from issue through wait0. The negative controls reject pending A rewrites.
Physical scalarization, implicit widths, spills and physical async lifetimes
still require complete fresh PTX/SASS analysis.

The predicted rawA4 bank payload is16,384 bytes and finalB7 is7,168 bytes, giving
23,552/bank and47,104/two banks. Four64-bit pipeline barriers predict32 bytes;
align128 predicts47,232 dynamic bytes. The28,672-byte typed FP32 P tail overlaps
all23,552 bank0 bytes plus5,120 bank1 bytes. Both operand banks must drain before
typed tail activation. These are source/ABI predictions; actual host sizeof,
ELF shared attributes, static/reserved shared and device limits remain pending.

Narrowing B gives64 vector tasks, while raw A still gives256 tasks. All128
producer threads nevertheless fence and commit every stage. Full/empty arrival
counts remain128/256, with both complete128-thread consumer WGs releasing only
after wait0. The pipeline constructor, initial CTA bar0, role-local nonaligned
CTA bar6 after producer_tail/both consumer drains, and common tail-publish bar0
retain384 participants. Source barrier count/scope must not be reduced to the
number of B tasks. P tail publication stays after the two-bank drain.

## Host eligibility and future integration

The candidate preserves the old metadata domain: m%128, n%64, k%64 and lda/ldb%8,
with the original common native metadata validation. Every run checks current,
nonnull A/B bindings and16-byte pointer alignment. The A K-half offset remains
multiple32; retained B N-half offset remains multiple32; BN16 CTA column starts
and every eight-element vector start are multiples8. No native tail predicates
are required inside this retained domain.

Natural BN16 fulltile geometry would allow n%(2*Cols)==0, hence n%32. That would
expand the domain to shapes such as parent N32; this candidate intentionally
retains n%64 and does not authorize that expansion. A future registered variant
must retain truthful native/P7/Dense selection: partial native metadata that
P7 accepts takes P7; bad vector strides or other unsupported metadata take
Dense; a current misaligned A/B binding takes Dense through Plan's binding gate.
Auto, certification/range/scalar/transpose rules, degenerate ordering, current
address spans, overlap rejection, four-byte D/active C0 alignment and compatible
exact C0==D requirements remain Plan responsibilities. The raw helper alone is
not the complete public contract.

Future Plan registration needs a new independent ID, disabled until fresh gates,
its own helper declarations/build source, all native/P7/prepare dispatch sites,
maximum policy handling and an accurate64x16 metadata name. Reporting must
record the actual bound-pointer route separately from metadata-only
algorithm_name. This preparation changes none of those formal files.

## Register, resource, and measurement protocol for a later approved run

q48/96/96 and launch_bounds(384,2) are explicit new hypotheses. They change
additional axes alongside BN16, so this candidate is not a single controlled BN
comparison against raw40. At hypothetical uniform initialI80,384*80=30,720 words,
equal to128*(48+96+96). Producer release4,096 equals two consumer requests2,048.
Algebraically sufficient initials80/88/96 respect quota directions, but that
list describes a one-CTA source pool only. It does not prove two CTA resources.
Neither old raw40/SS nor new hybrid41 compiled role maxima can certify q96 here;
subtracting the changed P payload from their register numbering is invalid.

Before any launch, obtain fresh two-dtype compilation and bind source, actual
feeding PTX/cubin, linked test/benchmark symbols, host attributes and ELF identity.
Check exact template layouts/sizeof, initial allocation/directions, every role's
complete register live range and implicit-width operands, zero local/stack/spill,
prequota accesses, full CFG, two ascending K16 issues/P/stage, distinct live A
until wait0, barriers, and tail aliasing. To support a two-CTA resource statement,
use actual dynamic/static/reserved shared, allocated registers/granularity,
threads/warps/CTA limits, launch attributes and device occupancy evidence. A
minblocks annotation, pool inequality or shared-only calculation is insufficient.

Reference36 is an N32 atom and cannot be called a same-atom N16 comparison.
Future validation needs a private SS N16 companion or explicitly independent
host mathematical evidence; the original FP64 gate and tolerances stay intact.
Only after compiled gates pass should the owner run native and P7/Dense tests,
both dtypes,14 stage-count domains and relevant partial-N/K cases, fresh original
A/B bindings, current D/C0, beta0/null, exact alias, fresh graph capture/replay,
numerical gates and sanitizers. No test or GPU numerical outcome is filled here.

At parent1024/leaf512, BN16 doubles grid CTAs128 to256. Logical source A reads
increase32 to64MiB; B stays16MiB, giving80MiB versus old48MiB. Actual L2/DRAM
traffic is unknown. A future timing run must include own matched ordinary/Graph
rows against raw40, any approved SS N16 companion and tuned same-row cuBLASLt,
with actual routes, fresh identities, repeat distributions and no inherited
occupancy, forward-progress, precision or speedup claims.
