# SW128 A with SIMT raw-copy control: frozen CPU preparation

This is a concrete private control derived from the actual frozen measured
iteration41 Mega45 source:
`solution/megakernel_wgmma_psplit_selective_tma_a.cu`, SHA256
`3c84b3104bf65fa90b6b3373343aa12d10b16bbe4d8c1859cc3e1d6c87958197`.
It is not a measured candidate, a public variant46, a Plan route, or an
iteration42 start. This task does not compile, link, launch, benchmark, run
sanitizers/ncu, edit formal source/history, or mutate Git. The old71 future
files and all47 bound current library/interface source files are preserved.
The preceding18-file role-clock driver package is never rewritten.

## Question the control can test

Measured45 simultaneously switched A's shared layout from SW32 to SW128 and
raw-A transport from the producer's SIMT vector loads/stores to four TMA boxes.
The observed45 improvement does not isolate which change helped. This control
keeps45's SW128 layout and storage/initialization, but restores original39's
raw-A vector8 SIMT fill. A future same-binary, same-input comparison of45,
this control and39 can distinguish whether45's benefit persists when the new
layout is used with SIMT transport, and whether the TMA path including its raw
transaction wait contributes additional benefit. It cannot, by itself, assign
all differences to a single hardware cause, prove bank conflicts, or establish
measured residency. No guessed latency, speedup or occupancy is provided.

The exact transport change also requires an explicit A pointer in the kernel
parameter list. The encoded map remains a grid-constant parameter but is unused
in the SIMT kernel. Compilation may change the parameter image, code, GPR
liveness, register pool, or scheduling. Those effects must be observed in the
future feeding binary; the source control does not promise identical resources.

## Exact retained and changed source

The candidate lives only in a distinct
`megafmm::sw128_simt_control_future41` namespace and exposes a colocated private
API header. The generator records16 ordered exact replacements, including
names/API isolation, comments explaining the unused map, the explicit current-A
argument, and transport replacement. Each replacement records before/after
SHA and old/new positions. Independent forward and inverse reconstruction
returns the complete actual45 source byte for byte, including the deleted raw
wait region. The complete unified diff and frozen45 copy are retained.

The retained source properties are:

- BM64/BN32/BK64,384 threads, quotas64/192/192, two operand banks and the same
  complete rawA4/finalB7 slot counts. A is SW128 and B remains SW32.
- The complete alignas1024 SharedStorage declaration, dynamic shared base
  alignment, operands/typed-tail union, PipelineAsync barriers, and two
  raw_done uint64 slots. The unused raw_done initialization by tid0 and its
  barrier-init fence remain unchanged. Source layout expects123904 bytes,
  matching the parent's layout; new actual compiled sizeof/offsets/resources
  remain null until compiled and checked.
- All original CTA epochs: initial collective CTA0, complete two-bank drain
  CTA6, then complete seven-P stores before final CTA0/U. There is no additional
  CTA barrier, new tail alias, or early publication.
- All B transformation math, owning A3+4 construction, FP32 RN operation
  order, one final cast, seven whole-K FP32 P values, ascending K16 atoms,
  wait0/postfences before release, both-WG empty-bank release, producer_tail,
  fixed U, alpha RN multiplication, active-beta FMA and D stores.
- Original45 metadata and current-pointer guards, including parent K%128,
  current AB16 alignment, lda/ldb multiples of8 and A lda strictly below2^39.
  The control deliberately does not expand to original39's larger rare-stride
  domain. No public fallback dispatch or Auto certification is added.

The new raw-A producer function is copied from original39
`solution/megakernel_wgmma_psplit_rsa.cu`, SHA256
`1f964339034c3e800455704b6d33482d072ea99a2b6eb9f08773df57f705f054`, with
names normalized only. Its read_eight/store_eight helpers also match original39
byte for byte after normalization. Every producer tid handles four of512 A
vector tasks, each loading/storing the four original quadrants with uint4
opaque16-bit payloads. No raw-A conversion or transform is inserted. Those
stores use45's SW128 tensor addresses, not original39's SW32 addresses.

There are no TMA copies, arrive-and-expect-transaction calls or raw transaction
waits in the control. All128 writers complete their own A loop, original B
loop, proxy fence, and original producer_commit. Only the128th collective
arrival publishes the bank. A leader completing its own A/B work cannot
publish all other writers' unfinished operands. Reuse still waits for both
consumer WGs. Inactive raw_done barriers remain initialized but receive no
transaction arm, completion or wait; they cannot block the control pipeline.

## Deliberately unused descriptor and host cost

Exactly one current-A cuTensorMapEncodeTiled call remains inside each ordinary
typed host call and each Graph capture, using the same datatype/dimensions/
stride/64x64 box/SW128 policy and error check as45. The map is passed by value
as the same declared grid-constant128B/64-aligned descriptor type. The control
kernel explicitly marks it unused and reads A through the new explicit current
pointer. **No encoded map is consumed by this control kernel.** This retained
encode is solely a declared host-cost/layout control; it must not be described
as a useful TMA operation or removed from a timed call to improve its result.

Ordinary calls re-encode the current A pointer each time. Graph replay uses
captured fixed addresses, and the explicit A pointer still reads fresh content
from those addresses. Any future score must include the complete host call
and complete GPU computation as required by the normal measurement contract;
setup/tuning/capture must be labeled separately. There is no input-dependent
cache, global matrix S/T/P workspace, separate leaf launch or Dense substitution
inside this native helper. Non-native metadata are rejected by this private
helper rather than represented as a newly certified public fallback.

## Actual CPU evidence and its limits

The CPU generator, independent source verifier and final raw-copy/order model
have actual exit0 and empty stderr logs. Initial successful model source and
logs are retained separately; that initial model is not mislabeled a failure.
The intermediate explicit-phase model attempt exited1 while constructing an
older negative fixture: it forged a complete arrival set but omitted the newly
required full-phase transition. Its source, empty stdout, actual traceback and
exit1 are retained as attempt2. The final fixture explicitly sets that phase;
candidate/generated source and valid schedule rules were not changed for this
correction.
The final model checks:

- 20,480 absolute SW128 A vector8 coordinates /163,840 halfwords across five
 1024-aligned bases, both banks and all four slots, including contiguous16B
 stores, complete payload bijection and no overlap.
- 81,920 consumer RS packed pairs and17,920 unchanged B vector8 coordinates.
- 655,360 opaque uint4 copies and5,242,880 global A halfword address checks plus
 5,242,880 raw-bit roundtrips over all8 A-row CTAs/all8 K stages and five
 lda domains, including1024, padded strides and the largest allowed2^39-8.
 Packed1024 addresses cover all1,048,576 A coordinates; column CTAs repeat the
 same A addressing because fill_a does not reference col_base.
- 112 valid independently ordered schedules across1,2,3,4,8,9,16 K stages,
 with128 producer lanes modeled separately and two collective WG actors.
 Publication requires all128 A/B/fence/arrival sequences. Bank reuse requires
 both releases, WGMMA retirement precedes release, both final banks drain
 before typed tail reuse, and final U waits for all seven-P stores. Explicit
 full/empty phases start0, with producer start1 and consumer start0.
- 19 negative controls, including missing writers/fences, early publication,
 one-WG bank reuse, tail/drain/U violations, pending-A/P release, wrong initial
 or consumer phase, aliasing raw_done, guard/alignment violations, and mixed
 SW32 writers with the retained SW128 consumer.

The mixed-layout negative controls are intentionally inconsistent combinations.
They demonstrate that this control must keep the paired SW128 address mapping;
they do not claim that original39's consistently paired SW32 implementation is
incorrect or slower. The512-aligned negative violates the declared1024 control
ABI, without claiming that a separate correctly paired SIMT kernel could never
work at a smaller alignment. No bank-conflict count is estimated or measured.

The order model is an abstract collective/dataflow model, not a CUDA compiler,
warp scheduler, mbarrier/WGMMA hardware simulator, SASS lifetime proof, numeric
GEMM check, or timing measurement. Source normalization proves retained text,
not actual generated register initialization or asynchronous operand lifetime.
Opaque payload checks do not certify BF16/FP16 Winograd accuracy.

## Required future GPU use

ROOT must grant a new sole owner lease after the current formal iteration has
closed. Recheck frozen source/headers, actual library source identities and the
existing107488 H100 allocation. Build a private standalone derivative with
CUDA12.4/sm_90a/pinned CUTLASS3.5.1 and the actual MegaFMM library. Do not edit
the preceding frozen role-clock driver or register46 merely to reuse a CLI.
No compiler or linker has yet validated this API or explicit-A/map ABI.

Before a first GPU launch, independently audit both actual BF16/FP16 feeding
objects/PTX/cubin/SASS: initial allocation/pool, DEC/INC ordering and complete
per-role GPR including U,64/192/192 fit, no early P initialization, zero
stack/local/spill, pending A/P access protections, raw vector stores, complete
publication/release/tail/U ordering, retained unused raw_done init, actual
shared sizeof/offsets/alignment and exact explicit-A plus grid-constant-map
launch ABI. **Do not inherit45 or39 CAN/resource numbers.** The parent's
compiled shared layout does not certify the new compiler's resource fit.

Then verify actual fresh raw-U/final-D bitwise consistency against same-atom45
and39, nontrivial specified epilogue/stride/current-binding cases where valid,
and memcheck/racecheck/synccheck on the actual control binary. Preserve the
original FP64 gate `finite && eF_SW <=3*eF_Dense+floor`, BF16 floor1e-3 and
FP16 floor1e-4; never promote a precision-failing row as a certified winner.

A future complete1024 comparison must use the unchanged full measurement
policy and independently tuned same-row Dense/cuBLASLt, with45/39/control in
the same binary and run. Ordinary complete-call results are the primary goal
evidence; Graph and host-batch wall results have their distinct meanings.
Record all trials and actual exits, without dropping groups or using the
unused descriptor to evade host-cost accounting. Only future measured data
can choose whether this direction advances the original goal. Current compile,
CAN, ABI/resource, GPU/SAN/precision/performance fields are null/unachieved.
