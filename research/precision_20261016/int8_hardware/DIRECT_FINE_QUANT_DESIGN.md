# P55 premeasurement: direct fine-group INT8 producer and fullK consumer

Prepared after P54 rejection/restoration and completed M15/M16 numerical screens.
No P55 C++ sources have been generated/applied, compiled or run. P51 remains the
current fully validated research reference; production/library/Auto unchanged.
Scope1024^3,contiguous/nontransposed,alpha1,beta0,original FP16/BF16,FP32 output.

## Evidence and hypothesis

M15/M16 fine A8/B8 fullK512 corrections pass168 fresh configurations and both
original Dense gate and fixed floors, all7main/8cross dots. Coarse geometry has
FP16 fixed-floor negatives despite passing original gates. Post/leaf order is
isolated in the models; both fine orders pass. M16's safe combined exponent
restoration changes none of425 paired arm eF values/correction bits: observed zero
factors multiply zero integers and are not these matrices' error source. Native
corrections remain more accurate; the fine candidate spends some accuracy margin
within unchanged gates in exchange for a potentially cheaper integer compute path.

EmuGEMM per-row/column scaling and on-chip integer accumulation inform the
representation. Its preprocessing limit at512–1024 and P35–P39 hardware history
prevent inferring speed. P35–P37 requantize packed integers with extra barriers;
P55 instead generates integers directly from original FP32 transforms. P54's
source instruction/traffic reductions failed to improve complete time, so every
cost and actual compiler resource capacity remains part of this hypothesis.

Primary reading: https://arxiv.org/html/2606.25453v1 and CUDA12.4 PTX8.4 WGMMA
https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions

## Workspace and direct producer

Seven native main planes per side (7*512^2*2B), nine INT8 planes per side
(9*512^2B): S/T high0..3,low4..7,rawA22/B22high8. Native main plane mapping remains
P51/P39. All native highs and A integers are logical row-major; B integers are
physically transposed as P39 so both integer MMA operands are K-major. Metadata
ea/eb each9*64 INT32: plane*64+logical Arow/8 or Bcol/8, constant fullK512. Retain
1024 original inside exponents and262144B diagnostic partial workspace. Total
caller workspace12329472B; no extra FP32 leaf workspace or recombination kernel.

One cooperative preparation kernel128CTAs,768threads/CTA: blocks0..63 pack B
512x8 at col=8*block;64..127 pack A8x512 at row=8*(block-64). First256 threads
of every CTA execute P51's exact original8-column inside reduction; all768 threads
participate in the CTA barrier outside that condition and in one grid.sync.
The inside body must not retain its old __syncthreads inside the first256 branch.
All128CTAs must actually fit cooperatively, checked outside run/capture; never
launch based only on nominal shared/register bounds.768threads aims to preserve
roughly P51's24active warps/SM despite fewer tasks; no occupancy/speed promise.

After the grid barrier all768threads process their4096 logical tuple positions,
tid+768*j<4096 (five or six positions/thread). First pass reloads the original four
rounded quadrants, applies the exact inside factors with safe pow2 fallback,
computes identical FP32 RN transforms and writes all seven native highs. Keep
only five thread maxima live: four transformed native highs and raw A22/B22 high.
Reduce five maxima across24warps to five exponents e=ceil(log2(peak/127)),zero->0;
each paired low exponent e-p with p11FP16/p8BF16. Write nine metadata entries.
Quantization must match the native-high exponent arithmetic and RN reference on
finite native bins, not introduce an untested approximate logarithm rule.

Second pass reloads original tuples and regenerates identical transforms/highs;
directly quantizes high and residual x-h with the saved powers, RN/clamp[-127,127].
No intermediate INT8 requantization or extra grid barrier. A integers write directly;
B writes nine512x12 shared planes (8 useful columns+4 padding), then transposes to
coalesced K-major global stores. Use one aligned shared union: inside maxima or
five24-warp maxima/5exponents/nine transpose planes, padded55808B total. Actual
sizeof/static_assert and compiler capacity must be checked before device use.
Finish the CTA stores/transpose and async proxy visibility correctly. Candidate
has two original-input passes, reductions and transpose cost; all are timed.

Standalone preparation diagnostic launch must use64CTAs/768threads per side;
update compare.cu's old standalone pack casts/grids/types. Its full-call gates,
input generation, repetitions and timing arms stay unchanged. Balance=false
diagnostics use the same direct producer without inside; balanced is the contract.

## Consumer: same small tile, integer accumulation across completeK

Start from frozen P39's64x32 tile, grid(16,8)=128,256threads, first128 producer/
idle,last128 one consumer WG. Eight actual24576B operand slots and ready/free
barriers remain. Preload per-CTA fine metadata into shared ae[9][4]/be[9][8]
(108ints/432B) before the existing all-thread init barrier. Shared rounds to197248B.
No per-fragment global exponent chasing: metadata is reused on chip. Capacities
are provisional until actual compilation, zero-spill and occupancy queries.

Seven native main products in order0,1,5,6,4,2,3 accumulate fullK512 with native
m64n32k16. Then five correction groups (0),(1),(2,3),(4,5),(6,7), retaining all
eight m64n32k32 INT8 dots. Preserve P39 qa_plane/qb_plane mapping. Remove adaptive
K-common dispatch and perK fallback: every fine group is fullK512 by construction.
Each correction group's INT32 fragment starts zero and sums all four K128 chunks
and both paired terms before one conversion/restoration. Absolute intermediate
bound16516096<2^24<2^31 holds regardless of accumulation order.

For each CLayout fragment coordinate mn, logical Arow=n0+mn/64 and Bcol=m0+mn%64.
Exponent is ae[ap][(mn/64)/8]+be[bp][(mn%64)/8]. Paired term exponent sums must
agree. Restore float(int32accumulator) using unchanged P51 scale_pow2 (normal
factor branch, general ldexpf fallback). No standalone subnormal factor required;
this is range protection, not a claimed M16 matrix accuracy improvement.
Apply the five corrections to the four output accumulators after native main
recombination (M15 fine/post order), preserving all signs and FP32 RN operations.

Both native and integer fullK loops use wait1/release previous slot when k>0,
finalwait0/release last; wait must precede shared reuse and fragment reads. Retain
compiler operand fences and hardware warpgroup_arrive at every fragment reset.
Actual P55 producer/layout probe must model all48batches/eightslot reuse and
verify integer planes as well as native planes and fine metadata indexing.

At fixed tile:224native+128integer=352source MMA instructions/CTA versus P51's480.
However nominal TMA payload remains1081344B/CTA (whole138412032B), while command
count grows120/CTA (whole15360) versus P51's88/11264. There is no claimed transfer
byte reduction. Twelve fullK groups need48 K128 batches versus28 in P51; more
TMA commands, scale reads and producer/reduction work can outweigh fewer MMAs.
Two complete kernels, four ordinary descriptors (two native/two integer), no
allocation/hostsync/input cache in run; Graph captures descriptor setup as before.

## Required order of validation

1. Isolated CPU source generation/identity and geometry checks, then freeze.
2. Compile only the own packing/scalar/TMA probes first; query actual768-thread
   cooperative capacity and197248B consumer capacity. Reject unsafe resource use.
3. Own CPU-oracle rounded-original packing comparisons: all native planes, all
   integer bytes/exponents, exact1024inside, guards, fresh14pattern seeds and
   extreme/native/subnormal bins; all signed quant saturation explicitly checked.
   Own scalar scale_pow2 integer/exponent bounds, including the M16 negative case.
4. Own native+integer TMA/shared swizzle/48batch ring-reuse/metadata/CLayout probes
   before first new MMA. No inherited native-only certificate covers INT8 layout.
5. Build full candidate, original78 FP64/noNF/exactinside gates, extra model-derived
   row/column patterns, then independent P51/P55 ordinary and Graph complete calls.
   Optional independent P39 control can diagnose combined producer/metadata effects
   but cannot identify one unique cause. Keep every timing arm, unlocked-clock limits.
6. Only if useful versus P51: own fresh dual-stream/workspaces/guards matrix stress
   and memcheck/racecheck/synccheck before retention; otherwise restore byte-exact
   P51 and new-process78 sanity. Own profile diagnoses remaining cost. GPU work serial.

No formal iter371, no production integration or performance acceptance at proposal
stage. Prepared CPU geometry is not generated C++ or GPU execution evidence.
