# P51 premeasurement: smaller tuple tiles with a 384-CTA cooperative grid

Written after P50 own complete acceptance and before any P51 source application,
compilation or measurement. P49 all-native tuple reuse alone regressed versusP47;
P50 A32x128/B64x32 subdivision improves ordinary2.61/2.78% andGraph2.38/1.77%
BF16/FP16 versus same-roundP47. P50 own78gates, ordinary/Graph312 each,54fresh
full-plane bits plus18edgepacking configurations,12-pattern144/1152freshstress
and48/96scoped tools each pass. RetainP50; none transfers to P51.

Stall reassessment reviewed N1–N3/P49/P50 frozen results and ownP50profile:
firstbalancedBF16prep18.464us/54regs/65536B/issue14.653062%,consumer17.440us/
143regs/196736B/issue31.567722%; both0localRW. Profiler timings excluded.
P50 prep issue is belowP49 despite faster fullcalls, so no unique occupancy or
barrier attribution/floor. Fresh primary CUDA12.4BestPractices10.1/10.3/10.4
supports testing latency hiding, not assuming occupancy guarantees speed:
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics

P51 halves rows again A32->16, B64->32, Cols128/32 unchanged, PT8A/4B.
Each tuple loads/scales once then emits all seven native high/four low planes,
with explicit RN transforms,p11FP16/p8BF16,all seven main/eight cross dots.
Exact per-column inside and its first128CTAs*8columns reduction stay unchanged.
Consumer,pow2,workspace,descriptors and restore/recombine order are unchanged.

New tile counts128A+256B=384 active packingCTAs. Existing grid288 cannot cover
these without loops. Use grid384 with B0..255,A256..383, divisionsB256/A128.
Astart256 divisible128 permits existing blockIdx%Tiles mapping. Group0 packs;
all384CTAs execute the existing unconditional cooperative grid.sync before
packing. Keep actual65536B shared request. Init must query actual compiled
resident capacity and reject if supported=false or activeBlocksPerSM*SMcount<384;
132SM*3=396 is only an expectation, not proof. No global spin barrier.
Joint nonbalanced launch also384. Legacy standalone component pack grids must
increase from96A/192B to128A/256B for complete coverage. This is the only
compare.cu change: diagnostic component launch coverage, no timing/gate changes.

B factor pos=r0+tid/32+8*(tid%4) prepares precisely four per-warp loop rows;
r0<=480, pos<=511, pos+512<=1023. Keeping lane%8 would read out of range.
A factor indexing col-based unchanged. Preserve fast-range guard/fallback RN
values with smaller guard scope; no coarse scale or missing bounds coverage.

Require pre-MMA54fresh poisoned-workspace full-plane comparisons against frozen
P47, seed112043+409trial+103pattern,311427072uint16+55296exps, zero errors.
Separate18fresh finite edge cases seed113051+419trial+107pattern: original
representable BF16 +127/-132, reversed, alternating32Kslabs extreme/unit;
FP16+14/-24 controls. The new32Kslabs mix P50 B64 guard but separate P51B32;
frozenP47 also supplies a bit-exact reference. Record actual outside-normal
counts/maxabs; all11planes andinside must agree. These are diagnostic only.
Do not execute P51MMA until both fresh packing diagnostics pass.

Then own standard original-input FP64/exactinside/noNF checks, independent
ordinary/Graph ABBA against frozenP50, including all current preparation and
ordinary descriptors. If retained, own12-pattern fresh dual-stream/workspaces,
128floatguards and scoped memcheck/racecheck/synccheck are mandatory, with new
seeds and32Kmixed edgepattern. Keep original Dense gate, optional stricter
fixedfloors must not replace it. Archive failure evidence and restore P50 if
unhelpful. All GPU tasks serial; no production modifications/formaliter371.
