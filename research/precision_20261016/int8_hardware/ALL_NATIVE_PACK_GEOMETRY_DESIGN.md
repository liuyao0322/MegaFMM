# P50: spatial subdivision of P49's all-plane producer

Written before applying/compiling P50. P49's own54fresh full-plane comparison,
ordinary78 gates and independent ordinary/Graph ABBA312checks per mode pass,
but complete calls regress versus P47: ordinary BF16/FP16 28.696840/28.824400
->30.314680/29.816440us; Graph28.654680/28.814800->30.146040/30.032279us.
Reject P49; no own fresh dual-stream/guard or sanitizer acceptance.

P49 own first balancedBF16 NCU17passes per kernel: prep20.096us profile-only,
54registers,65536Bdynamicshared,capacity3 byshared/4 byregisters,issue15.224558%;
consumer17.536us,143registers,196736B,1CTA byshared/registers,issue31.588867%.
Both0localRWsectors. Profiler-process timings excluded from ordinary/Graph.
Lower pack CTA count and lower aggregate issue rate are consistent with a
parallelism cost; they are not phase-specific occupancy proof or unique causal
attribution. Do not infer a physical floor or guaranteed gain from subdivision.

P50 starts frozen P49 and changes only producer row geometry/index coverage:
A64x128->32x128 and B128x32->64x32, halving per-thread tuple iterations to16/8.
Each active CTA still loads/scales a raw tuple once and emits all7high/4low.
Preserve cols,unroll4,RN operations,p11FP16/p8BF16,planes,workspaces,consumer,
descriptors,compare,pow2,65536Brequest,cooperative288grid and exact inside stage.
All7main/8crossdots remain. No K granularity or FP32 recombination changes.

P49 Group0 owns32A+64Btiles. P50 Group0 owns64A+128Btiles, giving192active
packingCTAs. Change group divisions accordingly in cooperative, joint and legacy
pack launchers. A block192..255 maps modulo64 to0..63 because192 is divisible
by64; B0..127 maps directly. Other blocks stop packing only after the existing
unconditional grid barrier; all288 participate. Existing diagnostic legacy grids
96A/192B are still >=new64A/128B, so complete plane coverage remains possible.
No grid launch count change in this candidate. Query actual resident capacity.

Critical B fast-exponent indexing: the old lane%16 prepared128rows and would
read beyond1024inside when r0=448 after halving the row tile. Use lane%8 instead,
preparing8Kpositions perwarp, precisely the8 loop iterations. Across8warps these
cover the current64rows. The all-warp normal-exponent guard retains identical
per-element fast/fallback RN values; its scope follows the smaller tile. A
factor indexing remains col-based and unchanged. This is required bounds-safe
geometry adaptation, not permission to coarsen exact inside or skip fallback.

Require own pre-MMA full native plane bit probe against frozen P47:54fresh
configurations,9patterns*3trials*2types, seed110017+389*trial+89*pattern, poison
newworkspace before preprocessing,311427072nativeuint16values and55296inside
exponents. Zero bitwise errors before new MMA. Then original-inputFP64/noNF/
exactinside and complete ordinary/Graph independent comparison to P47. Own
fresh streams/workspaces/guards and scoped tools are mandatory for retention.
P47 remains reference until the new candidate's complete acceptance. Preserve
P49 failure evidence and production identity. This is a code-derived control
informed by the prior CUDA12.4 latency-hiding guidance, not a proven bottleneck.


Additional targeted pre-MMA edge diagnostic, planned before any P50 MMA: the
legacy nine patterns do not drive exact inside beyond[-126,127]. Add18cases
(3patterns*3trials*2types), seed111031+397*trial+97*pattern. BF16 uses original
representable input exponents+127/-132, reversed exponents and alternating64K
slabs of those extremes versus unit-scale. FP16 uses representable+14/-24.
The mixed pattern forces the old B128 guard to fallback while the new B64 tile
may be wholly normal or wholly outside range. Compare all11native planes and
exactinside to frozenP47, recording actual out-of-normal exponent counts/maxabs;
require genuine BF16 fallback coverage. These are bitwise packing diagnostics,
not originalFP64 matrix, performance or independent sanitizer acceptance.
