# M15 results: fine scale geometry passes, hardware cost unresolved

Frozen premeasurement design/source preceded the serial GPU model run. All84
newly seeded original rounded-input configurations completed,420 arm rows. Both
first-uniform controls exactly match freshly evaluated M9 coarse/post and M13
native/leaf printed eF for each dtype. CPU double-log oracle agrees with86016
original per-column inside exponents. All outputs/correction operands finite,
no preclamp high/low quantization saturation; every integer accumulation is
within16516096<2^24<2^31. These are numerical model checks, not hardware evidence.

| dtype | arm | original gate passes | fixed floor passes | maximum eF |
|---|---|---:|---:|---:|
| FP16 | coarse/post |42/42|39/42|1.1588408493e-4|
| FP16 | coarse/leaf |42/42|39/42|1.1596610722e-4|
| FP16 | fine/post |42/42|42/42|7.2857092712e-5|
| FP16 | fine/leaf |42/42|42/42|7.2939963142e-5|
| FP16 | native/leaf |42/42|42/42|3.2074890162e-5|
| BF16 | coarse/post |42/42|42/42|8.9898580697e-4|
| BF16 | coarse/leaf |42/42|42/42|8.9898790169e-4|
| BF16 | fine/post |42/42|42/42|5.4322450117e-4|
| BF16 | fine/leaf |42/42|42/42|5.4322468873e-4|
| BF16 | native/leaf |42/42|42/42|1.6116498092e-5|

The six negative rows are three FP16 rowcol_cancellation inputs repeated across
the two coarse orders. They fail the stricter fixed1e-4 floor, while passing the
original gate3*Dense_eF+floor (limits3.335–3.528e-4). Fine geometry passes both
criteria on the same inputs. The process exits2 after writing all420 rows because
the script requires every arm's stricter floor; this is a recorded negative
control, not an interrupted run. Changing only post/leaf order does not repair
the coarse fixed-floor failures. No original-gate failure is asserted here.

The numerical evidence supports finer geometry as a candidate ingredient; native
corrections remain more accurate on the maximum-error cases. Models approximate
FP32 leaf accumulation and do not prove behavior of Hopper WGMMA, synchronization
or saturation under any untested inputs. All7main/8cross dots are retained.

BF16 tiny_cancellation seed123107 records16384 zero FP32 scale products for
coarse geometry and4096 for fine (counts repeated in post/leaf reports). These
counts do not tell whether the associated integer corrections were nonzero.
A targeted restoration ablation should compare multiplying two pre-rounded
FP32 scales against restoring an exact bounded integer with the combined binary
exponent, recording missing-nonzero corrections and original matrix gates.
The frozen M15 screen deliberately used M9 restoration to isolate geometry.

No new CUDA packing/consumer candidate or speed measurement exists. P51 sources
and binary remain current, production/library/Auto unchanged. A direct A8x512 /
B512x8 producer could expose128 tasks but must pay raw reloads, reduction,
metadata, transpose and conversion; the reduced source MMA count does not prove
complete latency. Hardware design follows the restoration check and retains
all original precision gates and full-call ordinary/Graph comparisons.
