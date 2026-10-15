# M15 premeasurement: fine scale groups with fullK512 INT8 corrections

This numerical screen is prepared after rejected P54 and restoration of P51.
No M15 numerical or hardware result is asserted here. Production is unchanged.
Scope and reference remain original rounded FP16/BF161024^3,FP32 output, alpha1,
beta0, all seven native main products and all eight correction cross dots.

## Hypothesis and history distinction

EmuGEMM §II-C1 uses per-row/column power-of-two scales, and §III keeps integer
partial results on chip. This suggests a testable hypothesis for our small
residuals: scales shared over8 rows of A or8 columns of B, each over fullK512,
may retain adequate residual information while allowing fullK integer accumulation
and one restoration per correction group. The paper's512–1024 preprocessing
limitation prevents a performance inference. PTX8.4 uses signed INT32 accumulators
and defines overflow behavior; our own bound is required.

- https://arxiv.org/html/2606.25453v1
- https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions

M9 already tested fullK512 with coarse A64-row/B32-column scales, but no direct
fine-group producer. P35–P37 actually requantized prior packed INT8 data, adding
rounding and barriers. P39 uses perK128 correction conversion. P51 instead uses
native corrections; P52/P53/P54 altered scheduling/tile shape and all lost latency.
M15 changes compensation representation, without omitting any cross dot or dtype.

## Controlled arms and arithmetic

Use the same input, exact per-original-column inside exponent, FP32 transforms,
native high cast and seven main products in every arm. Main products accumulate
four modeled FP32 K128 bmm slabs. For each transformed high h and residual l=x-h,
select e=ceil(log2(maxabs(h)/127)), with e=0 for an all-zero high group. Quantize
qh=RN(h*2^-e), ql=RN(l*2^(-e+p)), p11FP16/p8BF16, to signed[-127,127]. Count every
high/low preclamp saturation. Raw A22/B22 need only high quantization.

Paired cross dots share exponent eA+eB-p at each output coordinate. FullK512 sums
remain integer before restoring to FP32. A64/B32 versus A8/B8 is the only change
between the two quantization geometries. Scales may vary within a64x32 consumer
tile but remain constant over its complete K dimension. Bound every partial by
2*512*127^2=16516096<2^24<2^31, so TF32-disabled FP32 integer bmm/add models an
exact INT32 sum in this bounded screen. This is not a hardware WGMMA certificate.

Five arms: coarse/post, coarse/leaf, fine/post, fine/leaf, native/leaf control.
Post applies five restored corrections after main recombination (M9 order).
Leaf adds corrections to leaves2,3,4,5,6 before recombination (P51/M13 dependency
graph). This factorial comparison separates grouping from FP32 addition order.
Both integer geometries use the existing M9 FP32 power-of-two scale-product
restoration, including its subnormal behavior; changing restoration is a separate
experiment if it becomes necessary. Native control uses M13's p11/p8 low casts
and fourK128 accumulation. First uniform case for each dtype must exactly match
fresh M9 coarse/post and M13 native/leaf printed eF controls.

## Inputs and gates fixed before measurement

Three fresh seeds123107/123108/123109, both dtypes, all12 M12 patterns plus
rowcol_scale_bands and rowcol_cancellation. New patterns multiply rows of original
A and columns of original B by aligned8-element power-of-two groups cycling
[-4,0,4,-2]; cancellation uses the original near-cancellation generator first.
Round to the input dtype before reference. Total84 configurations,420 arm rows.

Reference is FP64 AB of those original rounded inputs. Record Dense FP32 eF,
original gate eF<=3*eF_dense+floor, and stricter fixed-floor pass, with floor1e-4
FP16/1e-3BF16, no nonfinite operands/output and no saturation. Also check1024
inside exponents per configuration against an independent CPU double-log oracle.
Record integer bounds and paired output deltas; preserve all failures. No timing
from PyTorch models establishes kernel performance or production acceptance.

## Conditional hardware implications, not an implementation proposal

If fine groups pass, direct packing can assign64 A8x512 tasks and64 B512x8 tasks.
These128 tasks may share a cooperative preparation launch with128 exactinside
tasks, but register/shared capacity, grid residency, raw-input reloads, metadata,
transpose/coalescing and actual producer cost need a concrete subsequent design.
Avoiding double quantization is intentional; reloading originals or intermediate
FP32 storage is not free and must be included in complete call timing.

At fixed64x32 MMA shape, seven native K16 products plus eight integer K32 cross
dots have352 source MMA instructions per output tile versus480 all-native;
this is a count, not a speed estimate. Fine-scale per-output restoration adds
metadata/vector work. No GPU candidate is authorized for retention by this model.
Any later implementation needs its own pre-MMA integer/scale/layout probes,
original matrix gates, independent ordinary/Graph complete timing, fresh streams
and tools before replacing P51. P51 remains the current research reference.
