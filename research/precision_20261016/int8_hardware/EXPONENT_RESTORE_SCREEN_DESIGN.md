# M16 premeasurement: integer correction restoration only

M15 finer A8/B8 scale geometry passes84 original and stricter-floor configurations,
but BF16 tiny_cancellation seed123107 produces zero rounded scale products. M15
does not establish whether their integer corrections are nonzero. This experiment
isolates restoration before any hardware proposal. P51 and production unchanged.

Two restoration modes share identical M15 inputs, exactinside, seven native main
products, all eight quantized cross dots, coarse/fine geometry, post/leaf orders
and native control. Product mode is exactly M15's integer*(scaleA*scaleB) in FP32.
Exponent mode computes RN_FP32(integer*2^(eA+eB)), where eA and eB are the exact
integer powers represented by the individual scales. For the model, perform this
scaling in FP64 then cast to FP32; bounded integers and powers of two in the tested
range are exact in FP64, so this models one final rounding and protects against
an unrepresentable standalone FP32 scale factor. It is not a CUDA ldexpf proof.

Paired dots must have equal exponent sums at every output coordinate. Integer
bmm/add and its bound16516096<2^24 remain unchanged. Count zero product scales,
nonzero integer values at those coordinates, and nonzero representable restored
corrections lost by product mode. Also count product/exponent correction bit
differences. Compare combined scaling against independent NumPy float64 ldexp
then FP32 on signed integers0/1/127/65535/16516096 and exponents-300..300. Explicit
negative control: integer2^20, exponent-152, standalone FP32 factor rounds to0
while the final correctly scaled value2^-132 is representable and nonzero.

Run84 fresh configurations: M15's14 patterns, seeds124139/124140/124141 and both
original input dtypes. Separately reproduce the old BF16 tiny_cancellation seed
123107 to explain M15's observed zero products. Each configuration has ten rows:
five common M15 arms under two restoration modes (native control identical).
Fresh first-uniform product rows must exactly match fresh unmodified M15 run.
The replayed old product rows must match archived M15 per-arm eF/zero counts.

All850 rows retain original rounded-input FP64 AB, original Dense gate and stricter
floors1e-4FP16/1e-3BF16, no nonfinite/saturation and CPU-oracle inside checks.
Preserve coarse fixed-floor negatives; coarse failure does not abort collection.
Exit0 requires full collection, controls/scalar oracle, all original gates and all
fine/native rows passing original+fixed floors. Record failures if these conditions
are not met. No numerical result establishes speed or GPU instruction behavior.

Conditional implication: a later hardware design should store integer exponents
and scale the converted accumulator safely, without constructing an underflowed
factor. That native CUDA operation still needs its own scalar/packing/consumer
probes. All prep, exponent reads, conversion and synchronization remain timed.
