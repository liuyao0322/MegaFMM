# M16 results: early scale underflow is not the observed matrix error source

Full serial run completed exit0:84 new configurations (seeds124139–124141,both
dtypes,14 patterns) plus one old BF16 tiny_cancellation seed123107 replay,850
rows under two restoration modes. Product mode matches fresh unmodified M15
for both first-uniform controls and matches archived M15 for the old replay.
All850 original gates pass, no nonfinite/saturation,87040 CPU-oracle inside
exponents agree. Fine/native arms all pass both original and stricter floors.
Coarse FP16 again has three stricter-floor negative cases per mode/order, while
passing original gates; all rows are preserved.

Combined-exponent scaling passes5409 scalar bit comparisons against independent
NumPy ldexp, including signed zeros, underflow and legitimate overflow. Constructed
integer2^20/exponent-152 demonstrates a lost representable value: standalone FP32
factor0, correctly restored value2^-132. This validates the mathematical model,
not the native CUDA implementation or a production fix.

On every tested matrix group, a zero scale product coincides with a zero integer
accumulator. Nonzero integers at zero scales=0, lost representable nonzero
corrections=0, product/exponent correction bit differences=0. All425 paired arm
eF values and their deltas to native/leaf agree exactly. Thus the zero products
seen in M15 do not explain its measured matrix errors. Do not claim an accuracy
improvement from this restoration change on these matrices.

Maximum eF on84 fresh configurations, identical between restoration modes:

| dtype | arm | original gate | fixed floor | max eF |
|---|---|---:|---:|---:|
| FP16 | coarse/post |42/42|39/42|1.1423525403e-4|
| FP16 | coarse/leaf |42/42|39/42|1.1428877042e-4|
| FP16 | fine/post |42/42|42/42|7.1093174615e-5|
| FP16 | fine/leaf |42/42|42/42|7.1179984423e-5|
| FP16 | native/leaf |42/42|42/42|3.1163955364e-5|
| BF16 | coarse/post |42/42|42/42|8.7736298674e-4|
| BF16 | coarse/leaf |42/42|42/42|8.7737826581e-4|
| BF16 | fine/post |42/42|42/42|5.1469090403e-4|
| BF16 | fine/leaf |42/42|42/42|5.1469351169e-4|
| BF16 | native/leaf |42/42|42/42|1.7062104596e-5|

Fine fullK512 geometry now passes168 newly seeded configurations across M15/M16,
both original gates and fixed floors, with all7main/8cross dots. These screens
do not certify arbitrary inputs or Hopper WGMMA, nor prove performance. A direct
fine-group hardware proposal may proceed, using integer exponent metadata and
safe restoration without claiming M16 changed matrix error. Its packing,
descriptors, metadata reads, integer conversion, synchronization and full timing
must be validated independently. Working P51, production/library/Auto unchanged.
