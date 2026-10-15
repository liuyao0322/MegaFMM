# Pair INT8 corrections before restoring scale

The working standalone P20 experiment follows M8's three-arm numerical screen.
The original production API and fixed original-input FP64 gates are unchanged.
The target is fewer correction waits, scale restorations and recombination leaves;
all seven native main products and eight INT8 dots remain.

For each transformed operand, let `eL=eH-b`, where b=11 for FP16 and8 for BF16.
For a two-sided residual correction, the two integer terms then share the exponent:

```
(eHA-b)+eHB = eHA+(eHB-b).
```

Within one K128 slab, accumulate both INT8 matrix dots into the same INT32 fragment
before waiting and restoring this common scale. Each dot is bounded by128*127²;
the paired bound4,129,024 is below2²² and well below INT32 overflow. Do not sum
integer dots across different K slabs or assume unmatched adaptive scales agree.
M8 separately controls the change to relative low scales and the change to paired
addition, with global per-column inside scaling throughout. Its passes are a screen,
not a universal range guarantee or actual WGMMA correctness proof.

The five correction groups are S4_low×B22_high; A22_high×T4_low; then the two
first-order terms for each of S1/T1, S2/T2 and S3/T3. Their output signs remain
unchanged. Each group accumulates four independently rescaled K128 results in
FP32, then recombines. Pairing changes floating-point rounding relative to the
old eight-leaf correction, so printed eF need not be identical to P17.

The existing24,576-byte TMA slot can hold a paired group without growing shared
memory. WGMMA's left64x128 INT8 operands (from transposed B) occupy offsets0 and8192;
the right32x128 operands (from A) occupy16384 and20480. These ranges are disjoint,
fill exactly24,576bytes, and start at1024-byte-aligned swizzle bases. Existing
single-term groups use only offsets0 and16384. The expected transaction byte count
is12,288 for the two single-term groups and24,576 for each paired group.

Seven main groups plus five corrections, each four K slabs, reduce stage-ready/free
barrier cycles from60 to48. Integer MMA count, integer operand bytes and total TMA
requests are unchanged. The second dot uses the same integer accumulator and is
issued before the one commit/wait for that pair. A slot is released only after its
whole group has completed. Per-column inside, native layouts, workspace and total
shared allocation are preserved. Native WGMMA accumulator ordering and subnormal
behavior remain hardware-dependent; the real device checks decide acceptance.

The [PTX8.7 WGMMA specification](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
defines INT32 accumulation, overflow behavior, warpgroup-uniform execution and
ordering through commit/wait. The pair has identical MMA shape, and no CPU access
to its accumulator occurs between the two dot sequences. Required register fences,
async proxy ordering and all existing waits remain.

Validation order: build and ordinary78 original-input cases including alternating
imbalance; exact79,872 inside-exponent comparisons; full-call interleaved timings;
then fresh dual-stream direct/Graph stress with guards and scoped sanitizers for a
retained candidate. A producer/consumer layout change is not certified by the M8
model or by P7's old Graph/sanitizer evidence. Performance still requires comparison
with approximately10us production SW, not merely an improvement within this prototype.
