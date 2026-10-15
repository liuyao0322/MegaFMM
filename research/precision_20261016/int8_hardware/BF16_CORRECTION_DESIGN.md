# M11: native main products with BF16 correction operands

P39 remains the fully validated ordinary/Graph research reference, about36us
versus productionSW about10us. P40/P41/P42's independent local controls have not
produced a3percent gain. Preserve the original FP16 and BF16 input requirements,
FP64 reference from original rounded inputs, fixed gates, seven main products,
all eight standard cross terms, caller workspace and complete-call accounting.

## Numerical hypothesis

Keep native main operands h=round_T(S), g=round_T(T), with T the original input
dtype. For correction only, encode high operands as BF16(h), and low operands as
BF16((S-h)*2^q), q in{0,16}. Raw A22/B22 correction operands are BF16 conversions
of their native values. Preserve the existing eight cross terms and five groups.
Accumulate each group over K in FP32 and multiply by2^-q once after completion.
No INT8 quantizer, per-tile correction exponent or INT32-to-FP32 restoration is
needed in this proposed representation. The original exact per-column inside
scaling still needs its reduction and grid boundary.

This changes quantization and accumulation order. Its outputs need not equalP39.
FP16 main high values lose bits when converted to BF16 for correction; scaled
low conversion also rounds. These errors and omitted low-low terms must be
measured on both original input dtypes, including cancellation and scale mismatch.

[Henry, Tang and Heinecke2019](https://arxiv.org/html/1904.06376v1), §II,
analyze decomposition into several BF16 components and the errors of skipping
products. [Ootomo and Yokota2023v3](https://arxiv.org/html/2203.03341v3), §3,
discuss residual scaling and Tensor Core accumulation. These motivate independent
representation/accumulation tests. Their SGEMM results do not prove this FMM
variant's accuracy, its H100 instruction behavior, or a speedup.

## Screen before hardware work

M11 uses fresh seeds6301–6303,12patterns, both original dtypes:72configurations,
three paired arms/216rows. Baseline is existingK128 pairedINT8, compared with
unscaledBF16low and fixed2^16 BF16low. Ten originalM9 patterns plus tiny and tiny
cancellation patterns retain range/mismatch/banded-scale coverage. Tiny inputs
scale both sides by2^-60 BF16 or2^-10 FP16 before original-input rounding/reference.
FP32 batched dot operations have TF32 disabled; they model separate paired-dot
rounding rather than promise exact Hopper WGMMA behavior. Baseline must match
M9 on the same first fresh uniform input for each dtype.

Screen gates are the original fixed floors1e-4 FP16/1e-3 BF16, zero nonfinite
outputs and zero nonfinite correction operands. A pass is numerical screening
only; do not inherit P39's matrix, Graph, stream, sanitizer or performance evidence.

Fixed scaling is not a full-range theorem. A negative control uses the finite
FP32 transform2^127+2^119, finite roundedBF16high2^127 and residual2^119. Scaling
this residual by2^16 overflowsBF16. Keep this counterexample even if the sampled
matrix suite passes. Before supporting arbitrary finite-range inputs, a new
range policy and its complete cost would be required; existing P39 likewise has
only its explicitly measured scope, not an arbitrary-range proof.

## Performance hypothesis and costs to preserve

Encoding all9correction planes as16bit doubles their bytes versusINT8, adding
about4.72MB of workspace writes per preparation across both sides and corresponding
consumer traffic. Separate native7planes remain. No buffer reuse is assumed yet.
All eight correction dots now use k16 rather than integerk32 WGMMA: nominal
static source work perCTA rises from224native+128integer to224native+256BF16 MMA
calls. These are theoretical loop counts, not executed hardware counters.

Potential benefit is removal of quantizer maxima/synchronization and perK
integer restoration/guard branches, allowing a correction's FP32 fragment to stay
untouched across allK like the native path. A pairedK128 slot needs49,152B, so
four slots plus ready/free metadata can fit196,736B; actual compiled resources,
slot ordering and phase retirement must be designed and checked independently.
More input traffic, fewer slots and more MMA calls may outweigh all savings.

No old SW implementation is restored, no production source is changed, and no
hardware implementation or speed is claimed by this design or numerical model.
