# M12 proposal: native correction high aliases, native scaled residuals

P43's hardware BF16 corrections improve accuracy but regress full-call latency.
P44 isolates removing the INT8-inherited B transpose; it does not remove duplicate
main/correction high storage. Both original input dtypes and all eight correction
dots remain mandatory. A native-typed representation is the next independent
numerical question, not an implementation or performance acceptance.

Let h=round_T(x), with original T=FP16 or BF16. Represent a correction high by h
itself, and its low by round_T((x-float(h))*2^p), p=11 forFP16 or8 forBF16.
Accumulate each correction group in FP32, then restore2^-p once perleaf. Keep
all eight existing cross terms, main casts and exact inside scaling. FP16 high
correction retains its original bits, rather than rounding them again toBF16.

The fixed p distinguishes this from M11's fixed16 arm. For finite x and finite
rounded h, RN error within a native binade with exponentE is at most2^(E-p).
Scaling by2^p bounds the low magnitude by2^E; the highest finite binade gives
32768FP16 or2^127BF16, both finite representable. The finite-high overflow
counterexample residual2^119 inM11 becomes2^127 atp8, rather than overflow atp16.
Subnormal h has a constant quantization step; the same bound is at most the
minimum normal after scaling. This is a no-overflow argument for finite h;
it does not guarantee no underflow, exact correction, finite h for arbitrary
transforms, or final matrix accuracy. Preserve independent range-negative controls.

A future hardware layout could use eleven planes per side: seven native main
high planes followed by four native scaled lows. Correction high descriptors
would address existing native planes, eliminating their duplicated conversion,
storage and preparation writes. Original BF16 and FP16 follow the same design.
With MN-major B asP44, neither side needs a physical transpose. Nominal plane
storage11MiB versusP43/P44's16MiB, and only two tensor maps need encoding. More
MMA work and four slots remain; these savings cannot be called speedups before
complete-call measurement. No old SW implementation or production edits.

First screen fresh seeds7301–7303,12patterns, both original input dtypes, three
paired arms: originalK128INT8, M11unscaledBF16 and native-scaled corrections.
Keep original rounded-inputFP64 reference, fixedfloors1e-4/1e-3, all terms, zero
nonfinite output and operands. Baseline and BF16 arms must exactly match M11's
same-input implementation on firstuniform7301perdtype. Add an independent scalar
range check for bothnative dtypes, including highest finite binades, signs, zero,
subnormals and midpoint neighbors; preserve M11's fixed16 negative control.
These are numerical screens, not WGMMA rounding, synchronization or speed evidence.
