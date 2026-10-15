# M6/P10 proposal: globally shared K128 inside scales

Status: REJECTED as a replacement. M6 passed42/42 old fixed-floor cases; P10
passed72/72 old hardware cases at54.794/54.421us. New within-slab alternating
imbalance M7 passes0/6 with coarse K128 inside, while per-column inside passes6/6.
Fresh hardware also confirms P9 passes6/6 and P10 passes0/6 on alternating
imbalance. The old72-case pass does not establish replacement acceptance.
P9 remains the retained FMM prototype (~65us), production unchanged.


## Why test this direction

P9 preparation is about 41us, including a strided per-column inside reduction and
per-element B inside scaling. F11 fusion repeats transformations across output
CTAs and remains about 229–245us. A global K128 scale could make both packers use
one pair of invariant factors while permitting coalesced partial maxima.

The exponent remains `round(.5*(log2(maxA)-log2(maxB)))`, with the same zero rule,
but maxima span all 1024 rows/columns and one K128 slab instead of one K column.
This changes the numerical algorithm; retain the original-input FP64 reference,
all precision floors, and full-call timings. The M6 screen uses P9 quantization
sizes A64x128/B128x32, adaptive independent high/low scales, FP32 approximate native
leaves and exact integer FP32 partial dots. It is only a numerical screen.

## Candidate implementation after the model gate

1. One partial-max kernel with 8 K slabs x32 row/column chunks, 256 CTAs. Each CTA
   reads A[32 rows,128 K] and B[128 K,32 columns], 4096 elements per side. Both
   inner dimensions are contiguous; reduce to two FP32 maxima per CTA. Store
   2x8x32 floats in the existing 1024-int inside workspace, with explicit layout.
2. Every packing CTA reduces the 32 partial maxima for each of its two original
   K halves, constructs the same two inside exponents and four normal factors
   once, then transforms and quantizes. SideA c0 and SideB r0 are K128 aligned.
   General exponent paths remain available. No additional finalize kernel.
3. The corrected consumer and its seven native/eight INT8 products stay the same;
   their metadata already rescales independent quantization blocks. A/B raw
   products must still represent cancelling powers of two without dtype overflow
   or underflow; original full hardware cases decide acceptance.

An alternative separate exponent-finalize launch should only be tested if the
repeated 32-way local reduction is actually costly. The launch cannot be omitted
from complete-call timing. Preserve failures and freeze source/binary identities.

## Remaining gates

- M6 all42 fixed-floor screen cases, including cancellation and K mismatch.
- Actual native hardware all72 original FP64 cases and nonfinite check.
- Full invocation including partial reduction, both packers, encoding and main.
- Independent repeats and validation only after an actual latency contender.

This proposal does not assert that coarser scaling always preserves accuracy or
that reducing preparation is sufficient to reach the production ~10us latency.
