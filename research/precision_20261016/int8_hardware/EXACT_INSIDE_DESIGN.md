# Next proposal: coalesced exact per-column inside scaling

Status: design only. Do not promote P10's coarser K128 scale. M7 identifies
within-slab alternating imbalance as a numerical counterexample. Preserve P9's
per-column exponents exactly and improve the way maxima are read/reduced.

## Proposed independent P11 experiment

Start from frozen P9, preserving both packers and the corrected consumer.

1. Extend the independent prototype workspace by262144bytes for two sets of
   1024K x32chunk FP32 partial maxima. Keep the original1024-int exponent buffer
   separate so finalization cannot overwrite a partial another CTA still reads.
2. Partial kernel grid8Kslabs x32row/columnchunks,256threads. A is read as32rows
   x128K: tid%128 fixes K and tid/128 fixes even/odd row, each thread loads16rows,
   and a shared256-float buffer reduces two thread maxima per K. B is read as128K
   x32columns: each warp handles16 K rows, one column per lane, then warp maximum.
   Both reads are contiguous across lanes. Store one A/B maximum per K and chunk.
3. Finalize with128 CTAs,256threads: one warp per K, one lane per chunk. Load and
   reduce32partial maxima per side; lane0 writes exactly the P9 exponent
   round(.5*(log2(maxA)-log2(maxB))), retaining the same zero rule and log function.
4. Run unchanged P9 A/B packing and main kernels. Both added kernels, extra
   workspace reads/writes, and descriptor encoding stay in full-call timing.

For finite inputs, max order does not affect positive absolute maxima; original
P9 and new per-column exponent arrays should agree element by element. Check
that equality on fresh ordinary, range, original mismatch and alternating cases,
then run the ordinary72 original FP64 checks and fresh alternating holdouts.
Do not infer correctness merely from the coalesced access pattern.

## Tradeoff and remaining optimization

An extra finalize launch may offset the faster reduction. Measure before adopting.
P10's packing improvements assume K128-invariant factors and cannot be copied as
if per-column scaling were invariant. For B, a separate future experiment can
hoist shared per-value exponent validation across its four original-quadrant
values, while retaining different factors at every K row. It must retain general
exponent semantics and the same rounded native values/residuals.

No predicted time here is a benchmark result. The end-to-end target remains near
the production SW ~10us, rather than merely beating P9 ~65us.
