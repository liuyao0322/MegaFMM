# M13 / proposed P46: reuse native high operands within each leaf

Premeasurement hypothesis: P45 loads native high operands for the main product
and reloads them for the residual cross terms. Fuse these operations within each
leaf, with separate FP32 main and residual accumulators; restore the residual
once by 2^-11 (FP16) or 2^-8 (BF16), add to the leaf, and then apply the same
seven-product Winograd schedule. All seven main products and eight cross dots
remain. Exact per-column inside scaling, native casts, original rounded-input
FP64 reference, dtype-specific gates and complete-call accounting are unchanged.

This changes FP32 recombination order. M13 first isolates that change with shared
main/residual fragments in two numerical arms, 12 adverse patterns, both original
dtypes and fresh seeds8301–8303 (72 configurations / 144 rows). Its P45-style arm
must exactly reproduce M12 on the first same-input uniform case per dtype.
TF32-disabled FP32 bmm/add is not exact WGMMA accumulation. No hardware acceptance
or performance conclusion follows from a model pass.

Proposed hardware: same 11 native planes per side, two TMA descriptors and four
49,152-byte slots. Each slot has Bhigh at0, Blow at16,384, Ahigh at32,768 and
Alow at40,960. Product0/1 loads2 planes (24KiB), product2 loads3 (32KiB), product3
loads3 (40KiB), product4/5/6 loads4 (48KiB). Producer performs28 K128 iterations
and88 TMA commands per CTA, versus48/120 in P45. Nominal operand bytes per CTA
are1,081,344 versus1,474,560. These are source counts, not measured DRAM traffic
or a latency prediction. Two16-FP32 fragments increase register pressure.

Each K batch includes the main and all applicable residual MMAs. Wait<1> releases
only the preceding batch's slot; wait<0> precedes scalar fragment reads, rescaling
and final slot release. Both fragments need operand fences and warp-group
ordering. Ready/free phases use execution sequence step*4+k, independent of the
logical product number. No interleaved plane layout or preprocessing change is
included in this control. Own original-input gates, exact inside exponents,
ordinary/Graph comparisons, fresh multi-stream/guard tests and scoped sanitizer
checks are required before retention. P45 stays the accepted research reference
until those checks; production and Auto remain unchanged.
