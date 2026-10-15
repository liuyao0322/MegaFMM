# P47: direct native high/low packing, retained research reference

Native scaled corrections no longer require cross-thread INT8 tile maxima or a
physical B transpose. P47 removes the inherited FP32 transform write/barrier/read
and writes each native high and scaled low immediately from its explicit RN
transform. Input addressing, per-column balancing including extreme-exponent
fallback, unroll4, native casts, low scaling and11plane mapping are preserved.
Column-max block/grid barriers and the async-proxy fence remain. `corrected.cuh`,
`compare.cu` and `pow2.cuh` are byte-identical to frozen P46: all seven native
main products and eight cross dots, two maps and corrected-leaf recombination.

This first control retains65,536B cooperative shared request to isolate direct
packing from a later resource-capacity control. Actual preparation registers are
45BF16/44FP16 versus P46's42; consumer143unchanged, zero stack/spill and no
C7515/C7510 warnings. The generated source reproduces all four frozen files
byte-for-byte in an isolated temporary directory.

| Independent full-call ABBA | BF16 P46→P47, microseconds | FP16 P46→P47, microseconds |
|---|---:|---:|
| Ordinary |30.37568→28.80904 (5.158% lower)|30.35412→28.81276 (5.078% lower)|
| Graph |30.40788→28.63972 (5.815% lower)|30.54520→28.85304 (5.540% lower)|

Each mode has four fresh processes,312original-input FP64 gates and319,488exact
inside-exponent comparisons, all passing. They repeat78deterministic numerical
configurations, with10timing groups per candidate/dtype. Clocks are unlocked,
without a confidence interval. Full preparation and compute remain timed;
Graph encodes current-pointer descriptors only during capture. Separate pairs
P45→P46 and P46→P47 are not a direct controlled P45→P47 comparison.

Own ordinary78checks and79,872exponents pass. All printed case errors match P46,
maximum BF16/FP16 eF1.61320431617e-5/2.16394614392e-5. A separate same-input
frozen-P46 control checks every16-bit value in the11high/low planes on both
sides, plus every inside exponent.54fresh configurations, ninepatterns,
both dtypes and seeds47999+trial*353+pattern*61 yield311,427,072native-value
comparisons and55,296exponents, zero bitwise errors. This is a pre-MMA diagnostic,
not another FP64 matrix trial or separately sanitized result. Source/build/
binary/rows are frozen in `p47_pack_bits_manifest.json`.

P47 owns seed47951+trial*347+stream*197+pattern*59, ninepatterns,
108configurations/864interleaved ordinary/Graph calls on two independent streams
and workspaces,128float guards before/after output. All pass. Scoped memcheck,
racecheck and synccheck (kns=int8_research) each pass36configurations/72calls,
zero errors; racecheck zero warnings/hazards. Trial0subsets are not additional
independent freshness. Graph78log is an exact copy of its own GraphABBA_b0,
not another process. Stress uses historical cuBLASGemmEx control; ordinary/Graph
use production tunedDense, all retaining original rounded-input FP64 AB/gates.
See `p47_validation.json` and manifests. No earlier certificate is transferred.

P47 is the retained ordinary/Graph research reference; P46/P45/P39 remain frozen
validated fallbacks. About28.8us still exceeds the roughly10us production SW
path. Production, Auto and static library remain unchanged, and the overall goal
is active. Only contiguous non-transposed1024cubed, alpha1/beta0 and original
FP16/BF16 inputs are tested. No arbitrary-range proof or physical floor follows.

Next proposed P48 changes only the cooperative scratch request to72floats/288B,
matching actual64A/8B maximum slots after staging was removed. It must measure
actual resources, device instructions and complete calls; the old288CTA grid
already fit old resident capacity, so a higher capacity query is not a promised
gain. No P48 source/binary/measurement is applied at this checkpoint. See
`COMPACT_NATIVE_SCRATCH_DESIGN.md`; column geometry or consumer changes stay
separate experiments.

Own first-balancedBF16 NCU,17passes per kernel: preparation18.848us/45registers/65,536B/3shared-limit CTAs, consumer17.760us/143registers/196,736B/1CTA limited by both registers/shared. Issueactive18.701/31.635percent, local read/write sectors0both. Preparation shared load/store wavefront counts2,669/2,075 and bank conflicts1,517/27; the kernel is not universally bank-conflict-free. The separate P46 capture had89,709/86,853wavefronts and6,637/2,885conflicts. Lower issueactive or larger long-scoreboard-per-issue ratios alone do not imply worse latency. All profiler-process timings are excluded from full-call comparisons; separate captures do not isolate exact causal savings or prove a physical floor. Seeprofile_p47_stats.json andmanifest.

Static CUDA12.4 cuobjdump corroboration: preparation BF16/FP16 instruction counts3744/3760→3368/3376, LDS49→9, STS82→2, BAR9→3. Both consumer functions retain1600instructions with identical normalized instruction text, including registers/operands. These are static counts, not executed instruction reductions or latency fractions. Tool identity/parser/binaries/results are frozen inp47_static_manifest.json.
