# INT8 hardware experiments

Production `solution/` unchanged; these are standalone research candidates, not production iteration371. Full original-input FP64 gate remains fixed, with fresh deterministic inputs in launch/Graph modes. Every timing row includes preprocessing and tensor-map setup. Sanitizer/NCU runs are excluded from ordinary latency evidence.

## P1 — first complete INT8 candidate

- Signed INT8 `m64n32k32` primitive:18 cases (six patterns, three scales), exactly matches CPU int64 dots and power-of-two rescaling. Bound2064512 reached without overflow. First compile failed due a nondependent CuTe descriptor host instantiation; archived and fixed with a dependent template boundary.
- Seven native main products plus eight first-order INT8 products, per-K128 rescaling, all FP32 output. Dynamic per-column inside scaling, seven packed high planes and nine INT8 planes per side, no allocation/host sync in run. Workspace12066176bytes.
- Full original six patterns plus six extended patterns ×three seeds ×two dtypes:72/72 launch checks passed. Maximum BF16/FP16 eF=5.0954e-4/6.5673e-5.
- Complete-call paired medians BF16/FP16=157.935/159.656us; currentSW10.208/9.941us, dense comparator6.522/6.555us. **Reject performance**. Diagnostic preprocessing82.603/82.828us is included in, not additional to, the complete-call time.
- 168registers/thread,32byte stack frame,0spill stores/loads for the corrected kernel; producer up to149registers,0spill. No sanitizer acceptance yet.
- Baseline source snapshot and logs in`p1/`; identities in`p1_manifest.json`. NCU one first BF16 corrected kernel, clocks/caches uncontrolled; all profiler-run timing arms excluded. Initial shared/tmp lock failure preserved, resolved by private TMPDIR; no other user's lock changed.
- Next change: replace common-range scalar `ldexpf` with exact power-of-two multiplication, retaining general exponent fallback and an independent GPU equivalence check. This is an implementation cost change, not a new numerical representation.

## P2 — normal-range power-of-two multiply

- Independent1,048,576-case GPU equivalence check against `ldexpf`, exponents−300..300, including normal/subnormal extrema and signed zeros:0errors. Fallback retained outside the range in which a single normal scale is representable.
- Full72/72 precision checks pass; every printed error metric identical toP1. Complete-call BF16/FP16 medians159.533/159.497us; preprocess92.460/92.860us. **No performance improvement**; the initial ldexp-dominant hypothesis is unsupported. P1/P2 timings are separate processes with unlocked clocks, not a paired source-to-source significance result.
- Snapshots and identities in`p2/` and`p2_manifest.json`. Next action is separate kernel timing and stage waits, not further math-function optimization. The overall FMM accuracy+performance goal remains unmet.

## P3 — packer nonalias promise and stall reassessment

- Packer input/output/workspace ranges are distinct by allocation; add restrict only to these valid disjoint ranges. Full72/72 checks pass. BF16/FP16 complete156.876/157.985us; no material solution to the latency gap. Preprocess91.220/91.708us. Treat this as a failed bottleneck hypothesis, not a win.
- Separate balanced BF16 kernels:inside9.820us, A pack62.514us, B pack17.837us. With no balance in a diagnostic process:A19–20us,B12us. These are isolated measurements, not additive overlapped role costs.
- NCU P1 corrected kernel80.96us, Tensor elapsed3.80percent, issue active17.75percent; not a performance acceptance run.
- Sparse same-CTA consumer trace of the P3/P2-equivalent core: median operand-ready6.144us, prepare/issue6.240us, exposed wait2.368us, post-MMA metadata/rescale35.456us, free1.408us, between-stage16.704us; same-CTA total68.576us. Median phase sums need not add to median total. Trace core74.694us,184regs vs168baseline, no spill but32byte local stack. One fresh BF16 FP64 check matches baseline eF exactly. Diagnostic timestamps perturb scheduling and exclude preprocessing/output tail.
- SASS has local stores/loads for runtime-indexed constexpr operand-plane tables, despite ptxas reporting0spill. The post-MMA interval includes these accesses and three global metadata loads, not just arithmetic. Next candidate moves32correction-K metadata entries into lane registers, broadcasts by shuffle, and replaces tables with arithmetic.
- Reassessment sources: [NVIDIA restrict discussion](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) cautions about register/occupancy tradeoffs; [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) defines async-proxy ordering and WGMMA completion. No removal of required proxy fences or waits based only on a latency guess.

## P4 — register metadata instead of runtime table/local/global chasing

- Replace operand-plane table lookup with an equivalent arithmetic mapping. Preload32correction/K metadata entries one per lane, broadcast from warp registers on demand. Corrected kernel stack32->0bytes, registers168->162, no spill.
- Full72/72 precision checks pass with identical printed eF toP1. Complete BF16/FP16 medians149.670/148.159us; improves this prototype but far above currentSW~10us. No production promotion.
- Next isolated change: hoist power-of-two exponent validation and scale construction outside the per-output-element loop. Preserve general exponent fallback and separate RN multiply/add.
