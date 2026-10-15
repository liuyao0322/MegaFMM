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

## P5 — validate/build scale once per correction-K entry

- Precompute exponent sums and normal power-of-two factors in lane registers; one common/exceptional branch per correction-K block, not per output element. Exceptional exponents retain the exact prior operation. Keep separate RN multiply/add, no new FMA rounding.
- Full72/72 checks pass, printed eF unchanged. Complete BF16/FP16 medians129.031/128.861us. Registers166,0stack/spill. Improvement is substantial within the rejected prototype, but still~13x currentSW; overall target not met.
- Next candidate specializes the packer's uniform plane selection before the element loop; no numerical representation or gate changes.

## P6 — compile-time packer plane bodies

- Dispatch each CTA to a compile-time plane body before its element loop. Numerical operations and full gate unchanged. Full72/72 checks pass with identical printed eF.
- Balanced preprocessing BF16/FP16=71.715/71.859us, down from~92us. A38.0/38.1us improves, B23.3/23.2us regresses from~17.8us. Template-local shared allocations raise B shared usage to41344bytes; A registers143/144. Do not attribute net gain solely to arithmetic count.
- Detailed complete-call aggregates in`p6_balanced.json`; performance still fails the currentSW target. Compiler unreachable-loop warnings refer to constant raw-plane early returns, not a failed build.
- Next isolated change applies P5's proven uniform-branch hoisting to INT8 quantization factors in the packer.

## P7 — hoist quantization scale construction and validate hardware

- Hoist quantization exponent checks and exact normal-range scale construction outside the packer's per-element loop; retain general exponent fallback. No representation, threshold, floating-point addition order or production change.
- All72 ordinary checks pass with identical printed eF toP1. Full-call BF16/FP16=94.593/94.380us versus currentSW10.192/10.023us. Preprocess58.200/57.815us, included in the full time. **Reject performance**, approximately9.3–9.4x slower than currentSW.
- Graph72/72 passes, full-call91.183/89.846us versus currentSW9.839/9.896us. Same seeds/configurations as ordinary; these are not72 additional independent numerical configurations.
- Separate fresh extended dual-stream/Graph stress:72cases,576 interleaved calls, original-input FP64 gate and128-float output guards all pass. Independent input seed set from compare.
- CUDA12.8 memcheck/racecheck/synccheck:each24cases/48calls, filtered to`kns=int8_research`; all0errors and racecheck0warnings. Do not describe this instrumentation scope as the full72case/576call stress.
- Audit verifies all P1–P7 source/result/binary identities, primitive18exact checks,1,048,576power-of-two equivalence cases, and all five production file hashes. Status explicitly`accuracy_validated_performance_rejected`; overall goal remains active.
- Evidence and limitations in`HARDWARE_REPORT.md`. Next work targets repeated inside-scale work in the packer, followed by consumer scheduling; no promotion based solely on numerical success.

## P8 — hoist invariant per-thread A inside scaling

- A pack has128columns and a256element per-thread stride; each thread reuses exactly two input column exponents across its32elements. Hoist these loads, exponent validation and scale construction, then select a compiled fast or general loop before reduction barriers. B's numerical body is unchanged. No gate or representation change.
- Full72/72 ordinary FP64 checks pass; every printed eF equalsP7. Complete BF16/FP16=77.415/76.751us, currentSW10.181/9.994us. Preprocessing41.269/41.198us; A11.255/11.087us, B17.036/17.111us, inside9.320/9.943us. Still reject performance.
- Source/binary/results frozen in`p8/` and`p8_manifest.json`. Graph/dual-stream/sanitizer evidence in the hardware report is specificallyP7, not silently transferred toP8.
- Next isolated experiment moves consumer product/term selection outside its16-output loops, preserving the scalar addition order.

## P9 — select recombination branch before output loops

- Move main-product and correction-term branch selection outside each16-output loop, preserving every scalar add/subtract and all barriers. Full72/72 ordinary checks pass, printed eF identical toP8.
- bf16: complete 65.711us versus currentSW10.230us.
- fp16: complete 65.457us versus currentSW10.024us.
- Corrected kernel registers166->196,0stack/spill. This isolated change does not address the end-to-end gap. P7 sanitizer/Graph acceptance is not transferred toP9.
- Next: screen a structurally fused design numerically, using local K128 inside scales and CTA-local quantization, before implementing a new on-chip producer/consumer pipeline.

## F1 numerical screen — local K128 inside scales for fused packing

- A output tile32rows and B output tile64columns define each output CTA. Select two independent inside powers perK128, for the two original K halves, from all four local raw A/B quadrants. Quantize with local32x128 A and128x64 B scales, instead of the global packer's64x128/128x32 blocks.
- 7patterns x3fresh seeds1701–1703 x2dtypes=42cases:all pass fixed floors, no nonfinite or low-part saturation. FP32 batched leaves approximate native accumulation; correction integer dots are exactly representable. Model has no performance/synchronization claim.
- This screen retains leaf-then-output recombination. Interleaving K and main/correction accumulation for an on-chip design is a separate numerical change and needs its own test. Evidence in`tile_inside_adaptive*`; manifest identifies the pre-interleaving script.
