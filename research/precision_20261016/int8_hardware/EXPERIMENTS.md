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

## F2/F3 — interleaving and pair-scale numerical ablations

- F2 keeps adaptive low scales but interleaves K128, native product, associated correction and output updates to match a fused schedule. All42cases pass fixed floors; maxFP16/BF16 eF=7.2822e-5/5.3049e-4. No saturation or nonfinite.
- F3 additionally fixes low exponent=high exponent minus10(FP16) or7(BF16), so paired cross terms share an output scale. Exact paired integer dot sums precede a single FP32 rescaling.39/42 pass the conservative fixed-floor screen; three FP16 cancellation cases fail, max1.20305e-4>1e-4. BF16max9.32634e-4. No saturation. Fixed-floor failure is not a measured original Dense-dependent hardware gate failure, but lacks sufficient numerical margin to pursue this simplification. Retain the failed evidence and use F2 adaptive scaling.
- F4: an initial paired fused source failed NVCC parsing at a conditional constexpr declaration, before any hardware execution. Source/build log frozen in`../fused_int8/f4_compile_failed`. No F4 accuracy or latency result. F5 returns to independently adaptive low scales and separate cross-term rescalings.

## M4 — recover one bit in relative low scales (model only)

- Follow up the failed F3 representation by using low exponent=high exponent minus11(FP16) or8(BF16), one bit finer than F3. Keep the same42configurations/seeds as a controlled ablation. Paired exact integer sums still share one rescaling per pair.
- All42pass fixed floors, zero nonfinite and zero low-part saturation in tested inputs. MaxFP16/BF16 eF=7.28217401e-5/5.30492964e-4, essentially the adaptive model's errors. This is not an independent42-case sample or a hardware result. F3's39/42failure remains preserved, not replaced.
- Representation is supported for subsequent hardware testing, not promoted. It still needs exceptional/subnormal-range behavior, real WGMMA original-input checks and full latency. Source identity in`m4_manifest.json`.

## M5 — a priori transform bounds for quantization: rejected

- Propagate original-quadrant maxima through positive upward-rounded FP32 transform bounds, derive native-high and residual-cell bounds, then choose INT8 scales without actual transformed maxima. Same42 numerical configurations as F1/M4; adaptive representation and interleaved arithmetic otherwise retained.
-36/42 fixed-floor passes; all six cancellation cases fail, maxFP16/BF16 eF2.85228e-4/2.12624e-3. No high/low saturation and no nonfinite: the failure is lost resolution, not clipping.
- CPU constant-block counterexample has S1=1/256 quantized high integer0 with coarse upper-bound scale1/64, discarding its T1 residual cross term. FP16 model eF4.60506e-4; adaptive scale is exact for this construction. Original rounded-input reference retained. No hardware/performance claim for M5.

## M6 / P10 — global K128 inside, coalesced partial reduction

- M6 screen uses globally shared K128 inside and P9 quantization tiles A64x128/B128x32, with adaptive high/low scales and leaf-then-recombine FP32 model.42/42 fixed-floor passes, maxFP16/BF16 eF7.33055e-5/5.30556e-4. This screen changes both tile sizes and inside scope from F1/F2; do not attribute the full difference to a single variable.
- P10 replaces the strided per-column reduction with256 coalesced partial-max CTAs, storing512floats in existing workspace. Each packing CTA reduces the32partials for each of its two K halves, constructs powers once, and uses invariant factors for both A and B common paths. Exceptional exponent paths retained; corrected consumer unchanged.
- Preflight inspection found the component-timing harness still launched the old1024-CTA reduction. The first compilation step107488.2031 was stopped before any device execution, terminal137; log retained as p10_preflight_stopped_build.log. The diagnostic grid was updated to8x32 with the actual invocation, then rebuilt.
- Ordinary72 original FP64 checks all pass, every printed eF equals P9 on these configurations. Complete BF16/FP16 calls54.794/54.421us versus SW10.234/10.021us. FP16 preparation diagnostic30.257us, individual inside/A/B stages3.768/13.135/11.697us; individual stage times are diagnostics, not additive substitutes for full calls.
- Despite lower latency than P9, coarser inside is not accepted. M7 introduces within-slab alternating imbalance, which the prior fixed patterns did not cover; model rejects the coarser scale. Full precision/performance goal remains active.

## M7 — adversarial within-slab imbalance (fresh numerical inputs)

- Fresh seeds2701–2703, two dtypes, six configurations each measured with globalK128 and per-column inside. Alternating K columns reverse which original K half has a2^-12 factor, with reciprocal B scaling. FP64 reference always uses the original rounded A/B, and both variants use identical numerical operations and quantization tile sizes.
- K128 passes0/6 fixed floors; maxFP16/BF16 eF2.11315e-2/1.55680e-1. Column passes6/6, max6.70704e-6/5.54585e-5. Zero nonfinite and saturation. This identifies scale granularity as the cause and invalidates treating P10's old72-case pass as a general improvement.
- Fresh hardware comparison uses different seeds2801/3072/3343, frozenP9/P10 device sources with only host dispatch stripped to avoid compiling unrelated unbalanced variants, and the unchanged production Dense gate. Actual fresh hardware result: P9 passes6/6, P10 fails6/6. Maximum P10FP16/BF16 eF2.09349e-2/1.54619e-1, versus P9 6.65145e-6/5.55399e-5. All nonfinite0. No timing or sanitizer acceptance inferred from this diagnostic. Frozen source/log/binary identities in alternating_hardware_manifest.json. Working headers, compare driver and compare binary restored to frozenP9; P10 failure and old timings remain archived.

## P11 — coalesced partial reduction while preserving per-column exponents

- Return to P9's exact numerical algorithm. Add262144bytes of caller workspace for A/B maxima per K column and row/column chunk. A reads contiguous32x128 tiles and combines two thread maxima per K; B uses warp reductions over contiguous32columns. A second128-CTA kernel finalizes the same1024 per-column exponents. P9 packers and corrected consumer remain unchanged.
- Default research checks append the new alternating-imbalance pattern:13patterns x3trials x2dtypes =78 original-input FP64 checks, all pass. Also compare every generated exponent against the original P9 reduction on each input:79,872 values, zero differences. Old72-case printed eF values remain identical to P9.
- Complete BF16/FP16 calls66.192/66.433us versus currentSW10.334/10.086us, no measured gain over P9 65.711/65.457us. FP16 partial/finalize/A/B diagnostic stages7.564/2.555/11.397/18.414us. Extra finalize launch and partial-buffer traffic are included in the full invocation; no timing removed.
- Partial32regs/1024B shared, finalize16regs,0stack/spill. Ordinary78-case evidence only, no Graph/dual-stream/sanitizer transfer. Freeze P11, then test P12 changing only partial-buffer access layout and staging B warp maxima in shared memory before contiguous stores.

## P12 — contiguous partial-max stores, same exact per-column scaling

- Keep P11's numerical operations and final1024 exponents. Stage128 B warp maxima in shared memory, then have128threads store both A/B maxima contiguously in chunk-major order. Update finalize loads to the matching layout. Partial shared memory increases1024→1536bytes;32regs,0stack/spill. Finalize16regs, consumer196regs,0stack/spill.
- All78 original-input FP64 checks pass, including six alternating-imbalance cases.79,872 per-column exponents equal the original P9 reduction exactly. All old72 printed eF values remain identical to P9.
- Complete BF16/FP16 calls63.512/63.246us versus currentSW10.334/10.104us. This is a modest measured reduction from P9 65.711/65.457us and P11 66.192/66.433us; no independent cross-process performance certification or claim of production competitiveness.
- FP16 partial/finalize/A/B diagnostics4.812/2.812/11.060/18.325us. Full-call timing includes every launch, descriptor encoding, workspace traffic, packer and main kernel. The workspace is12,328,320bytes (+262144 from P9).
- Retain frozenP12 as the current global-packing research source/binary. P10 remains rejected by fresh hardware counterexamples. No P7 Graph/dual-stream/sanitizer validation transfers to P12; production unchanged and complete accuracy/performance objective remains active.

## Frozen P12 B-packer profile (diagnostic only)

- Standalone driver runs only one balanced BF16 B pack after exact inside setup, using frozenP12 headers. NCU17passes, caches/clocks left uncontrolled, profiled28.86us versus ordinary component~18us; not a performance acceptance measurement.60regs,0stack/spill,41.32KB shared in this separately compiled driver.
- DRAM1.29%,memory14.06%,compute11.42%,issue-active24.85%,eligible warps0.36 per scheduler. No-instruction-fetch wait7.7cycles,37.31% of average20.74-cycle issue gap. Kernel startup and code jumps can both contribute; cannot attribute all of this to one source line.
- One selected B kernel has25,464 static SASS instructions,2,160 MUFU.EX2 and3,345 BRA. Static presence does not measure which rare exponent branches execute. Frozen driver/source/binary/NCU/SASS identities in profile_b_p12_manifest.json. Readable details export has trailing whitespace removed; raw NCU report is unchanged.

## P13 — warp-distributed B inside factors

- Each B warp visits16 K rows. Cache the row-specific exponent/factor pair in lanes0..15 (duplicated in16..31), guard all16 rows once with warp-all, and shuffle the two factors for each element iteration. General exponents retain the original per-element path. Per-column exponent computation, numerical representation, metadata and main kernel unchanged.
-78/78 FP64 checks pass, all printed eF values identical toP12,79,872 per-column exponents equal the original reduction. CompleteBF16/FP16 54.909/54.805us vsSW10.341/10.104us. FP16 B packing diagnostic9.480us vsP12 18.325us, A10.822us.
- B register usage rises60→122 BF16 /59→118 FP16,0stack/spill, shared41344bytes. The simpler common path still reduces measured full latency. No independent performance certification or Graph/dual-stream/sanitizer transfer. RetainP13 as next research base; still performance-rejected versusSW.

## P14 — outline exceptional packing exponents (no latency gain)

- Only packing uses a compact normal-power wrapper with one noinline ldexpf device function for other exponents. Shared pow2.cuh and the corrected consumer are unchanged, so existing fusion/source dependencies retain their original helper.
-1,048,576 full-bit float tests over exponents[-300,300], signed zero, normal/subnormal extrema and arbitrary bit patterns match directldexpf; NaN payload differences treated as equivalent as in the original test.0errors.
-78/78 original FP64 checks pass; all printed eF equalP13. CompleteBF16/FP16 54.869/54.974us vsSW10.281/10.115us: no meaningful improvement overP13 54.909/54.805us. FP16 A/B packing11.230/9.596us, effectively unchanged.
- BF16 B entry static SASS shrinks28,264→13,952 instructions (outlined helper body separate),497CALL instructions in that entry; static call presence is not execution count. B117/115regs, A158/146regs,0stack/spill. Code compaction alone does not identify the current common-path bottleneck.
- FreezeP14 and its compact-scaling validation. P13 remains the best measured latency reference; P14 is a semantically equivalent experimental base for a new bulk-packing design that can reuse four loaded/scaled quadrants. No production changes or broader validation claims.


## P15/P16 — reuse original quadrants in one combined packing launch: rejected

- P15 processes one A or B quantization tile per CTA, retaining four exactly scaled FP32 original quadrants and writing all seven native planes plus nine INT8 planes. A/B packing shares one96-CTA launch. Per-column inside and the corrected consumer remain unchanged; explicit RN transform order, actual transformed maxima and transposed global B layout are preserved.
- P15 uses256threads. Both dtypes use255regs/thread; balanced FP16 has1472B stack,1656B spill stores/1492B spill loads, BF16 has936B stack,1180/1124B spill stores/loads. Static shared8448B. Full78/78 precision checks and79,872 exponent comparisons pass; all printed eF equalP14. FullBF16/FP16 calls69.080/69.588us, slower thanP13~55us. FP16 combined packing33.422us; reject latency.
- P16 halves per-thread elements by using512threads; all tile and quantization boundaries are unchanged. Balanced FP16 uses128regs,736B stack,944/848B spill stores/loads; BF16 uses128regs,496B stack,488/488B spill stores/loads. Static shared8576B. Again78/78 precision and79,872 exact exponent checks pass, printed eF identicalP15. FullBF16/FP16 calls58.509/64.778us. FP16 combined packing27.667us; still slower thanP13. Spill reduction alone does not establish a faster design, and these two dtypes show different full-call effects.
- P15/P16 snapshots, build/results and binary identities are preserved in their manifests. No production promotion or transfer of P7 Graph/stream/sanitizer evidence. Register pressure and measured spills argue against further bulk packing as currently implemented, though no profiling establishes spills as the sole cause.
- Stall reassessment afterP14/P15/P16: outline-only code compaction was neutral; quadrant reuse introduced spills; additional threads did not beatP13. Return to frozenP13 for a new B-packer profile. NVIDIA's [shared-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks) describes bank serialization and row padding; the [Hopper guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) documents the64K-register budget. Hypothesis to measure: B INT8 shared transpose has32-byte rows, causing an8-bank stride between adjacent lanes reading rows; pad rows to36bytes while preserving all numerical and global-layout contracts. Profiling and the full ordinary gate precede any acceptance.


## B-packer reassessment on frozen P13

- One BF16 balanced B pack with the same fresh uniform seed as the P12 profile; frozenP13 packing code, no corrected consumer. NCU17passes, clocks/cache controlnone;14.304us profiler duration is diagnostic, not an ordinary latency measurement.
- Measured shared-load bank conflicts516,142; shared-load wavefronts593,326; store conflicts39, store wavefronts79,207. The leading516,096 predicted excess load wavefronts for nine quantized planes at8-way bank serialization matches the order and almost all observed conflicts, but the measurement is aggregate and includes metadata access. B122regs,41,320B static shared, issueactive36.30percent.
- P17 will change only B shared transpose row pitch32→36bytes on frozenP13. A path, global layout, quantization and per-column inside remain unchanged. Arithmetic has no intended change; full78 checks,79,872 exponent comparisons, printed-eF comparison and complete-call timings remain required. New profile needed to verify actual conflict reduction.


## P17 — pad only B INT8 shared transpose rows

- FrozenP13 numerical implementation with32→36byte shared transpose row pitch. No operation order, per-column inside, quantization, corrected consumer or global layout change. Both balanced B packers use114regs,46,464B static shared in the full comparison build,0stack/spill; A/consumer unchanged.
- All78/78 original-input FP64 checks pass, all79,872 inside exponents equalP9, and all printed eF equalP13. CompleteBF16/FP16 calls53.807/53.499us vsSW10.354/10.113us. FP16 B pack8.662us vsP13 9.480us. This approximately2percent full-call change is a small, unlocked-clock cross-process result, not independent performance certification. P17 is the working research candidate; production performance remains rejected.
- NCU on one frozenP17 BF16 B pack, same driver seed/options asP13: shared-load conflicts974 versus516,142, shared-load wavefronts78,158 versus593,326; store conflicts93 versus39. Reduction of measured load conflicts>99.8percent supports the address-layout hypothesis. Residual metadata conflicts remain; do not claim zero conflicts. Issueactive47.38percent vs36.30percent. Profiler duration13.760us vs14.304us is diagnostic and excluded from ordinary latency evidence. Separately compiled static shared46,440B; full comparison has46,464B.
- Sources/results/binaries in p17_manifest.json; profiling driver, reports and metrics in profile_b_p13_manifest.json/profile_b_p17_manifest.json. Still no later-candidate Graph/dual-stream/sanitizer transfer. The large remaining full-call gap requires a different reduction of producer/consumer work.


## P16 bulk producer diagnostic: actual local-memory traffic

- One frozenP16 BF16 joint A/B pack, fresh uniform seed1901901, NCU17passes, no corrected consumer. Measured local-load331,776sectors and local-store249,856sectors (32B sectors); shared-load conflicts536,348. Compiler reports for this separate diagnostic kernel differ in static shared usage from the full comparison, so do not transfer exact allocator counts between binaries.
-29.376us profiler duration excluded from ordinary timings. Register count128, static shared8,464B, issueactive26.63percent. Actual local accesses on normal input support reducing temporary lifetime, while bank conflicts remain a second cost. This does not establish that all local sectors are spills or all time is caused by them.
- P18 changes only P16 temporary storage: retain four transformed arrays, recompute the identical dtype cast and RN residual after reduction rather than keeping two additional high/low arrays across it. An empty read/write register-operand asm prevents CSE from extending the old cast/residual lifetimes. Thread count512, tile geometry, inside scales, output/quantization layout and consumer remain unchanged. Original fixed gate and complete-call timing required before acceptance.


## P18 — recompute casts/residuals rather than retaining high/low arrays

- P16 baseline with the temporary-lifetime change described above. Same512threads, all quantization and scale granularity unchanged. Balanced FP16 still128regs,512B stack,520B spill stores/552B spill loads; BF16 still128regs,384B stack,388B spill stores/412B spill loads. Static shared8576B. Fewer compiler-reported spills thanP16, but spills remain.
- Full78/78 original-input FP64 checks and79,872 exact exponent comparisons pass; all printed eF equalP17. CompleteBF16/FP16 calls60.306/61.444us versusP16 58.509/64.778us; no improvement overP17 53.807/53.499us. Reject performance. Dtype-dependent effects prevent claiming a general latency reduction from this rematerialization alone.
- FrozenP18 sources/build/results/binary identities preserved. Next controlled comparison can use256threads with the same rematerialization: the smaller block raises the per-thread register budget and completes the block-size × temporary-array-lifetime ablation. P17 remains the fastest measured research reference; no production change.


P17独立ABBA复测（P13→P17→P17→P13，四个新进程）共312次精度检查通过，仍是78个确定性数值配置。两次进程共10个组的完整耗时中位数：BF16 P13/P17为54.571/54.378μs，FP16为54.810/54.042μs；降幅分别0.35%/1.40%，小于首轮约2%。GPU未锁时钟，没有统计置信区间，不宣称显著性。精度、hash及全部组保留于[p17_abba.json](p17_abba.json)及manifest；保留P17作为内部转置布局已修复的研究候选，整体性能目标继续拒绝。


## P19 —256-thread rematerialized bulk pack: rejected

- Same rematerialization asP18,256threads asP15; updates both actual preprocessing and balanced/unbalanced diagnostic launch branches before compilation. Runtime numerical algorithm unchanged. Balanced FP16255regs,1096B stack,1088B spill stores/1124B spill loads; BF16255regs,672B stack,664/664B spill stores/loads. Shared8448B. Register pressure still produces spills.
- All78 fixed original-input FP64 checks and79,872 exponent comparisons pass; printed eF exactly equalsP17. FullBF16/FP16 calls65.606/74.240us. Reject performance and preserve frozenP19 manifest.
- The completed two-axis comparison isP15(256, retained high/low),P16(512, retained),P18(512, rematerialized),P19(256, rematerialized). None beatsP17; thread-count changes are not a robust workaround for this bulk producer. Restore frozenP17 source/binary after archiving. Next algorithmic screen retains per-column scaling and pairs INT8 dots before scale restoration, seeking fewer correction waits/rescalings while preserving all eight integer dots.


## M8 — per-column relative scales and five correction leaves (new model inputs)

-48 fresh numerical configurations: eight patterns including alternating K-column imbalance, seeds3901–3903, both dtypes. Same original rounded-input FP64 reference and fixed floors. Three controlled arms share each original input: adaptive8leaves, relative8leaves, relative5leaves. All144 measured rows pass, representing48 configurations rather than144 independent inputs. Zero nonfinite and low saturation.
- Relative low exponent=high exponent minus11(FP16)/8(BF16). The paired arm really sums three two-dot pairs while still integer within eachK128, then rescales once and accumulates each resulting correction leaf acrossK; it does not silently use the older unpaired leaf-then-recombine path. It still executes eight integer dots, grouped into five correction leaves.
- MaxFP16/BF16 eF: adaptive6.76968e-5/4.97240e-4, relative-unpaired6.76981e-5/4.97240e-4, relative-paired6.76981e-5/4.97240e-4. Largest measured paired integer156,986; absolute bound2*128*127²=4,129,024<2²² guarantees exact representability in the FP32 numerical model and avoids INT32 overflow. No hardware or speed claim yet.
- M8 uses global per-column inside andP17 quantization tiles64x128/128x32; this closes the coarse-inside and small-tile gaps in earlierM4/M6 screens. FP32 bmm models native leaves rather than Hopper's exact accumulation. Sources/imports/results frozen bym8_manifest.json; next hardware experiment must retain fresh original-input gates and all preparation costs.
