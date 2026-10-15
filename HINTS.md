# Hints

## Multi-size design handoff — 2026-10-15

- User asked how to generalize the 1024-specific implementation, then requested a design document for another agent to continue. The handoff is `research/size_generalization/DESIGN.md`.
- This handoff turn is documentation only; no new size implementation or GPU benchmark has been performed. The document proposes the scope for the next implementation task, rather than reporting new support or results.
- When that task is started, the proposed aligned multi-size scope supersedes the older 1024-only experiment scope for that task. Preserve original accuracy gates, Auto=Dense until acceptance, and the stopped status of FP8 residual correction.

## Current user directives — 2026-10-14 architecture reset

These directives supersede the previous experiment-specific instructions to preserve
all historical implementations or continue speed-only iterations. They come from
the user's requests in this conversation:

- Delete old code; redesign the SW architecture next. All old SW implementations,
  including Mega45 and Mega53, are removed. Do not silently restore them.
- Investigate a unified intermediate buffer and separate TMA, Tensor Core and
  CUDA Core warp roles. Current implementation and its measured scope are recorded
  in README and the first-version report.
- Numerical acceptance is a prerequisite for declaring a winner. Report numerical
  failures prominently; performance-only rankings do not establish a valid solution.
- First validate the new numerical representation and complete original-input
  precision gate, then optimize the accepted implementation's performance.

## Preserved baseline and contracts

- Auto and Dense remain ordinary tuned cuBLASLt GEMM. Experimental SWPipeline is
  explicitly forced and supports only contiguous non-transposed 1024^3 / alpha=1 /
  beta=0. Passing the measured numerical cases does not promote it to Auto.
- A/B are BF16 or FP16; computation/accumulation and output are FP32.
- The reference computes ordinary AB in FP64 from the same original rounded A/B;
  never transform reference inputs to make an SW candidate pass.
- Preserve the fixed gate eF_SW <= 3*eF_dense + floor, BF16 floor=1e-3 and
  FP16 floor=1e-4; require all fresh trials and no nonfinite. Do not relax gates.
- Preserve explicit tuning, caller workspace, current pointer validation,
  alpha/beta semantics, and run() without allocation or host synchronization.
- Time complete invocations, including dynamic transformations. Compare matching
  dtype, shape, output, workspace and execution mode against independently tuned Dense.
- Future optimized SW enters Auto only after numerical and performance acceptance.
- Build and run CUDA on assigned compute resources, using cluster-llm-ops guidance.
  Existing scripts default to H100 allocation107488/CUDA12.4; verify allocation first.

## History and optimization workflow

- Prior committed directives are available in Git history; current directives are above.
- PLAN.md and ITERATIONS.md retain historical contracts and evidence. Architecture,
  route and resource descriptions there do not imply current source implementations.
- Keep formal historical results and trajectory snapshots. Avoid accumulating old
  candidates in solution/ merely to preserve history.
- When optimization resumes, follow AKO4ALL's bench -> iteration log -> Git commit
  protocol. The current reset is cleanup, not a new performance iteration.
- Prior session ncu probe failed. Use analytical evidence and runtime statistics
  without repeating the known failed probe unless new evidence warrants it.

## First implementation scope — latest user request

- Implement the proposed unified shared-buffer, TMA/CUDA-Core/WGMMA warp-specialized first version.
- Only test and benchmark M=N=K=1024 in this task, for BF16 and FP16.
- Compare complete invocations against tuned cuBLASLt Dense, with original FP64 reference and unchanged numerical gates.
- The first implementation and comparison are complete. The latest user goal
  supersedes the previous stopping instruction: use AKO4ALL to continuously optimize
  the current kernel, while preserving 1024-only scope and numerical acceptance.

## Active optimization goal — latest user request

- User explicitly requested sustained AKO4ALL optimization. Continue beyond the first version.
- Rank actual SW complete-call latency, retain failures, and preserve all fresh-input gates.
- On every iteration: benchmark with iter-N label, immediately log, immediately commit.
- Keep Auto on Dense until numerical and end-to-end performance acceptance.

## NCU profiling — explicit user request after optimization pause

- User paused sustained optimization, then explicitly requested NCU profiling. Profiling does not resume optimization.
- NCU2025.1.1 from CUDA12.8 successfully profiled iter76 on allocation107488/g08,39 passes per dtype. Prior failed NCU probes no longer imply NCU is unavailable.
- The initial /tmp/nsight-compute-lock failure was resolved by a private TMPDIR after checking no other ncu process was active, following NVIDIA's documented workaround. Do not remove another user's lock file.
- Evidence: trajectory/20261014_ncu_iter76/REPORT.md, summary.json and .ncu-rep artifacts. Production source remains iter79; profiling used a separate iter76 snapshot.
- Key findings: Tensor elapsed activity~18.8percent; occupancy24.93percent; eligible warps~0.32/scheduler; DRAM active<1percent in this warm-cache capture. Long-scoreboard hotspots include product-ready mbarrier consumers, not just global input loads. P stores have8-way bank conflicts and401408 excessive wavefronts.
- Dense profiling must retain the actual independently tuned configuration. Under NCU, retuning picked a slower BF16 split-K algorithm; exclude that initial capture from comparisons. Fixed Dense id39/tile18/stage35/cluster10 (BF16) and id21/tile18/stage18 (FP16) were captured instead.
- Kernel replay, cache-control none, clock-control none. Short-kernel samples and unlocked clocks limit exact percentages; profiler timing is diagnostic and does not replace normal complete-call benchmarks.

## P shared bank conflict — latest narrow user request

- User explicitly asked to fix the eight-way bank conflict when storing P to shared. This authorizes this fix and its validation; sustained optimization remains paused.
- Final iter81 (58743df) starts from profiled iter76 (456ce9d), uses row-local XOR swizzle `row*32+(col^((row&1)|((row&6)<<2)))` for both P producer and consumer, retaining row-major global output. Other arithmetic and barriers match iter76.
- NCU confirms all16 P STS and42 P LDS per dtype are1-way,0 excessive wavefront. P store excessive wavefront drops401408->0. Kernel-wide hardware bank-conflict counters are not all zero; do not conflate those counters with P instruction evidence.
- Full invocation BF16/FP16=21.216/21.661us versus historical iter76=21.245/21.883us: no significant overall speedup established. Still slower than tuned Dense12.381/9.992us. Auto remains Dense.
- Original-input numerical gates pass, independent72-check launch/Graph stress passes, memcheck/racecheck/synccheck quick tests pass. Only1024^3 tested.
- Evidence: trajectory/20261014_validation-iter81-p-banks/REPORT.md and NCU artifacts.

## Streaming recombination — latest user request

- User asked to try the proposed next version. Implement and validate the first step: P1,P2,P6,P7,P5,P3,P4 scheduling with streaming recombination and early output.
- Keep BK256, two operand slots, two conflict-free P slots and current warp roles for this experiment; do not conflate it with deeper WGMMA pipelining or warp-count changes.
- Advance operand/product slot phases by execution sequence, not logical P number. Retain original recombination operation order and original-input precision gates.
- Only1024^3 BF16/FP16; measure full invocations against tuned Dense, independent launch/Graph correctness and NCU evidence. This request does not resume an indefinite optimization goal.

- Iter82 (77e1489) implements this candidate and passes original numerical gates, independent72-check launch/Graph stress and memcheck/racecheck/synccheck. P STS/LDS remain1-way,0 excessive wavefront.
- Full standard launch BF16/FP16=21.350/21.462us, tuned Dense=12.194/9.978us. Contemporaneous ABBA comparison against rebuilt iter81 shows~1.60%/1.49% lower launch time; Graph~0.84%/1.44% lower. Small improvement, below3% threshold; no performance winner certification.
- Compiled registers remain110/thread with0 local/stack and213120 dynamic shared; Tensor activity~19percent unchanged; dynamic instructions increase~7percent. No stage timing was added.
- Evidence: trajectory/20261014_validation-iter82-streaming/REPORT.md. Keep this correct candidate; deeper WGMMA pipelining and warp-count changes have not been implemented.

## Production-chain diagnosis — latest explicit user request

- User asked to distinguish TMA supply waits from per-K WGMMA full waits. This authorizes isolated diagnostic timing, not a new optimization loop. Production solution remains iter82 unchanged.
- Independent sparse per-stage timestamps and same-invocation cumulative timing agree: per-CTA TC production-loop median BF16/FP16~13.344/13.408us; operand waits2.432/2.464us; prepare+issue8.544/8.576us; explicit wait0 residual0.352/0.352us. Sparse cross-check residual sum~0.448us.
- This corrects the prior hypothesis that per-K wait0 is the main bottleneck. Among waits, operand supply wait is~7x larger; the largest measured production interval is descriptor preparation and WGMMA issue, with execution overlapped. Do not equate this entire interval with pure descriptor cost or removable latency.
- Producer-stage fractions are~64% prepare/issue,18% operand waits,2.7% explicit wait0; these are not whole-kernel percentages. CUDA consumers still wait for P. TMA also waits for free operand slots.
- Baseline clone has identical2800 SASS instructions and operands to production for both dtypes. Trace output matches baseline bitwise on3fresh inputs; cumulative timing raises registers110->118 without spill or occupancy change; event-time perturbation~0.5–0.7% average.
- Evidence: trajectory/20261014_profile-production-chain/REPORT.md, compressed raw traces, generated sources and reproduce scripts. Only1024^3 tested.

## Descriptor issue optimization — latest narrow user request

- User asked to optimize based on the preceding production-chain findings (prompt typo interpreted in that context). Optimize WGMMA preparation/issue first; this does not resume an indefinite optimization goal.
- First candidate constructs base GMMA descriptors once per matrix term and uses CuTe descriptor iteration for K tiles. Preserve arithmetic, precision gates, warp roles and 1024^3-only BF16/FP16 validation. Measure complete invocations and verify generated instructions before claiming a gain.

- Iter83 (9d678d8) canonical descriptor iteration validates a small gain. Iter84 (e35b759) failed to compile due to an undeducible CUTLASS raw_pointer_cast template; archived, then corrected in iter85 (2870c75).
- Current iter85 keeps high descriptor word invariant and updates only low-word K address offsets; shared operands fit below256KiB. Original FP64 gates,72-check stress and all3quick sanitizers pass.
- Against rebuilt iter82, A/B/B/A full-call launch BF16/FP16=21.300/21.311->20.653/20.858us (3.04%/2.13% lower); Graph20.843/20.644->20.120/20.316us (3.47%/1.59%). Against iter83, launch improves1.03%/0.70%, BF16Graph improves0.89%, FP16Graph regresses1.20%; do not claim universal best.
- PerCTA issue interval8.608/8.608->7.776/7.744us, ready~2.4us, explicitwait0~0.352us. NCU dynamic instructions6119946/6144374->5395148/5419281; R2UR remains995968/dtype. BF16 IMAD.X319488(iter83)->3072(iter85). Registers110,local/stack0,shared213120; allP STS/LDS1-way0excess. No residency increase.
- Formal launch SW20.847/21.012us vs Dense12.506/10.012us, still slower; Auto remains Dense. Evidence: trajectory/20261014_validation-iter85-descriptors/REPORT.md. This completes the narrow optimization request; indefinite goal remains paused.

## Fixed-slot issue and staged operand readiness — user approved next steps

- User explicitly agreed to the proposed order: first specialize fixed operand slots and descriptor reuse, then try high/low operand staged readiness if the first direction is insufficient. This is a bounded follow-up, not resumption of the paused indefinite goal.
- Inspect generated SASS before claiming R2UR savings; retain changes only on complete-call comparisons with original FP64 gates. Keep1024^3 BF16/FP16, all arithmetic order, warp roles and P layout. Revalidate independent launch/Graph stress and sanitizers for any retained synchronization change.

- Iter86 (ecb7b5c) explicitly unrolls TC q=0/1, static HGMMA48->96; R2UR995968->487424/dtype, registers110->72. Its small complete-call gain justified proceeding to the already-approved readiness experiment.
- Current iter87 (c9380c3) uses high-ready and low-ready transaction barriers; issue high TMA operands before residuals, execute high-high before waiting low, then corrections in unchanged order. Advance low barrier phases even on zero-byte stages; free slots only after all WGMMA completes. Shared remains213120 bytes due to existing padding.
- Three-version A/B/C/C/B/A comparison: iter85->87 launch BF16/FP16=20.58592/20.70080->19.28576/19.30112us (6.32%/6.76% lower); Graph20.25632/20.60000->18.75104/18.79936us (7.43%/8.74%). Iter86->87 launch5.61%/5.79%, Graph4.74%/5.02% lower.
- NCU current R2UR178944/dtype (82.03% below85), real64x32WGMMA245760 unchanged; dynamic instructions4673556/4681806, registers79, local/stack0, oneCTA/SM; allP instructions1-way0excess. Staged barriers also changed code generation; do not attribute the entire gain only to TMA overlap.
- TC cumulative operand wait (high+low)2.95->2.368us versus86; issue~6.8us, explicitwait0~0.96us, TCloop11.744us. These diagnostic medians have unlocked-clock and trace perturbation limits; issue includes asynchronous execution overlap.
- Original FP64 gates,72-check stress and quick memcheck/racecheck/synccheck all pass. Formal SW19.316/19.283us versus Dense12.303/9.949us, still slower. Auto remains Dense. Final evidence: trajectory/20261014_validation-iter87-staged-ready/REPORT.md; bounded approved experiments complete, indefinite goal remains paused.

## Latest request: continue profiling and optimization from iter87

- User explicitly requested continued profiling and optimization. Continue from validated iter87, keeping1024^3 BF16/FP16, original-input FP64 gates, full-call timing and Auto=Dense until acceptance.
- Re-profile the current critical path before new candidates; investigate consumer polling/warp allocation and remaining producer waits with one change per iteration. Preserve failed candidates and validate retained synchronization changes independently.

- Iter88–96 completed the latest profiling round. Current96 is verbatim92(bf34e5a):32bit paired input transform + combined A TMA; regs79, shared213120, no spill. ActualTMA110->44/CTA; predicated-on counts required. Consumer reductions,64bit IO,restrict,two-group delayed retirement,descriptor prefetch rejected.
- Four paired campaigns vs87: launchBF16~1.3–1.7% lower,FP16~0.8–1.5%; Graph smaller and one FP16 slight regression. Still slower than tuned Dense; Auto=Dense.
- Critical new diagnostic:92->94 wait0~.992->.544us but operandwait2.336->4.192/4.352us, production11.552->12.896/12.832us. Deferring slot0 release harms refill overlap; lowering explicit WGMMA waits alone is not a performance criterion.
- Final96 originalFP64 gates,72stress,all3quick sanitizers and3DAG tests passed. Full evidence trajectory/20261014_validation-iter96-supply/REPORT.md. No physical-floor or exhausted-space claim; this round ends with six tested optimization directions and best source retained.

## User request 2026-10-15: seven-product ablation

- User explicitly asked to remove extra compensation and measure performance. Test1024^3 BF16/FP16 seven main products, omitting residual generation/loads/waits/multiply terms. Preserve original FP64 numerical gates and report failures plainly. Keep workspace/shared/roles unchanged to isolate compensation cost. This is a measurement experiment, not authorization to weaken accuracy acceptance.

- User-requested no-compensation ablation completed as97(503c7de):7products, low generation/stores/TMA/waits removed, same shared/workspace/roles. A/B/B/A full-call launch BF16/FP16~14.176/14.401us vs15term18.921/19.153us (~25% lower); Graph13.566/14.061us vs18.603/18.811us. Still slower than Dense. Both FAIL original gates(eF.0060152/.000754626,limits.00100256/.000103753). Source restored verbatim to validated96 after experiment; final gates PASS. Evidence trajectory/20261015_validation-iter97-no-compensation/REPORT.md.

- Latest user steering: explicitly KEEP the seven-product no-compensation version as current source, despite reported accuracy failure. This supersedes the earlier automatic restoration decision; do not restore compensated96 as current. Preserve original accuracy gates and Auto=Dense.

## Current seven-product profile, user request2026-10-15

- Profile only, no solution edits or compensated restore. NCU confirms56regs,no spill,shared213120,oneCTA/SM,~25%warpoccupancy,Tensoractivity~13%,HBM<.4%,L2hit97–98%,P16STS+42LDSall1way0excess.
- Current TC same-invocation median BF16/FP16 ready2.880/2.912us,issue2.368/2.368,wait0 1.408/1.408,total7.936/7.936; transform+grid inclinit2.624us. Sparsecrosscheck ready2.688,issue2.368,wait01.312. TMAfreewaitstage-median-sum4.992/5.056us,consumerPwait4.448us,TC Pfreewait.384us; overlapping,notadditive.
- Current bottleneck is more exposed operand supply/slot reuse latency + synchronization, not prior15term issue-dominant chain. HBM not bandwidth-bound in testedwarmcache workload; no claim about coldcache. Shared has98304B unused lowoperand space; reclaim may enable deeperpipeline, but128CTAon132SM means2CTA/SM capacity alone doesnot improve achievedoccupancy. No candidate implemented.
- Diagnosticbaseline SASS identicalproduction(2416instructions/dtype),270timingrowsbitexact; independentFP64gate stillFAILboth. Evidence trajectory/20261015_profile-iter97-seven-products/REPORT.md.

## User-authorized seven-product optimization round,2026-10-15

- Explicit directions: reclaim unused shared and test deeper operand prefetch with timely release; tune K blocking and issue/wait rhythm; reduce input-transform/grid-sync front cost. Keep current seven-product no-compensation semantics, originalFP64gates and Auto=Dense. Report accuracy FAIL plainly; compare bitwise with97 to catch synchronization/arithmetic regressions separately. Use complete1024^3 BF16/FP16 timings and independent diagnostics/sanitizers on retained changes.

- Completed authorized98–108round,selected100(76b4d09):4compactoperand slots,BK256,immediatewait0/free perK. Recovered98304B reusedfor2extraslots; finalshared213120,regs53,nospill,oneCTA/SM.
- Finalpaired vs97launchBF16/FP16~14.238/14.386->13.060/13.248us,Graph13.565/14.064->12.273/12.534us(~8%launch,9–12%Graph). Still slowerDense.
- CurrentTCready1.440/1.472us,issue3.648/3.680,wait0 .512/.512,loop7.072/7.104;front2.592us. Dynamicinstructionsup3.204->3.921M despitegain. Do not attributeallgainonlytoTMAorclaimissueisallsoftware.
- K128sixslots,K512twoslots,twoMMAgroups,64bittransform,splittransform,initCTAbarrierremoval,TCproductunroll rejected.104invalidtruncatedstorefixed105; exclude104timings. Frontcostnotimproved.
- Final100matches97bitwise72checks and passes3quicksanitizers;270diagnosticexact,DAG3pass. OriginalFP64gates STILLFAIL,eF.0060152/.000754626. Userexplicitlykeepsuncompensatedroute,AutoDense. Evidence trajectory/20261015_validation-seven-prefetch/REPORT.md.

## Latest sustained goal,2026-10-15

- User explicitly requested continued optimization until faster than Dense. Resume sustained optimization from100; target both BF16/FP16 ordinary full-call1024^3 faster than contemporaneously independently tuned cuBLASLt Dense. Track Graph separately. Keep seven-product uncompensated route, original FP64 gates and Auto=Dense; accuracy failure must stay explicit. This supersedes earlier bounded-round scope.

## Sustained-goal checkpoint through162 (goal still active)

- Selected162(d8e6261):140 asynchronous608-worker pack/128-consumer architecture plus4MiB compact high planes and no terminal CTA barrier.64regs,213120Bshared,no local sectors. Current original FP64 gates stillFAILboth; AutoDense.
- Paired140/162 A/B/B/A:162 launchBF11.33696us vsDense12.19088;FP11.54240 vs9.96544. GraphBF10.95920 vs11.26336;FP11.16416 vs9.50496. BF wins,FP stillneeds~1.577us/13.66percent SW-time reduction. Goal NOT complete.
-162 passed72bitwise97checks,all3quick sanitizers,256two-stream/Graph launches withpoisonedworkspaces. Private CUDA12.4 driver-workspace barrier is an internalABI dependency; no cross-call solution state.
- Full162NCU~2.519Minstructions,Tensor16.7–16.9percent,L2throughput~50.8percent,hitrate~98percent,PsourceSTS/LDS1way0excess. No floor claim.131 phase diagnostics predate asyncfrontend and mustnot be called162phase timings.
- Rejected newer directions:K128eightstages142/143,K512145;DSM split-product cluster146/147;multicastA150;local shared transform153/154;dualaccumulator interleave158/159;warpMMA160/161.151/152numericallyinvalid swappedP3/P4 fixed153.146NCULaunchFailed no reliablecounterdata.153/158/160spilldespitecuobjdumpLOCAL0; inspectSTACKandSTL/LDL.
- Evidence trajectory/20261015_optimization-goal-checkpoint/REPORT.md. Continue current critical-path diagnostics; next256consumerworkers has simpleconstant8row chunk offsets and has not been tested. ThirdPslot also untested on this architecture. Do not stop or claim exhausted directions based on this checkpoint.

## Sustained-goal checkpoint through208 (goal active, not paused)

- Restored exact162(d8e6261) in208(cf5c90e). Fresh ordinary launch SW BF/FP11.32544/11.52032us versus independently tuned Dense12.29984/9.92480us; BF1.0860x,FP0.8615x. FP needs1.59552us/13.85percent SW-time cut. No new winner; AutoDense; both original gatesFAIL.
- Current162 phase diagnostics replace131 stale frontend figures: ready3.136us,issue3.328BF/3.360FP,wait0 .608us,TCloop8.608/8.640us;72regs vs64,no spill,3freshdatasets bitexact. Do not add overlapping intervals or call issue entirely software cost.
-165–207 tested many rejected directions.194float2flat;196N64singleCTA14us;197/198dualTC12.2/11.8us;199/200cluster8 insufficient active capacity (15vs16);201persistent15clusters34us. NCU201LaunchFailed/nan is invalid.
-202private PTX product accumulators eliminateC7515 but80regs8Bstack with real local traffic and3.638Minstructions.203hoisted dispatch13.1us;204full static TC PTX11.32/11.63us, no paired gain certified.205pingpongwait1 11.61/11.90us despite74regs,0spill,1.895Minstructions. No reason to tune205 block size just for spills.206A4/B5 and207A5/B3 independent producers12.10/12.03FPus,not retained.
- Parallel leaf189rowmajor+TCpack19us;192/193TMAreduce15–16us and unordered FP32 sums not retained.186spills;18779regs no spill still32us; epilogue9–14us instrumentationperturbed,188coalescing cuts10us but remains21us.
- Evidence trajectory/20261015_optimization-goal-checkpoint-208/REPORT.md includes new current162 diagnostics and187/198/202/205 raw profiles. Rejected new candidates were NOT independently certified by162 tests.
- Next iteration209. Goal remains active; no physical-floor/exhausted-directions claim. Continue investigating current critical path and new designs from validated162; local transform/shared reuse with controlled register lifetime remains investigable.

## Sustained-goal checkpoint through248 (goal active)

- Selected224(62142fc), restored verbatim247(bd5330a); Graph refresh248(294f13d). Latest ordinary BF/FP SW10.97760/11.08960us vs independently tuned Dense12.27520/9.97504us. BF1.1182x, FP0.8995x; FP still needs1.11456us/10.05percent SW-time cut. Graph10.51136/10.87744 vs11.36832/9.44064us. Goal NOT complete, no floor/exhaustion claim.
-224 is218 transposed B_leaf^T*A_leaf^T private static PTX chain,512pack workers/672threads,4 BK256 slots,196736Bshared,64regs/local0,4MiBworkspace. No P/C shared. Immediate wait0/free eachK, no terminal CTA barrier. Original eF .0060152/.000754626 and gates FAIL, AutoDense; private CUDA12.4 invocation workspace publication ABI dependency unchanged.
-218 vs162 paired ordinary improves2.86/3.49percent;224 vs218 adds only0.10/0.44percent ordinary and1.28/2.01Graph, not a3percent new win. Selected224 passed72bitexact162 checks,three quicksanitizers,256two-stream/Graph launches; source normalization verifies validation candidate equals selected source. These do not certify original FP64 accuracy or rejected candidates.
- Fresh218 sparse phase trace90exactchecks/115200rows: ready3.424BF/3.360FPus,issue2.368,drain1.344; first raw and first transformed blocks dominate. Traced72regs vsbase64,local0;10-node same-inputGraph regime differs from standard100-call rotated inputs and can run faster, so no nonperturbation claim; stage medians overlap and cannot sum to fullcall. Full224NCU64regs/local0,~1.700Minst,Tensor~17.5percent elapsed,L2~50percent;221 lower1.468Minst but ordinary slower.
- Rejected209–246: local transform21020us, N64/64CTA22713.6us, BK51222812.4us,dualaccumulator23011.43FP,splitK23441us->23521us->23615.82us;atomic23713.5us withFTZlimitation;orderedflags23842us.235trace30exactchecks shows final merge~7.82us,fence~.9,counter~.64; no spill. Final-Cshared24011.20FP/twoslots24111.46;shared-LHSdualTC24214.45FP/dualN16fullCTA24312.56;threeoperands24411.41;pairwait1 24511.29/fullKonegroup24611.37. Keep timely slot release. Profile24264regs/local0, halfSMgrid and expanded perCTAfrontwork; no spill attribution.
- Invalid220 initial FP speedup comes from unstable Dense12.83->10.09us; stablepairedreject.226/239compilefailed,212generatorfailedbenchunchanged211; exclude them as performance evidence.226protocol slip explicitly corrected/logged.197/198 prior dualproduct groups differ from242shared-LHS groups.
- Evidence trajectory/20261015_optimization-goal-checkpoint-248/REPORT.md,135evidencefiles+hashes. Next249; continue dependency-aware S/T publication or lower-cost local transform with proven per-call initialization/memory-order protocol. No authorization needed to continue; do not repeat explored axes without new cause.

## Sustained-goal checkpoint through270 (goal active)

- Selected266(18f9a6f), restored verbatim269,Graph270. This adds four descriptor prefetches before compute initialization to new254 PDL architecture. No original FP64 gate passes; AutoDense, target NOT met. Ordinary paired BF/FP10.50944/10.36512us vsDense12.32128/9.80560us; FP stillneeds.55952us/5.40percent SW reduction. Graph10.03136/10.13264 vs11.32624/9.34624us.266 vs254 ordinaryFP improvesonly.62percent andBFflat,GraphFP regresses.60percent; not3percent/universal win.
-254 replaces224 private driver workspace publication with two official PDL kernels,128x512transform and128x256compute; both100percent sharedcarveout essential. RawP1/P2 before first transformed grid dependency wait.4MiBworkspace,196736Bshared,4BK256slots,TC128–255 privatePTX originalmath. No run allocations/hostsync/cacheofinputs. Ordinary254 improves3.96BF/5.93FPpercent vs224 infullABBA.
-254/266 independentlypassed72bitexact162 checks,3quicksanitizers,256two-stream/Graphcalls;266 validation source matches selected.266 NCU60compute regs/32primary,local0,~1.0697Minst; SASS4UTMACCTL.PF. Do not compare minimal266replay duration to full254 as acceleration evidence.
-254 batchtrace10calls rotated2inputsets confirmsordinaryoverlap(TCstart~.256us beforemaxpacklane0finish),Graph~.544us. Single-callNSysnooverlap doesnotgeneralize. traced64vsbase60regs. New sparsecriticalblocktrace72regs,stack0,notprofiledlocal: ordinaryfirstrawready1.024/issue.192/drain.128us,firstST.256/.192/.096,last.128/.160/.096;120finaloutputsbitexact. Diagnosticnotformalperformance/noperturbationclaim, mediansoverlapcannotadd.
-249 scopedcounters failed~13us from1.8–1.9us globalinitializationbarrier; padding251still12.5. New clusterjointfree26119us,PDLwithoutcarveout25212.2,withoutPDL25513.2FP; N64/48,K512,transformgrid256x256/64x512,pairretirement,hosttemplates,staticTMA,earlyfirstTMA,160threadroles notretained. Archive249–270 includes146evidencefiles+hashes attrajectory/20261015_optimization-goal-checkpoint-270/REPORT.md.
- Next271. Investigate PDL fine-grain publication without249 globalinit barrier only after proving per-call initialization/memoryordering, or other current-critical-path directions. No physicalfloor/exhaustion evidence, no permission needed to continue, goal active.


## Sustained-goal checkpoint through291 (goal active)

- Selected282(1bdfb29), restored verbatim290(17191e9),Graph291(fd074a0). Ordinary paired BF/FP10.27280/10.23840us vsDense12.22848/9.80672us; FP stillneeds.43168us/4.22percent SW reduction. Graph9.97616/9.99264 vs11.27376/9.34080. Original FP64 gates bothFAIL, AutoDense; target NOT achieved.
-279 changes266 to8BK128slots,196736Bshared,seq8 firstSTwait;60compute/32primaryregs,local0.266/279pairedordinary BF improves2.43percent andFP.70percent;GraphBFflat/slightlyregresses andFP.75percentbetter.282 adds pair-rounded final conversions, tiny279/282pairedordinary.47/.35percent gain andGraph.32/.05; primary dynamicinstructioncount identical552960, so do not claim fewer primarySASS instructions. No3percentbreakthrough.
-279/282 EACH independentlypassed72bitexact162checks,three quicksanitizers,256two-stream/Graphcalls;282validation source matches current selected. API oldalgorithmname64x32x256 islegacy; actualmetadata bk128stages8.
-282 sparsebatchphase trace76800rows,120finalbitexactoutputs. Ordinary rawready.512/issue.096/drain.064us;firstST.128/.096/.064;last.128/.128/.032;TCchain6.496BF/6.528FPus. Trace68vsbase60regs,stack0/notprofiledlocal. Different regimes/perturbation; no adding overlapping medians. Profile279/282minimal60regs,local0,~1.267Minst.
- Fullsame-round266vs275 M-XORmap:both60regs0local,~1.070Minst;FP L2sectors1.646M->2.331M,longscoreboard/issue2.038->4.379,replay9.856->13.312us. Notspill; do not uniquely claim L2bankcause from aggregate counters.281low32descriptor reduces~1.070->.989Minst but80regs/local0 andno ordinarygain.
- Rejected271serialflags16.79FP,272batchflags12.49,273initprefix11.09;274N40,275M-XOR,276TMAorder,27764bitpack,278N16twoCTASM14.80,280K128lookahead10.29,281low32desc10.37,283allrawdataprefetch12.73,284ninthslot10.32,285unified128threadinlineTMA11.10,286twoTMAproducers10.39,287K192tail12.10,288normalcachehint10.27,289leafreorder10.31. RejectedcandidatesNOTindependentlycertified.271–273 no-safe-init-free flags breakthrough; prefixfencealreadycosts~.6us.
- Evidence trajectory/20261015_optimization-goal-checkpoint-291/REPORT.md plushashedarchives. Next292. Continue primary trigger/thread scheduling, first-block-only prefetch or materially simpler cross-CTA reuse protocol. Old261multicast19us mustnotrepeat unchanged; grant-free rank1local-release+rank0joint-release is a possible distinct protocol, needs memory/lifetime review. No physicalfloor/exhaustion evidence, goal active, no permission needed.


## Sustained-goal checkpoint through313 (goal active)

- Selected282(1bdfb29), verbatim restored313(79dad0b). Fresh ordinary BF10.29344vsDense12.46912us;FP10.24576vs9.79072us, needs.45504us/4.44percentSWcut. OriginalFP64gatesbothFAIL;AutoDense; nofloor/exhaustion/goalcompletion.
-292/295fullNCU selected282vs294pairgroup:60regs/local0;barrier/issue1.48->.89butlongscoreboard1.1->1.59, no ordinarygain.302leaderwait62regs/local0/1.423Minstslower.303postwaitwarp-sync removal independentlypassed72bitwise,3quick sanitizers,256streams, butfullABBA ordinary10.29696BF/10.25584FPvs28210.27952/10.22128 soREJECTED. Graphseparately10.00256/10.01648vs10.00624/10.05104.
-299N64cluster4+3uses124regs/local0/.863Minstbut16usfullcall;noadoption.300firstblockprefetch,301256x512primary,304asymK128+K384nogain.305generatorfailedandshellrepeated304:stalesourceexplicitlyexcluded;usefail-fastsubsequentcommands.306/307compilefailures.
-308-312newlocalsharedtransformroute112persistentaccumulators,384threads172160shared,globalS/Tunused.308138regs/local0/5.223Minst+C7515serialization24us;309staticK removesC7515~18us;310physicaloffset16.4FP;311K64doublebufferWRONGrawphaseoutput1e36excluded;312phasecorrect15.70FPstilltooslow. Rejectednotindependentlycertified.313restoredselected282,hashandcheck282normalizationmatch.
- Evidence trajectory/20261015_optimization-goal-checkpoint-313/REPORT.md +hashedarchives. Next314testnewPDLglobal4+3splitN64:rank1precomputesALLthreeleavesbeforeacquiringU2/U3,perwarpglobalflagsinitializedbyprimarybeforegriddependencycompletion,noinitialgridflagbarrier,nocluster/DSM,6MiB+8192workspace. Detailedproposalarchived. Old176/299lessonmustreview. Continuegoalwithoutpermission,nofinishclaim.

## Latest numerical-cost feedback

- User states that high/low compensation is too expensive. Prioritize investigation of representations or formulas that keep seven leaf products rather than assuming the full fifteen-product compensation route is the next implementation. This records the user's cost concern; original FP64 gates and honest accuracy reporting remain unchanged.
- CPU numerical diagnostic at trajectory/20261015_numerical-alternatives-seven-products/REPORT.md isolates transform quantization with FP64 leaf dots/recombination on three bounded uniform1024^3 seeds. ClassicStrassen improves error only~27percent and remains above both gate floors. BF16 original inputs with FP16 Winograd operands reduce median error~.00603645 to.000316776, but have NOT passed productionGPU gates or demonstrated speed; range and mixed-format operand handling are unresolved. Does not solve FP16-input precision. No solution source changed by this diagnostic.

- Latest user correction: FP16 input precision still fails; a BF16-only fix is not a complete solution. Keep FP16 original acceptance mandatory. Three-seed diagnostic enumerating all256 subsets of eight standard ideal residual cross terms finds best14-product eF1.52466e-4 above original~1.03753e-4 gate; all8 terms needed in this tested family. Does not prove other correction algorithms impossible. Evidence selective.py/json/log and REPORT.md in trajectory/20261015_numerical-alternatives-seven-products/. No GPU source or benchmark contract changed.

## Active precision/performance research goal — 2026-10-15

- User explicitly requested trying the layout/supply and selective-fusion lessons from FalconGEMM. This is a bounded comparison from current iter330: test coordinated S/T workspace layout and CTA access, then selective on-chip transformation with full-call costs. Preserve seven products, both dtype gates, Auto=Dense and the stopped FP8 correction route. Reject slow/incorrect candidates and restore the measured baseline if neither direction wins.

- Completed bounded331–336 Falcon experiments.332 row-interleaved S/T yields only0.15%BF/0.11%FP ordinary latency reduction in an independent ABBA repeat (first round0.76%/0.22%); no significant L2 request reduction.333 64element striping regresses ordinaryFP.335 selective on-chip S3/T3 passes72 bitwise checks,3quick sanitizers and256 two-stream/Graph launches but slows ordinaryBF/FP~84%/75%, with extra TMA payload/instructions and no spill.334 had invalid scratch swizzle placement, corrected335; exclude334 timing.336 restores entire solution verbatim from331/iter330, original FP64 gates bothstillFAIL. Evidence research/falcon_layout_fusion/REPORT.md; no precision solution or physical-floor claim, broader research goal active.

- Latest explicit user instruction: stop the FP8 residual-correction/fusion route because its cost is too high. Do not implement the proposed fully fused FP8 prototype or continue dense/sparse FP8 correction experiments. Preserve existing evidence. This stops that route; it does not authorize relaxing the original accuracy gates or declaring the broader goal complete.

- User explicitly requests sustained research into fixing accuracy without significantly affecting performance. Preserve both FP16 and BF16 original-input FP64 gates and complete-call timing. Prioritize materially cheaper numerical representations/corrections; do not silently settle for accuracy-only compensation or speed-only seven products.
- Research diagnostics live in `research/precision_20261015/`. No production solution or benchmark gate changed in this research phase. Screened DPS equations98/101/102 pass tensor identities but FP16 transform quantization still exceeds the original floor with ideal FP64 leaf dots.
- Scaled E4M3 first-order residual correction has promising numerical evidence, including actual H100 FP8 Tensor Core products on independent seeds; this is not production acceptance or a demonstrated cheap implementation. Eight independently launched correction GEMMs and cuBLASLt batched8 timing are too expensive as append-only implementations. All-2:4 residual pruning fails both floors. Enumerated mixed dense/sparse corrections require further performance and hardware-sparse validation.
- The research goal remains active until numerical correctness and low complete-call overhead are jointly verified. See research report for exact measurements, limitations and next experiment.

## Active MNK tuning goal — 2026-10-15

- Latest user goal: build a tuning strategy for different M/N/K and strive to stably outperform cuBLAS for every size. This supersedes historical 1024-only scope for this task; use bench/shapes.json and both BF16/FP16, with aligned, rectangular, boundary and tiny cases.
- Preserve original-input FP64 gates, independently tuned cuBLASLt comparisons, full-call costs, fresh data, and no run-time allocation/host synchronization. Do not claim success from selecting the cuBLAS fallback or from speed-only SW results.
- Add precision-valid dense Hopper candidates and empirical per-shape selection alongside the experimental SW routes. Keep existing Auto behavior until acceptance evidence warrants promotion; expose experimental tuning explicitly.
- Existing multi-size WMMA SW has implementation tests/memcheck passing but 2048^3 takes4.7–4.9ms and original gates fail. It is a functional baseline, not a tuned winner.
- FP8 residual correction remains stopped. No relaxation of numerical contracts or restoration of deleted old SW implementations is authorized.

## MNK checkpoint through iter344 (goal active)

- Branch `opt/mnk-tuning-337`; iter343 `7679b69` adds five alternating100-call holdout batches after short ranking, rejects the previously unstable4096³ custom selection. Full28 combinations pass original FP64 gates; fallback is not a custom win.
- iter344 `575d56f` adds K128/K256 Hopper tiles and direct-store epilogues (candidate count33). Seven-shape screening passes14/14 original precision checks;512³ BF/FP provisional1.126/1.135x,1024³1.49/1.19x,2048³1.097/1.047x. Other dimensions remain unresolved. New narrow basic candidates are slow at4096 and should be bounded after gathering full-size evidence.
- Current344 CTest5/5 passes: gpu_correctness,exact_winograd_dag,sw_multisize_implementation,tuned_correctness,tuning_candidates_correctness. All33 forced candidates bothdtypes fresh data,Graph/direct dualstreams,inplace,padding and multipleKtiles pass; compute-sanitizer memcheck0errors. Known experimental SW1024 original precision failure is not changed or certified by this subset.
-343 ownWMMA ids24/25 independently passes memcheck0 and racecheck0errors/0warnings; current tests run them in344 all-candidate memcheck too. Profiler fixedid25 512³ FP16:126regs,6.25% achievedoccupancy,32% excess global sectors and32% excesssharedwavefronts;13.47us replay duration is diagnostic,not ordinary timing. Ordinary~9.6–10us vsDense~5.8, boundary514 scalar~24–30us vsDense~8.8. Not a winner.
- Historical339 independent3fullsweeps certified9/28 custom wins. Never transfer that certification to later selection policies. Current344 independent3fullsweep acceptance is running in `results/mnk344_acceptance/`, driver `results/mnk344_acceptance_driver.log`, tool session46572. Do not modify solution/build binary until it finishes; script checks binarySHA. Firstsweep512 bothdtypes ~1.12x, precisionpass sofar. Registry exit2 means unresolved combinations, not an execution failure.
- Validation343 archive `trajectory/20261015_mnk-validation-343/` plus pending README/script changes need a separate evidence commit after current acceptance. Script now fingerprints benchmark source, shapecontract and itself in addition tosolution/header.
- Pending345 prototype only in `results/dense_wmma345.cu` and generator `results/prepare345.py`: add4byte cp.async.ca for nontransposed even input strides and32x32 outputblocks to increase small-GEMM parallelism; not compiled or production. After current acceptance, consider applying as candidates33/34 and rerun formaliter345. Must add direct4byte alignment/tail tests, restrict WMMA routing names toactualids, adjust `--wmma` test filter, memcheck/racecheck. No claim this prototype is correct or fast.
- After344 all-shape data, prune slow larger-K narrow kernels above2048 if evidence agrees. Profile fixed large candidate12/21 before trying64x128/64x256 basicS2/S3 two-CTA-residency alternatives. Do not reinstate stoppedFP8 correction. Goal remains active; no permission needed to continue.
- Additional pending345 preparation: `results/dense_wmma345.cu` now also contains explicit CuTe SM80 MMA+ldmatrix32x32 kernels with direct register-to-global output, using `results/manual_mma345.inc`. `results/apply345.py` is prepared but NOT EXECUTED: installs this prototype as ids33–36 (twoWMMA/twoMMA), enables4byte copy, bounds larger-K ids26–32 to2048, and adds even-stride/nontransposed tail tests plus `--new`/`--wmma` filters. Do not rerun `prepare345.py` alone because it regenerates only the pre-manual prototype. Wait for session46572 complete before applying/building. Continue formal bench→log→commit immediately; then force tests+memcheck+racecheck before counting any gains.

-344 acceptance FINISHED (session46572 exit2 is expected unresolved status): all28 combinations precisionpass in all3sweeps;11/28 stablecustomwins (all15pairedgroups >=1.03). Archive `trajectory/20261015_mnk-validation-344/acceptance/registry.json`, binarySHA919c063cb17df73657023f60aa8bd6e3be9e588333a947bb5603a3756155c02a. Stable512 BF/FPmin1.11451/1.10911,1024min1.48336/1.19072,2048min1.07778/1.04432,1024x8192x512 BFmin1.06445,odd1025x1537x769 BF/FPmin1.10938/1.07482,tinyBF/FPmin1.06777/1.07420. RectFP intermittentlyfallback;17combinations unresolved, goalACTIVE. Production may now be changed for formal345.

-345 `9275759` completed full28 rows, alloriginalFP64pass, no newwin on boundary. id36 explicitMMA boundary13.7–13.8us vsDense8.8;aligned512~9us vsretainedHopper5.2. All37forcedcandidates+TunedAPI pass; `--wmma` memcheck0 andracecheck0errors/0warnings. Archive `trajectory/20261015_mnk-validation-345/`. FixedNCU boundaryid36:114regs,5.649394Millionissuedwarpinstructions,19.18%occupancy,35%excessglobalsectors/23%excesssharedwavefronts,16.99us replay. Largeid12 8192³FP16:168regs,215040Bdynamicshared,14.06%occupancy,Tensor91.368%elapsed,localld/stsectors0,1.54ms replay (not ordinary timing). This is compute-pipeline pressure, not proof of global memory saturation or all viable routes exhausted.
- Next346 prototype `results/dense_wmma346.cu` generated by `results/prepare346.py`: dedicated nontransposed/even-input MMA with32bitproblem/indices and compile-timecopywidth4/16; restrictdimensions<=2048andstrides<=2^20 tokeepoffsetsafe. Not production/compiled yet. Need ids37/38 routing, aligned+even-tailtests; transpose mustfallback/reject. Also profile supports a bounded cluster4/8 andClusterN2 experiment forlarge matrices; no measuredgainclaimed. 344 eleven stablewins remain the last complete independent certification. GoalACTIVE,17unresolved.

## MNK checkpoint through347;348 running (goal active)

-346 `8da480a` full9shape18rows precisionpass,43forcedcandidates+APIpass,warp memcheck0/racecheck0. Narrow-index evenMMA ids37/38 cutboundary~13.8to10us butDense8.8stillwins. Clusters39–42 (CM4/CM8/CN2/CM2CN2) do not win on8192/16384; no additionalstablewins. Evidence `trajectory/20261015_mnk-validation-346/`.
-347 `a5c0494` short16-wide Ktail andfloat2 output underD/C eightbytealignment gates. Sixshape12rowsoriginalFP64pass;boundary9.3–9.5us stillfallback, noadditionalstablewin. All43forcedcandidates includingK2,tail,Graph/dualstreams/inplace/padding plusAPIpass;warp memcheck0/racecheck0. Profilefixedid38 boundaryFP16:64regs,3.575145M issuedwarpinstructions,19.57%occupancy,0.81eligiblewarps/scheduler,25%excesssharedwavefronts,12.16us replay diagnostic. Validationarchive347.
- Stall re-assessment after345–347: reviewediterationhistory, reprofiled347, searchedofficialNVIDIA CUTLASS efficient-GEMM docs. Smalltiles improvewavefill butaddcopy/reusecost; nextdistinctdirection usesXORsharedlayout toaddressmeasuredsharedexcess plus smallerCTA outputtiles. Nofloor/exhaustionclaim.
-348 is RUNNING, toolsession89252, driver `results/iter348_driver.log`, formalcommand6shapes512;514x770x258;1024;2048;odd1025x1537x769;tiny, bothdtypes,2inputsets, full100x5/3fresh. Production changes via `results/apply348.py`: `HopperCandidateCount=46`;new43=swizzled32x32x64,44=swizzled16x32x64,45=swizzled32x16x64. XORoffset `(r*Stride+c)^((r&7)<<3)` onpower2A/B pitches,8-half contiguous segments preserve16bytecopy/ldmatrixalignment. Blocks128threads;Nperwarp8fornewsmaller512outputblocks. Existing37/38 remaindefault nonswizzled. Metadata/tests updatednewIDs; sourceisUNCOMMITTEDpendingformalcompletion.
- Once348 finishes: immediatelyNEXTtoolcall `scripts/log_iteration.py348 ...`, thenNEXTtoolcall gitaddITERATIONS+specificsolution/tests+trajectory348andcommit `[iter348]...`, evenifcompile/numericfails. Do notprobeuntilclosed. Aftercloseforceall46candidates,API,`--wmma` memcheck/racecheck; thisfilter includesold24/25,33–38,new43–45. Pending`--new` starts33 andincludesclusters39–42too. Currentmostrecentcomplete independentcertification remains344 elevenof28customwins; goalACTIVE17unresolved. Need newindependentacceptanceifnewwinner, nobinarymutationduringacceptance.

-348 `ed2e71a` completed12/12 originalFP64pass,46forcedcandidates+APIpass,warp memcheck0/racecheck0. XORid43 boundary10.1–10.3us versus retained347id38 9.3–9.5us; smaller44/45 11.2–11.8us, allfallback. NCUid43 cuts sharedexcess25%to14% butincreasesissuedwarpinst3.575Mto4.128M,56regs,19.53%occupancy,13.15us replay. Archivevalidation348. Noadditionalstablewin.
-349 RUNNING toolsession63773, `results/iter349_driver.log`;6shape12rows fulltiming/freshFP64 protocol. `results/apply349.py` adds48totalcandidates,46/47 register-only per-warp GEMM: tiles16x32/32x16,4warps, nativeSM80 m16n8k16, raw original-input globalhalf2 loads;4warp-shuffles+2byteperm pack Bfragment;shared0/noCTAbarriers,whole-warp earlyreturnonly. Nontranspose/even/alignment/32bitlimits identical to restricted evenMMA; alpha/beta,float2 output andcurrent-pointer binding. Kernel in `solution/dense_wmma.cu`, detailedprototype `results/register349.inc`. Uncommittedproduction349 whilebenchruns. Existingtestfilters automaticallyincludeids46/47 underid>=43 for aligned,even-tail,K2,metadataalignment/overflow rejection. Mustforcevalidationafterformalclose.
-349 finish protocol: NEXTtoolcall log_iteration.py349 thenNEXTtoolcall gitaddITERATIONS+specificmodifiedsolution+trajectory349 andcommit `[iter349]...`; no probes betweencompletion/log/commit. Thenall48forced+API+`--wmma`memcheck/racecheck beforeprofile/claim. Last complete independentcertification remains34411/28,17unresolved,goalACTIVE. No stoppedFP8 routes, noaccuracyrelaxation.

-349 `0f84bd5` sixshape12rows originalFP64pass;48forced+APIpass,warp memcheck0/racecheck0. Raw register ids46/47 boundary13.5–14.7us notwinner. NCUid47:32regs,2.906409Mwarpinst,37.07%occupancy but0.37eligiblewarp/scheduler and64%excessglobalsectors,17.50us replay. Reducinginstructioncount/SMEMalonefailed; globaltransactiongeometry/latencydominatesnewrawfamily. Archivevalidation349. Bestboundarystill347id38 9.3–9.5us vsDense~8.8. Lastcompleteindependentcertification34411/28,goalACTIVE17unresolved. Next350 can testBK128 (trendBK32~10.6→BK64~9.3) andnativeSM80GEMMiterators withdifferentwarpoutputreuse; raw-registerfurthermicrotuningnotprioritizedwithoutnewcause.

## MNK checkpoint iter351 (goal active)

-350 d213888 failed compilation only: AMP32x128 used too many warps for shared-copy map.351 cb680ed fixes WarpM32, passes six-shape twelve-row original FP64 gates. New candidate52 SM80 64x64x32 wins boundary514x770x258 provisionally1.300xBF/1.303xFP; no additional full-suite certification yet.56forced candidates and API pass, relevant CTest5/5, memcheck0/racecheck0errors0warnings. Archive trajectory/20261015_mnk-validation-351/.
- Independent frozen-binary three-sweep acceptance RUNNING session40701, results/mnk351_acceptance_driver.log and results/mnk351_acceptance/. Do not modify solution or build benchmark until completion. Latest COMPLETE certification remains34411/28; fallback never counts. After acceptance archive registry/logs and update README accurately, then profile fixedcandidate52 boundary for new-direction evidence and pursue unresolved large/rectangular sizes. Goal ACTIVE, no accuracy relaxation or stopped FP8 route.

-351 acceptance COMPLETE: three independent full sweeps,84/84 original precision checks pass,13/28 stable custom wins (all15pairedgroups>=1.03). New514x770x258 BF/FP median1.294439/1.299526,min1.287713/1.293235. Other15unresolved. Evidence archived trajectory/20261015_mnk-validation-351/acceptance/. Goal ACTIVE. Binary may now change after fixedcandidate52 profile. Prepared next experiment only results/prepare352.py (not applied): shallow-stage Hopper plus2workers/SM direct epilogues and beta0 source-disabled TMA variants; must inspect diff, formalbench/log/commit then force allnew candidates includingbeta0 rejection andsanitizers. Never transfer351certificationtolaterpool.

-352 6501946 complete:12/12 original precisionpass; all62forced+API and CTest5/5 pass; memcheck0/racecheck0errors0warnings. New61 beta0-only128x256auto improves provisional1024x8192x512 BF1.0904/FP1.0601; not independentcertification.56–60 shallow/direct/2worker configurations slow on4k/8k andrect.61 large8192 BF1.512ms/FP1.605ms shortscore stillbehindrerankedLt;4096 shortgain failslongholdout andfallbackcorrectlyselected. Sourcecooperativekernel hardcodes producer40/consumer232registerbudget, making2CTA residency unlikely despitelowerSMEM. Profile351archived, profile352 fixed61 large+rect and60resource RUNNING session82665. Latestcomplete independent35113/28,15unresolved,goalACTIVE. Nextconsider beta0 raster/swizzle and128128auto afterprofiles; boundnegative56–60above2048. No precision changes.

-352 profiles complete:61 8192FP16 uses168regs,231424sharedbytes,92.354%Tensor elapsed,L2hit78.92%;replay1.525msdiagnostic.61 rect8192x1024x512 Tensor41.80%,L2hit87.42%,25.568usreplay.60 loweredSMEM100352fits2blocks butregisterlimit1, actualoccupancy14.06%,4.359ms;2workerlaunchdoesnotgive2residentCTAs. Archive352profiles.
-353 RUNNING session54287, results/iter353_driver.log; sixshapes512/514boundary/4096/8192/tworectangles, fulloriginalprotocol. Productionuncommitted70candidates:62–65sourcefree128256sw1/sw4/rasterM/rasterN;66sourcefree128128auto;67/68pingpong64x256sourcefreeauto/S3;69sameCM2. Negative56–60boundedabove2048. Existingtestsforid>=60exercisebeta0andnonzerobetarejection, newCM2testbase129x256x520crossescluster tail. Finishformalbench→immediate log353→immediatecommitbeforeprobes. Thenforce70/API andsanitizersbeforeacceptance. Latestcompletecertification35113/28, goalACTIVE15unresolved;352rectangleFPgainisprovisionalonly.

-353 8b11978 complete12/12originalprecisionpass. New67/69pingpong wins8192x1024x512 BF1.0829/FP1.0912,1024x8192x512 BF1.1377/FP1.1125 provisionally. Positive narrowing of per-warpaccumulator to64M avoids old128M pingpongserialization; large square stillfallback.70forced+API/CTest5pass,memcheck0. CUDA12.4racecheck HOSTheap abort reproducibly isolated69; no GPUhazardreport, notcountedpass. Samebinary passes CUDA12.8 ComputeSanitizer2025.1 full --wmma racecheck0errors0warnings and isolated69 memcheck/racecheck0. Logs archive353; no runtime/kernel toolchainchange. Testhelpernow accepts --candidate ID forisolation.
- Next full353 frozenbinaryacceptance3sweeps; no solution/build changes untilcompletion. Latest COMPLETEcertification still35113/28. Pendingresearchonly results/hopper_reg_budget_kernel.cuh (upstreamlicensedcooperativekernel exposedregbudgets) and results/regbudget354_prefix.cuh (104consumer/32producer registers, min2blocks) NOTcompiled/applied. MustresourcechecknumRegs budget andoccupancybeforefirstGPUlaunch toavoidsetmaxnregCTA-pool deadlock; officialPTX saysincblocksifCTA poolinsufficient. No productionclaims.

-353 acceptance COMPLETE:16/28 stablecustomwins,all28combinations precisionpass eachof3fullsweeps (all15pairedgroups>=1.03). Added8192x1024x512 BF/FP min1.061058/1.076948 and1024x8192x512 FPmin1.093060; otherrectBFmin1.109149. Archivetrajectory/20261015_mnk-validation-353/acceptance/registry.json.12unresolved are4096/8192/16384squares andthreebigKrectangles,bothdtypes. GoalACTIVE. Frozenbinarymaynowchangeafterresearchpreflight. Next resourceguardedprototype results/regbudget354_probe.cu, prefix+licensedkernelcopy inresults, NOTcompiled; includes128128S2,12864S3,anddirect128128S3, math104/load32/minblocks2. Also prepared results/profile_lt354.cu toprofileactualLtbeforefurtherlarge-kernel changes (tuneoutsideNCUrange; zero diagnosticinputs,notacceptancetiming). ConsiderStreamKnext: currentzero-workspace-only candidatepathcannotbalanceKtails; mustproperlyallocate/reset/costscratchandGraph/dualstreamtests.

- Research354: exact cuBLASLt winning8192FP16 kernel128x256x64,384threads,168regs,231424sharedbytes,cluster2,Tensor91.964%elapsed,1.535msZEROinputreplaydiagnostic; no physicalfloor claim. Archivedtrajectory/20261015_mnk-research-354/. Original104/32regbudget(min2blocks) failsC7600target80 evenRelease, notproduction. Newlicensednine-warp source-free adaptation movesconsumersfirst256threads/producerlast32, removesallsetmaxnreg; Releaseprobe90regs,local0,100352shared,occupancylimit2,exact129x256x520ones+paddingpass. InitialnonRelease spill/serializationlognotperformanceevidence.
-354 RUNNING session77581, results/iter354_driver.log, sourceUNCOMMITTED77candidates. Adds70–75sourcefree128256CM2/CN2/2x2/CM4/rasterM/rasterN fromLtprofile;76ninewarp128128S2,2workers. Header solution/hopper_nine_warp_kernel.cuh retainsCUTLASSlicense, source-free/2consumer/cluster1staticasserts. Tests70–75use1025x520x128clustertails;76uses129x256x520;all>=60beta0/rejectnonzerobeta. Formal6shapesoriginalprotocol. Finish→NEXTlog354→NEXTcommitbeforeprobes. Thenall77/API,CUDA12.8memcheck/racecheckincludingnewclusters/ninewarp, profilecandidate76largeifcorrect. LatestCOMPLETEcertification35316/28;12largeK combinations unresolved, goalACTIVE. No stoppedFP8 or gatechanges.
