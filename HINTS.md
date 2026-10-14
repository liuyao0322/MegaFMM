# Hints

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
