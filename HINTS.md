# Hints

## User directives (2026-10-13)

- From the user's prompt: implement Strassen–Winograd according to `PLAN.md` and continuously optimize with AKO4ALL.
- `PLAN.md` defines the numerical, API and timing contract: BF16/FP16 inputs, FP32 sums before each final operand cast, FP32 accumulation/output; dynamic A/B transforms remain inside every timed invocation.
- Compare against tuned cuBLASLt with identical output dtype and timing mode. Fix the initial BF16/FP16 numerical gates before looking at candidates; report failures rather than widening gates.
- Use the existing H100 allocation 107488 only after verifying it, with CUDA 12.4 and CUTLASS 3.5.1. Downloads on login node; build, test and profile on assigned compute resources.
- Automatic dispatch selects ordinary GEMM unless a shape/dtype/mode has passed both numerical and measured end-to-end performance gates. Unsupported shapes/transpose/ranges fall back safely.
- Session profiler probe: ncu2024.1 failed with `InterprocessLockFailed` because `/tmp/nsight-compute-lock` is not writable. Per AKO4ALL, no repeated ncu probe this session; use end-to-end/stage event or Nsight Systems timing and compiler-resource analysis.
- User resumed from the final large-shape report: continue optimizing and focus on small-matrix speedup. Prioritize128/256/512/1024/2048/4096 squares and small rectangular controls; ordinary launch and CUDA Graph remain separate. Numerical failures must remain visible, but are not a reason to stop the requested execution-performance experiments.
- From the user's explicit active goal: continuously optimize the SW operator until all four square sizes 128, 256, 512 and 1024 are faster than cuBLAS. Keep every size in scope and verify against the same-row independently tuned ordinary cuBLASLt baseline under the existing dtype/output, numerical and timing contracts. A near-unity noisy observation, Dense fallback or an improvement at only a subset of sizes does not complete this goal.

## Current experimental direction (iteration27)

- Cooperative complete-P M1 controls share one transformed A/B bank per CTA/K16 stage: Mega26 uses32x16 output tiles and Mega27 uses32x32. Preserve all older CUDA implementations and historical evidence.
- Retain original FP32 S/T dependencies, one RN conversion for each final operand, seven complete-K FP32 P products, fixed U, and explicit alpha RN multiply followed by active-beta FMA. Public certification, overlap, current-pointer fallback and Auto Dense rules remain unchanged.
- Keep128/256/512/1024 in each diagnostic comparison, with special attention to shared-transform reuse at512/1024. Rank implementation changes by their own SW latency against the historical envelope; slower Dense measurements cannot establish a code benefit.

## Current experimental direction (iteration28)

- From the continuing one-level1024 optimization task: Hopper9/10 fuse original-input transforms into one uniform seven-product leaf launch, writing full-K FP32 P; the existing combine is the required second launch. Measure both together, preserving all older CUDA implementations and evidence.
- Use native strict full tiles and current A/B/P16 guards with P-only caller workspace. Incompatible metadata/bindings use ordinary GEMM. Keep the original FP32 DAG/final RN cast, frozen FP64 gate, Lt metadata/tuner, Auto Dense and all four target sizes unchanged.
- From the user's explicit1024 comparison request: irrespective of the usual >=3% quick-screen trigger, run same-source/binary full1024 end-to-end measurements for both fused controls and packed-once Lt1/H7, both dtypes and launch/Graph. Use20warm/100calls/5groups/3fresh, shared8GiB cap and32heuristic budget; report same-row Dense/SW, p10/p90/CV and unchanged numerical gates.

## Current experimental direction (iteration29)

- From the user's request to compare both one-level1024 input-placement routes: add Hopper11/12 matched compact-input leaves using the same32x32/32x64 BK32 MMA/shared layout and consumer order as fused Hopper9/10. Every complete run freshly packs original rounded A/B, executes the leaf, then combines P (three launches). Preserve all older implementations and frozen evidence.
- Keep the original FP32 S/T DAG and one final RN conversion, complete seven FP32 P products, original FP64 numerical gate, caller-owned workspace, current binding guards, tuned Lt protocol/metadata and Auto Dense unchanged. Rank changes by their own SW latency, rather than a slower Dense denominator.
- Full1024 is required irrespective of the usual >=3% quick-screen trigger: same source/binary Hopper9/10/11/12 and packed-once Lt1/H7, both dtypes, launch/Graph,20warm/100calls/5groups/3fresh,8GiB cap and32 Lt candidates. Report all24 same-row Dense/SW results, p10/p90/CV, workspace and failed numerical gates. This is a matched leaf control for input placement; it is not a dispatcher certification or completion of the four-size goal.

## Active goal (2026-10-14)

- From the user's explicit active objective: continuously optimize the megakernel at square size1024 until it is faster than independently tuned cuBLAS under the existing dtype/output, numerical and complete-call timing contracts. Prioritize the single-launch megakernel path; packed-once/multiple-launch SW and ordinary Dense remain controls, not substitutes for this objective.
- Keep BF16 and FP16 results separate, preserve the original FP64 reference and fixed numerical gates, and include fresh input-dependent transforms in every measured invocation. Numerical failures stay visible and do not stop execution-performance experiments, but a failing candidate cannot certify completion or enable Auto SW.
- The earlier four-size request and its frozen evidence remain recorded. This active goal focuses the next implementation work on1024; it does not certify the earlier four-size objective as achieved.

## Current experimental evidence (iteration30)

- Native Mega28/29 use BM64 BN16/32 BK32,256threads split128producer/128consumer and two final7operand shared banks with PipelineAsync2. Input transforms stay inside every one-launch complete run; no globalS/T/P or native matrixworkspace. Original FP32RN DAG, finalonecast, complete7P/U and explicitalphaRNbetaFMA remain unchanged.
- Frozen samebinary1024 full20warm100calls5groups3fresh shows Mega29 at25.1098/25.4029µs ordinary and24.8368/25.0938µs Graph BF16/FP16; compared with retainedMega27 ownlatency reduces19.219%/15.827% ordinary and18.925%/17.016% Graph. Same-rowDense/SW remains0.482655/0.387950 ordinary and0.456742/0.376342 Graph. All20rows60fresh originalnumericalgates FAIL/NF0; AutoDense and activegoal remain.
- Mega28 has no improvement overMega27. Multi-launchLt1 remains the observedallSW control winner and must not substitute for trueMegakernel goal completion. Rank next single-launch candidates against ownMega29 latency and tuned same-rowDense, while preserving failedgates; do not infer improvement from Dense denominator variation.
- Compiled native initialREG96/160 and perCTAownedpool24576/40960 safely support producer64/consumer96-or160; independentfullCFG/implicitwidth audits maxR58/89 or54/145 and linked/objectinstruction equality PASS. Dynamiclaunchshared71808/86144B and ELFstaticSHARED1024B are distinct. Resource-based possibleCTA residency is not measuredoccupancy.
- Default2080CASE859PASS, newstrict60=36PASS24FAIL/NF0, six targetedCUDA12.8sanitizers zeroerrors. The rawU/finalD native29 reference uses matchingN32oldMega0 atalpha1beta0 followed by independenthostRNmul/floatFMA for specifiedactiveepilogue; oldactivebetaM1 contraction differs and is retained asdiagnostic. Do not replace the original FP64 numerical gate with these implementation-consistency tests.

## Current experimental direction (iteration31)

- Add native TMA raw-alias Mega30/31 BM64 BN16/32 BK32 with256threads split128producer/128consumer; two finalbanks, eight real TMA copies perCTA/stage, rawbarriers, complete7P/fixedU, no globalSTP/nativeworkspace. Keep old26CUDA TUs and source23–30 evidence bytepreserved.
- Ordinary currentbindingrun performs two real cuTensorMapEncodeTiled calls then one kernel; both encodes remain inside repeated timedhostloop. Graph captures fixeddescriptoraddresses and each replay consumes freshcontents with all transforms insidekernel; report event and batchwall/call medians separately, not single-callwall.
- Do not inherit previous vector producerregister quota proof. BeforefirstGPUlaunch prove new initialREG/direction/perCTAownedpool/allroleCFG highestR includingimplicit64/128/TMA/HGMMA width and registerinitialization from finalobject/linkedSASS. Native uses512B sharedalignment/SW64A andSW32-or64B; CPU/pinned API review does not replace compiled/GPU gate.
- Samebinary full1024 new30/31, retained29/27 and Lt1 is mandatory irrespective quick3percent signal: two dtypes, launch/Graph,20warm100calls5groups3fresh,32candidates8GiB cap. Rank newownMega latency against retained29 and same-row independentDense; allprecisiongatefails remain visible, AutoDense.

## Current experimental evidence (iteration31)

- Samebinary full1024 TMA31 is the faster of new30/31, but retained SIMT Mega29 remains faster in all four dtype/mode combinations. New31 ordinary BF16/FP16 event25.702400/26.413441µs versus29 25.239041/25.427520µs, regression1.836%/3.877%; Graph24.929600/25.605760µs versus29 24.776001/25.054400µs, regression0.620%/2.201%. Dense/SW31 is0.476880/0.372029 ordinary and0.452782/0.365843 Graph. Do not promote TMA or infer gain from Dense denominator variation.
- Ordinary run encodes both currentA/currentB maps inside timedhostloop before one kernel. New31 batchwall/call25.77972/26.49597µs includes submission and batchsync; event-to-wall delta is not an isolated encode benchmark. Graph fixed descriptors consume fresh captured-address contents but do not re-encode per replay.
- All20rows60fresh originalFP64gate FAIL/NF0; newstrict60=36PASS24FAIL/NF0,48native12Densefallback. Default2126CASE933PASS, new36 sameatom rawU/host specifiedepilogue groups and40currentmapchecks,12Graph36fresh,6CUDA12.8SAN zeroerrors. AutoDense and activegoal remain; original precision/gates retained.
- Actual compiled TMA producer/consumer maximumR43/89 N16 and46/145 N32, initialREG96/160, ownedpool24576/40960≥20480/28672; compiled64align256B gridconstant descriptors and512alignshared reviewed before firstlaunch. Dynamicshared72192/86528B plus separateELFstatic1024B,STACK/LOCAL/spill0. Static8TMA/14HGMMA perfunction is not measured traffic or occupancy.
- Firstbuild sentinel FAIL was proof-domain misuse of flagged byte-swizzle at element coordinates. All assertions remain; repair only uses raw layout_b element→byte layout_a→element mapping, not a device-layout change. Actual pinnedCPU exhaustivecoordinates/vector8/absolute512bases PASS and rebuilt fullcompiled gate precedeGPU. Preserve failed diagnostics and old26CUDA/source23–30 evidence bytes.

<!-- User-supplied behavior directives. The skill reads this at session start
     and respects any constraint named here. Examples:
     - Optimization constraints or focus areas (e.g., "Prefer Triton over raw CUDA")
     - Strategies to try or avoid     (e.g., "Avoid shared memory")
     - Agent behavior controls         (e.g., "Stop after 5 iterations")
     - Dependency policies             (e.g., "Do not install new packages")
     - Environment constraints         (e.g., "ncu is unavailable on this host",
                                             "8GB device memory limit")

     The skill's own protocol (iteration steps, stall handling, ncu fallback,
     stopping rules) lives in SKILL.md — do not duplicate it here.
-->

## Current experimental direction (iteration32)

- Isolate vector producer consumer scheduling: native Mega32/33 retain source30 BM64 BN16/32 BK32 two-bank256thread producer128/consumer128; read/retire states allow two committed WGMMA groups, wait1 releases only the old operand bank, final wait0 drains before genericP/U. No TMA or dual-consumer axis combined. Preserve all27 old CUDA TUs and source23–31 evidence bytes.
- Re-prove compiled perCTAowned register pool, allocation direction, fullroleCFG implicit widths, accumulator initialization and no genericP spill/reload/read while latest group is inflight before firstlaunch. CPU/API wait1 proof is not a GPU/resource/performance certificate.
- Samebinary full1024 new32/33 plus retained29, TMA31 and Lt1 mandatory regardless quick signal; both dtypes/modes20warm100calls5groups3fresh32LtCandidates8GiB. Keep original FP64 gates, fresh transforms, explicit alphaRNbetaFMA, currentbinding/overlap, AutoDense and failedgates visible. Rank ownMegalatency versus29, not slowerDense denominator.

## Current experimental evidence (iteration32)

- Vector wait1 Mega33 is faster thannew32 butregresses retained29 inallfour combos: ordinary BF16/FP16 29.158080/29.407041µs vs25.134721/25.401919 (+16.007%/+15.767%); Graph28.604801/28.983040 vs24.854400/25.129600 (+15.089%/+15.334%). Same-rowDense/SW33 .420703/.337084 ordinary,.396633/.325929 Graph. Keepretained29 singlelaunchwinner; Lt1 multiplelaunchcontrol notgoalsubstitute.
- All20rows60fresh originalFP64gateFAIL NF0; strictnew60=36PASS24FAIL NF0 (48native12Denserangefallback). Default2188CASE1075PASS/new62CASE142PASS,96sameatom rawU-host specifiedepiloguegroups,40currentABchecks,20Graph60freshonekernel covering1/2/3/4/16/17/32/33stages,6CUDA12.8SANzeroerror/hazard/warnings andmemleak0. Structuralconsistency doesnotcertifyoriginalprecision; AutoDense andactivegoal retained.
- NewcompiledroleCFG/implicitwidth ownedpool/direction/accinit andzero genericPaccessuntiltailwait0 gate independentlyPASS; initialREG96/160,prod58/54 cons89/145 highestR,stack/local/spill0,dynshared71808/86144B+ELFstatic1024B align128. Staticinstructions1168/1504 and14HGMMA do notprove measuredoccupancy/traffic orwait1 slowdownmechanism. Keepold27CUDA andsource23–31evidence byteidentical.

## Current experimental direction (iteration33)

- Add native Mega34/35 three-bank vectorproducer wait1 BM64 BN16/32 BK32, preserving full128/empty128 andtwo-commitoldbankretirement/tailwait0. OnlyStages2to3; norelease1/TMA/twoConsumer/Psplit combined. Keep28oldCUDA TUs andsource23–32 evidence bytepreserved.
- New independent compiled initialCTAownedpool/direction/fullroleCFGimplicitwidth/accinit/no genericPaccessorspillwhileinflight, actualdynamic/staticshared, modulo3phaseand3producertail acquires andobject/linkedbench/teststream gate requiredbeforefirstlaunch. CPUmodelstdout and actualprocess exit both archived.
- Mandatory samebinary full1024 new34/35, retained29 andtwo-bank33, multi-launchLt1 bothdtype ordinary/Graph20warm100calls5groups3fresh32candidates8GiB regardlessquick3percent. Rank ownlatency vs29+33 andsame-rowDense/SW. KeeporiginalDAG/finalonecast/FP64gate/publiccurrentbinding/AutoDense; allnumericalfailsvisible.

## Current experimental evidence (iteration33)

- Newthree-bank35 improves ownlatency versus samewidthtwo-bank33 by12.679%/12.454% ordinaryBF/FP and12.878%/12.286% Graph, but observed25.407679/25.605760µs ordinary and24.951680/25.332479 Graph remain0.27–1.19% slower thanretained29. New34 stillslower; do notpromote baseddifferentDense denominators or treat multi-launchLt1 as trueMegagoal.
- Mandatoryfull20rows60fresh alloriginalnumericalFAILNF0; strictnew60=36PASS24FAILNF0(48native12Dense). Default2262CASE1269PASS/new74CASE194PASS,132sameatomrawU+specifiedhostepi,40ABchecks,20epirebind60runs,new36Graph108fresh/all168Graph504fresh, sixCUDA12.8SANzeroerrors/hazards/warnings andmemleak0. Structural/sanitizerconsistency doesnotcertifyoriginalprecision; AutoDense andactivegoal retained.
- Actualfreshcompiledquota/initialCTAownedpool/fullroleCFGimplicitwidth/accinit/no genericPuntiltailwait0/mod3phase/6barriers/3tailacquires independentlyPASS; REG96/160 prod58/54cons89/145, dyn107648/129152+static1024align128,stack/local/spill0. Complete5424instructionstreams object/bench/test exact; rawheaderflags differing retained. Thirdbankbenefit doesnotprove isolatedstall/traffic/occupancymechanism. CPUmodelhas archivedactualexit0 andstdout, old28CUDA/source23–32evidencebyteidentical. Next hypotheses require separatefreshcompiled/GPU gates.

## Current experimental direction (iteration34)

- NativeMega36/37 bothBM64BN32BK32,384threads128producer/two128consumers,full128/empty256,twooperandbanks,WG0P1–3/WG1P4–7eachwholeK/wait0. ShareoriginalSTonceperCTA/stage andpublishall7P into typeddrained-bankunion onlyafterretirebarrier, thenpublishbarrier anduniqueownerfixedU; no globalSTPworkspace. Sym64/96/96 min1 versusasym64/80/96 min2, noadditionalaxis.
- ActualnewinitialCTApool/effectiveI/direction/fullroleCFGincludingproducer-tailU/prequota/implicitwidth/accinitafterinc/zeroinflightgenericPspill anddrainunion/shared/barriers/objectbenchtest gates beforefirstlaunch. For37effectiveinitial80 required, ELFreportedREG orallocationrounding alone insufficient. OriginalCPUmodels actualstdout+exit archivedwithoutmutatingoldfrozenprototypes/snapshots.
- Mandatory full1024 new36/37+retained29+threebank35+Lt1bothdtypes/modes20warm100calls5groups3fresh32LtCandidates8GiB, independentoriginalgate andDense/SW; compare ownvs29/vs35 and37vs36. Preserve29oldCUDA/source23–33bytes/currentguards/fallback/AutoDense. If compiledrouteunsafe recordactualmissingrows, neverunsafe launch orfakecomplete20.

### Iteration34 final scope before timing

Mega36 restores64/96/96 after consumer-only accumulator scope repair; same whole-K7P/fixedU and two-bank ST. Three collective CTA epochs are initbar0, nonaligned drainbar6 count384, publishbar0. Mega37 64/80/96 still has actual STACK128/spill384store256load/C7512 and is publicly disabled before degenerate handling; raw helpers reject. Measure legal16 of expected20 rows with four explicit compiled-safety exclusions. Do not execute blocked37 or claim original60strict/sixSAN: current strict30 and threeSAN scope is Mega36 only. Originaldefault/strict/precision gates remain unchanged for enabled candidates.


## Current experimental evidence (iteration34)

- Final legal Mega36 P3+4 dualconsumer, producer64/WG96/WG96 and consumer-only accumulators/nonaligned drain6, improves own complete1024 latency versus samebinary29 by7.394%/7.039% ordinary BF/FP and7.217%/7.821% Graph. Ordinary23.426239/23.645761us, Graph23.054399/23.157120us; original FP64gate stillFAIL. Mega37 compiledSTACK/spill/C7512 remains publiclydisabled/no native37 launches. AutoDense and goalactive unchanged.
- All16 legalrows48fresh originalFAIL NF0, strict36only30=18PASS12FAIL NF0, default2298CASE1361PASS, selected36/93, threeCUDA12.8SANzeroerror/memleak0. Full1560 is unexecuted source-domain; scope doesnotclaim blocked37 checks. PreserveallfiveDense groups, including ordinaryFP DenseCV26.632093%; SWownlatency determines gain.

## Current experimental direction (iteration35)

- From the authorized continuing1024 singlelaunch goal: add onlyMega38 BK64 at BM64/BN32,384threads/producer64/two96consumers/twooperandbanks, preserving successful36consumer-onlyscope/nonalignedbarrier.sync6,384/threeCTAcollectiveepochs/wholeascendingP-K16/fixedU/FP32STonecast/RNalphaFMA/currentguards/AutoDense. OnlyBK32to64,Atask/4&3to/8&7,parentKguard%64to%128; noTMA/threebank/emptyrelease1/precisionaxis.
- Requiredbeforefirstlaunch: newactualfeedingPTX/ELF/objectbenchtest/full3roleCFGimplicitwidth/ownedinitialpool/direction/afterincaccinit/noinflightgenericP/STACKLOCALspill0 and actualsharedhostoptin. Predicted172160 is notactualproof. EachP has4K16perstage,1024stage8; tailallbank0 butbothbankdrain remains. ID38 explicitlyN32/reference36, registered37 remainsblocked beforedegenerate, max38invalid39.
- Mandatorysamebinaryfull1024 new38/currentbest36/control29/multi-launchLt1 bothdtype ordinaryGraph16rows,20warm100calls5groups3fresh/rotate1/32Lt8GiB regardlessquick3percent. ReportSWown changesvs36/29,same-rowindependentlytunedDense/SW,event+100callbatchwall andalloriginalFAIL/NF; originalFP64gate/floors unchanged. Default/selected38,newstrict30/3serialCUDA12.8SAN/CPUactualstdout+exit/newindependentcompiled/datafinalreviews required. Preserve31oldCUDA/source23–34evidencebytes.

## Iteration35 measured BK64 outcome

- New38 BM64/BN32/BK64 dual-consumer P3+4 passed a fresh independent compiled gate: producer/WG0/WG1 implicit GPR maxima58/93/93, effective initial96, CTA-owned pool36864 vs required32768, zero stack/local/spill, actual172160 dynamic+1024 static shared. Original37 remains disabled. No ncu retry.
- Mandatory1024 full16 same-binary comparison (20warm/100calls/group/5groups/3fresh): BK64 was slower than successfulBK32 Mega36 by ordinary BF0.8624%/FP2.3005%, Graph BF1.6970%/FP3.0219%. Do not infer a stall or occupancy cause from static shared-memory size. Current fastest measured singlelaunch remains36.
- New38 ordinary eventmedians BF23.7651205063us/FP24.0908789635us; same-row independently tuned cuBLASLt Dense/SW0.518352920711/0.415931668502. New38 BF Dense CV2.3983% has all five timing groups retained; code benefit uses own SW latency.
- All48 full fresh original numerical gates FAIL/NF0; new38-only strict30=18PASS12FAIL/NF0. Structural bitwise/hostepilogue and3zero-errorSAN are separate evidence, not FP64 numerical certification. Auto remainsDense and goal remainsactive.
