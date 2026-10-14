# Sustained optimization checkpoint: iterations109–162

The user requested continued optimization until faster than Dense. **The goal remains active and incomplete: BF16 is faster, FP16 is still slower.** This is a validated checkpoint, not a stopping or physical-floor claim.

Current selected source: **iter162, d8e6261**, based on independently validated140, with compact operand planes and removal of the terminal CTA barrier. The user explicitly keeps the uncompensated seven-product route. **Both original FP64 numerical gates still FAIL.** No accuracy-qualified winner is declared, and Auto remains tuned ordinary Dense.

## Complete-call comparison

H10080GB,132SM, allocation107488/g08, CUDA12.4. Full20warmup,5groups×100calls,3fresh numerical trials, rotating inputs. Each executable independently tunes Dense against the current pointers. Compare140/162 in A/B/B/A order, separately for ordinary launch and Graph. Values below are medians of the two complete benchmark runs for162; the Dense column is the matching contemporaneous median. Do not substitute NCU replay timing.

| Mode | Input | SW162 μs | Tuned Dense μs | Dense/SW |
|---|---|---:|---:|---:|
| ordinary launch | BF16 |11.33696|12.19088|1.07532×|
| ordinary launch | FP16 |11.54240|9.96544|0.86338×|
| Graph | BF16 |10.95920|11.26336|1.02775×|
| Graph | FP16 |11.16416|9.50496|0.85138×|

FP16 ordinary launch still needs1.57696μs, or13.66% reduction in current SW time, to reach this Dense result. Graph needs1.65920μs. The BF16 Graph margin is modest and should be rechecked as optimization proceeds.

Paired140→162: ordinary BF16/FP16=11.42224/11.59008→11.33696/11.54240μs; Graph=11.05520/11.34032→10.95920/11.16416μs. This is a small improvement, below the skill's3% stall threshold. The earlier three-version100/131/140 campaign is also archived;100→140 provided the substantial gain in this sustained run.

## What improved

- Restrict P consumers to128threads. The original larger consumer group contributed redundant waiting and more complex address generation.
- Simplify each consumer's16 fixed chunk addresses; keep only two row-swizzle bases. Registers fell110→64, without spill.
- Use768threads, with608 CUDA pack workers balancing exactly2048 transform elements per CTA. Only128 continue to consume P.
- Overlap raw-input P1/P2 with input transforms. TMA leaders acquire the local pack barrier, then publish once globally; TC/consumer warps do not join this publication barrier.
- Compact four S and four T high planes, removing obsolete residual-plane holes. Caller workspace falls8→4MiB.
- Remove the terminal CTA barrier. CTA shared storage remains live until the last role thread returns; all TMA/GMMA and per-slot synchronization stays in place.

Publication currently uses CUDA12.4 cooperative-groups **internal driver workspace** (`get_grid_workspace()->barrier`), a per-invocation initialized counter. This is a CUDA12.4-specific ABI dependency, not a portable CUDA API guarantee. No persistent solution counters, cached computed values, extra allocations, or host synchronization are introduced. Current-pointer tensor maps are encoded each invocation. Graph/multistream tests exercise invocation isolation; a future toolkit port must revisit this dependency.

## Numerical and synchronization validation

Frozen gate remains `eF_SW <=3*eF_dense + floor`, with floors1e-3(BF16),1e-4(FP16), ordinary original-input FP64 AB reference.

| Input | SW eF, representative worst fresh trial | Gate limit | Result |
|---|---:|---:|---|
| BF16 |0.00601519652935|0.00100255679669|FAIL|
| FP16 |0.000754625769821|0.000103752760587|FAIL|

Nonfinite=0. These approximation errors match retained uncompensated97. Implementation checks do **not** turn these failures into numerical acceptance.

Selected162 passed:

-72 bitwise comparisons with97:6 input patterns ×3fresh datasets ×ordinary/Graph ×2dtypes. Workspace/output are poisoned; captured graphs replay three times. Candidate allocation is exactly4MiB, baseline8MiB.
-Quick Compute Sanitizer memcheck, racecheck and synccheck:0errors, racecheck0warnings.
-Two nonblocking streams with independent inputs, outputs and4MiB workspaces, ordinary and independently captured Graph modes,32replays per stream:256 candidate launches total, final outputs bitwise identical to97. Workspace/output are poisoned before each invocation.

The same72-check and sanitizer workflow also passed140 before further experiments.

## Profiling evidence

NCU12.8 full set,39passes per dtype, kernel replay, cache-control none and clock-control none. These profiler times diagnose execution; they are not performance verdicts.

Selected162:64registers/thread,213120B dynamic shared, one resident CTA/SM,0local-memory load/store sectors. Approximately2.519M dynamic warp instructions, Tensor active16.7–16.9%, L2 throughput50.8–50.9%, L2 hit97.95–98.03%, eligible warps0.41–0.42/scheduler. P STS/LDS source instructions remain1-way with0excessive wavefronts; aggregate shared-bank counters also include other operations and must not be attributed to P stores without source evidence.

Earlier131 cumulative instrumentation measured TC operand waits~1.47μs, prepare/issue~3.42μs, explicit wait0~0.58μs, TC loop~6.78μs. That instrumentation predates the asynchronous140/162 front end: **do not present those intervals as current162 phase timings.** Prepare/issue includes asynchronous execution overlap and is not pure removable descriptor cost. Updated role timing is still needed to locate the remaining1.6μs FP16 gap.

## Rejected directions and invalid evidence

Every edit/benchmark was logged and committed individually. `iterations109_162.json` links the complete per-iteration trajectories.

-110:two TC warp groups;111/123:uniform/parameter descriptor schemes;113:smaller N16 tiles. No gain.
-114/115:asynchronous packing with larger consumer groups. Later restricting consumers and simplifying addresses produced the140 win.
-116–120:continuous groups and TC-owned recombination; some had C7515 serialization, fully unrolled119 removed the warning but remained slower. Do not equate removal of a warning with end-to-end improvement.
-125/126:distributed wider leaf tiles with global P and recombination; extra publication and output costs were slower.
-132–137:vector P exchange, warp-leader product publication,64consumers, partial TC unrolling and host metadata reuse. No reliable gain.
-141:1024threads;142/143:K128/eight slots with wait0/wait1;144:validated static descriptor start plus partial unrolling;145:K512/two slots. All slower than140.
-146/147:two-CTA DSM cluster, four/three product split, wider64×64 tile. Correct baseline errors, but15.72/15.86 and14.74/14.97μs. NCU full and application replay both failed `LaunchFailed`; retain logs and do not invent cluster profiling counters.
-148:64bit balanced transform IO, no gain.149 terminal barrier removal was a small candidate, subsequently combined into162.
-150:multicast A across adjacent CTAs with cross-CTA free-slot retirement. Correct baseline errors, but14.67/14.95μs; saved reads did not pay for coordination.
-153/154:correct local shared transformations with seven register accumulator tiles, eliminating grid publication.22.96/21.81 and26.67/25.12μs.153 NCU found~6.4M instructions,168registers,96B stack with real STL/LDL and~200k local sectors, Tensor~8%. Fewer threads did not improve overall performance.
-156:explicit paired16bit conversion;157:non-returning release reduction publication. Correct, flat versus155.
-158/159:two independent accumulators, explicit slot order0,2,1,3 and wait1 retirement. No C7515 warning, but17.65/17.57 and13.71/14.09μs.158 had80registers and264B stack; `LOCAL:0` in cuobjdump is insufficient to claim no spill.
-160/161:warp-level MMA sync with ldmatrix, fully unrolled/bounded K loop. Correct baseline errors, but~30/~29μs, far slower than WGMMA.160 NCU~6.55M instructions,80registers,96B stack, Tensor~9%.

Exclude invalid trials entirely from speedup conclusions:109 accidentally measured Dense;112/127 failed compilation;121 used wrong shared descriptor base and produced invalid output;124 failed residency setup;128 direct producer was numerically wrong.151/152 local fusion swapped P3/P4 (off-diagonal outputs wrong, eF~1.58), fixed153. Their timings are not legitimate competing solutions.

## Continuation

Goal remains active.162 is the verified checkpoint. Next investigate current per-role backpressure, especially whether128consumer threads throttle P-slot reuse; test256consumers with simple constant8-row chunk offsets before adding a third P slot. Keep original FP64 gates, complete-call timing, all fresh inputs and independent Dense tuning. There is no evidence of a physical floor and no claim that all viable directions have been exhausted.
