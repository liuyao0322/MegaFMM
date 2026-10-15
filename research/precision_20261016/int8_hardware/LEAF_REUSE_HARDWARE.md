# P46: corrected leaves reuse native high operands

P46 changes only P45's consumer schedule, operand staging and FP32 recombination.
`pack.cuh`, `compare.cu` and `pow2.cuh` are byte-identical to frozen P45. All seven
native main products, eight native scaled residual cross dots, exact per-column
inside exponents and original FP64 AB gates remain. Low scaling is still p11
for FP16 and p8 for BF16; 11 native planes per side and two TMA maps are unchanged.

Four49,152B slots hold Bhigh/Blow/Ahigh/Alow at offsets0/16,384/32,768/40,960.
Products0/1 load24,576B, product2 loads32,768B, product3 loads40,960B and paired
products4/5/6 load49,152B. There are28K128 batches and88TMA commands per CTA,
versus48/120 in P45; nominal operand bytes1,081,344 versus1,474,560. These are
source counts, not hardware traffic measurements. The seven main plus eight
cross products still issue480native WGMMA instructions per CTA by source count.

Main and correction use separate16-FP32 fragments. A K batch commits all work
for both fragments; wait<1> precedes previous-slot release and wait<0> precedes
both fragments' scalar reads and final slot release. Both fragments carry CuTe
operand fences and the hardware warpgroup ordering. The corrected leaf is
`RN(main + RN(correction * 2^-p))`, before the original streaming product order
0,1,5,6,4,2,3. This changes FP32 addition order. M13 separately isolates that
change on72fresh numerical configurations, with two same-input M12 controls;
its model does not certify exact Hopper accumulation.

The actual consumer uses143registers per thread for both dtypes versus P45's150,
196,736B shared memory, zero stack/spill and no C7515/C7510 warnings. An additional
source accumulator does not imply higher compiled register count. Preparation
uses42registers and65,536B shared; actual residency capacity is checked outside
run/capture in every validation process.

| Independent full-call ABBA | BF16 P45→P46, microseconds | FP16 P45→P46, microseconds |
|---|---:|---:|
| Ordinary |35.11308→30.30504 (13.693% lower)|35.12012→30.27388 (13.799% lower)|
| Graph |35.44780→30.24224 (14.685% lower)|35.43436→30.24876 (14.634% lower)|

Each mode has four fresh processes and312original-input precision checks plus
319,488exact inside-exponent comparisons, all passing. They repeat78deterministic
configurations, with10timing groups per candidate/dtype, unlocked clocks and no
confidence interval. All preparation/compute costs remain included; Graph encodes
descriptors during capture only. No profile or sanitizer-process timings enter
these comparisons. P46 ordinary maximum BF16/FP16 eF is1.61320431617e-5/
2.16394614392e-5, slightly higher than P45, still below the unchanged gates.

P46 owns new seed46943+trial*347+stream*197+pattern*59, ninepatterns,
108configurations/864interleaved ordinary/Graph calls, two independent streams
and workspaces,128float guards before/after each output. All pass. Scoped
memcheck/racecheck/synccheck (kns=int8_research) each pass36configurations/72calls,
zero errors; racecheck has zero warnings/hazards. Tool subsets reuse trial0,
not additional independent fresh configurations. Graph78log is an exact copy
of its existing GraphABBA_b0, not another process. Ordinary/Graph use production
tunedDense controls; stress retains historical cuBLASGemmEx, with original
rounded-input FP64reference in both. See `p46_validation.json` and manifests.

P46 is the retained ordinary/Graph research reference. P45/P39 remain frozen
fully validated fallbacks. About30.3microseconds still exceeds the roughly10us
production SW path; production integration, Auto promotion and the overall goal
are not achieved. Only contiguous non-transposed1024cubed, alpha1/beta0 and the
two original dtypes are tested. No arbitrary-range numerical theorem follows.

Own first-balancedBF16 two-kernel NCU capture (17passes each) reports preparation20.512us/42registers/65,536B/3shared-limit CTAs and consumer17.536us/143registers/196,736B/1CTA by both shared and register limits. Issueactive26.166/31.519percent, local read/write sectors0both. These are profiler diagnostics, excluded from all ordinary/Graph comparisons; separate profiler sessions do not isolate exact causal costs or prove a floor. Raw metrics and units are frozen inprofile_p46_stats.json.
