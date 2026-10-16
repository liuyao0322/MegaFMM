# Native fusion hardware evidence

At the native-fusion checkpoint, P47 was the fully validated ordinary/Graph
research reference. The later independently validated P56 native consumer schedule is now the
reference (~25.1us; see ../int8_hardware/STATIC_LEAF_SCHEDULE_HARDWARE.md), with
its own14-pattern168/1344stream/guard and56/112scoped-tool evidence. P51 remains a
historical validated fallback.
All N1/N2/N3 comparisons below retain their original frozen P47 controls. Production
and its static library remain unchanged. The accuracy/performance goal is active:
P47 takes~28.8us versus production SW~10us, while production SW still fails the
original accuracy requirement. Native fusion N1/N2 pass their own numerical
checks but fail performance and do not replace P47.

Both new candidates preserve original FP16/BF16 inputs, exact per-column inside,
all seven main products and eight native residual cross dots, p11/p8 restoration,
FP32 output and original rounded-input FP64 AB/gates. M14's numerical screen
motivates K64 streaming product recombination; actual WGMMA is checked separately.
N1/N2 share that recombination order. This is separate from earlier INT8 F13.

| Candidate / mode | Same-round P47 BF16 / FP16 us | Candidate BF16 / FP16 us | Decision |
|---|---|---|---|
| N1 ordinary | 28.549200 / 28.815040 | 616.205154 / 615.401924 | Reject,21.58x/21.36x P47 cost |
| N1 Graph | 28.493920 / 28.754160 | 616.138519 / 615.559593 | Reject,21.62x/21.41x |
| N2 ordinary | 28.589639 / 28.783001 | 118.346798 / 117.528841 | Reject,4.14x/4.08x |
| N2 Graph | 28.703360 / 28.740720 | 117.920678 / 116.977077 | Reject,4.11x/4.07x |

Each row uses its own independent four-process ABBA, ten complete-call timing
groups per dtype/candidate, original78configurations repeated across processes.
Per candidate and mode312 original gates and319488 exact exponent comparisons
pass. Unlocked clocks, no confidence intervals. Ordinary calls include inside,
transform, all MMAs, descriptors and output; Graph descriptors are created at
capture and replay includes all captured device work. There is no allocation,
host synchronization or input-derived cache in run. Profiler-process timings
never enter these results. N1/N2 do not have own fresh dual-stream/output-guard
or sanitizer acceptance; no such certificate transfers from P47.

Each candidate's separate first ordinary78 suite passes and has maximum BF16 /
FP16 eF7.76549370193e-6 /8.86329900005e-6,0nonfinite and79872 exact exponents.
All per-case printed errors match N1 versus N2; this does not prove full-output
bit equality. Independent pre-MMA diagnostics each cover54fresh configurations:
9patterns*3trials*2types, seeds103001+359*trial+67*pattern(N1) and
104003+367*trial+71*pattern(N2). Each compares3737124864 uint16 operand values
including CTA reuse and55296exponents, zero errors against accepted P47 planes.
These repeated operand positions are not billions of independent input values
and are not FP64 matrix or sanitizer evidence.

N1 has two49152B raw and two24576B native leaf slots,147584B total shared;
N2 has one raw and one135168B all-plane native pack,184448B total. Both have
136registers,0stack/spill and no C7515/C7510 in their own builds. Both actual
cooperative resident queries report132SM*1CTA/SM, supporting128CTAs. Workspace
is4096B in both. Source raw requests64TMA/393216nominalbytes per CTA, nativeMMA480,
versus P47's88TMA/1081344operandbytes. These counts do not predict physical
traffic, parallelism, latency or a hardware floor.

N1 re-reads/balances the same raw tuple for seven leaves and hands off56times.
N2 emits all7high/4low planes from each balanced tuple once and hands off8times,
but serializes transformation of the next pack until TC releases the current
pack. Raw prefetch can overlap consumption; full next-pack transform cannot.
The separately measured reduction from~615us to~118us supports investigating
this duplicate work; it is not an independent direct N1-vs-N2 paired estimate.

Own first balancedBF16 NCU12.8 captures,17passes each:

| Metric | N1 | N2 |
|---|---:|---:|
| Profile-only duration us | 634.752 | 125.984 |
| Issue active percent | 16.230425 | 22.267105 |
| Shared LD / ST wavefronts | 5746787 /2166856 | 787584 /2168450 |
| Shared LD / ST bank conflicts | 0 /2120 | 0 /3714 |
| Local read / write sectors | 0 /0 | 0 /0 |
| Long-scoreboard / issue-active ratio | 5.918134 | 4.611615 |
| Wait / issue-active ratio | 2.973386 | 1.677844 |

These are aggregate diagnostic counters, not elapsed-time fractions or unique
causal attribution. Lower shared-read count accompanies removing repeated
raw transformations; it does not establish all remaining cost as exponent loads.

Initial N1 preflight failed before any MMA: generic shared indexing omitted
CuTe's flagged byte-to-element swizzle conversion, a hazard already recorded in
old F13 diagnostics. Preserve failing source/binary/log inpreflight_01; repair
withas_position_independent_swizzle_layout for generic indexing, retaining
flagged layouts for make_tensor/WGMMA descriptors. All54cases then pass. The
first probe compile also needed typed casts of P47 void pointers. N2's first
link failed because its local binary directory did not yet exist; standalone
build now creates it. NCU's first N1 filter used 'false' while its demangled
name rendered '(bool)0', so it captured no kernels; that attempt is archived
and excluded. Corrected N1/N2 profiles each explicitly report one17pass kernel.

After P48/N1/N2 fail to improve P47, the own N2 profile, failure-history review
and new CUDA12.4 primary reread motivate N3's isolated shared exponent cache.
N2 repeatedly accesses global inside values after the grid barrier. Caching may
help dependent loads with few active warps, but adds staging/shared costs. The
CUDA read-only cache assumes immutable data over the entire kernel; these
exponents are written in the same kernel, so forcing __ldg is not this proposal.
See N3_DESIGN.md. Only N3's own device evidence can support acceptance.

References: NVIDIA[PTX12.8 WGMMA/proxy ordering](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html),
[CUTLASS flagged pointers](https://github.com/NVIDIA/cutlass/blob/main/include/cute/pointer_flagged.hpp),
[CUDA12.4 memory and latency guidance](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html).
Code-specific hypotheses and counts are our inferences. See immutable per-candidate
manifests and independent audit.py for actual measured scopes and production hashes.

## N3 completed: exponent storage control

Own54fresh layouts (seed105019+373*trial+73*pattern) compare3737124864uint16
operands including CTA reuse and55296exponents,0errors, before its own MMA.
Own first ordinary78gates/exponents pass; every printed corrected eF matches
N1/N2 on the same standard cases, with no claim of bitwise full-output equality.
Independent ordinaryABBA P47→N3 BF1628.573920→106.078319us and
FP1628.765720→105.360960us; Graph28.670400→105.615918us and
28.690600→104.887001us. Each mode312gates/319488exponents pass. N3 remains
3.66–3.71times P47 cost, rejected. It has no own fresh dual-stream/output guards
or sanitizer acceptance. N2→N3 rough~118→105us is across separate rounds, not a
direct paired estimate of exponent-cache effect.

Own build136registers,188544Bshared,0stack/spill/noC7515/C7510 and actual
132SM*1CTA cooperative capacity. Own first balancedBF16 NCU17pass capture:
113.600us profile-only, issueactive24.865674percent, sharedLD/STwavefronts
1180800/2176503,bankconflicts0/7671,localRWsectors0/0, long-scoreboard ratio
3.799002,wait ratio1.566830. Shared caching raises shared reads as expected,
while long-scoreboard ratio falls; these aggregate observations cannot assign
a causal duration or prove that global exponent loads were the unique bottleneck.

All three raw-fusion variants materialize11native planes per CTA perK64. Across
128outputCTAs and8chunks this writes69206016native values, versus5767168unique
global native values in P47, a12x source store count. A values are repeated over
8B-column tiles, B over16A-row tiles. P47 itself reloads raw tuples in three
separate groups; therefore12x emitted native values is not12x raw-load/scaling
work, nor12x physical traffic. This structural replication is a remaining cost
that local inside caching does not remove. Keep P47 and return to testing shared
raw-tuple reuse in its global native producer; ALL_NATIVE_PACK_DESIGN.md records
the separate unimplemented P49 proposal. No P49 performance or acceptance exists.

N2/N3 generators reproduce all five generated source/build files each, byte-exact
in isolated CPU temporary directories. Per-candidate layout, build, ordinary,
ABBA, source/binary and profile identities are frozen in nine manifests. Local
executables and profiler containers are retained by hash, not committed binaries.
The first N1 failed preflight remains distinct and never executed MMA. Audit
keeps production and P47 source/binary hashes unchanged, goal_achieved=false.
