# P49 all-native raw-tuple reuse: reject complete-call performance

P49 applies the frozen ALL_NATIVE_PACK_DESIGN.md generator to P47. Only Group0
packs, writing all seven native high and four low planes from one raw tuple;
96 actual packing CTAs remain within the unchanged 288-CTA cooperative grid.
Exact per-column inside, all RN transforms, p11FP16/p8BF16 lows, both original
dtypes, seven main/eight cross dots and the consumer remain unchanged.

Own pre-MMA 54 fresh poisoned-workspace configurations compare 311427072 native
uint16 values plus 55296 exact inside exponents against frozen P47: zero errors.
Own ordinary78 original-input FP64 checks and 79872 exponent comparisons pass;
max eF BF16/FP16 1.61320431617e-5 / 2.16394614392e-5. BF16/FP16 prep54/47regs,
consumer143regs; zero stack/spills and no C7515/C7510. Runtime reports actual
3residentCTAs/SM and sufficient capacity for288 on132SMs.

Independent four-process A/B/B/A results, baseline P47 versus P49 (us):

| mode | BF16 P47 | BF16 P49 | FP16 P47 | FP16 P49 |
|---|---:|---:|---:|---:|
| ordinary |28.696840048|30.3146800995|28.82439994825|29.816440343875|
| Graph |28.65468001375|30.146039963|28.81479978575|30.03227949125|

Each mode has312 passing original-input checks and319488 exact exponent
comparisons. Ordinary worsens5.638%/3.442%, Graph5.205%/4.225%. Unlocked clocks,
no confidence interval. Reject P49; no own fresh-stream/guard or sanitizer
acceptance. P47 evidence does not transfer.

Own first balanced BF16 profile,17 replay passes per kernel: prep20.096us,
54regs/65536B, shared-resource limit3CTAs/SM, register limit4, issue15.224558%;
consumer17.536us,143regs/196736B, limit1 by registers/shared, issue31.588867%.
Both zero local read/write sectors. Profiler-process timings are excluded from
ordinary/Graph results. Reduced packing parallelism is a plausible cost, not
uniquely proven by aggregate issue rate, and no physical floor follows.

The initial ABBA driver failed126 because copyfile omitted executable mode on
the frozen binary. p49_abba_permission_failure.log preserves this failure;
chmod restored mode without changing binary bytes, then all four comparison
processes were rerun. No failed-driver data support acceptance. Isolated CPU
regeneration reproduces all four frozen files, see p49_p50_reproduction.json.
P49 manifests freeze its own sources/binary/logs/probe/ABBA/profile identities.
Production sources and library remain unchanged; this is no formal iter371.
