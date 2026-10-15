# P45: native correction highs alias the main operands

M12 screens native T corrections at p11FP16/p8BF16 on72fresh configurations,
216paired rows and515,064finite-native-high scalar samples; it is not a hardware
certificate. P45 starts from frozenP44 and keeps original FP16 andBF16 inputs,
exact per-column inside scaling, seven native main dots, eight correction dots,
original rounded-inputFP64 reference, gates and complete-call costs.

Each side has eleven native-T planes: seven main high planes then four scaled
lows. Preprocessing writes mainhigh once and each low as
T((FP32transform-float(nativehigh))*2^p). Correction high operands reference
existing main high planes; rawA22/B22 do not require another copy. Both sides keep
P44's original row-major orientation, without a transpose or quantizer maxima.
Historical metadata is allocated but unused, and qa/qb point to low-plane tails.
High and low restricted pointer accesses have disjoint actual plane ranges.

Correction plane indices preserve all eight terms:

| term | correction A | A plane | correction B | B plane |
|---|---|---:|---|---:|
|0|S4 low|10|B22 native|2|
|1|A22 native|3|T4 low|10|
|2|S1 low|7|T1 native|4|
|3|S1 native|4|T1 low|7|
|4|S2 low|8|T2 native|5|
|5|S2 native|5|T2 low|8|
|6|S3 low|9|T3 native|6|
|7|S3 native|6|T3 low|9|

All correction dots use original T's MN/K WGMMA type, rather than forcingFP16
highs throughBF16. The same four49,152B operand slots, TMA byte counts, all48K128
steps, accumulator fences, ready/free phases and previous-slot retirement apply.
After allK in each of the five correction groups, one scalar2^-p restoration
precedes the original recombination. Every correction dot has exactly one scaled
low operand. No adaptive scale detection, fallback or hidden host synchronization.

The native main and correction consumers share two eleven-plane maps. Both maps
are encoded for current pointers on every full run or Graph capture. B uses
rank3box64x128; A uses rank4box64x32x2, each128B inner dimension. Descriptor count
falls fromfour totwo; no cached computed value depends on prior inputs. Plane
data falls16MiB to11MiB, with original metadata/partial allocations retained.
MMA source counts remain480perCTA; traffic/storage reductions are hypotheses,
not measured dynamic counters, DRAM traffic, or performance acceptance.

The finite-high residual no-overflow argument has a limited domain, explicitly
recorded inNATIVE_CORRECTION_DESIGN.md. It does not guarantee arbitrary transforms
stay finite, eliminate underflow, or predict Hopper rounding or matrix gates.
Require P45's own precision, exact inside exponents, compiled resources, full
ordinary/Graph comparisons and fresh stream/guard/scoped checks before retention.
P43/P44 certificates cannot transfer. Initial P45 syntax-only compile failure is
frozen separately and has no device execution or matrix/performance evidence.
