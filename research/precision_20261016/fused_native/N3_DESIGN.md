# N3: cache exact inside exponents after the grid barrier

Premeasurement reassessment after P48/N1/N2 fail to improve P47 by3percent.
P48's shared capacity alone regressed Graph. N1 eliminated global native planes
but repeated raw tuple scaling seven times and took~615us. N2 computes all
planes once and takes~117–118us, still~4.1times independent same-mode P47.
Both fusion candidates keep all native terms and pass their own layout and
original-input gates. Their failures do not justify weakening inside granularity.

Own N2 first balanced BF16 NCU17pass capture:136registers,184448Bdynamic shared,
1CTA/SM capacity,0localRWsectors,125.984us profile-only, issueactive22.267105percent,
sharedLD/STwavefronts787584/2168450, LD/STbankconflicts0/3714. Long-scoreboard /
issue-active ratio4.611615 and wait ratio1.677844. N1's comparable separately
captured sharedLD count5746787 falls strongly, while storednativeplanes/counts
remain. Aggregate ratios are not elapsed-time fractions or a unique root cause.
The profile is not a performance-floor argument and its latency is excluded
from ordinary/Graph independent ABBA.

New primary reread after this reassessment: NVIDIA CUDA12.4 Programming Guide
§5.2.3 latency hiding, §5.3.2 shared memory, §7.10 and hardware global memory
read-only cache semantics. Off-chip operand dependencies may be costly at low
active-warp count; shared caching trades loads/staging/shared resources. The
read-only-data-cache recommendation specifically assumes data read-only for the
entire kernel lifetime. Our inside data is written in the first phase of this
same kernel, so N3 does not force __ldg on it. This is a new code-specific
inference, not a performance result established by the guide.
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#hardware-multithreading
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#device-memory-accesses
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#read-only-data-cache-load-function

N3 changes only N2 exponent storage after the existing exact inside grid barrier.
All256threads copy1024ints from global inside to an appended4096B shared array,
then the existing proxy fence/block barrier publishes the regular shared cache.
Only producer transforms read that cache; diagnostic global inside1024 remains
unchanged for exact reference comparison. Original raw and native plane offsets,
all barriers/phases, WGMMA/register fences, RN arithmetic, all7main/8crossdots,
native p11/p8 residuals, K64recombination, original types/shapes/gates stay N2.
No input-derived state survives between calls and every complete call includes
inside reduction, cache staging, descriptor creation and cooperative launch.

Expected structure188544B; query actual cooperative capacity>=128 after compiling,
not inferred from N2. Source N2 transforms request98304globalints/CTA across8K64
chunks; N3 staging reads1024/CTA then uses shared reads. These are source request
counts, not physical traffic or timing. The extra staging/barrier/resource cost
may erase gains. No N3 speed or accuracy claim before its own measurements.

Own layout preflight54fresh cases seed105019+373*trial+73*pattern,9patterns and
both dtypes. Compare69206016 native values (CTA reuse included) and1024exponents
per case before MMA. Then own original FP6478gates, exact inside comparisons,
complete ordinary/Graph and independent P47 comparison. Retention also needs
own fresh streams/workspaces/output guards and scoped three-tool acceptance.
