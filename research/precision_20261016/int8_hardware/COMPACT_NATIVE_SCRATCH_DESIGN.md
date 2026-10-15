# Next unmeasured P48: cooperative scratch capacity control

P47's direct native writes no longer access `PairPackShared::values`. The actual
cooperative kernel uses scratch only for `ma[64]` and `mb[8]` during exact column
maxima,288bytes total, before the existing block and grid barriers. Its65,536B
dynamic request was deliberately retained to isolate the staging change.

Proposed P48 changes only that cooperative function's max-dynamic-shared-memory
attribute, occupancy-query parameter and launch request from65,536 to288bytes.
The host init report must print the actual request and query actual capacity.
Unused historical type sizes and standalone diagnostic pack requests can remain;
they are not the cooperative launch's actual byte count. The consumer, workspace,
input scaling, all explicit RN arithmetic, global writes, barriers/proxy fences,
11native planes and two descriptors stay identical to frozen P47.

Smaller scratch may alter resource capacity or cache partitioning; neither is
assumed to improve complete calls. The288CTA grid already fits the old resident
capacity, so a larger maximum-CTA query is not evidence of more useful parallel
work or a latency gain. Measure actual compiled device instructions/resources,
both dtypes, exact exponents, unchanged original FP64 gates and full calls in
ordinary/Graph mode. If retained, require own fresh-stream/guard tests and scoped
tools; no prior certificate transfers. Keep P47/P46 frozen fallbacks.

This is a future proposal, not an implemented/measured candidate at creation.
The reasoning uses actual P47 buffer accesses and the resource-sensitivity
discussion in NVIDIA's [CUDA12.4 Best Practices §10.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#effects-of-shared-memory),
not a physical performance floor or a promised speedup. Do not combine it with a
new inside-column tile geometry, unroll factor or WGMMA pipeline change.
