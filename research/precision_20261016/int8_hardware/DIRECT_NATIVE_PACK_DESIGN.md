# Proposed P47: remove FP32 staging after quantization was removed

P46 retains P45 preparation. Its three groups per operand load/balance each
element, write one/two FP32 transforms to shared memory, synchronize, then read
the same thread's transforms to write native highs and scaled residual lows.
Native corrections no longer need INT8 tile maxima, cross-thread quantization
or a B transpose. Therefore a new control can cast/write the high and scaled low
immediately from each computed FP32 value, retaining each explicit RN operation.

The source identity is native = T(v), low = T(RN(RN(v-float(native))*2^p)), with
p11FP16/p8BF16 and the same7high/4low plane mapping. Raw groups write the same
three native blocks, without a residual. Global input/output coordinates, tiles,
loop unroll4, exact column maxima and all balancing fast/fallback paths remain.
The cooperative grid barrier, column-max block barrier and global async-proxy
fence stay. Only the transform-staging barrier goes because there are no shared
transform readers or writers. Consumer, two descriptors and complete calls stay
byte-identical to P46.

P47 deliberately retains65,536B dynamic shared allocation and existing init/
residency queries, even though only72floats are needed by column maxima. This
isolates removal of staging from a later shared-capacity experiment. Earlier
P25/P26 needed shared storage for tile maxima/quantization and avoidance of
large private arrays; their performance results do not directly predict this
per-element native store path. More live values and different store scheduling
may change compiled register pressure and latency. No pass or speed is assumed.

Primary reread: NVIDIA CUDA12.4 [Best Practices §9.2.3 and §10.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#shared-memory)
describes shared memory as useful for reuse/reordering and discusses occupancy
sensitivity via dynamic allocation in [§10.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#effects-of-shared-memory). [§9.2.7.1](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html#register-pressure)
and [Programming Guide §5.2](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#maximize-utilization)
explain register/resident-warp tradeoffs. These motivate measuring resources and
full calls; they do not establish a speedup or a physical floor for this kernel.
The direct-store opportunity is an inference from our actual native code.

Require own original-input gates/exact exponents, ordinary/Graph ABBA, fresh
multistream guards and scoped sanitizer checks before retention. P46 stays the
reference until all checks pass. No production changes or omitted correction.
