# P43: independent BF16 correction hardware control

The passing M11 numerical model has no hardware certificate. P43 preserves
original FP16/BF16 main operands, exact per-column balancing, seven main dots,
all eight standard correction dots and their five recombination groups.
Correction high/low operands are BF16(native high) and BF16(transform-high),
with no fixed residual scaling. No old SW implementation or production edit.

The preprocessing path retains P39's288 cooperative CTAs and one grid barrier.
It removes adaptive quantizer maxima, integer exponent metadata, clamping and
restoration. A and B transformed FP32 values retain the shared preload barrier.
B's two 128x34 BF16 transpose buffers each have publication and retirement
barriers before cross-warp reads/reuse. A does not reuse shared values afterward.
Historical scale buffers remain allocated, unused; host diagnostics do not read
uninitialized metadata or report fictitious scale eligibility.

Nine16bit correction planes per side double their previous INT8 storage, adding
4.5MiB of preparation writes and corresponding nominal consumer reads. Workspace
plane data is16MiB plus retained metadata/partial scratch. Nominal source loops
perform224native+k16 and256BF16+k16 MMA calls perCTA versus224native+128INT8+k32;
these are source counts, not measured dynamic instructions or DRAM traffic.

Four49,152B slots share the existing196,736B aligned shared-memory budget. Native
operands use B[0,16384), A[16384,24576); corrections use firstB[0,16384),
secondB[16384,32768), firstA[32768,40960), secondA[40960,49152).
Only paired groups submit the second correction operands. Barrier expected bytes
are24KiB for native/single correction and48KiB for paired correction.
All48K128steps use slot=seq%4 and ready phase=(seq/4)%2. Producer waits on free
phase=((seq-4)/4)%2 before reusing a slot. Four consumer-warp leaders arrive only
after wait<1> retires the preceding batch, or finalwait<0> retires the last.
Accumulators have hardware fences before asynchronous MMA and after completion
before any scalar read/reuse; a warp shuffle is not a memory-ordering substitute.

Both correction operands initially use K-major BF16 WGMMA. With128B swizzle,
rank3 K128 boxes would have an invalid256B fastest dimension. The measured
version uses rank4 dims64x512x8x9, boxes64x{64,32}x2x1, byte strides1024,128,
Plane*2 and coordinates(0,M,k*2,plane). Native main descriptors stay unchanged.
The [CUDA12.4 tensor-map API](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
limits the fastest bounding-box byte extent to the swizzle size. The invalid
initial rank3 version and its host-encode failure are separately frozen in
p43_descriptor_preflight_manifest.json: no matrix or timing acceptance.

All caller workspace, input transforms, descriptor encoding and computation stay
in the measured full call. Graph encoding occurs only at capture. No allocation,
host synchronization or cached input-dependent values are added inside run.

Required acceptance: own compiled resources/residency and ordering diagnostics;
own original-inputFP64 checks and exact1024inside exponents for both dtypes;
complete ordinary/Graph independent timing; fresh multistream guards, extreme
small values and scoped tools before retention. In particular M11's FP32 bmm
cannot certify BF16 WGMMA subnormal/underflow behavior. Fixed2^16 scaling remains
disabled; its finite-high overflow counterexample is preserved in M11.
