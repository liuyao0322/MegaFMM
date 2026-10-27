# CPU-only future RS-A BK32 candidates

This is a future experiment prepared while iteration36 compiles its independent
BK64 RS-A implementation. Neither file has a public variant ID, CUDA compile,
GPU launch, hardware resource gate, numerical certification, or timing result.
It must not enter the current iteration36 measurement or certify the active goal.

The primary candidate is derived from the frozen BK64 RS-A CPU candidate
`4e32c7cf4bd7b916ae5fad3cab8aba601fb629aaca41ffd310ce40014113fca1`.
The recorded inverse of all parameter/comment edits reproduces that parent byte
for byte. BK becomes32, each whole stage holds two ascending K16 atoms per P,
the register-input builder receives the BK template parameter, and the native
parent-K guard becomes `%64`. Producer task loops already derive from BK.
BM64/BN32,384 threads, the P3+4 split, two operand banks, all full-K P sums, final
onecast A/B mathematics, the fixed U, explicit alpha RN then beta FMA, output
ownership and alias/current-binding requirements remain the parent structure.

The primary source keeps the parent source quota64/192/192. Its lower A payload
does not reduce its required source register pool:57344 words. A legal uniform
initial allocation for that hypothesis must still be152,160 or168, with all
actual role implicit register widths, allocation directions and pool ownership
verified in the compiled object and linked binaries before any GPU launch.

`candidate_160.cu` is a separate quota hypothesis:64/160/160. Its source pool
requirement is49152 words. Algebraically allowed uniform initials are
128,136,144,152 or160. Initial168 is invalid for its `setmaxnreg.inc160`
direction. These lists use384 threads,65536 register words and eight-register
quota granularity; they do not prove ptxas allocation, residency, forward
progress, zero spills or speed. The two candidates must not be described as one
experiment that changes only BK.

For BK32, each consumer thread must retain24/32 packed A words and48/64 FP32 P
words respectively, yielding72/96 payload lower bounds before addresses,
descriptors, transformation temporaries and common U. Every P/atom has distinct
A registers; no generic read, write, copy or liveness fence may touch an A source
from issue until its `wait0` completes. Preissue and postwait tensor fences,
alignas16 backing arrays without initializers, and real async ordering still
need compilation and complete physical PTX/SASS inspection.

The analytic rawA4+finalB7 operand bank is30720 bytes; both banks total61440.
With the predicted32-byte barrier structure and align128, sizeof is predicted
as61568. This is not actual CuTe/ABI evidence. The57344-byte P tail overlaps
30720 bytes of bank0 and26624 bytes of bank1. Both banks and both consumer groups
must be retired before nonaligned CTA bar6, tail stores and the common publish
bar0. Padded/minimum storage, static shared, reserved shared and the launch opt-in
budget must all be checked in actual host/ELF evidence.

At1024 there are128 CTAs and16 BK32 stages. Whole-CTA logical operand writes stay
491520 bytes, total S/T RN operations stay262144 and final onecasts stay196608,
the same totals as the parent BK64 RS-A model. The change affects stage size and
count, not cross-CTA input reuse. Generic raw A shared references stay458752
bytes/CTA. These are source-level analytic counts, not physical traffic, stalls,
occupancy or performance predictions.

The actual CPU script imports only the frozen parent's function definitions,
never its artifact-writing top-level body. It protects the parent prefix from
writes, checks BK32 vector/RS packed-pair lane addresses and generic shared-load
bank patterns, checks virtual pending A lifetimes over14 stage-count domains,
rejects early four-word overwrite and early generic-fence negative controls,
and verifies the two pool/direction candidate lists. Its stdout,stderr and exit
are separate artifacts. No new numerical emulation is claimed: the geometry
inverse and unchanged helper arithmetic preserve the prior CPU hypothesis,
which itself did not prove CUDA conversion or the original FP64 numerical gate.

Next evidence required: fresh two-dtype CUDA API instantiation and exact host
layout/sizeof, initial/role/pool/implicit-width/zero-spill compiled gate, physical
A/P lifetime and synchronization, source/object/benchmark/test identity, actual
GPU progress/alias/fallback/fresh Graph/numerical tests and sanitizers, followed
by complete ordinary/Graph1024 timings against tuned same-row cuBLASLt.
