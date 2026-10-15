# Proposed P49: reuse raw tuples in P47's global native producer

Premeasurement proposal after native fusionN1/N2/N3 precision passes but ordinary
andGraph performance rejection, each with own profile. Keep P47 as the fully
validated research reference and production unchanged. P49 is not applied to
working sources, compiled or measured at this checkpoint.

N2/N3 emit all11native planes from each raw four-quadrant tuple once perCTA,
but A is replicated across8output-column CTAs and B across16output-row CTAs.
Their69206016native shared stores per complete call are12times P47's5767168
unique global native stores. P47 itself reads/scales raw tuples in three groups;
this is not a12times raw-compute or physical-traffic comparison. N3's exponent
cache lowers complete cost only to~105us, still~3.7times P47. This supports
returning to the global producer reuse boundary, not weakening original terms.

P49 tests all7high/4scaledlow stores from one tuple in the global producer,
using P47's direct_native_store helper, exact same RN float transforms and
p11FP16/p8BF16 residuals. Preserve cooperative grid288, exact first128*8inside
columns, grid barrier,65,536B request, metadata/workspace, all helper load order,
fast normal-exponent shuffle/guard/fallback, tile sizes64x128A and128x32B,
unroll4, plane indices and global async-proxy publication. Consumer, descriptors,
compare and pow2 source remain byte-exact P47. All7main/8crossdots unchanged.

Within existing logical groups, only Group0 packs all planes; Group1/2 do no
packing after the existing grid barrier. Active packing becomes64B+32A CTAs
instead of192B+96A, while all288still participate in inside/grid synchronization.
A block192..223 and B0..63 preserve P47 modulo tile indexing. A compile-time
Group!=0 return is local to pack_pair after the collective grid barrier; there
are no later block/grid barriers in this helper and inactive blocks can finish.
The same helper is used by legacy diagnostic launchers, which must likewise
write each plane once without races. Both actual occupancy and cooperative
capacity must be queried anew; do not assume register count or grid residency.

This simultaneously reduces duplicate raw tuple reads/scales and active packing
parallelism because one CTA now writes all its planes. More stores/live values,
less latency hiding, instruction footprint and unchanged inside/grid overhead may
erase savings. Existing raw inputs remain original native values and all global
native output planes are retained. No performance prediction follows from the
source reuse count. This is distinct from raw-input fused TC/producer kernels.

First require own layout/native-plane bit comparisons against frozen P47, both
dtypes, fresh adversarial cases and exact1024inside exponents; do not execute a
new MMA until layout passes. Then original-inputFP64 gates/nonfinite checks and
complete ordinary/Graph independent ABBA. Retention requires own fresh streams,
workspaces/output guards and scoped memcheck/racecheck/synccheck. No certificate
from P47 or N1–N3 transfers. Keep the rejected candidates' failures archived.

Evidence reviewed: N1–N3 own hardware/profiles, P45–P48 producer history and
NVIDIA CUDA12.4 §5.2.3/§5.3.2 on latency hiding/shared and global memory tradeoffs;
all-plane global producer reuse is our code-specific inference. EmuGEMM2026v1
§III/V-G operand reuse/preparation observations informed M14 but its INT8
speedups do not certify native FMM. The proposed store schedule is new research.
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html
https://arxiv.org/html/2606.25453v1
