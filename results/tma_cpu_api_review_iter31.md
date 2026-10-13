# Iteration31 TMA CPU/API/source review

Reviewer `/root/design1024`; current status
**PASS_CPU_PINNED_LAYOUT_API_SOURCE_ONLY**, restored after owner unit repair,
actual pinned-layout CPU evidence and this incremental source review. The owner of
formal implementation, build, GPU, benchmark, logger and git remains
`/root/core1024`. This review only reads formal files and writes this document
and its JSON companion. It performs no build, GPU, git or formal-file write.

The first actual CUDA template build failed the dense-box sentinel assertions.
Pinned `smem_ptr_flag` upcast retains a byte-address Swizzle, but the sentinel
directly calls the placeholder `ComposedLayout` on element indices. Actual
`make_tensor` moves that Swizzle into the pointer and uses `layout_b` for
element offsets. For example A SW64 row2/k0 has raw element64 and physical
byte128 XOR16=144, giving element72; direct placeholder invocation applies
the byte Swizzle to64 and returns64. The repaired retained assertion checks
actual `layout_b` raw dense offsets and applies `layout_a` to twice that
offset before converting back to elements. Vector sentinels use the same
units. Core owns the actual pinned layout check, formal fix and CUDA rebuild.
The earlier review missed this assertion misuse; the independent intended
physical-offset CPU model did not instantiate the failing assertion.

Core's compiled actual pinned-layout CPU check exited0. It enumerates the
actual half_t A14,336/B16 3,584/B32 7,168 elements and vector8 groups,
plus absolute byte topbases0/512/1024/8192, with physical/vector mismatch0.
The old placeholder invocation mismatches10,752/2,688/5,376 coordinates.
The reviewer read the complete source, log and exit file; the reviewer did
not compile or execute this check. BF16 uses the same16-bit atom construction;
four actual CUDA instantiations and resources remain separate owner gates.
Source/log/binary/exit hashes are recorded in the JSON companion.

Frozen scratch CU `4911ade686966e598f7e7d5685a5e2033ed1c978a58aa63f6e929c69ef966d4c`
and MD `54e2270d7beb1ab8447dc377a3585f1be90a7283375f2e961b4136215dfb1cfa`
match the independently reviewed alias prototype. Formal TU
`670ec8012736de5abeadfec0794c5cf0e00c6f7ec56003046896c72bbd43a955`
differs in two comments and two constexpr verification helpers, retaining
all dense/vector assertions and adding explicit actual raw-density checks.
No host or device execution body changes.
The JSON companion binds all six reviewed formal files by SHA-256.

Independent CPU evidence enumerated 78,848 compact offsets, 9,856 contiguous
16B-aligned vector8 groups, 630,784 BF16/FP16 final seven-slot bit checks,
3,072 consumer coordinates and 256 alias raw-box instances. Additional checks
covered 630,784 absolute SW32/SW64 addresses across eight align512 topbases,
1,132 raw phase/count/tail transitions, and 66 abstract two-bank producer/
consumer interleaving states. These passed without assertion failures.
CPU rounding and phase models do not prove hardware numerics or synchronization.

The implementation issues four complete A64x32 and four complete BBNx32 TMA
copies per CTA/K32. A K_SW64, BN32 B MN_SW64 and BN16 B MN_SW32 match the
64B/64B/32B inner boxes. All operand regions, banks and product slots are
512B multiples and shared declarations use align512. Pinned CuTe's absolute
pointer swizzle and fixed-zero descriptor base offset agree at these origins.

CUDA12.4 rank2 descriptor parameters use current parent A/B dimensions,
byte strides 2*lda/ldb and element steps(1,1). Both128B descriptors are
align64 and passed in a256B immutable grid-constant aggregate. The launch
re-encodes both current pointers inside every ordinary complete host run;
Graph capture binds those fixed pointers while replay reads their fresh data.
Graph capture legality still requires GPU evidence.

Two separate count1 raw transaction barriers expect20,480/24,576 bytes and
wait at producer-phase XOR1. All vector tasks read four raw snapshots after
completion before overwriting their own A21/S3 or B12/T1 aliases. The three
retained original slots receive no SIMT re-store. An explicit compiler memory
fence follows raw wait; it does not add a hardware barrier. The final
PipelineAsync2 uses128 producer and128 consumer arrivals. Per-writer proxy
fences precede final publication, and all WGMMA readers wait0 before release.

The full seven P, fixed RN U, one final operand cast, alphaRN/betaFMA,
beta0/nullC0 and exact C0=D loads-before-stores contract is preserved.
Only preparation queries the device and sets dynamic shared opt-in.

Formal variants30/31 use explicit branches before28/29 and configuration
variant-30. Native metadata failure selects prepared packed7; common current
AB16 failure selects zero-scratch Dense. Public span-end/overflow/overlap,
output alignment and exact in-place guards remain. Auto, levels0, empty and
degenerate scale policies remain, and megakernel caller workspace stays zero.
Four header APIs match their definitions; max31 rejects32; CMake adds the
TMA TU and retains driver linkage. Benchmark compact cudaMalloc-aligned
bindings support its native actual labels. `algorithm_name()` is metadata
based and alone cannot report a current AB+2 rebound Dense execution.

Known template complexity risk was addressed in the frozen scratch by bounded
sentinels; independent CPU full geometry remains separate evidence. Register
budgets are hypotheses pending actual compile evidence: producer64,
consumer96/160 and legal initial direction plus owned CTA pool. This review
does not inherit iteration30 resource results. `/root/pipeline_review30`
owns the independent initial quota/role SASS audit before first GPU use.

No CUDA kernel compilation, actual descriptor lowering, GPU Graph/correctness/sanitizer,
original FP64 numerical gate or timing result is certified here. The separate
frozen two-consumer axis is excluded from iteration31.
