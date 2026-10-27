# Wait1 prototype independent CPU and pinned API review

Status: **PASS_CPU_PINNED_API_SOURCE_ONLY**. Reviewer `/root/design1024`, 2026-10-14.
This is the future vector-producer scheduling axis. Iteration31 keeps its separate
TMA implementation. No template compilation, resource/SASS audit, GPU check,
Graph/sanitizer result, numerical pass or performance is certified here.

The source is bound to these hashes:

- `solution/megakernel_wgmma_pipeline.cu`: `3243483e1e84e1fc2785c38904eba6b565b0c8f72876d1664e4c88881b4f4fde`
- `.scratch/megakernel_wgmma_wait1.cu`: `80e4ee8f6df3ba738ef10101b9224c701abbb517de993a7ea7d1d710bbd8ae91`
- `.scratch/megakernel_wgmma_wait1.md`: `206ccea3a888b7b6bfd76fa250166e385ade246f9d1b4de8281ac556dc0d105e`
- `.scratch/wgmma_wait1_next.md`: `c041c3cec675dc8e3dc75fd98c42feff8a7206eec63535f3d2c73e5310b3403b`
- `.scratch/wgmma_wait1_state_model.py`: `f2ddcefde43fc7fc5193d74a39e447df9bfb2a64b95c87e8972dd9f73fbc4df6`
- `.scratch/tma_two_consumer_next.md`: `7d6b3838a2f7ae33f7892305c8571a97a6ffc94a488126db6509372caa9af736`

Seven normalized regions exactly match frozen formal30: layouts and vector
helpers; init/producer/proxy publication/producer_tail; quota/fragments/clear;
consumer descriptors and pre-issue operand fences; seven gemm calls; fixed U,
alpha/beta/C0/D epilogue; native host API. Name, comments and error-text changes
are cosmetic. The sole execution change is read/retire/pending and wait1/tail0.

Each read bank feeds fourteen same-shape WGMMAs: seven P times two ascending K16
substeps. Exactly one group is committed for that bank. The first stays owned;
the second commit makes pending2. Wait1 proves completion of the older submitted
group, then its128 collective releases make that bank reusable. The newest group
stays owned even when it completed early. Pending counts committed unreleased
groups, including groups physically completed, and never estimates actual
incomplete hardware groups. After the final commit/oldest retirement, wait0
covers the final group before drain and first generic U access.

The pinned [CUTLASS3.5.1 SS collective](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)
sets K_PIPE_MMAS1, separates read/release state, retains a prologue group, uses
wait1 before oldest release and tail wait0. [CUDA12.4 PTX8.4](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations)
supports same-shape accumulator ordering and older-prefix completion for wait1.
Thus a newer group completing first alone cannot justify releasing the old bank.
The source has no generic P access within the loop: only WGMMA and empty
compiler operand asm constraints. Final operand fences precede U. Actual emitted
accumulator instructions/spills still need a fresh SASS review.

The author's frozen model was independently rerun:1,570 states/2,302 transitions
for counts0..17,416 randomized sequences/114,464 actions including512-stage wraps,
all PASS. Its producer_tail performs two explicit independent acquire events,
including early acquisition before consumer drain. It deliberately allows group
completion out of submission order and detects three unsafe release/U mutations.

A separate inline stdlib BFS was written without importing the author's model.
Its nine-tuple carries produced/committed/retired counts, tail acquire count,
consumer mode, exact bank owners, completion flags and explicit empty/full phase
bits. Publication/release toggle these bits; independent count-derived checks
verify parity. Wait1 and its oldest collective release are one atomic event,
while tail wait0 and releases remain separate. Arbitrary completion of either
group is enabled; source-derived collective arrivals are aggregate events.
It exhaustively covers0..17,32,33,64,512 stages: **6,442 states/9,598 transitions**,
one fully drained terminal each, no nonterminal deadlock. New-before-old
completion states and early producer_tail states are present. At1024,16 stages
cover132 states/195 transitions in this separate model. Counts and invariants
are stored in the companion JSON; no simulation result is a hardware proof.

Before a launch, initial owned quotas must obey legal inc/dec direction and
cover the per-CTA role pool:20,480 registers forBN16 and28,672 forBN32. Without
another reserved pool this implies initial80..96 or112..160, subject to actual
allocation granularity. Producer64 and consumer96/160 are proposed role quotas;
new source lifetimes can change registers. Fresh role SASS must include calls,
wide operands, accumulator materialization and spill paths. Formal30/31 resources
cannot certify this source. The inherited flagged-layout edge sentinel is bounded
and not a full physical-pointer proof; layouts are unchanged and actual new
instantiations remain pending.

Numerical chronology, whole-K seven P, original S/T RN with one cast, fixed U,
RN alpha, active-beta FMA, all-four-C0-read-before-stores and native current-binding
API are retained. This is a source chronology claim. Raw P/U/output comparison,
original numerical gates, sanitizer, C0 alias/fresh AB/beta0 and Graph validation
remain independent. Report any experimental failed gates explicitly. Certified
speedup requires these gates and complete-call matched timings vs source30 and
independently tuned cuBLAS. No speedup or Auto eligibility is claimed.

The quick hybrid-doc review also passes its CPU-only boundary: directSW64 child
raw group+16 elements; original-parentSW32 exact36-copy path; changed splitcompact
SW32 12-copy matched controls; full128/empty256; rolepool32768/uniforminitial88
and possible one-CTA limitation. SW64 child actualtemplate/descriptor/GPU and
performance evidence remain pending. No hybrid change enters iteration31.

Only `.scratch/wait1_review_next.json` and this Markdown were written by this
reviewer; formal sources, prototype files, builds, GPU jobs and git were not
mutated or executed.
