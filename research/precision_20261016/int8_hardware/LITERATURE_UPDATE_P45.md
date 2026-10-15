# Targeted primary-source update after P43–P45 (2026-10-16)

This supplements the frozen parent literature review without rewriting it. It is
a targeted engineering search/read, not a systematic review. No published speedup,
accuracy certificate or machine-model optimum transfers to MegaFMM.

## EmuGEMM, Lu et al., 2026 v1

[Primary text](https://arxiv.org/html/2606.25453v1), dated24June2026;
read§II-B,§III-A/B and§V-G. Interleaving slices can reduce TMA commands and reuse
on-chip operands across products, but additional accumulators and shared staging
compete for capacity. Their measured INT8 pipeline-depth/tile-width trade-offs
are not measured BF16/FP16 WGMMA characteristics. Their reported square-matrix
results cover2048–16384;§V-G explicitly notes preprocessing dominance at512–1024.

MegaFMM already keeps partial products/recombination in registers, so the paper's
removal of global INT32 intermediate round-trips is not an available saving here.
The applicable hypothesis is narrower: reuse a native high operand when evaluating
its corresponding residual cross terms. This may reduce repeated operand loads
without eliminating any of the eight dots. It costs an additional correction
accumulator and changes recombination order if the corrected leaf is combined
early. Test that order numerically before a separate hardware implementation.

## Twill, Soi et al., 2025 v1

[Primary text](https://arxiv.org/html/2512.18134v1), dated19December2025;
read§1,§4.3,§5.3/5.4 and§6.1/6.2. Scheduling depends jointly on operation latency,
warp assignment, memory capacity and communication/synchronization. Optimality
is relative to the supplied machine/dependence constraints; tile choices and
full CUDA lowering remain separate. Current support assumes singly nested loops
without extra control flow, while MegaFMM has multiple product branches.

Use an explicit buffer-lifetime and accumulator-resource model for any reuse
schedule, including fences and slot retirement. This does not justify claiming
that P45's aggregate stall counters establish a floor, nor that a paper's attention
schedule is optimal for this FMM.

## NVIDIA CUTLASS: Efficient GEMM / Hopper specialization

[Official documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
read pipelining and Hopper warp-specialization sections. Producers must wait for
empty buffers; consumers signal release only after asynchronous use completes.
Register/shared capacity and tile dimensions constrain available overlap. Use
these as correctness/resource constraints, then measure complete calls. P45's
lower registers alone do not establish lower latency or additional resident CTAs.

## Next hypothesis, unimplemented

Starting from independently validated P45 only, group each native transformed leaf
with its one or two residual cross terms. Load the native high once perK128 for
both main and correction work. Retain seven main products, all eight correction
dots, originalFP64 reference/gates and full cost accounting. A first control can
keep eleven-plane storage and existing descriptors, avoiding a simultaneous
layout change. Source-level estimates are28rather than48K128producer steps and
88rather than120TMA commands perCTA; prove exact maps and counts before relying
on them. No performance prediction follows. A separate model must test corrected
leaf recombination/FP32 rounding, then own hardware validation if it passes.
