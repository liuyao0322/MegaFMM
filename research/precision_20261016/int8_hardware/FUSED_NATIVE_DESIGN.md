# M14 and proposed native fusion N1

P47 preserves precision but complete calls remain about28.8us versus production
SW about10us. P48 shows smaller shared request alone does not resolve this gap.
Next numerical hypothesis changes the K traversal to permit raw-input fusion,
using native p11FP16/p8BF16 corrections without INT8 quantization. All seven
main products and eight standard cross dots, exact per-column inside balancing,
original rounded-input FP64 AB/gates and both original types remain mandatory.
This is new research, not restoration of stopped FP8 or removed production SW.

First M14 tests72fresh configurations (12patterns, both dtypes,seeds9301–9303)
and four paired numerical arms: existing full-K corrected leaves with K128 bmm,
full-K leaves with K64 bmm, K64 partial-output recombination then accumulation,
and K64 streaming product accumulation directly into four output tiles. This
isolates changed FP32 accumulation granularity from recombination placement.
The first same-input uniform baseline per dtype must reproduce M13's corrected
leaf exactly. Disable TF32, require fixed floors/nonfinite checks as well as the
original gate. FP32 bmm/add models neither exact WGMMA accumulation nor latency.

The streaming arm avoids four extra partial-output tiles (64FP32 registers per
TC thread) that a separate chunk-output accumulator would require. For each K64
chunk it executes products0,1,5,6,4,2,3 with their complete one/two residual terms,
restores residual scale and updates the four persistent output fragments. This
changes addition order and increases fragment handoffs versus full-K leaves;
numerical and compiler behavior cannot be assumed from P47 acceptance.

Only after a model pass, propose a separate `fused_native` hardware prototype:
128cooperative CTAs,256threads,64x32output quadrants. All256threads first compute
the same exact eight inside columns per CTA and publish1024exponents, then one
grid barrier. Actual capacity must cover all128CTAs on the assigned132SM H100.
Afterward, one128thread group performs TMA/raw transforms and one128thread group
performs native WGMMA, producing the four FP32 output quadrants.

Two current-pointer TMA maps address the original1024x1024native inputs. At K64,
four32x64A blocks occupy16KiB and four64x64B blocks32KiB,48KiB raw per chunk.
Retain the raw blocks while generating all seven corrected leaves. A high/low
operand slot is24KiB, using native MN-major B and K-major A as P47. Two raw slots
plus two leaf slots nominally occupy144KiB before barriers/alignment/maxima;
this is a layout proposal, not compiled shared usage. Explicitly prove all TMA,
regular shared-memory and async WGMMA proxy/fence/barrier lifetimes, including
read-to-reuse accumulator ordering if double-buffered fragments are attempted.

Source proposal counts64raw TMA commands/CTA and393,216nominal input bytes,
versus P47's88commands/1,081,344operand bytes, while480native MMA instructions
remain. Removing global transformed planes could reduce workspace/traffic and
a kernel boundary, but recomputes transforms across output CTAs and adds shared
traffic, register pressure and per-leaf handoffs. These costs can erase savings;
no speedup, occupancy or physical-floor claim follows from these counts.

Prior F13 INT8 fusion was numerically accepted after layout repair but took
roughly224–240us. Its tile maxima/quantization and shared publication are not
required by native corrections, but its failure is evidence against assuming
fusion is automatically fast. Do not copy its acceptance certificate or restore
removed production implementations. Require independent native layout checks,
original-input precision/exponents, ordinary/Graph complete-call comparisons,
fresh streams/guards and scoped tools before retention or production promotion.

Primary reread: [EmuGEMM2026v1 §III and §V-G](https://arxiv.org/html/2606.25453v1)
supports considering operand reuse and accumulator/resource tradeoffs, while
reporting that preprocessing can dominate at512–1024. Its INT8 results do not
validate native FMM or raw-input fusion. Our outputs already stay on-chip, so
its removal of global INT32 partials is not an available saving here. The raw
fusion proposal and all byte counts above are inferences from our code.

N1 is unimplemented at this checkpoint. M14 results, hardware acceptance and
performance must be recorded separately; keep P47 as the complete reference.
