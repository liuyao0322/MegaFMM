# Iteration34 compiled audit: 36 only may launch; 37 blocked

The final96 consumer-only scope implementation passes a fresh compiled safety gate for **Mega36 BF16 and FP16**. **Mega37 remains blocked** by actual STACK128/spill384-store256-load/C7512 evidence; its retained instruction identity is not a hardware gate pass. Forced37 is rejected before empty/scale, raw native helpers reject it, and Auto remains Dense. This audit does not certify numerical accuracy or speed.

Final source36 `fdaae75a…`, disabled source37 `4bd6786d…`, Plan `8ef27f41…`, tests `54a26b03…`; bench `e5004b3f…`, tests binary `33f31ae9…`. The adjacent JSON contains full SHA bindings for all eight sources, both binaries, feeding PTX/cubins, object/linked streams/resources/ELF, preserved build failures and pinned headers.

| Final36 fresh compiled fact | BF16 | FP16 |
|---|---:|---:|
| BN / BM / BK / stages / CTAthreads | 32/64/32/2/384 | 32/64/32/2/384 |
| REG initial / producer / WG0 / WG1 quota | 96/64/96/96 | 96/64/96/96 |
| Initial owned CTA pool / role request | 36864/32768 | 36864/32768 |
| Highest actual GPR incl implicit producer / WG0 / WG1 | 58/93/93 | 58/93/93 |
| Prequota highest GPR | 2 | 2 |
| Full role CFG PC counts producer / WG0 / WG1 | 679/552/590 | 679/552/590 |
| STACK / LOCAL / spill-store / spill-load | 0/0/0/0 | 0/0/0/0 |
| Static / dynamic shared bytes | 1024/86144 | 1024/86144 |
| Complete static instructions | 1216 | 1216 |
| WG0 / WG1 HGMMA per committed group | 6/8 | 6/8 |

All current object, bench and test instruction streams match completely by PC, predicate and instruction text, including retained disabled37. Current streams also match the freshly audited revision6. Header differences remain in raw artifacts; no claim of complete raw-text equality between object and linked files is needed. No prior30–33 role quota proof was inherited.

The full PC worklist covers init, quota retries, branch-role separation, both K loops, all mbarrier wait stubs, common tail publication, beta paths and exits. It propagates fresh TID lineage and invalidates overwritten TID/predicate facts, exploring both unresolved branch edges. Implicit widths include CS2R pairs, 64/128-bit memory data, wide/address pairs and all16 GPRs of each HGMMA64x32 accumulator. There are no local-memory instructions, spills, helper calls or indirect controls on enabled36.

Initial quota96 is the linked ELF/resource allocation on an eight-register boundary, jointly bound to actual feeding maxntid384/minnctapersm1, launch binding and actual USETMAXREG CTAPOOL instructions. EIATTR_MAXREG_COUNT168 is a compiler ceiling, not initial quota or measured occupancy. Pool36864 exceeds128×(64+96+96)=32768; producer64 releases4096 registers, consumers request the same96 quota, and all direction/range checks pass. WG0 initializes48 FP32 registers R40–87 and WG1 initializes64 R24–87 only after successful inc96; first CS2R is0x250, first HGMMA0x1300/0x930.

WG0 commits six HGMMA and completes wait0 at0x1560; WG1 commits eight and completes wait0 at0xc50. From first issue to each wait0 there is zero generic live-accumulator access. Both K loops release only after completion, and producer-tail retires both banks. Actual feeding PTX uses nonaligned `barrier.sync 6,384` at the two role sites, with compiler memory clobbers. Pinned NamedBarrier emits aligned bar.sync and is not a legal replacement here.

There are three collective CTA events per lane: bar0 initialization, bar6 drain, bar0 publication. Consumer drain is0x15d0 and producer drain0x3d00; first shared P store is0x1950, after drain, and common publication is0x3d30 before U. Fresh tail TID S2R at0x15b0 was legally scheduled before bar6, after the whole-K loops; register/address calculations do not reuse the union payload. The gate requires shared tail writes after drain, not an artificial S2R PC order.

Both operand banks total86016B; seven full-P FP32 tail occupies57344B and overwrites14336B of bank1, so both banks must drain. Four mbarriers at86016/86024/86032/86040 remain outside the typed union. Full128/empty256 comes from the pinned PipelineAsync per-thread arrival protocol. Two consumer WGs independently own P1–3 and P4–7; unique coordinate/XOR mapping covers14336 words. All384 U owners cover2048 tile coordinates and8192 quadrant outputs without overlap. Original source30 input helper bodies, operand layouts, RN S/T, two ascending K16 perK32, whole-P order and fixed U DAG are preserved.

Compiled active-beta C0 loads0x45c0/4600/4640/46a0 all precede earliest D store0x4880, preserving exact C0=D. Beta0 executes no C0 load. RN alpha multiply and active-beta FFMA remain explicit. Common U remains within producer64 and consumer96 actual bounds. Source/API native currentAB binding and actual packed7/Dense fallbacks, one ordinary launch, zero matrix workspace, original timing/ref/FP64 gate and Graph replay content contract are preserved. The 29 pre-existing CUDA files match the frozen on-site manifest; Git history binding belongs to owner/data audit.

Failed revisions are preserved. Initial and revision1 spill32/48 accumulator words; moving descriptors, separating whole-K loops, delaying coordinates and raising36 quota128 did not remove the failures. Revision5 initially failed C++ compilation because the tail pointer moved into consumer scope while public U still referenced it. Both independent source reviews corrected the earlier protocol-only PASS distinction; the actual diagnostic is retained. Adding only a new public tail pointer after publish fixes lexical visibility. Revision5b consumer-only scope yields zero spill36 at128; final revision6 restores96 with a fresh zero-spill result. This is an observed comparison, not a causal dynamic-profile claim.

GPU validation and performance are outside this initial compiled-gate review; the owner has subsequently reported default/selected36 execution, to be independently bound in the final audit. The next complete measurement can contain16 of20 requested rows, with four37 dtype/mode keys explicitly missing. New strict candidate scope is36-only30, not requested two-route60; full1560 is a code-domain count, not an execution result. No promotion, numerical certification, goal completion or measured occupancy is asserted.

The reviewer read source/artifacts and ran pure CPU parsing/enumeration, wrote only the authorized adjacent scratch JSON/MD, and performed no build, GPU, benchmark, sanitizer, object extraction, Git or formal mutation.

Owner archived host object disassembly independently confirms both dtype launch paths set `r8d=0x15080` (86144 dynamic shared bytes), `rdx=0x100000180`/`ecx=1` (block 384×1×1) before `__cudaPushCallConfiguration`. Both dtype prepare paths pass `edx=0x15080` and attribute id8 to `cudaFuncSetAttribute`. This agrees with actual source storage size and launch geometry; the separately reported 1024 static shared bytes are not included in that dynamic allocation. The artifact and object hashes are bound in the JSON. This is a host compiler artifact check, not runtime occupancy.
