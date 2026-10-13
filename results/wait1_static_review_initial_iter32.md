# Iteration32 independent static/compiler launch gate

PASS_COMPILED_CANLAUNCH_FOUR_INSTANCES for current frozen source80e4ee8f, bench8fe268ef and test95d8e95b. This review performed no build, GPU, benchmark, Git mutation or formal write; only the authorized scratch JSON/Markdown records are written. New32 compiled evidence is independently derived;30/31 quota safety was not inherited.

| dtype | BN | InitialREG | producer/consumer quota | Actual max R incl implicit | Initial CTA pool / request | first HGMMA / wait1 / tailwait0 |
|---|---:|---:|---|---|---|---|
| fp16 | 32 | 160 | 64/160 | 54/145 | 40960/28672 | 0x870 / 0xd60 / 0xe80 |
| fp16 | 16 | 96 | 64/96 | 58/89 | 24576/20480 | 0x6a0 / 0xbf0 / 0xd10 |
| bf16 | 32 | 160 | 64/160 | 54/145 | 40960/28672 | 0x870 / 0xd60 / 0xe80 |
| bf16 | 16 | 96 | 64/96 | 58/89 | 24576/20480 | 0x6a0 / 0xbf0 / 0xd10 |

All four streams have zero stack/local/spill. Object and linked benchmark SASS are byte-identical SHA4390c07edff9b286c62c0854f7f15fc62bc7062551a9c8d1deea88cf7236af39; four resource records match exactly. All CTA owned pools cover both128-thread roles and setmaxnreg directions are legal. Full actual CFG follows predicate/unconditional branches, backward loops and compiler-emitted full/empty wait stubs; role paths are disjoint. Implicit CS2R64, vector64/128 data, global64 address, IMAD.WIDE pairs and complete HGMMA8/16 accumulator spans are counted.

The critical asynchronous accumulator window starts at the first HGMMA and includes each wait1, loop backedge and full-barrier wait stub, terminating only at the final unconditional wait0. Ordinary instruction reads or writes of P registers are zero throughout this conservative reachable window. Seven fragments are cleared after successful consumer allocation before first issue. Each group has exactly14 original same-shape HGMMA operations, two ascending K16 operations for each P, with gsb0 commit only on the last instruction. Read UR15 and retire UR16 remain separate: first group is retained, pending2 leads to wait1 then oldest empty arrival, and final wait0 dominates first U FADD. No spill/reload or generic accumulator materialization was introduced by compiler operand fences.

Shared128 alignment is preserved in source, debug PTX and ELF; static1024 reservation precedes raw dynamic base0x400. Dynamic71808/86144 includes original two final banks and barriers. These are static compiler/source facts; occupancy and dynamic execution counters are not measured.

Independent parallel source/API review found no blocker. Formal TU is identical to frozen scratch. All six normalized source30 numerical/producer/API regions match; only consumer read/retire scheduling changes. Public32/33 map to config0/1; invalid34 precedes empty/scale; current AB16 and public full-span/overflow/overlap/D4/activeC04/C0=D rules remain. Native helper rejects nonnative bindings; integrating Plan chooses actual packed7/Dense fallback. Auto stays Dense. Benchmark differs only in33 CLI range, CMake only adds new TU; full-call event, batchwall, original FP64 gate and independently tuned equal-budget Dense contracts are unchanged. Native path has no globalS/T/P, descriptor encoding or matrix workspace. All27 retained CUDA files hash-match owner-preserved map; data reviewer owns historical Git binding.

Current new test binary95d8e95b and source71162d69 are bound. Its selected four complete instruction address/predicate/text streams independently match object/benchmark exactly:5344 instructions. Raw headerflags differ under filtered extraction, so this claim is instructionstream equality, not complete test-SASS file byte equality. Initial compiled approval covers the reviewed four instances in benchmark and test. No old31 test binary is treated as current32. GPU default/targeted/Graph/freshbinding/rawU-hostepilogue/tail checks, strict and sanitizers, full1024 timings and final report remain independent pending evidence. CPU/source/compiler PASS does not certify original numerical gate, raw seven independent P slots, speedup, Auto promotion or goal completion. Historical source-only full1410 is not an executed suite; future empty-release1 candidate is outside iteration32.

The companion JSON contains full hashes, four instance records, methodology and boundaries.
