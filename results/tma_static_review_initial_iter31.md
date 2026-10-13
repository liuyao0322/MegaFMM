Iteration 31 first-launch compiled/static review: **PASS**.

The owner may enter the already-authorized GPU correctness sequence on kernel `670ec8012736de5abeadfec0794c5cf0e00c6f7ec56003046896c72bbd43a955`, test source `d1638e0498473c7b8fb5bf7ea354d177e2b64bc8a12368a8c865df3b77f33b04`, benchmark `9e3b6a7245217aa4f728622961e9b45640346fbf4945ed76019eef472c200785` and test binary `9a8d4fae96ddf1b9138cd5ed65a2e82e1d0c712ddb86398cff80568a8aee7ed8`. This review performed no build, GPU execution, benchmark, git mutation or formal-file write.

| Both BF16 and FP16 | BN16 | BN32 |
|---|---:|---:|
| ELF initial registers/thread | 96 | 160 |
| Producer / consumer quota | 64 / 96 | 64 / 160 |
| CTA-owned initial pool | 24,576 | 40,960 |
| Final role demand | 20,480 | 28,672 |
| Producer highest actual GPR | R43 | R46 |
| Consumer highest actual GPR | R89 | R145 |
| HGMMA implicit accumulator highest | R79 | R135 |
| Role CFG overlap | 0 | 0 |
| Dynamic shared launch bytes | 72,192 | 86,528 |
| Separately accounted static shared | 1,024 | 1,024 |
| Static+dynamic shared total | 73,216 | 87,552 |
| Stack / local | 0 / 0 | 0 / 0 |
| Raw TMA static instructions | 8 | 8 |
| Raw transaction expected bytes | 20,480 | 24,576 |
| HGMMA static instructions | 14 | 14 |

All predicated branch arms, uniform TMA election loops, full/empty/raw wait stubs, role loop backedges and producer drain paths were included in the independent CFG. The two role sets are disjoint. The audit includes CS2R and .64 pairs, LDS/STS.128 four-register widths, global64 address pairs, IMAD.WIDE destination/addend pairs and 8/16-register HGMMA accumulator widths. Each role remains below its post-setmaxnreg quota. Direction is legal (64 ≤ initial ≤ consumer); the required register pool belongs to the CTA. Consumer allocation precedes CS2R initialization of all seven full fragments R24..R79 or R24..R135.

The ordinary PTX dump contains no executable PTX. Actual compiler PTX is embedded in the existing ELF `.nv_debug_ptx_txt`, which was inspected. It declares the 256-byte map parameter with `.align 64`, uses two `cvta.param` pointers with B at A+128, and has no local declaration/instruction. ELF map offset0x70 plus CBANK base0x210 is0x280; the second descriptor is0x300, both64 aligned. SASS obtains parameter addresses and feeds uniform TMA operands directly; no LDL/STL or hidden per-thread descriptor copy exists.

Actual PTX declares external shared storage with512 alignment. Each of the four ELF shared sections has size0x400 and alignment0x200; SASS uses base0x400. All bank, A/B region and product-slot starts remain multiples of512. Payload plus32 final-barrier bytes and16 raw-barrier bytes is rounded to512 by SharedStorage. The separate static1024 bytes is included in the total above.

Each compiled instance issues8 TMA boxes after the raw barrier's expected20,480/24,576 bytes. Raw wait uses producer phase XOR1. Every producer's generic shared stores are fenced before the final-ready barrier arrives. All256 threads reach the single CTA barrier before dividing into roles. The consumer executes14 HGMMA operations, waits group0 before release, and the producer waits both final banks before EXIT. These are compiled/static observations, not measured dynamic counts.

Object and linked benchmark SASS are exactly equal, including full four-function text, SHA-256 `1be160e8db30d1bb1a4447a9c42a22718870643aa9370a0f8620a29dbf1a6c98`. Four resource records match field by field. Object resource SHA is `14732cce0899d69f1ee34f0219e94035ba9dd4c150d84f65d3127486efa2f47b`; linked resource SHA is `6461f8f042464ba62cc5a24b6bf5dc2358693efd5585f5b429345553011255e6`; ELF/debug-PTX SHA is `c8908b37a866c3d289b352fff006e39244b9f9405feb865b92075b74785bb5b2`. Full source/artifact bindings are in the companion JSON.

Source/API routing review also passes: new policies30/31 explicitly select this route, invalid32 is rejected before empty/scale, native current pointers are encoded into two maps inside each ordinary run/capture, native launch is one kernel with zero global workspace, and tile/current-binding fallbacks retain packed7/Dense. All26 older CUDA TUs were independently byte-frozen against the iteration30 checkpoint. The original FP32 RN transform DAG, one final operand cast, complete seven P over ascending K/2, fixed U and alpha RN/active-beta FMA contract remains. Source tests compare30→28 and31→29 within the same atom chronology, separately test rawU and the specified host epilogue, change both current input bindings and replay fresh Graph input contents.

The first library build failed because its sentinel fed element offsets into a retained byte swizzle. The owner preserved all assertions, corrected the helper to `layout_b()` then `layout_a()(2*raw)/2`, and added raw==dense_plain. Actual pinned-layout exhaustive CPU evidence has zero physical mismatches; device/host execution code is unchanged. That CPU evidence was never substituted for the independent final compiled review.

**No compiled/static launch blocker remains.** Before measurements the owner should update the declared full1024 plan from the superseded38134 source and empty binary mapping to this freeze. GPU correctness, fresh Graph/rebinding tests, strict FP64 numerical gate, sanitizer and end-to-end speedup are still pending; this record certifies none of those. A separate final review must inspect their actual evidence.

