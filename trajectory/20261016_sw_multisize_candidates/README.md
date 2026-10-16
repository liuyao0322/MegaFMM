# Multi-size seven-product SW candidate archive

C372–C405 are standalone candidate identifiers, not formal AKO iteration numbers.
They link their candidate objects ahead of the frozen iter371 library. No dense full-AB replacement is promoted.
The production consolidation is formal iter372. All timings retain 100 trials, 5 groups, 20 warmups, 3 fresh original FP64 checks and 2 rotating inputs.

Known invalid candidates: C374/C376/C377 have layout errors; C387/C390 fail public leaf-view validation; C388 omits the A22 K offset and its timings are invalid; C402 fails compilation (namespace), corrected in C403. C392/C404 were staged but not run. Missing CSV never establishes performance.

Correct but rejected directions include two-consumer warp groups with global/shared intermediates (C373/C378/C382/C396/C397/C398), local fused transforms (C379/C380/C381), BK256 short-K (C391), and interior store splitting (C399). Shared-U3 C396/C397 show 456-byte stack frames. NCU was attempted earlier and failed on its interprocess lock; no new profiler success is claimed.

Promoted boundary implementation: C400 paired guarded packing, cap512, padded N16/N64, true seven-product WGMMA; 1x1x1 executes seven SIMT leaves. Promoted aligned consumer split-first runtime K loop: C372; row raster from C386 for K>=2048, group2 plus asymmetric cache hints from C403 only for K>=16384. C401 group8 improved over uncached group2 but was slower than C403 at 16384 cubed. Restored sw1024.cu remains byte-for-byte unchanged.

C403 completed and passed its own implementation tests. C394/C405 retain seven cuBLASLt half-size leaf products with streaming recombination; C405 additionally packs raw quadrants. Both passed their own 4096-cubed dual-stream/Graph fresh exact checks (24 calls total per candidate); neither reached full-AB cuBLAS performance, and neither is promoted to the production kernel path. Their original FP64 gates fail. No tool certificates are transferred to these candidates. Original numerical gates remain unchanged and mostly fail; implementation-oracle checks do not certify original-input precision.

Analysis behind final consolidation: large N128 consumers have 254 registers and wait for each WGMMA K group, limiting occupancy and latency hiding; their observed sub-cuBLAS performance is not a hardware lower bound. At 16384 cubed raster group8 substantially reduces runtime relative to group2; no claim is made that cache traffic was measured. The current pass exhausted three distinct structural directions (local transforms, multi-warpgroup accumulation with global/shared intermediates, and variable-K/raster/packing schedules) without reaching cuBLAS for aligned large shapes. These failed measurements remain available for the next design pass; broad performance target remains unmet.
