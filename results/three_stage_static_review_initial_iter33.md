Iteration33 independent compiled initial review: PASS / CAN_LAUNCH

All four new FP16/BF16 BN16/32 instances received fresh full role-CFG, register quota and inflight accumulator checks. Initial96/160 allocation and producer64 use the CTA owned pool. Including64/128-bit implicit widths, producer maxima58/54 and consumer89/145 remain below quotas. All accumulators are initialized before first HGMMA; every reachable generic live P read/write from first HGMMA through wait1, loops and waitstubs until tailwait0 is absent. No stack/local/spill/call path exists.

Pinned PipelineState and PipelineAsync directly support Stages3 with compare3/reset0/phaseXOR, six8-byte barriers (empty base=full+24) and three producer tail acquires. Actual new SASS implements all of these. Three banks retain max2 unretired committed groups. Actual host launch/prepare constants allocate107648/129152 dynamic bytes; extern shared alignment128 and static shared1024 are recorded separately.

The complete four address/predicate/instruction streams match object, benchmark and current test (1528/1184/1528/1184). Object raw header flags differ and are retained; benchmark/test SASS files match byte for byte. Source/API/benchmark only add34/35, invalid36 and thirdbank; original DAG, aliases, fallback and timing/reference contract persist. All28 old current CUDA TUs match the owner base map.

Frozen source f78b1fbf1541583315ce084cac3e05afd7b5d5887f64b928c0f504c9902afd23; benchmark8d0d7ad7b728537962c78c880cbb67df8dcef59c283a895764cff4b383d0dd0b; current testd31a70941bf7212abbff172bc2e54eb0960c3b9f9c5b7d1c0ef7ad61ba6879de. JSON contains exact compiler/source bindings and per-instance instruction addresses.

Reviewer performed no build/GPU/benchmark/Git/formal mutations. CAN_LAUNCH permits owner first execution; it does not certify original FP64 precision or promote performance. Auto remains Dense. No measured occupancy, causal profiling or current full1530 execution claim is made.
