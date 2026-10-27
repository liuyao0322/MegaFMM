#!/usr/bin/env python3
"""CPU-only BK64 geometry against the frozen consumer-only iteration34 rev6 source.

Imports frozen Python functions without invoking either old main. No C++/CUDA
instantiation, GPU operation or file write occurs in this model. An audit hook
rejects file mutations, including accidental writes by an imported old main.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ".scratch/psplit_bk64_scope34_baseline.cu"
BASE_SHA = "fdaae75a80893ee161a9db202d4746e86daf3d38135332759098bd9f7faa7f91"
CANDIDATE = ".scratch/psplit_bk64_scope34_candidate.cu"
CANDIDATE_SHA = "d119f6cf9e922f870a2acca90803231feb330d8cba134ea91942e101b4b4baeb"
GEOMETRY_MODEL = ".scratch/psplit_bk64_model.py"
GEOMETRY_SHA = "1eb0197f886ad7c160728564cc18c853c2b60792c7283695e3e24de23270c8f7"
OLD_MANIFEST = ".scratch/psplit_bk64_manifest.json"
OLD_MANIFEST_SHA = "2caddaef0db928f9c2fab85cf1ff000dbee1699141dd622714fa434b43548e6b"


def reject_mutation(event, args):
    if event == "open":
        _, mode, flags = args
        if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or \
                (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise RuntimeError("CPU model rejects any file write: " + str(args[0]))
    if event in ("os.rename", "os.remove", "os.rmdir", "os.mkdir", "os.link", "os.symlink"):
        raise RuntimeError("CPU model rejects mutation: " + event)


sys.addaudithook(reject_mutation)


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def frozen_check():
    assert digest(BASE) == BASE_SHA
    assert digest(CANDIDATE) == CANDIDATE_SHA
    assert digest(GEOMETRY_MODEL) == GEOMETRY_SHA
    assert digest(OLD_MANIFEST) == OLD_MANIFEST_SHA
    manifest = json.loads((ROOT / OLD_MANIFEST).read_text())
    for path, expected in manifest["file_sha256"].items():
        assert digest(path) == expected
    return {**manifest["file_sha256"], OLD_MANIFEST: OLD_MANIFEST_SHA}


def load_geometry():
    spec = importlib.util.spec_from_file_location("_frozen_bk64_geometry_no_main", ROOT / GEOMETRY_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # Import only. Neither frozen main is invoked.
    return module


def source_check(geometry):
    base = (ROOT / BASE).read_text()
    source = (ROOT / CANDIDATE).read_text()
    normalized = source
    for new, old in (("static constexpr int BK = 64;", "static constexpr int BK = 32;"),
                     ("const int row = task/8, k = (task&7)*8;", "const int row = task/4, k = (task&3)*8;"),
                     ("p.k%128 == 0", "p.k%64 == 0")):
        assert normalized.count(new) == 1
        normalized = normalized.replace(new, old)
    assert normalized.encode() == (ROOT / BASE).read_bytes()
    # Source only: the q storage is lexically confined to consumer else scope.
    producer_start = source.index("  if (tid < T::ProducerThreads) {")
    consumer_start = source.index("  } else {", producer_start)
    producer = source[producer_start:consumer_start]
    consumer = source[consumer_start:source.index("  // Complete all14336", consumer_start)]
    assert not re.search(r"\bq[0-3](?:_data)?\b", producer)
    alloc = consumer.index("cutlass::arch::warpgroup_reg_alloc<96>();")
    for q in range(4):
        assert consumer.count(f"float q{q}_data[cosize_v<QLayout>];") == 1
        assert consumer.index(f"float q{q}_data[cosize_v<QLayout>];") < alloc
        assert consumer.index(f"clear(q{q});") > alloc
    assert "cutlass::arch::warpgroup_reg_dealloc<64>();" in producer
    assert "params.consumer_arv_count = T::AllConsumerThreads;" in source
    assert "static constexpr int AllConsumerThreads = 256;" in source
    # Two lexical nonaligned barrier call sites implement ONE all-role event.
    named = 'asm volatile("barrier.sync 6,384;" ::: "memory");'
    assert source.count(named) == 2 and producer.count(named) == consumer.count(named) == 1
    assert producer.index("pipeline.producer_tail(state);") < producer.index(named)
    assert consumer.index(named) < consumer.index("unsigned tail_tid;") < consumer.index("float* tail = storage.tail_products;")
    assert 'asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tail_tid) :: "memory");' in consumer
    assert source.count("__syncthreads();") == 2
    assert source.count("warpgroup_wait<0>();") == 2
    assert source.count("pipeline.consumer_release(state);") == 2
    assert source.count("k_base += T::BK") == 3  # producer + independent whole-K loops
    assert source.count("gemm(mma,") == 7
    for p in range(7):
        assert f"const float p{p+1} = tail[psplit_tail_word({p},local_row,local_col)];" in source
    # Re-execute the old pure geometry derivation, whose same three arithmetic
    # changes have now been independently byte-bound to the new rev6 baseline.
    old_geometry = geometry.source_geometry_check()
    return {**old_geometry,
            "baseline": BASE, "baseline_sha256": BASE_SHA,
            "candidate_sha256": CANDIDATE_SHA,
            "normalized_exact_frozen_source_match_after_three_geometry_reverts": True,
            "source_symbols_and_comments_unchanged": True,
            "consumer_only_q_lexical_scope_preserved": True,
            "source_quotas": [64,96,96],
            "barrier_sync_6_384_memory_clobber_call_sites": 2,
            "syncthreads_call_sites": 2,
            "CTA_collective_events": ["all-role initial pipeline publication",
                                      "producer-tail and both consumers operand retirement through role-local barrier6",
                                      "all-role complete P publication before U"],
            "CTA_collective_event_count": 3,
            "independent_consumer_whole_K_loops": 2,
            "volatile_tail_tid_and_complete_U_epilogue_unchanged": True,
            "actual_thread_CFG_memory_order_or_compiler_register_proof": False,
            "inherited_task_count_comments_scope": "Exact source retains old BK32 comments; checked BK64 actual A tasks512/B tasks256 are reported in layout_expectation."}


def tail_check(geometry, protocol):
    report = geometry.tail_and_output_check(protocol)
    # Enumerate the actual unchanged U topology as well, independently of the
    # consumer coordinate enumeration inside the frozen P writer checker.
    reads = bytearray(7 * 64 * 32)
    coords = set()
    for tid in range(384):
        for linear in range(tid,64*32,384):
            row,col = divmod(linear,32)
            assert (row,col) not in coords
            coords.add((row,col))
            for product in range(7):
                word = protocol.tail_word(product,row,col,True)
                assert 0 <= 4*word < 57344 < 86016
                reads[word] += 1
    assert len(coords) == 2048 and all(count == 1 for count in reads)
    return {**report,"U_read_words_explicitly_enumerated":len(reads),
            "unique_U_owners_per_CTA":len(coords),
            "P_writers_and_U_reads_same_full_word_set":True,
            "bank0_contains_all_P_tail_words":True,
            "barrier_scope": "Three logical CTA collective events, four lexical call sites (two role-local retire calls)."}


def main():
    frozen = frozen_check()
    geometry = load_geometry()
    protocol = geometry.load_frozen_protocol()
    source = source_check(geometry)
    layout = geometry.literal_layout_check()
    stages = list(range(18)) + [31,32,33,63,64,65,511,512]
    rows,witnesses = [],{}
    for count in stages:
        row,current = protocol.exhaustive(count)
        rows.append(row)
        if count == 3: witnesses = current
    random_counts = (0,1,2,3,4,7,8,15,16,17,32,33,64,512)
    actions = sum(protocol.randomized(count,100000*count+seed) for count in random_counts for seed in range(16))
    tail = tail_check(geometry,protocol)
    resources = geometry.resources()
    assert resources["predicted_align128_SharedStorage_bytes"] == 172160
    assert frozen_check() == frozen
    paths = (BASE,CANDIDATE,".scratch/psplit_bk64_scope34_model.py",GEOMETRY_MODEL,
             ".scratch/cooperative_st_once_psplit_model.py",
             "third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp",
             "third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
             "third_party/cutlass/include/cute/layout.hpp",
             "third_party/cutlass/include/cute/layout_composed.hpp",
             "third_party/cutlass/include/cute/pointer_flagged.hpp",
             "third_party/cutlass/include/cute/pointer_swizzle.hpp",
             "third_party/cutlass/include/cute/swizzle.hpp",
             "third_party/cutlass/include/cute/arch/mma_sm90.hpp",
             "third_party/cutlass/include/cute/algorithm/gemm.hpp")
    print(json.dumps({"status":"PASS_CPU_SOURCE_GEOMETRY_AND_ABSTRACT_PROTOCOL_ONLY",
                      "formal_policy_id":None,"CUDA_or_host_compilation":False,
                      "GPU_Graph_SAN_FP64_bitwise_performance_evidence":None,
                      "evidence_bindings":{path:digest(path) for path in paths},
                      "old_seven_BK64_files_frozen_hashes_verified":frozen,
                      "old_main_invoked":False,"model_file_writes":"Rejected by Python audit hook",
                      "source_review":source,"layout_expectation":layout,
                      "synchronization":{"exhaustive":rows,
                          "total_states":sum(row["states"] for row in rows),
                          "total_transitions":sum(row["transitions"] for row in rows),
                          "randomized_stage_counts":random_counts,"randomized_traces":16*len(random_counts),
                          "randomized_actions":actions,"mutation_counterexamples":witnesses,
                          "1024_corresponds_to_stage_count":8,
                          "model_scope":"Three collective128-thread roles; not actual per-thread PCs, role CFG, named-barrier memory order or hardware resource progress."},
                      "tail_output_alias":tail,"resources_prediction_only":resources,
                      "actual_BK64_gates":{"CuTe":"PENDING","codegen_register_CFG":"PENDING",
                                           "initial_CTA_owned_pool":"PENDING","spills_or_C7512":"PENDING",
                                           "bitwise_FP64":"PENDING","Graph_SAN":"PENDING",
                                           "ordinary_Graph_complete1024_latency":"PENDING"},
                      "rev6_BK32_build_evidence_inherited_as_BK64_gate":False},indent=2))


if __name__ == "__main__":
    main()
