# SW tuner completion evidence

Objective: implement seven-product SW configuration tuning by dimensions and actual hardware measurements.

| Requirement | Authoritative evidence | Current status |
|---|---|---|
| Explicit SW-only tuning entry; no full-AB dense candidate | include/megafmm/gemm.h, solution/plan.cpp, solution/sw_tuning.inc | Implemented; native and restored seven-product launch routes audited |
| Use all M/N/K, dtype, workspace and actual GPU resources | sw_configs(), sw_config_workspace(), setup_sw_config(); candidate resource CSV | Implemented; actual CUDA function attributes/occupancy queried |
| Preserve restored 1024 specialization | manifest.json; original SHA256 ebc37e9dd171dd5855cdaed91fbb030d250c052a4c459746dd33b82cc634264b | Verified byte-identical |
| Correct native candidates across aligned/padded/rectangular/longer-K inputs | sw_tuner_candidates CTest, ctest.xml | Passed, both dtypes, all eligible configurations, 3 fresh exact-reference inputs, ordinary/Graph |
| Strict original-input FP64 gates; no valid candidate must throw | sw_tuner_selection CTest and solution/tuning_check.cu | Passed, both dtypes/modes; failed retuning clears prior choice |
| Diagnostic mode retains failed precision; no nonfinite acceptance | Random-input diagnostic tests; api_edges.log | Passed both dtypes, including explicit NaN rejection with nonfinite_rejected telemetry |
| Complete-call empirical ranking and minimum confirmed-median selection | Candidate 5x20/5x100 raw samples, selected_id in benchmark CSV; audit.json | Passed full 28-row audit, 2,128 candidate records and 106 confirmed candidates |
| Ordinary/Graph tuning and fresh replay | CTest selection/candidates; graph.csv and graph.csv.sw_tuning.csv | Functional tests passed; 8 supplemental Graph rows completed, winners checked against confirmation medians |
| Memory/race/sync safety | memcheck.log, racecheck.log, synccheck.log | Passed on native architecture subset, 204 calls/tool, no errors, race no warnings |
| Accurate reporting of pre-existing precision failure | tests.log, ctest.xml | Full CTest 9/10; restored sw_1024_correctness remains known 24/36 failures per dtype |
| Full target matrix, original timing/precision contracts | results.csv/benchmark.log; scripts/validate_sw_tuner.sh | Completed and audited exact 28/28, 2 rotating inputs, 3 fresh checks, 5x100 samples. Precision 2/28 passes, 26/28 diagnostic failures |
| Frozen source/binary identity throughout measurement | manifest.json; final report_sw_tuner.py audit; edges_identity.log | All40 measured source/binary hashes unchanged; supplemental library/binary/test-source hashes unchanged |
| Public policy/workspace/binding/reentrancy edge behavior | test_api_edges.cu; api_edges.log; strict_reject.log and sidecar; strict_scalar.csv | Passed both dtypes: budget/workspace, invalid policies/layouts, rebinding, dual streams, capture rules, nonfinite rejection. Strict CLI rejects 90/90 on 512 BF16 and accepts scalar both dtypes |
| Usable documented API and candidate telemetry | docs/sw_tuner.md; results.csv.sw_tuning.csv; sw_tuning_results(); REPORT.md | Implemented/documented; final measurements, resources, per-set errors, timing samples and supplemental evidence archived |

Completion here concerns the tuner. It does not certify arbitrary-input SW accuracy,
all-shape cuBLAS speedups, a hardware-global optimum, a learned cost model, or a
cross-process configuration cache. Those are not claimed by the implementation.
