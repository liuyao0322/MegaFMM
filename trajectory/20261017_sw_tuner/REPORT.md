# SW empirical tuner validation

The complete-call measurements below are one 28-row launch sweep, not independent repeated speedup certification. Every chosen execution route is seven-product SW. Diagnostic selection does not waive numerical gates.

Original fresh-input precision: 2/28 passed. All original thresholds and 3 fresh trials remain enabled.

Candidate audit: all eligible registry entries were checked; the selected route has the lowest confirmation median among the measured finalists. Per-candidate errors, gates and 5x20/5x100 samples are retained in results.csv.sw_tuning.csv. Source and binary hashes match manifest.json; restored 1024 source is unchanged.

| M,N,K | dtype | candidates | SW us | cuBLAS us | speedup | fresh precision | max eF | max gate | tune s | selected |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 512,512,512 | bf16 | 90 | 7.774 | 5.561 | 0.7153 | FAIL | 0.00604926 | 0.00100118 | 0.373 | sw_native;leaf_terms=7;tile=64x32;bk=128;stages=8;padded=0;pack_cap=512;raster=0 |
| 512,512,512 | fp16 | 90 | 7.690 | 5.636 | 0.7328 | FAIL | 0.000756941 | 0.000101933 | 0.171 | sw_native;leaf_terms=7;tile=64x32;bk=128;stages=8;padded=0;pack_cap=128;raster=0 |
| 1024,1024,1024 | bf16 | 91 | 10.364 | 11.917 | 1.1499 | FAIL | 0.0060152 | 0.00100256 | 0.300 | sw1024_pdl;leaf_terms=7;source=iter313_79dad0b |
| 1024,1024,1024 | fp16 | 91 | 10.437 | 9.579 | 0.9178 | FAIL | 0.000754626 | 0.000103753 | 0.304 | sw1024_pdl;leaf_terms=7;source=iter313_79dad0b |
| 2048,2048,2048 | bf16 | 90 | 51.526 | 31.782 | 0.6168 | FAIL | 0.00602495 | 0.00100557 | 1.345 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=2 |
| 2048,2048,2048 | fp16 | 90 | 51.841 | 32.034 | 0.6179 | FAIL | 0.00075359 | 0.000107374 | 1.368 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=2 |
| 4096,4096,4096 | bf16 | 90 | 336.884 | 233.966 | 0.6945 | FAIL | 0.0060269 | 0.00101207 | 9.084 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=2 |
| 4096,4096,4096 | fp16 | 90 | 356.089 | 216.867 | 0.6090 | FAIL | 0.000753133 | 0.00011463 | 9.049 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=2 |
| 8192,8192,8192 | bf16 | 90 | 2709.112 | 1621.871 | 0.5987 | FAIL | 0.00602339 | 0.00102565 | 68.521 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 8192,8192,8192 | fp16 | 90 | 2802.000 | 1693.794 | 0.6045 | FAIL | 0.000753411 | 0.000129068 | 69.813 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 16384,16384,16384 | bf16 | 90 | 20650.110 | 12847.953 | 0.6222 | FAIL | 0.00602276 | 0.00105372 | 707.780 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 16384,16384,16384 | fp16 | 90 | 21337.334 | 13298.000 | 0.6232 | FAIL | 0.000753079 | 0.000157995 | 713.193 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=2 |
| 8192,4096,8192 | bf16 | 90 | 1354.012 | 835.324 | 0.6169 | FAIL | 0.006025 | 0.00102565 | 34.196 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=2 |
| 8192,4096,8192 | fp16 | 90 | 1428.805 | 876.229 | 0.6133 | FAIL | 0.000753171 | 0.000129069 | 34.445 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=2 |
| 4096,8192,8192 | bf16 | 90 | 1417.498 | 832.018 | 0.5870 | FAIL | 0.0060256 | 0.00102565 | 34.250 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 4096,8192,8192 | fp16 | 90 | 1454.244 | 828.171 | 0.5695 | FAIL | 0.000753411 | 0.000129069 | 34.864 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 8192,8192,4096 | bf16 | 90 | 1376.317 | 824.873 | 0.5993 | FAIL | 0.00602418 | 0.00101208 | 33.370 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 8192,8192,4096 | fp16 | 90 | 1443.631 | 916.572 | 0.6349 | FAIL | 0.000753089 | 0.000114633 | 34.147 | sw_native;leaf_terms=7;tile=64x128;bk=64;stages=4;padded=0;pack_cap=512;raster=4 |
| 8192,1024,512 | bf16 | 90 | 35.241 | 25.272 | 0.7171 | FAIL | 0.00605665 | 0.00100118 | 0.790 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=4 |
| 8192,1024,512 | fp16 | 90 | 35.284 | 25.617 | 0.7260 | FAIL | 0.000754911 | 0.00010193 | 0.799 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=4 |
| 1024,8192,512 | bf16 | 90 | 33.908 | 27.275 | 0.8044 | FAIL | 0.00602494 | 0.00100118 | 0.796 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=0 |
| 1024,8192,512 | fp16 | 90 | 34.659 | 26.901 | 0.7762 | FAIL | 0.000753215 | 0.00010193 | 0.799 | sw_native;leaf_terms=7;tile=64x128;bk=256;stages=2;padded=0;pack_cap=512;raster=0 |
| 1025,1537,769 | bf16 | 36 | 18.833 | 38.634 | 2.0513 | FAIL | 0.00601373 | 0.00100109 | 0.222 | sw_native;leaf_terms=7;tile=64x64;bk=128;stages=4;padded=1;pack_cap=512;raster=0 |
| 1025,1537,769 | fp16 | 36 | 18.853 | 36.454 | 1.9336 | FAIL | 0.000751472 | 0.0001028 | 0.310 | sw_native;leaf_terms=7;tile=64x64;bk=128;stages=4;padded=1;pack_cap=512;raster=0 |
| 514,770,258 | bf16 | 36 | 8.676 | 8.834 | 1.0182 | FAIL | 0.00604418 | 0.00100057 | 0.097 | sw_native;leaf_terms=7;tile=64x16;bk=128;stages=8;padded=1;pack_cap=512;raster=0 |
| 514,770,258 | fp16 | 36 | 8.714 | 8.870 | 1.0179 | FAIL | 0.000758417 | 0.000101023 | 0.096 | sw_native;leaf_terms=7;tile=64x16;bk=128;stages=8;padded=1;pack_cap=512;raster=0 |
| 1,1,1 | bf16 | 1 | 2.496 | 3.475 | 1.3921 | PASS | 0 | 0.001 | 0.010 | sw_scalar;leaf_terms=7 |
| 1,1,1 | fp16 | 1 | 2.463 | 3.412 | 1.3855 | PASS | 0 | 0.0001 | 0.009 | sw_scalar;leaf_terms=7 |

The displayed fresh eF and gate are separate maxima across the three fresh trials. PASS requires every trial to satisfy its own original gate; comparing these two maxima alone does not establish acceptance.

Full CTest: 9/10 passed, including both new tuner tests. The sole failure is the known restored sw_1024_correctness precision failure: 24/36 original precision checks fail in each dtype. Native candidate tests cover 612 configurations with three fresh bindings each (1,836 calls including Graph). Memcheck, racecheck and synccheck each cover the 68 native architecture/shape/dtype cases with three fresh bindings (204 calls/tool); all report zero errors and racecheck reports zero warnings. This sanitizer scope does not claim all 2,128 measured packing/raster variants were instrumented. Raw evidence is in ctest.xml, tests.log and the three sanitizer logs.

Heuristic screening diagnostic: the selected configuration lies within the first 8 model-ranked entries on 11/28 rows. Using only those first 8 entries would have a maximum 17.67% gap against the best all-candidate short-ranking median in this dataset. This is an offline estimate, not a separate budget-8 tuning run. Default budget=0 retains full empirical search.

Current search retains one compute warpgroup and per-K wait0. It chooses within this implemented space; it does not establish a hardware-global optimum. No cross-process tuning cache or learned cost model is claimed.

Supplemental public API checks passed for BF16 and FP16: shortlist budget=1, workspace filtering, unsupported policy/layout rejection, fresh aligned rebinding, independent workspaces/streams, fresh Graph replay, rejection of tuning during capture, and rejection of nonfinite results even in diagnostic mode. The library, benchmark and external test source still match edges_before.sha256.

Eight supplemental Graph rows completed with the unchanged 2-input/3-fresh/5x100 contract; all retain their original precision failures. The serialized validation driver checked that each selected configuration has the minimum confirmed median. Strict CLI testing rejects all 90 candidates on 512^3 BF16, emits the rejection sidecar and performs no performance ranking. Strict scalar testing passes both dtypes.

| Graph M,N,K | dtype | speedup | fresh precision |
|---|---|---:|---|
| 1024,1024,1024 | bf16 | 1.0991 | FAIL |
| 1024,1024,1024 | fp16 | 0.9191 | FAIL |
| 514,770,258 | bf16 | 1.0056 | FAIL |
| 514,770,258 | fp16 | 1.0116 | FAIL |
| 1025,1537,769 | bf16 | 1.9935 | FAIL |
| 1025,1537,769 | fp16 | 1.9073 | FAIL |
| 1024,8192,512 | bf16 | 0.7693 | FAIL |
| 1024,8192,512 | fp16 | 0.7554 | FAIL |
