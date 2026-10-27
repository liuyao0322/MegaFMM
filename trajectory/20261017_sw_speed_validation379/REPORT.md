> WITHDRAWN AS ACCEPTED BF16 EVIDENCE: this version converts BF16 to FP16, violating a prior user constraint reaffirmed2026-10-17. Raw measurements are retained for audit; this is not the current production implementation.

# Current seven-product SW tuner evidence

The full 28-row acceleration objective remains unmet unless every row passes both precision and speed. All rows execute seven-product SW. Same-run independently tuned cuBLASLt is only the performance/error reference. BF16 FP16-intermediate rows are labeled explicitly; no original precision gate was relaxed.

Fresh precision passes: 15/28. Single-sweep precision-valid ratios above one: 3/28; these are not all independent stable wins. All28 goal achieved: False.

| M,N,K | dtype | SW us | cuBLAS us | speedup | eF | gate | precision |
|---|---|---:|---:|---:|---:|---:|---|
| 512,512,512 | bf16 | 8.360 | 5.571 | 0.6664 | 0.00031865 | 0.00100118 | PASS |
| 512,512,512 | fp16 | 7.715 | 5.652 | 0.7326 | 0.000756941 | 0.000101933 | FAIL |
| 1024,1024,1024 | bf16 | 13.916 | 11.965 | 0.8598 | 0.000316337 | 0.00100256 | PASS |
| 1024,1024,1024 | fp16 | 10.369 | 9.625 | 0.9282 | 0.000754626 | 0.000103753 | FAIL |
| 2048,2048,2048 | bf16 | 55.702 | 31.799 | 0.5709 | 0.000316915 | 0.00100557 | PASS |
| 2048,2048,2048 | fp16 | 52.271 | 32.084 | 0.6138 | 0.00075359 | 0.000107374 | FAIL |
| 4096,4096,4096 | bf16 | 351.220 | 215.018 | 0.6122 | 0.000316786 | 0.00101207 | PASS |
| 4096,4096,4096 | fp16 | 358.984 | 216.769 | 0.6038 | 0.000753133 | 0.00011463 | FAIL |
| 8192,8192,8192 | bf16 | 2643.856 | 1662.149 | 0.6287 | 0.000316902 | 0.00102565 | PASS |
| 8192,8192,8192 | fp16 | 2789.184 | 1704.191 | 0.6110 | 0.000753411 | 0.000129068 | FAIL |
| 16384,16384,16384 | bf16 | 20073.005 | 12730.432 | 0.6342 | 0.00031687 | 0.00105372 | PASS |
| 16384,16384,16384 | fp16 | 21302.798 | 13306.156 | 0.6246 | 0.000753079 | 0.000157995 | FAIL |
| 8192,4096,8192 | bf16 | 1354.110 | 846.519 | 0.6251 | 0.000317004 | 0.00102565 | PASS |
| 8192,4096,8192 | fp16 | 1416.590 | 894.882 | 0.6317 | 0.000753171 | 0.000129069 | FAIL |
| 4096,8192,8192 | bf16 | 1400.652 | 821.538 | 0.5865 | 0.000316907 | 0.00102565 | PASS |
| 4096,8192,8192 | fp16 | 1479.886 | 882.501 | 0.5963 | 0.000753411 | 0.000129069 | FAIL |
| 8192,8192,4096 | bf16 | 1334.345 | 869.470 | 0.6516 | 0.00031676 | 0.00101208 | PASS |
| 8192,8192,4096 | fp16 | 1453.688 | 909.247 | 0.6255 | 0.000753089 | 0.000114633 | FAIL |
| 8192,1024,512 | bf16 | 39.046 | 25.340 | 0.6490 | 0.000316716 | 0.00100118 | PASS |
| 8192,1024,512 | fp16 | 35.653 | 25.673 | 0.7201 | 0.000754911 | 0.00010193 | FAIL |
| 1024,8192,512 | bf16 | 37.902 | 27.558 | 0.7271 | 0.000317369 | 0.00100118 | PASS |
| 1024,8192,512 | fp16 | 34.745 | 26.932 | 0.7751 | 0.000753215 | 0.00010193 | FAIL |
| 1025,1537,769 | bf16 | 18.849 | 38.602 | 2.0480 | 0.000315565 | 0.00100109 | PASS |
| 1025,1537,769 | fp16 | 18.811 | 36.481 | 1.9394 | 0.000751472 | 0.0001028 | FAIL |
| 514,770,258 | bf16 | 8.860 | 8.834 | 0.9971 | 0.000319713 | 0.00100057 | PASS |
| 514,770,258 | fp16 | 8.744 | 8.849 | 1.0120 | 0.000758417 | 0.000101023 | FAIL |
| 1,1,1 | bf16 | 2.466 | 3.012 | 1.2215 | 0 | 0.001 | PASS |
| 1,1,1 | fp16 | 2.457 | 2.985 | 1.2148 | 0 | 0.0001 | PASS |

Precision PASS requires each fresh trial to satisfy its own gate; the table reports separate maxima. All five group means, three fresh inputs,20 warmups,100 trials and two rotating bindings are retained.

Related CTest2/2 passed. The full native candidate sweep already passed before the formal benchmark. Sanitizer coverage is actual native architecture/shape/dtype cases, not every packing/raster variant. Counts and full logs are retained in audit.json. Original restored1024 known precision failures are not claimed fixed.

| Shape | mode | SW us | cuBLAS us | median paired speedup | worst paired speedup | all15 groups >=1.03 |
|---|---|---:|---:|---:|---:|---|
| 1025,1537,769 | launch | 19.098 | 38.131 | 2.0034 | 1.9843 | True |
| 514,770,258 | launch | 8.489 | 8.653 | 1.0179 | 0.8782 | False |
| 1025,1537,769 | graph | 19.264 | 36.981 | 1.9196 | 1.9074 | True |
| 514,770,258 | graph | 8.205 | 8.120 | 0.9919 | 0.9869 | False |

FP16 intermediate candidates have reduced exponent range. Tuning certifies supplied values, not arbitrary rebound data. Changed numerical range/distribution needs retuning. High-range BF16 rejection and stale-winner invalidation are tested. run recomputes transforms and products for every input, allocates no device memory and performs no host synchronization.

The search uses compiled resource legality and actual complete-call measurements. It does not prove a global optimum or cuBLAS superiority. Full empirical search can take many minutes at16K; positive shortlist budgets can miss valid or fast candidates. Only H100 has been measured here.
