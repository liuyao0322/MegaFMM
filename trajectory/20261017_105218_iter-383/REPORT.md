# Iteration 383: native narrow tiles and exact two-block loops

Six native narrow/short-K fused candidates were added to the existing pool; the three exact-two-block candidates require original K512. All prior native/restored/scalar candidates remain available. BF16 operands remain BF16. There are no residual/correction products or full-AB replacements.

Full single-sweep evidence: 28 shape/dtype rows, 2316 candidate records, 84 fresh original-input FP64 checks. 5 rows have all five paired timing groups at least 1.03x cuBLASLt. This is not independent-repeat certification.

The original numerical threshold is report-only. Finite output and implementation checks remain required. Removing a threshold does not improve accuracy.

Completed prechecks: 1836 original native calls and 224 batched-leaf calls, 450 new narrow/short-K calls and 90 matching-native bitwise comparisons. See precheck_audit.json for exact observed scope. No sanitizer result is implied.

| M,N,K | dtype | SW us | cuBLASLt us | speedup | min paired speedup | eF | selected ID |
|---|---|---:|---:|---:|---:|---:|---:|
| 512,512,512 | bf16 | 6.685 | 5.561 | 0.8318 | 0.8307 | 0.0060492632 | 500003 |
| 512,512,512 | fp16 | 6.511 | 5.633 | 0.8651 | 0.8616 | 0.00075694057 | 500003 |
| 1024,1024,1024 | bf16 | 10.321 | 11.957 | 1.1585 | 1.1556 | 0.0060151965 | -2 |
| 1024,1024,1024 | fp16 | 10.408 | 9.582 | 0.9206 | 0.9193 | 0.00075462577 | -2 |
| 2048,2048,2048 | bf16 | 50.684 | 31.163 | 0.6149 | 0.6120 | 0.0060249492 | 16 |
| 2048,2048,2048 | fp16 | 51.915 | 32.353 | 0.6232 | 0.6179 | 0.00075358965 | 16 |
| 4096,4096,4096 | bf16 | 348.495 | 208.598 | 0.5986 | 0.5536 | 0.0060268975 | 100001 |
| 4096,4096,4096 | fp16 | 344.709 | 216.355 | 0.6276 | 0.5885 | 0.00075313254 | 52 |
| 8192,8192,8192 | bf16 | 2121.467 | 1672.746 | 0.7885 | 0.7701 | 0.0060233938 | 100001 |
| 8192,8192,8192 | fp16 | 2171.510 | 1727.923 | 0.7957 | 0.7867 | 0.00075341126 | 100001 |
| 16384,16384,16384 | bf16 | 17539.122 | 12701.755 | 0.7242 | 0.7207 | 0.006022762 | 100001 |
| 16384,16384,16384 | fp16 | 18654.642 | 13251.233 | 0.7103 | 0.7069 | 0.00075307867 | 100001 |
| 8192,4096,8192 | bf16 | 1165.383 | 824.657 | 0.7076 | 0.6794 | 0.0060249992 | 100001 |
| 8192,4096,8192 | fp16 | 1198.510 | 879.164 | 0.7335 | 0.7272 | 0.00075317067 | 100001 |
| 4096,8192,8192 | bf16 | 1151.712 | 833.235 | 0.7235 | 0.7177 | 0.0060256023 | 100001 |
| 4096,8192,8192 | fp16 | 1185.253 | 875.668 | 0.7388 | 0.7199 | 0.00075341087 | 100001 |
| 8192,8192,4096 | bf16 | 1231.044 | 857.390 | 0.6965 | 0.6833 | 0.0060241771 | 100002 |
| 8192,8192,4096 | fp16 | 1287.233 | 873.481 | 0.6786 | 0.6645 | 0.00075308887 | 100002 |
| 8192,1024,512 | bf16 | 35.583 | 25.307 | 0.7112 | 0.7072 | 0.00605665 | 53 |
| 8192,1024,512 | fp16 | 35.355 | 24.995 | 0.7070 | 0.7050 | 0.00075491108 | 53 |
| 1024,8192,512 | bf16 | 33.829 | 26.906 | 0.7954 | 0.7924 | 0.0060249368 | 51 |
| 1024,8192,512 | fp16 | 34.556 | 26.530 | 0.7677 | 0.7612 | 0.00075321501 | 52 |
| 1025,1537,769 | bf16 | 19.223 | 38.026 | 1.9782 | 1.9731 | 0.0060137307 | 15 |
| 1025,1537,769 | fp16 | 18.968 | 36.383 | 1.9181 | 1.9135 | 0.00075147179 | 15 |
| 514,770,258 | bf16 | 8.829 | 8.858 | 1.0033 | 1.0018 | 0.0060441812 | 6 |
| 514,770,258 | fp16 | 8.687 | 8.831 | 1.0166 | 1.0146 | 0.0007584174 | 6 |
| 1,1,1 | bf16 | 2.487 | 2.894 | 1.1638 | 1.1612 | 0 | -3 |
| 1,1,1 | fp16 | 2.468 | 2.867 | 1.1618 | 1.1451 | 0 | -3 |

Raw candidate measurements and confirmation medians are in results.csv.sw_tuning.csv. Raw full-call timing and original-input errors are in results.csv and output.txt. Source/support hashes are in manifest.json and support_manifest.json.
