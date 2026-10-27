# Iteration 384: seven independent SW accumulators across product boundaries

Two native K512 candidates with seven independent FP32 accumulators and cross-product wait1 were added to the existing pool. The original producer, transform arithmetic and reconstruction order remain unchanged. All prior native/restored/scalar candidates remain available. BF16 operands remain BF16. There are no residual/correction products or full-AB replacements.

Full single-sweep evidence: 28 shape/dtype rows, 2328 candidate records, 84 fresh original-input FP64 checks. 5 rows have all five paired timing groups at least 1.03x cuBLASLt. This is not independent-repeat certification.

The original numerical threshold is report-only. Finite output and implementation checks remain required. Removing a threshold does not improve accuracy.

Completed prechecks: 1836 original native calls and 224 batched-leaf calls, 450 narrow/short-K calls and 90 matching-native bitwise comparisons, plus 140 new cross-product calls and 28 matching-native bitwise comparisons. See precheck_audit.json for exact observed scope. No sanitizer result is implied.

| M,N,K | dtype | SW us | cuBLASLt us | speedup | min paired speedup | eF | selected ID |
|---|---|---:|---:|---:|---:|---:|---:|
| 512,512,512 | bf16 | 6.561 | 5.561 | 0.8476 | 0.8385 | 0.0060492632 | 600001 |
| 512,512,512 | fp16 | 6.431 | 5.624 | 0.8746 | 0.8739 | 0.00075694057 | 600001 |
| 1024,1024,1024 | bf16 | 10.345 | 12.003 | 1.1603 | 1.1555 | 0.0060151965 | -2 |
| 1024,1024,1024 | fp16 | 10.327 | 9.585 | 0.9281 | 0.9218 | 0.00075462577 | -2 |
| 2048,2048,2048 | bf16 | 50.978 | 31.537 | 0.6186 | 0.6159 | 0.0060249492 | 16 |
| 2048,2048,2048 | fp16 | 51.954 | 32.102 | 0.6179 | 0.6034 | 0.00075358965 | 16 |
| 4096,4096,4096 | bf16 | 336.319 | 229.590 | 0.6827 | 0.5830 | 0.0060268975 | 52 |
| 4096,4096,4096 | fp16 | 362.218 | 218.868 | 0.6042 | 0.5645 | 0.00075313254 | 52 |
| 8192,8192,8192 | bf16 | 2128.428 | 1658.649 | 0.7793 | 0.7703 | 0.0060233938 | 100001 |
| 8192,8192,8192 | fp16 | 2187.425 | 1759.158 | 0.8042 | 0.7932 | 0.00075341126 | 100001 |
| 16384,16384,16384 | bf16 | 18069.181 | 12736.028 | 0.7048 | 0.7033 | 0.006022762 | 100001 |
| 16384,16384,16384 | fp16 | 18281.405 | 13285.728 | 0.7267 | 0.7260 | 0.00075307867 | 100001 |
| 8192,4096,8192 | bf16 | 1154.889 | 845.379 | 0.7320 | 0.7096 | 0.0060249992 | 100001 |
| 8192,4096,8192 | fp16 | 1201.202 | 877.998 | 0.7309 | 0.7097 | 0.00075317067 | 100001 |
| 4096,8192,8192 | bf16 | 1151.540 | 844.569 | 0.7334 | 0.7241 | 0.0060256023 | 100001 |
| 4096,8192,8192 | fp16 | 1183.301 | 872.868 | 0.7377 | 0.7271 | 0.00075341087 | 100001 |
| 8192,8192,4096 | bf16 | 1237.388 | 873.855 | 0.7062 | 0.6830 | 0.0060241771 | 100001 |
| 8192,8192,4096 | fp16 | 1291.806 | 893.266 | 0.6915 | 0.6681 | 0.00075308887 | 100002 |
| 8192,1024,512 | bf16 | 35.116 | 24.957 | 0.7107 | 0.7098 | 0.00605665 | 53 |
| 8192,1024,512 | fp16 | 35.818 | 25.533 | 0.7129 | 0.7119 | 0.00075491108 | 52 |
| 1024,8192,512 | bf16 | 34.180 | 26.933 | 0.7880 | 0.7859 | 0.0060249368 | 52 |
| 1024,8192,512 | fp16 | 34.532 | 26.795 | 0.7759 | 0.7710 | 0.00075321501 | 51 |
| 1025,1537,769 | bf16 | 19.191 | 37.957 | 1.9779 | 1.9720 | 0.0060137307 | 15 |
| 1025,1537,769 | fp16 | 18.932 | 36.386 | 1.9219 | 1.9108 | 0.00075147179 | 15 |
| 514,770,258 | bf16 | 8.814 | 8.847 | 1.0037 | 0.9631 | 0.0060441812 | 6 |
| 514,770,258 | fp16 | 8.700 | 8.835 | 1.0155 | 1.0112 | 0.0007584174 | 6 |
| 1,1,1 | bf16 | 2.452 | 2.888 | 1.1778 | 1.1702 | 0 | -3 |
| 1,1,1 | fp16 | 2.483 | 3.420 | 1.3772 | 1.3454 | 0 | -3 |

Raw candidate measurements and confirmation medians are in results.csv.sw_tuning.csv. Raw full-call timing and original-input errors are in results.csv and output.txt. Source/support hashes are in manifest.json and support_manifest.json.
