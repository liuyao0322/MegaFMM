# Iteration 382: measured seven-product batched leaves

Four native-dtype seven-product candidates were added to the existing pool. All prior native/restored/scalar candidates remain available. BF16 operands remain BF16. There are no residual/correction products or full-AB replacements.

Full single-sweep evidence: 28 shape/dtype rows, 2232 candidate records, 84 fresh original-input FP64 checks. 5 rows have all five paired timing groups at least 1.03x cuBLASLt. This is not independent-repeat certification.

The original numerical threshold is report-only. Finite output and implementation checks remain required. Removing a threshold does not improve accuracy.

Completed prechecks: 1836 original native calls and 224 new batched-leaf calls. See precheck_audit.json for exact observed scope. No sanitizer result is implied.

| M,N,K | dtype | SW us | cuBLASLt us | speedup | min paired speedup | eF | selected ID |
|---|---|---:|---:|---:|---:|---:|---:|
| 512,512,512 | bf16 | 7.859 | 5.584 | 0.7105 | 0.7091 | 0.0060492632 | 3 |
| 512,512,512 | fp16 | 7.672 | 5.639 | 0.7351 | 0.7318 | 0.00075694057 | 3 |
| 1024,1024,1024 | bf16 | 10.350 | 11.942 | 1.1538 | 1.1501 | 0.0060151965 | -2 |
| 1024,1024,1024 | fp16 | 10.367 | 9.611 | 0.9270 | 0.9241 | 0.00075462577 | -2 |
| 2048,2048,2048 | bf16 | 51.314 | 31.421 | 0.6123 | 0.6100 | 0.0060249492 | 16 |
| 2048,2048,2048 | fp16 | 52.007 | 32.684 | 0.6285 | 0.6276 | 0.00075358965 | 16 |
| 4096,4096,4096 | bf16 | 337.121 | 227.254 | 0.6741 | 0.6113 | 0.0060268975 | 52 |
| 4096,4096,4096 | fp16 | 346.905 | 215.647 | 0.6216 | 0.5841 | 0.00075313254 | 52 |
| 8192,8192,8192 | bf16 | 2137.484 | 1647.049 | 0.7706 | 0.7690 | 0.0060233938 | 100001 |
| 8192,8192,8192 | fp16 | 2173.048 | 1763.402 | 0.8115 | 0.7870 | 0.00075341126 | 100001 |
| 16384,16384,16384 | bf16 | 17510.964 | 12792.283 | 0.7305 | 0.7272 | 0.006022762 | 100001 |
| 16384,16384,16384 | fp16 | 18684.987 | 13330.818 | 0.7135 | 0.7063 | 0.00075307867 | 100001 |
| 8192,4096,8192 | bf16 | 1161.201 | 845.570 | 0.7282 | 0.7174 | 0.0060249992 | 100001 |
| 8192,4096,8192 | fp16 | 1195.569 | 875.781 | 0.7325 | 0.6855 | 0.00075317067 | 100001 |
| 4096,8192,8192 | bf16 | 1145.417 | 838.223 | 0.7318 | 0.7175 | 0.0060256023 | 100001 |
| 4096,8192,8192 | fp16 | 1181.204 | 874.553 | 0.7404 | 0.7378 | 0.00075341087 | 100001 |
| 8192,8192,4096 | bf16 | 1233.156 | 874.366 | 0.7090 | 0.6836 | 0.0060241771 | 100001 |
| 8192,8192,4096 | fp16 | 1302.546 | 873.860 | 0.6709 | 0.6614 | 0.00075308887 | 100001 |
| 8192,1024,512 | bf16 | 35.205 | 25.143 | 0.7142 | 0.7120 | 0.00605665 | 53 |
| 8192,1024,512 | fp16 | 35.648 | 25.142 | 0.7053 | 0.7029 | 0.00075491108 | 53 |
| 1024,8192,512 | bf16 | 34.213 | 27.158 | 0.7938 | 0.7914 | 0.0060249368 | 52 |
| 1024,8192,512 | fp16 | 34.741 | 26.777 | 0.7708 | 0.7605 | 0.00075321501 | 52 |
| 1025,1537,769 | bf16 | 19.148 | 37.979 | 1.9835 | 1.9696 | 0.0060137307 | 15 |
| 1025,1537,769 | fp16 | 18.985 | 36.494 | 1.9222 | 1.9129 | 0.00075147179 | 15 |
| 514,770,258 | bf16 | 8.827 | 8.849 | 1.0025 | 0.9991 | 0.0060441812 | 6 |
| 514,770,258 | fp16 | 8.729 | 8.890 | 1.0184 | 1.0167 | 0.0007584174 | 6 |
| 1,1,1 | bf16 | 2.476 | 2.897 | 1.1704 | 1.1591 | 0 | -3 |
| 1,1,1 | fp16 | 2.460 | 2.900 | 1.1784 | 1.1725 | 0 | -3 |

Raw candidate measurements and confirmation medians are in results.csv.sw_tuning.csv. Raw full-call timing and original-input errors are in results.csv and output.txt. Source/support hashes are in manifest.json and support_manifest.json.
