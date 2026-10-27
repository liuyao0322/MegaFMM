# Native seven-product tuning policy — iteration381

Full14 shapes × BF16/FP16, H10080GB, CUDA12.4. All28 finite; every default
SW candidate ranked regardless of the historical precision threshold. No kernel
arithmetic, dtype, compensation or registry change.2128candidate records and
84fresh original-input FP64 checks audited. Selected lowest confirmed median
on every row; restored1024source digest unchanged.

Five rows won all five groups by at least3% in this single sweep. Independent
repeat certification is pending; most sizes still lose to cuBLASLt.

| M,N,K | dtype | SW us | cuBLASLt us | speedup | minimum group speedup | eF |
|---|---|---:|---:|---:|---:|---:|
| 512,512,512 | bf16 | 7.833 | 5.634 | 0.7193 | 0.7162 | 0.0060492632 |
| 512,512,512 | fp16 | 7.700 | 5.716 | 0.7423 | 0.7388 | 0.00075694057 |
| 1024,1024,1024 | bf16 | 10.327 | 12.040 | 1.1658 | 1.1647 | 0.0060151965 |
| 1024,1024,1024 | fp16 | 10.355 | 9.558 | 0.9230 | 0.9224 | 0.00075462577 |
| 2048,2048,2048 | bf16 | 51.591 | 31.767 | 0.6158 | 0.6151 | 0.0060249492 |
| 2048,2048,2048 | fp16 | 52.183 | 32.997 | 0.6323 | 0.6259 | 0.00075358965 |
| 4096,4096,4096 | bf16 | 331.129 | 223.211 | 0.6741 | 0.6223 | 0.0060268975 |
| 4096,4096,4096 | fp16 | 358.685 | 219.006 | 0.6106 | 0.5907 | 0.00075313254 |
| 8192,8192,8192 | bf16 | 2733.302 | 1626.082 | 0.5949 | 0.5887 | 0.0060233938 |
| 8192,8192,8192 | fp16 | 2797.054 | 1702.767 | 0.6088 | 0.6023 | 0.00075341126 |
| 16384,16384,16384 | bf16 | 20572.612 | 12735.500 | 0.6191 | 0.6180 | 0.006022762 |
| 16384,16384,16384 | fp16 | 21327.034 | 13318.500 | 0.6245 | 0.6217 | 0.00075307867 |
| 8192,4096,8192 | bf16 | 1367.579 | 858.006 | 0.6274 | 0.5887 | 0.0060249992 |
| 8192,4096,8192 | fp16 | 1423.039 | 876.593 | 0.6160 | 0.5934 | 0.00075317067 |
| 4096,8192,8192 | bf16 | 1424.136 | 792.546 | 0.5565 | 0.5461 | 0.0060256023 |
| 4096,8192,8192 | fp16 | 1444.476 | 871.870 | 0.6036 | 0.5632 | 0.00075341087 |
| 8192,8192,4096 | bf16 | 1373.913 | 829.547 | 0.6038 | 0.5702 | 0.0060241771 |
| 8192,8192,4096 | fp16 | 1449.254 | 907.660 | 0.6263 | 0.5911 | 0.00075308887 |
| 8192,1024,512 | bf16 | 35.258 | 25.504 | 0.7233 | 0.7233 | 0.00605665 |
| 8192,1024,512 | fp16 | 35.520 | 25.515 | 0.7183 | 0.7129 | 0.00075491108 |
| 1024,8192,512 | bf16 | 33.892 | 27.560 | 0.8132 | 0.8093 | 0.0060249368 |
| 1024,8192,512 | fp16 | 34.089 | 26.478 | 0.7767 | 0.7720 | 0.00075321501 |
| 1025,1537,769 | bf16 | 18.867 | 38.599 | 2.0458 | 2.0351 | 0.0060137307 |
| 1025,1537,769 | fp16 | 18.841 | 36.505 | 1.9375 | 1.9324 | 0.00075147179 |
| 514,770,258 | bf16 | 8.832 | 8.833 | 1.0002 | 0.9969 | 0.0060441812 |
| 514,770,258 | fp16 | 8.722 | 8.843 | 1.0138 | 1.0110 | 0.0007584174 |
| 1,1,1 | bf16 | 2.464 | 2.913 | 1.1821 | 1.1710 | 0 |
| 1,1,1 | fp16 | 2.450 | 2.924 | 1.1937 | 1.1719 | 0 |

Historical gate passes only the2scalar rows. The latest objective removes this
threshold from eligibility; these errors are reported, not claimed improved.
See policy_audit.json for IDs/configs and raw results/output for samples.

Post-benchmark sw_tuner_selection CTest PASS (5.28s), covering strict and default
report-only policies, stale-winner invalidation, changed inputs and Graph tuning
for both dtypes. Full candidate precheck612configurations×3fresh calls had already
passed during the build/benchmark. CPU exact_winograd_dag and confirmation_policy
also passed; no full sanitizer or independent-repeat certification claimed.
