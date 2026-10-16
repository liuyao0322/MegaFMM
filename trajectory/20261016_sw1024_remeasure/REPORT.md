# Restored specialized 1024 SW performance remeasurement — 2026-10-16

Measured the restored seven-product SW, not direct dense GEMM. All 12 result rows identify `sw1024_pdl` and 4 MiB workspace. Source and benchmark binary SHA256 hashes match before/after. No kernel modifications or optimization iterations.

H100; BF16/FP16 input, FP32 accumulation/output; contiguous 1024^3, alpha=1, beta=0. Ordinary and Graph each 3 independent benchmark processes, each freshly tunes cuBLASLt. 20 warmups, 5 groups x100 complete calls, 2 rotating input sets and 3 fresh original-input FP64 checks per dtype/process. Pair order uses the existing randomized harness. No clock locking, no confidence intervals.

The aggregate table takes the median of all 15 group means per arm, then Dense/SW. Both modes time all SW input-dependent GPU transformations and multiplication/recombination. Ordinary calls include per-call tensor-map encoding; Graph encoding occurs during capture and is not repeated on replay. Graph times are kept separate.

| mode | dtype | cuBLASLt us | SW us | Dense/SW | min/max paired group ratio |
|---|---|---:|---:|---:|---:|
| launch | bf16 | 12.17340 | 10.14880 | 1.199492 | 1.185574 / 1.207822 |
| launch | fp16 | 9.58720 | 10.06050 | 0.952955 | 0.946899 / 0.961905 |
| graph | bf16 | 11.23680 | 10.02780 | 1.120565 | 1.111262 / 1.126812 |
| graph | fp16 | 9.29440 | 10.00800 | 0.928697 | 0.925432 / 0.939006 |

## Every independent run

| mode | repeat | dtype | cuBLASLt us | SW us | Dense/SW |
|---|---:|---|---:|---:|---:|
| launch | 0 | bf16 | 12.21472 | 10.14336 | 1.204209 |
| launch | 1 | bf16 | 12.14528 | 10.20672 | 1.189930 |
| launch | 2 | bf16 | 12.16096 | 10.14176 | 1.199098 |
| launch | 0 | fp16 | 9.64224 | 10.05792 | 0.958671 |
| launch | 1 | fp16 | 9.54112 | 10.04096 | 0.950220 |
| launch | 2 | fp16 | 9.58720 | 10.06176 | 0.952835 |
| graph | 0 | bf16 | 11.21024 | 10.03840 | 1.116736 |
| graph | 1 | bf16 | 11.26496 | 10.03232 | 1.122867 |
| graph | 2 | bf16 | 11.26144 | 10.01600 | 1.124345 |
| graph | 0 | fp16 | 9.39456 | 10.00896 | 0.938615 |
| graph | 1 | fp16 | 9.29440 | 10.01376 | 0.928163 |
| graph | 2 | fp16 | 9.23680 | 9.96416 | 0.927002 |

## Accuracy and limits

All 36 fresh dtype checks fail the unchanged original-input FP64 gate, no nonfinite. Maximum eF in this benchmark input set is BF16 0.00601519652935 (0.60152%), FP16 0.000754625769821 (0.0754626%). Each process exits 2 for this known precision failure, not a runtime failure. These are performance measurements of an approximate implementation, not numerical acceptance. The wider six-pattern restoration test has larger worst-case errors; see ../20261016_restore_sw1024/validation.json.

All samples, per-run configurations, wall timings, raw logs and source/binary hashes are retained in this directory.
