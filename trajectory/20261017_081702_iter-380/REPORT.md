# BF16 dtype constraint correction — iter380

The prior user constraint forbids BF16-to-FP16 intermediate storage and computation. Iter378/379 mixed-path accuracy and acceleration are withdrawn as accepted outcomes. Their raw measurements remain labeled historical rejected evidence.

All production solution files are byte-identical to the same-dtype4c3898e baseline. The empirical SW tuner and restored original1024 source remain. BF16 input uses BF16 intermediate storage and BF16 leaf products with FP32 transforms/accumulation/output. Original FP64 gates are unchanged. Strict tuning rejects an empty valid set; no full-AB dense substitute is used.

## Current1024 full-call diagnostic measurement

H100;two rotating input sets,20 warmups,5 groups x100 calls,3 fresh original-input FP64 checks. Both selected routes are original sw1024_pdl. These timing ratios do not establish numerical acceptance.

| dtype | SW us | cuBLASLt us | ratio | SW eF | gate | precision |
|---|---:|---:|---:|---:|---:|---|
| bf16 | 10.238 | 12.452 | 1.2163 | 0.0060151965 | 0.0010025568 | FAIL |
| fp16 | 10.096 | 9.533 | 0.9442 | 0.00075462577 | 0.00010375276 | FAIL |

## Verification

Full clean build and all612 native candidate configurations x3 fresh direct/Graph checks passed (1836 calls). BF16 seed2 direct and seed3 Graph use exact inputs scaled2^20, above FP16 exponent range;306 BF16 configurations therefore receive612 high-range calls. The independent analytical original-AB oracle and output guards remain active. These integer implementation checks do not solve ordinary random-input precision.

Related CTest sw_pipeline_implementation and sw_tuner_selection2/2 passed,25.94s. The independent candidate registry/finalist selection audit passed for both measured rows (182 candidates). Source/binary hashes remained unchanged across checks. Full28 performance and new sanitizers were not rerun for this correction; prior mixed-path certificates are not transferred. The all-size precision-valid cuBLAS acceleration target remains unmet.
