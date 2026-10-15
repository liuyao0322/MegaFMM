# INT8 hardware experiments

Production `solution/` unchanged; these are standalone research candidates, not production iteration371. Full original-input FP64 gate remains fixed, with fresh deterministic inputs in launch/Graph modes. Every timing row includes preprocessing and tensor-map setup. Sanitizer/NCU runs are excluded from ordinary latency evidence.

## P1 — first complete INT8 candidate

- Signed INT8 `m64n32k32` primitive:18 cases (six patterns, three scales), exactly matches CPU int64 dots and power-of-two rescaling. Bound2064512 reached without overflow. First compile failed due a nondependent CuTe descriptor host instantiation; archived and fixed with a dependent template boundary.
- Seven native main products plus eight first-order INT8 products, per-K128 rescaling, all FP32 output. Dynamic per-column inside scaling, seven packed high planes and nine INT8 planes per side, no allocation/host sync in run. Workspace12066176bytes.
- Full original six patterns plus six extended patterns ×three seeds ×two dtypes:72/72 launch checks passed. Maximum BF16/FP16 eF=5.0954e-4/6.5673e-5.
- Complete-call paired medians BF16/FP16=157.935/159.656us; currentSW10.208/9.941us, dense comparator6.522/6.555us. **Reject performance**. Diagnostic preprocessing82.603/82.828us is included in, not additional to, the complete-call time.
- 168registers/thread,32byte stack frame,0spill stores/loads for the corrected kernel; producer up to149registers,0spill. No sanitizer acceptance yet.
- Baseline source snapshot and logs in`p1/`; identities in`p1_manifest.json`. NCU one first BF16 corrected kernel, clocks/caches uncontrolled; all profiler-run timing arms excluded. Initial shared/tmp lock failure preserved, resolved by private TMPDIR; no other user's lock changed.
- Next change: replace common-range scalar `ldexpf` with exact power-of-two multiplication, retaining general exponent fallback and an independent GPU equivalence check. This is an implementation cost change, not a new numerical representation.
