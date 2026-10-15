# Seven-product numerical alternatives after user feedback on compensation cost

User feedback: high/low compensation is too expensive. This diagnostic changes no solution, benchmark contract, reference, gate or selected kernel. It is not a formal GPU kernel iteration.

Three fresh independent seeds,1024x1024 A/B uniform[-1,1] rounded to original BF16 or FP16. Reference uses FP64 multiplication of those same original inputs. Alternative transform operands rounded as described, with FP64 leaf dots and FP64 output recombination to isolate transform/operand quantization. Thus these numbers are NOT production GPU numerical acceptance or latency. Original benchmark input distributions and fresh-case contract were not replaced.

Median relative Frobenius error:

| Original input | Seven-product method | Diagnostic eF |
|---|---|---:|
|BF16|current Winograd / BF16 operands|0.00603645|
|BF16|classic Strassen / BF16 operands|0.00439335|
|BF16|Winograd / FP16 operands|0.000316776|
|FP16|current Winograd / FP16 operands|0.000753290|
|FP16|classic Strassen / FP16 operands|0.000549398|

Classic Strassen helps modestly but remains above the existing~1e-3 BF16/~1e-4 FP16 floors even with FP64 leaf dots and recombination. Merely increasing accumulator/output precision will not repair discarded transform bits.

BF16 original inputs with FP16 transformed operands are promising for the BF16 floor without adding products: median error~19x lower than BF16 Winograd here. Not evidence of GPU pass or speed. BF16 exponent range is much wider: blindly converting arbitrary BF16 operands to FP16 can overflow or lose tiny terms. Scaling/range handling needs proof and validation. WGMMA cannot simply mix BF16 and FP16 operands in one instruction; implementation must preserve BF16 P1/P2 or convert raw inputs, and handle the raw A22/B22 operands of P3/P4 consistently. Additional conversion/packing traffic must be included. This does not solve the FP16-input precision gap, since current transform operands already use FP16.

No evidence of a universally cheap seven-product solution meeting both original gates. Goal stays active. Next potential investigation is mixed-format BF16 representation with original FP64 gate and complete-call timing; FP16 needs a separate numerical representation or justified selective correction, not a claim that formula replacement alone suffices.

## FP16 selective correction follow-up

User explicitly rejects the BF16-only representation proposal because FP16 inputs still fail. FP16 acceptance is a required part of the solution; no gate changes or source changes were made.

`selective.py` enumerates all 256 subsets of the eight standard first-order residual cross terms of the current Winograd formula, using three independent seeds 1,2,3, uniform[-1,1] 1024x1024 FP16 inputs. Residuals and correction products are ideal FP64, favoring compensation; this is a diagnostic rather than production GPU validation. For each number of extra products, select the subset minimizing the worst relative Frobenius error across these three seeds.

| Extra products | Total leaf products | Best worst-case eF across three seeds |
|---:|---:|---:|
|0|7|7.54034e-4|
|2|9|5.00901e-4|
|4|11|3.85183e-4|
|6|13|2.15499e-4|
|7|14|1.52466e-4|
|8|15|1.11328e-7|

The historical original FP16 GPU gate is about 1.03753e-4. Even ideal 14-product standard selective compensation remains above it on these diagnostic inputs; all eight standard cross terms are required within this tested family. This does not prove that other algebraic formulas, encodings, or correction algorithms cannot succeed more cheaply. The 15-product diagnostic result is not a GPU pass or a speed prediction, since production residual quantization and FP32 accumulation differ.

A BF16-only representation change does not resolve the task. Plain TF32 has the same ten explicit mantissa bits as FP16 and does not by itself supply higher precision for FP16 transforms. Any new proposal must address FP16 transform rounding directly and pass the original production FP64 gates before being called an accuracy-qualified winner. Current seven-product speed results remain accuracy-failing experimental results; Auto remains Dense. Next numerical investigation needs a materially different representation or correction mechanism, not another subset of these eight standard cross terms.
