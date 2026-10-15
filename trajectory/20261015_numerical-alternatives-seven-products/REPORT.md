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
