"""Exhaust signed INT8 and scale shifts for P36's exact cap-at-eight identity.

q is integral in[-128,127]. For shift>=8, |q*2^-shift|<=.5 and the only
half tie (-128,8) rounds to even zero. All capped products are exact binary FP32.
This is scalar arithmetic proof/control, not packed GPU or matrix acceptance.
"""
import hashlib
import json
import math
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
errors=0;checks=0
for q in range(-128,128):
    for shift in range(301):
        expected=round(math.ldexp(float(q),-shift))
        capped=round(math.ldexp(float(q),-min(shift,8)))
        errors+=expected!=capped;checks+=1
assert errors==0
out=dict(scope=__doc__,signed_int8_values=256,shifts_inclusive=[0,300],checks=checks,errors=errors,
         hardware_accepted=False,nonnegative_shift_reason='common=max of four integer exponents')
(h/'requantized_integer_arithmetic.json').write_text(json.dumps(out,indent=2)+'\n')
files=['requantized_integer_arithmetic.py','requantized_integer_arithmetic.json']
(h/'requantized_integer_arithmetic_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))
