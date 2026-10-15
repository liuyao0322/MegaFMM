"""CPU evidence for unsigned-bit max of nonnegative binary32; not GPU acceptance."""
import hashlib
import json
import random
import struct
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
def value(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
rng=random.Random(41173)
special=[0,1,0x007fffff,0x00800000,0x3f7fffff,0x3f800000,0x3f800001,0x7f7fffff,0x7f800000]
errors=0;warps=32768
for i in range(warps):
    source=([0]*32 if i==0 else [special[(i+j)%len(special)] for j in range(32)] if i<32
            else [rng.randrange(0x7f800001) for _ in range(32)])
    actual=max(source);expected=bits(max(value(x) for x in source))
    errors+=actual!=expected
assert errors==0
negative=[-1.,0.];invalid=value(max(bits(x) for x in negative));correct=max(negative)
assert invalid!=correct
result=dict(scope='CPU bit-pattern reduction only, no GPU/matrix/performance/synchronization acceptance',
    proof='For binary32 with sign0 and no NaN, unsigned bits order exponent then fraction in the same order as value, including positive subnormals, +0 and +inf. Existing local maxima start at +0 and fmax absolute values, so the proposed full-warp inputs lie in this domain. This does not permit replacing arbitrary signed/NaN-valued float reductions.',
    seed=41173,generated_warps=warps,generated_values=warps*32,bitwise_errors=errors,
    signed_negative_control=dict(values=negative,invalid_unsigned_bit_max=invalid,correct_float_max=correct))
target=h/'nonnegative_max_arithmetic.json';target.write_text(json.dumps(result,indent=2)+'\n')
files=[Path(__file__).resolve(),target]
(h/'nonnegative_max_arithmetic_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(result,indent=2))
