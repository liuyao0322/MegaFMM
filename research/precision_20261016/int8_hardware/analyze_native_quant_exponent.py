"""Audit exhaustive native-value GPU exponent screen, not matrix acceptance."""
import hashlib
import json
import re
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
rows=[dict(re.findall(r'(\w+)=([^ ]+)',x)) for x in (h/'quant_exponent_check.log').read_text().splitlines() if x.startswith('QUANT_EXP ')]
assert len(rows)==2
for x,count,fast,fallback in zip(rows,[31744,32640],[31743,31488],[0,1151]):
    assert int(x['cases'])==count and int(x['fast'])==fast and int(x['fallback'])==fallback
    assert x['zeros']=='1' and x['errors']=='0' and x['first_error_code']=='65536'
out=dict(scope='actual H100 sm90a CUDA12.4 exhaustive nonnegative finite native input values; arithmetic only, no matrix/performance/synchronization acceptance',
    cases=64384,errors=0,rows=rows,
    input_domain='Quantizer high maxima after native FP16/BF16 conversion, initialized at +0; not arbitrary FP32 transformed values or low residual maxima',
    unchanged_fallback='peak bits below0x04800000, nonfinite/negative exceptional patterns; zero explicitly0',
    note='Matches original compiled ceil(log2(peak/127.f)) on every finite nonnegative native value. Actual matrix maximum selection/quantization and new GPU implementation still require independent gates.')
target=h/'native_quant_exponent_arithmetic.json';target.write_text(json.dumps(out,indent=2)+'\n')
files=[Path(__file__).resolve(),h/'quant_exponent_proposal.cuh',h/'check_native_quant_exponent.cu',h/'build_quant_exponent_check.sh',h/'quant_exponent_check_build.log',h/'quant_exponent_check.log',h/'bin/check_native_quant_exponent',target]
(h/'native_quant_exponent_arithmetic_manifest.json').write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))
