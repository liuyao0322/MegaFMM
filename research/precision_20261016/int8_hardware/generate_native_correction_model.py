"""M12 paired numerical model from frozen M11; neither is WGMMA simulation."""
from pathlib import Path
h=Path(__file__).resolve().parent
s=(h/'bf16_correction_model.py').read_text()
start=s.index('"""');end=s.index('"""',start+3)+3
s='''"""M12: seven native main products, eight native scaled correction dots.

Original K128 INT8 and unscaled BF16 controls share fresh inputs with a native
high/native scaled residual arm, p=11 FP16 or8 BF16. FP32 bmm/add with TF32disabled
is numerical screening, not exact Hopper WGMMA behavior or performance evidence.
"""'''+s[end:]
s=s.replace("run as wide_control)","run as wide_control)\nfrom bf16_correction_model import run as m11_control")
s=s.replace("'bf16_low_scale16'","'native_low_scaled'")
s=s.replace("for arm,power in [('bf16_low_unscaled',0),('native_low_scaled',16)]:", "for arm,typ,power in [('bf16_low_unscaled',torch.bfloat16,0),('native_low_scaled',dt,bits)]:")
s=s.replace('def high(x):return x.to(torch.bfloat16).float()', 'def high(x):return x.to(typ).float()')
s=s.replace('def low(x):return (x*(2.**power)).to(torch.bfloat16).float()', 'def low(x):return (x*(2.**power)).to(typ).float()')
s=s.replace("if arm=='native_low_scaled':d=d*(2.**-16)","if arm=='native_low_scaled':d=d*(2.**-bits)")
s=s.replace('for seed in (6301,6302,6303):','for seed in (7301,7302,7303):')
s=s.replace("if pattern=='uniform' and seed==6301:","if pattern=='uniform' and seed==7301:")
s=s.replace("old=wide_control(dt,pattern,seed)[0]\n                    assert fresh[0]['ef']==old['ef'] and fresh[0]['max_absolute_integer']==old['max_absolute_integer']\n                    print('CONTROL '+json.dumps(dict(dtype=str(dt),seed=seed,baseline_matches_m9=True)),flush=True)",
 "old=m11_control(dt,pattern,seed)\n                    assert fresh[0]['ef']==old[0]['ef'] and fresh[0]['max_absolute_integer']==old[0]['max_absolute_integer'] and fresh[1]['ef']==old[1]['ef']\n                    print('CONTROL '+json.dumps(dict(dtype=str(dt),seed=seed,int8_and_bf16_match_m11=True)),flush=True)")
needle='def main():'
scalar='''def scalar_range(dt):
    n,p=(31744,11) if dt==torch.float16 else (32640,8)
    values=torch.arange(n,device='cuda',dtype=torch.int16).view(dt).float()
    next_values=(torch.arange(n,device='cuda',dtype=torch.int16)+1).view(dt).float()
    half_gap=(next_values-values)*.5
    half_gap[-1]=2.**(15-11 if dt==torch.float16 else 127-8)
    mid=values+half_gap
    lo=torch.nextafter(mid,torch.full_like(mid,-float('inf')))
    hi=torch.nextafter(mid,torch.full_like(mid,float('inf')))
    x=torch.cat([values,mid,lo,hi,-values,-mid,-lo,-hi])
    high=x.to(dt).float();mask=torch.isfinite(x)&torch.isfinite(high)
    scaled=((x[mask]-high[mask])*(2.**p)).to(dt).float()
    nf=int((~torch.isfinite(scaled)).sum().item())
    assert nf==0
    return dict(dtype=str(dt),native_positive_finite_bins=n,power=p,
        sampled_transforms=x.numel(),finite_native_high=int(mask.sum().item()),
        excluded_nonfinite_native_high=int((~mask).sum().item()),scaled_low_nonfinite=nf,
        scope='all nonnegative finite native bins and signed midpoint-neighbor transforms; not exhaustive FP32 interior')

'''
s=s.replace(needle,scalar+needle)
s=s.replace("assert torch.isfinite(x) and torch.isfinite(h) and not torch.isfinite((lo*(2.**16)).to(torch.bfloat16))\n    print('RANGE_NEGATIVE_CONTROL '+json.dumps(dict(finite_transform=True,finite_native_high=True,scaled_low_bf16_nonfinite=True)),flush=True)",
 "assert torch.isfinite(x) and torch.isfinite(h) and not torch.isfinite((lo*(2.**16)).to(torch.bfloat16)) and torch.isfinite((lo*(2.**8)).to(torch.bfloat16))\n    print('RANGE_CONTROL '+json.dumps(dict(finite_transform=True,finite_native_high=True,fixed16_overflow=True,native8_finite=True)),flush=True)\n    for dt in (torch.float16,torch.bfloat16):print('SCALAR_RANGE '+json.dumps(scalar_range(dt)),flush=True)")
(h/'native_correction_model.py').write_text(s)
print('generated M12 three paired arms; native-scaled correction p11/8, independent scalar range controls')
