"""Derive M10's paired model from frozen M9 without editing its evidence."""
from pathlib import Path
h=Path(__file__).resolve().parent
s=(h/'wide_quant_model.py').read_text()
a=s.index('"""');b=s.index('"""',a+3)+3
s='''"""M10: baselineK128, exactwideK512 and K128-to-K512 double-RN INT8.

Keep original per-column inside, seven modeled native leaves and eight integer
correction dots. Compare two different quantization algorithms on shared fresh
original rounded inputs. This model is not WGMMA/synchronization/speed evidence.
"""'''+s[b:]
s=s.replace('from paired_column_model import run as reference_control','from wide_quant_model import run as reference_control')
insert='''

def requant_pair(high, low, mantissa, fixed_low, side_a):
    # Existing K128 quantizers first; common exponent is max of stored exponents.
    # Zero-slab scale1 is retained, as a hardware metadata reduction would do.
    axis=-1 if side_a else -2
    pieces=[quant_pair(high.narrow(axis,k*128,128),low.narrow(axis,k*128,128),mantissa,fixed_low) for k in range(4)]
    common=[torch.stack([p[2+i] for p in pieces]).amax(0) for i in range(2)]
    merged=[]
    for i in range(2):
        out=[]
        for p in pieces:
            ratio=p[2+i]/common[i]
            assert bool(((ratio>0)&(ratio<=1)).all().item())
            q=torch.round(p[i]*ratio)
            assert int((q.abs()>127).sum().item())==0
            out.append(q)
        merged.append(torch.cat(out,axis))
    return merged[0],merged[1],common[0],common[1],sum(p[4] for p in pieces)
'''
pos=s.index('\ndef run(');s=s[:pos]+insert+s[pos:]
s=s.replace('for width in (128, 256, 512):','for arm,width in (("baseline_k128",128),("exactwide_k512",512),("requantized_k512",512)):')
old='''            sq = [quant_pair(tile_a(h, offset, width), tile_a(x-h, offset, width), bits, True)
                  for x, h in zip(ss, sh)]
            tq = [quant_pair(tile_b(h, offset, width), tile_b(x-h, offset, width), bits, True)
                  for x, h in zip(tt, th)]'''
new='''            def quant(x,h,side_a,fixed):
                tile=tile_a if side_a else tile_b
                high,low=tile(h,offset,width),tile(x-h,offset,width)
                return (requant_pair(high,low,bits,fixed,side_a) if arm=="requantized_k512"
                        else quant_pair(high,low,bits,fixed))
            sq=[quant(x,h,True,True) for x,h in zip(ss,sh)]
            tq=[quant(x,h,False,True) for x,h in zip(tt,th)]'''
assert s.count(old)==1;s=s.replace(old,new)
old='''            ar = tile_a(ah[3], offset, width)
            br = tile_b(bh[3], offset, width)
            ar = quant_pair(ar, torch.zeros_like(ar), bits, False)
            br = quant_pair(br, torch.zeros_like(br), bits, False)'''
new='''            ar=quant(ah[3],ah[3],True,False)
            br=quant(bh[3],bh[3],False,False)'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('quant_k=width, block_a=', 'arm=arm, quant_k=width, block_a=')
s=s.replace('(4801, 4802, 4803)', '(5201, 5202, 5203)').replace("seed == 4801","seed == 5201")
old='''                    control = [x for x in reference_control(dt, pattern, seed)
                               if x['arm'] == 'relative_paired'][0]
                    assert fresh[0]['ef'] == control['ef']
                    assert fresh[0]['max_absolute_integer'] == control['max_absolute_paired_integer']
                    print('CONTROL '+json.dumps(dict(dtype=str(dt), seed=seed,
                          same_input=True, quant_k128_equals_m8=True)), flush=True)'''
new='''                    control=reference_control(dt,pattern,seed)
                    for index,oldindex in [(0,0),(1,2)]:
                        assert fresh[index]['ef']==control[oldindex]['ef']
                        assert fresh[index]['max_absolute_integer']==control[oldindex]['max_absolute_integer']
                    print('CONTROL '+json.dumps(dict(dtype=str(dt),seed=seed,same_input=True,
                          k128_and_exact_k512_equal_m9=True)),flush=True)'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("    rows = []\n    for dt in",'''    # Rounding twice differs from direct wider quantization: it needs a new screen.
    first=round(1.49);twice=round(first/2);direct=round(1.49/2)
    assert twice==0 and direct==1
    print('NEGATIVE_CONTROL '+json.dumps(dict(input=1.49,old_scale=1,new_scale=2,twice=twice,direct=direct)),flush=True)
    rows = []
    for dt in''')
(h/'requantized_model.py').write_text(s)
print('generated M10 numerical screen from unchanged M9')
