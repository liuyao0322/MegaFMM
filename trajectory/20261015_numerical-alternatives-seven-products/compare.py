import numpy as np,json
from pathlib import Path

def quant(x,dt):
 x=np.asarray(x,dtype=np.float32)
 if dt=='fp16':return x.astype(np.float16).astype(np.float64)
 u=x.view(np.uint32);u=(u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)
 return u.view(np.float32).astype(np.float64)
def blocks(x):return x[:512,:512],x[:512,512:],x[512:,:512],x[512:,512:]
def evaluate(a,b,dt,classic=False):
 aa,ab,ac,ad=blocks(a);ba,bb,bc,bd=blocks(b);q=lambda x:quant(x,dt)
 # Float64 leaf dots/recombination isolate transform/operand quantization from GPU accumulation.
 if classic:
  p=(q(aa+ad)@q(ba+bd),q(ac+ad)@q(ba),q(aa)@q(bb-bd),q(ad)@q(bc-ba),q(aa+ab)@q(bd),q(ac-aa)@q(ba+bb),q(ab-ad)@q(bc+bd))
  x,y,z,w,v,t,u=p
  return np.block([[x+w-v+u,z+v],[y+w,x-y+z+t]])
 s1=ac+ad;s2=s1-aa;s3=aa-ac;s4=ab-s2
 t1=bb-ba;t2=bd-t1;t3=bd-bb;t4=t2-bc
 p1=q(aa)@q(ba);p2=q(ab)@q(bc);p3=q(s4)@q(bd);p4=q(ad)@q(t4)
 p5=q(s1)@q(t1);p6=q(s2)@q(t2);p7=q(s3)@q(t3)
 u2=p1+p6;u3=u2+p7;u4=u2+p5
 return np.block([[p1+p2,u4+p3],[u3-p4,u3+p5]])
rows=[]
for dt in ['bf16','fp16']:
 for seed in [1,2,3]:
  rng=np.random.default_rng(seed);a=quant(rng.uniform(-1,1,(1024,1024)),dt);b=quant(rng.uniform(-1,1,(1024,1024)),dt)
  ref=a@b;den=np.linalg.norm(ref)
  variants=[('Winograd_current',dt,False),('Classic_Strassen',dt,True)]
  if dt=='bf16':variants.append(('Winograd_FP16_all_operands','fp16',False))
  for name,operand_dt,classic in variants:
   out=evaluate(a,b,operand_dt,classic);ef=float(np.linalg.norm(out-ref)/den)
   row=dict(input_dtype=dt,seed=seed,variant=name,relative_frobenius=ef,nonfinite=int(np.count_nonzero(~np.isfinite(out))))
   rows.append(row);print(row,flush=True)
Path('results/numerical_alternatives/results.json').write_text(json.dumps(dict(scope='CPU numerical diagnostic only: 1024 cubed; float64 leaf dots and recombination; not production GPU precision or performance acceptance; BF16-to-FP16 range only verified on bounded uniform inputs',rows=rows),indent=2)+'\n')
