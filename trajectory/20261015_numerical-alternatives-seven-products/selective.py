from pathlib import Path
import numpy as np,itertools,json
exec(Path('results/numerical_alternatives/compare.py').read_text().split('rows=[]')[0])
terms=['P3_Slow','P4_Tlow','P5_Slow','P5_Tlow','P6_Slow','P6_Tlow','P7_Slow','P7_Tlow']
data=[]
for seed in [1,2,3]:
 rng=np.random.default_rng(seed);a=quant(rng.uniform(-1,1,(1024,1024)),'fp16');b=quant(rng.uniform(-1,1,(1024,1024)),'fp16');ref=a@b
 aa,ab,ac,ad=blocks(a);ba,bb,bc,bd=blocks(b)
 ss=[ac+ad,ac+ad-aa,aa-ac,ab-(ac+ad-aa)];tt=[bb-ba,bd-(bb-ba),bd-bb,(bd-(bb-ba))-bc]
 sh=[quant(v,'fp16') for v in ss];th=[quant(v,'fp16') for v in tt]
 sl=[v-h for v,h in zip(ss,sh)];tl=[v-h for v,h in zip(tt,th)]
 # Ideal exact residuals and FP64 correction products favor selective compensation.
 corr=[sl[3]@bd,ad@tl[3],sl[0]@th[0],sh[0]@tl[0],sl[1]@th[1],sh[1]@tl[1],sl[2]@th[2],sh[2]@tl[2]]
 signs=[(0,1,0,0),(0,0,-1,0),(0,1,0,1),(0,1,0,1),(0,1,1,1),(0,1,1,1),(0,0,1,1),(0,0,1,1)]
 cs=[]
 for v,sg in zip(corr,signs):cs.append(np.block([[sg[0]*v,sg[1]*v],[sg[2]*v,sg[3]*v]]).ravel())
 cs=np.asarray(cs);e=(evaluate(a,b,'fp16')-ref).ravel();den=float(np.dot(ref.ravel(),ref.ravel()))
 data.append((float(e@e),cs@e,cs@cs.T,den))
 print('prepared',seed,flush=True)
rows=[]
for count in range(9):
 best=None
 for subset in itertools.combinations(range(8),count):
  mask=np.zeros(8);mask[list(subset)]=1
  errors=[float(np.sqrt(max(0,e2+2*(mask@dot)+mask@gram@mask)/den)) for e2,dot,gram,den in data]
  score=max(errors)
  if best is None or score<best['max_ef']:best=dict(extra_products=count,terms=[terms[i] for i in subset],max_ef=score,seed_errors=errors,below_floor_all=score<=1e-4)
 rows.append(best);print(best,flush=True)
Path('results/numerical_alternatives/selective.json').write_text(json.dumps(dict(scope='diagnostic only: ideal exact residuals and FP64 correction products; enumerate all subsets of the eight standard cross terms; not proof for other numerical algorithms or GPU pass/performance',rows=rows),indent=2)+'\n')
