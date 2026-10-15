"""Warp-leader instruction-path intervals; diagnostic, not phase acceptance timing."""
import hashlib
import json
import re
import statistics as st
import subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[2]
log=(h/'trace_p33.log').read_text().splitlines()
sass_function='_ZN14phase_research30preprocess_cooperative_columnsIN7cutlass10bfloat16_tEEEvPKT_S5_PS3_S6_PaS7_PiS8_S8_Pf'
sass=subprocess.run(['/share/app/cuda/cuda-12.8/bin/cuobjdump','--dump-sass','--function',sass_function,str(h/'bin/trace_p33')],capture_output=True,text=True,check=True)
assert re.findall(r'^\s*Function\s*:\s*(\S+)',sass.stdout,re.M)==[sass_function]
(h/'ncu/trace_p33_preprocess_sass.log').write_text(sass.stdout)
(h/'ncu/trace_p33_preprocess_sass_warnings.log').write_text(sass.stderr)
timer_lines=[x.strip() for x in sass.stdout.splitlines() if 'SR_GLOBALTIMERLO' in x]
barrier_lines=[x.strip() for x in sass.stdout.splitlines() if 'BAR.SYNC.DEFER_BLOCKING' in x]
assert len(timer_lines)==6 and barrier_lines

def rows(prefix):return [dict(re.findall(r'(\w+)=([^ ]+)',x)) for x in log if x.startswith(prefix)]
checks=rows('TRACE_CHECK ');assert len(checks)==10
assert all(x['pass']=='1' and x['different']=='0' for x in checks)
raw=rows('PHASE ');assert len(raw)==288*8*10
out=[]
def stats(values):
 v=sorted(values);return dict(n=len(v),median_us=st.median(v)/1000,min_us=min(v)/1000,max_us=max(v)/1000,
                             p10_us=v[int(.1*(len(v)-1))]/1000,p90_us=v[int(.9*(len(v)-1))]/1000)
for dtype in ['bf16','fp16']:
 for sample in range(5):
  r=[{k:(v if k=='dtype' else int(v)) for k,v in x.items()} for x in raw if x['dtype']==dtype and int(x['sample'])==sample]
  assert len(r)==2304 and len({(x['block'],x['warp']) for x in r})==2304
  active_all=[x for x in r if x['block']<128];idle_all=[x for x in r if x['block']>=128]
  active=[x for x in active_all if x['warp']==0];idle=[x for x in idle_all if x['warp']==0]
  leader=[x for x in r if x['warp']==0]
  assert max(x['t3'] for x in active)<=min(x['t4'] for x in leader), 'warp0 global boundary consistency'
  premature=sum(x['warp']!=0 and x['t4']<next(y['t4'] for y in leader if y['block']==x['block']) for x in r)
  assert all(all(x['t'+str(i)]<=x['t'+str(i+1)] for i in range(5)) and x['t0']>0 for x in active_all)
  assert all(x['t1']==x['t2']==x['t3']==0 and 0<x['t0']<=x['t4']<=x['t5'] for x in idle_all)
  intervals={name:stats([x['t'+str(b)]-x['t'+str(a)] for x in active]) for name,a,b in [('a_path',0,1),('b_path',1,2),('cta_finalize_path',2,3),('active_grid_wait_path',3,4)]}
  intervals['inactive_until_grid_return']=stats([x['t4']-x['t0'] for x in idle])
  packs={name:stats([x['t5']-x['t4'] for x in leader if lo<=x['block']<hi]) for name,lo,hi in [('b_pair0',0,64),('b_pair1',64,128),('b_raw',128,192),('a_pair0',192,224),('a_pair1',224,256),('a_raw',256,288)]}
  out.append(dict(dtype=dtype,sample=sample,intervals=intervals,pack_paths=packs,
    excluded_nonwarp0_early_grid_return_stamps=premature,
    global_span_us=(max(x['t5'] for x in leader)-min(x['t0'] for x in leader))/1000,
    until_last_grid_return_us=(max(x['t4'] for x in leader)-min(x['t0'] for x in leader))/1000))
time=rows('TRACE_TIME ');assert len(time)==40
perturbation=[]
for dtype in ['bf16','fp16']:
 groups=[]
 for group in range(5):
  r=[x for x in time if x['dtype']==dtype and int(x['group'])==group];assert len(r)==4
  v={m:st.mean(float(x['us']) for x in r if int(x['method'])==m) for m in [0,1]}
  groups.append(dict(group=group,baseline_us=v[0],trace_us=v[1],relative_change=v[1]/v[0]-1))
 perturbation.append(dict(dtype=dtype,baseline_median_us=st.median(x['baseline_us'] for x in groups),trace_median_us=st.median(x['trace_us'] for x in groups),
  median_paired_relative_change=st.median(x['relative_change'] for x in groups),groups=groups))
result=dict(scope='H100-only globaltimer warp0 instruction-path diagnostics; other warp leaders retained as raw observations but excluded from phase interpretation, not pure component costs. No extra CTA/grid barriers. Target-specific timer behavior is not portable. Instrumented full-call timings only calibrate perturbation, not candidate ranking.',
 static_trace_function=sass_function,static_timer_instructions=timer_lines,static_deferred_barrier_count=len(barrier_lines),trace_cases=10,previous_timing_input_cases=2,new_seed_cases=8,bitwise_output_comparisons=10*1024*1024,bitwise_differences=0,
 all_cases_original_rounded_input_fp64_below_fixed_floor=True,checks=checks,warp_records=len(raw),interpreted_warp0_records=2880,cases=out,perturbation=perturbation,
 limitations='Only warp0 interval statistics are interpreted: non-warp0 post-grid timer stamps are observed before the corresponding warp0 return and would misattribute grid waiting to packing. The cause is not isolated; BAR.SYNC.DEFER_BLOCKING exists in SASS, possible independent instruction motion is a hypothesis, not a proved cause. Warp0 stamps pass max active pre-grid <= min post-grid consistency on every case. Warp leaders stamp instruction-path boundaries; other lanes may still have work. Active grid waits include skew/visibility, inactive waits include reduction time. Intervals include instrumentation and do not add to a kernel latency. Baseline source recompiled in same trace binary, not frozen compare_p33 binary. No Graph/two-stream/sanitizer acceptance or production performance claim.')
(h/'trace_p33_summary.json').write_text(json.dumps(result,indent=2)+'\n')
files=['generate_p33_phase_trace.py','analyze_p33_phase_trace.py','trace_p33_driver.cu','build_trace_p33.sh','trace_p33_build.log','trace_p33.log','trace_p33_summary.json','p33_manifest.json','bin/trace_p33','ncu/trace_p33_preprocess_sass.log','ncu/trace_p33_preprocess_sass_warnings.log']
files+=['trace_p33/'+f for f in ['pack.cuh','corrected.cuh','pow2.cuh']]
(h/'trace_p33_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(dict(checks=10,bitwise_differences=0,perturbation=perturbation),indent=2))
for x in out:
 if x['sample']==0:print(json.dumps(x,indent=2))
