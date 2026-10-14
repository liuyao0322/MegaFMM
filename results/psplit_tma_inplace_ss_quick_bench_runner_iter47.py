from pathlib import Path
import subprocess,os,json,csv,io
p=Path('results/psplit_tma_inplace_ss_bench_iter47.txt');assert not p.exists()
env=os.environ.copy();env.update(MEGAFMM_ALGORITHM='mega',MEGAFMM_SHAPES='1024,1024,1024',MEGAFMM_DTYPE='both',MEGAFMM_MEGA_VARIANT='51',MEGAFMM_TRIALS='20',MEGAFMM_WARMUP='20',MEGAFMM_GROUPS='5',MEGAFMM_CORRECT_TRIALS='1',MEGAFMM_ROTATE_INPUTS='1',MEGAFMM_GRAPH='0')
cmd=['bash','scripts/bench.sh','iter-47']
with p.open('w') as f:
 f.write('COMMAND '+json.dumps(cmd)+'\n');f.write('MEGAFMM_ENV '+json.dumps({k:v for k,v in env.items() if k.startswith('MEGAFMM_')},sort_keys=True)+'\n');f.flush();code51=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
p.with_suffix('.exit').write_text(str(code51)+'\n')
rows51=Path('results/current.csv').read_bytes();Path('results/psplit_tma_inplace_ss51_quick_original_iter47.csv').write_bytes(rows51)
archive=sorted(Path('trajectory').glob('*_iter-47'))[-1]
cmd52=['bash','scripts/on_gpu.sh','./build/megafmm_bench','--algorithm','mega','--mega-variant','52','--shapes','1024,1024,1024','--dtype','both','--rotate-inputs','1','--warmup','20','--trials','20','--groups','5','--correct-trials','1','--output','results/psplit_tma_inplace_ss52_quick_original_iter47.csv']
p52=Path('results/psplit_tma_inplace_ss52_quick_original_iter47.log')
with p52.open('w') as f:f.write('COMMAND '+json.dumps(cmd52)+'\n');f.flush();code52=subprocess.call(cmd52,stdout=f,stderr=subprocess.STDOUT)
p52.with_suffix('.exit').write_text(str(code52)+'\n')
first=list(csv.reader(io.StringIO(rows51.decode())));second=list(csv.reader(Path('results/psplit_tma_inplace_ss52_quick_original_iter47.csv').read_text().splitlines()));assert first[0]==second[0]
f=io.StringIO();w=csv.writer(f,lineterminator='\n');w.writerows(first+second[1:]);combined=f.getvalue().encode()
Path('results/current.csv').write_bytes(combined);(archive/'results.csv').write_bytes(combined)
with (archive/'output.txt').open('ab') as f:f.write(b'\nCOMBINED_QUICK52_SECOND_SERIAL_CALL\n');f.write(p52.read_bytes())
Path('results/psplit_tma_inplace_ss_combined_quick_iter47.json').write_text(json.dumps({'schema':'megafmm.iter47.combined-quick.v1','variants':[51,52],'quick51_exit':code51,'quick52_exit':code52,'archive':str(archive),'rows':4,'second_command':cmd52,'protocol':{'warmup':20,'trials':20,'groups':5,'fresh_checks':1,'ordinary':True}},indent=2)+'\n')
print('AKO47_COMBINED_CLOSED',code51,code52,str(archive),flush=True)
raise SystemExit(max(code51,code52))
