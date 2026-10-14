from pathlib import Path
import csv
rows=list(csv.DictReader(Path('results/current.csv').open()))
archive=sorted(Path('trajectory').glob('*_iter-44'))[-1]
for suffix,src in [('csv',Path('results/current.csv')),('log',archive/'output.txt'),('exit',Path('results/psplit_tma_bn16_bench_iter44.exit'))]:Path(f'results/small_psplit_tma_bn1648_launch_iter44.{suffix}').write_bytes(src.read_bytes())
proto=archive/'prototype';proto.mkdir(exist_ok=True)
for p in Path('results').glob('psplit_tma_bn16_frozen_*_iter44.*'):(proto/p.name).write_bytes(p.read_bytes())
text='\n### Iter44 — BN16 selective rawA TMA with whole-stage RS operands\n\n'
text+='- **Hypothesis:** Decrease BN32 toBN16 while retaining four rawA SW128 TMA boxes, SIMT finalB7, currentA map encode inside each ordinary call. The smaller N tile changes B/P/tail perCTA and grid shape; no isolated causality or residency claim.\n'
text+='- **Changes:** ONLY48, BM64BN16BK64/2banks/wholeP3+4/fourascendingK16/finalonecast/fixedU/RNalphaFMA/AutoDense and original floors retained. Initial40/128/128 actual REG128 STACK168spill224; stage-local volatile thread-id reconstructs the same rawA coordinates, with independent B onecast scheduling preserving original unrounded FP32 DAG. Revision1 retains40/128/128 and eliminates STACK/LOCAL/spill. Fresh I128 owned49152 exceeds requested37888; no two-CTA residency claim. Every failed source/build/SASS is archived; final new compiled gate required before first48 kernel.\n'
text+='- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.\n\n| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in rows:text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.\n- **Next:** Mandatory samebinaryfull20 for48/45/47/39/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text);Path('.scratch/psplit_tma_bn16_data_quick_archive_iter44.txt').write_text(str(archive)+'\n');print('ITER44_LOGGER_COMPLETE',len(rows),'rows',archive)
