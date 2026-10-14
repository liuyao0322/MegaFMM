from pathlib import Path
import csv
rows=list(csv.DictReader(Path('results/current.csv').open()))
archive=sorted(Path('trajectory').glob('*_iter-43'))[-1]
for suffix,src in [('csv',Path('results/current.csv')),('log',archive/'output.txt'),('exit',Path('results/psplit_tma_bn64_bench_iter43.exit'))]:Path(f'results/small_psplit_tma_bn6447_launch_iter43.{suffix}').write_bytes(src.read_bytes())
proto=archive/'prototype';proto.mkdir(exist_ok=True)
for p in Path('results').glob('psplit_tma_bn64_frozen_*_iter43.*'):(proto/p.name).write_bytes(p.read_bytes())
text='\n### Iter43 — BN64 selective rawA TMA with whole-stage RS operands\n\n'
text+='- **Hypothesis:** Increase BN32 toBN64 while retaining four rawA SW128 TMA boxes, SIMT finalB7, currentA map encode inside each ordinary call. The doubled N tile changes B/P/tail perCTA and grid shape; no isolated causality or residency claim.\n'
text+='- **Changes:** ONLY47, BM64BN64BK64/2banks/wholeP3+4/fourascendingK16/finalonecast/fixedU/RNalphaFMA/AutoDense and original floors retained. Initial40/224/240 actual STACK128spill168; fresh stage-local address reconstruction reduced STACK/spill to8, then48/216/240 same64512 pool eliminated spill. Every failed source/build/SASS is archived; final new compiled gate required before first47 kernel.\n'
text+='- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.\n\n| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in rows:text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.\n- **Next:** Mandatory samebinaryfull20 for47/45/39/36/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text);Path('.scratch/psplit_tma_bn64_data_quick_archive_iter43.txt').write_text(str(archive)+'\n');print('ITER43_LOGGER_COMPLETE',len(rows),'rows',archive)
