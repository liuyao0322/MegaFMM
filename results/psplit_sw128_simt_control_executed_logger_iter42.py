from pathlib import Path
import csv
rows=list(csv.DictReader(Path('results/current.csv').open()))
archive=sorted(Path('trajectory').glob('*_iter-42'))[-1]
for suffix,src in [('csv',Path('results/current.csv')),('log',archive/'output.txt'),('exit',Path('results/psplit_sw128_simt_control_bench_iter42.exit'))]:Path(f'results/small_psplit_sw128_simt_control46_launch_iter42.{suffix}').write_bytes(src.read_bytes())
proto=archive/'prototype';proto.mkdir(exist_ok=True)
for p in Path('results').glob('psplit_sw128_simt_control_frozen_*_iter42.*'):(proto/p.name).write_bytes(p.read_bytes())
text='\n### Iter42 — SW128 rawA SIMT control with unused currentA encode\n\n'
text+='- **Hypothesis:** Isolate rawA delivery with parent45 SW128 geometry/math/domain/host currentA map encode retained. New46 rawA vector8 SIMT and B SIMT, all128 writers each proxyfence before fullcommit; no kernel TMA issue/rawTX wait. The encoded map is deliberately unused, not a no-encode route.\n'
text+='- **Changes:** ONLY46 from frozen CPU candidate, BM64BN32BK64/2banks/wholeP3+4/ascending4K16/onecast/fixedU/RNalphaFMA/AutoDense and original floors preserved. Fresh compiled gate before first kernel, prior CAN45 not inherited.\n'
text+='- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.\n\n| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in rows:text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.\n- **Next:** Mandatory samebinaryfull20 for46/45/39/36/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text);Path('.scratch/psplit_sw128_simt_control_data_quick_archive_iter42.txt').write_text(str(archive)+'\n');print('ITER42_LOGGER_COMPLETE',len(rows),'rows',archive)
