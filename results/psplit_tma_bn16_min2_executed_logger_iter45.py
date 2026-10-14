from pathlib import Path
import csv
rows=list(csv.DictReader(Path('results/current.csv').open()))
assert len(rows)==2 and all(x['variant']=='49' for x in rows)
archive=sorted(Path('trajectory').glob('*_iter-45'))[-1]
for suffix,src in [('csv',Path('results/current.csv')),('log',archive/'output.txt'),('exit',Path('results/psplit_tma_bn16_min2_bench_iter45.exit'))]:
 Path(f'results/small_psplit_tma_bn16_min249_launch_iter45.{suffix}').write_bytes(src.read_bytes())
proto=archive/'prototype';proto.mkdir(exist_ok=True)
for p in Path('results').glob('psplit_tma_bn16_min2_frozen_*_iter45.*'):(proto/p.name).write_bytes(p.read_bytes())
text='\n### Iter45 — BN16 minCTA2 with atom-local RS inputs\n\n'
text+='- **Hypothesis:** Test register residency and shorter transformed-A lifetime on BN16/BK64 while retaining four rawA TMA boxes and SIMT finalB7. MinCTA2 is a source constraint; occupancy API predicts2, measured residency is unknown.\n'
text+='- **Changes:** ONLY49 q40/88/112, BM64BN16BK64/2banks/384threads. Known48 stage-local raw coordinates and B finalcast scheduling preserve RN. Firstcompile actual I80 STACK24/spill40FP16/36BF16 and newC7512 prohibited launch; failed source/PTX/SASS/resources retained. Revision1 uses one K16 A atom per P, four ascending atom windows each build/fence/3or4HGMMA/commitwait0/postfences, release after all4; whole FP32P/wholeK/fixedU/RNalpha-betaFMA/original floors unchanged. Fresh roles37/77/71, zeroSTACKLOCALspill, owned30720=request30720. Legal initial-owned commonP R32..55 preINC zeros recorded truthfully; no meaningful growth access. Fresh independent CAN49 only.\n'
text+='- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; all originalgate failures retained.\n\n| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in rows:text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Complete own SW latency evaluates the combined resource/lifetime change. No measured residency/stall/traffic cause or numerical certification is inferred. Current A encode remains inside each ordinary complete call.\n- **Next:** Mandatory samebinaryfull20 for49/48/45/39/Lt1,20warm100calls5groups3fresh regardlessquick3%; final plan and once-complete identity created after this quick Git checkpoint but before full timing. SourceGit proves seven executed sources; later unique prestatus Git binds plan/identity. Originalstrict30 andserialSAN3, coverage/profile/report and independentfinal audits follow.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text)
Path('results/psplit_tma_bn16_min2_quick_archive_iter45.txt').write_text(str(archive)+'\n')
print('ITER45_LOGGER_COMPLETE',len(rows),'rows',archive)
