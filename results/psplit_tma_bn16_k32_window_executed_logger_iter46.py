from pathlib import Path
import csv
rows=list(csv.DictReader(Path('results/current.csv').open()))
assert len(rows)==2 and all(x['variant']=='50' and x['actual']=='megakernel_wgmma_psplit_tma_bn16_k32_window_64x16' for x in rows)
archive=sorted(Path('trajectory').glob('*_iter-46'))[-1]
for suffix,src in [('csv',Path('results/current.csv')),('log',archive/'output.txt'),('exit',Path('results/psplit_tma_bn16_k32_window_bench_iter46.exit'))]:
 p=Path(f'results/small_psplit_tma_bn16_k32_window50_launch_iter46.{suffix}');assert not p.exists();p.write_bytes(src.read_bytes())
text='\n### Iter46 — paired K16 inputs, two K32 windows on BN16 TMA\n\n'
text+='- **Hypothesis:** Retain two successive K16 A atoms until one matching wait0, reducing four input windows in49 to two in50. Geometry64x16x64,2banks/384threads,40/96/96/minCTA2. More live A may increase register pressure; no speed or measured residency is inferred from the source constraint.\n'
text+='- **Changes:** ONLY50 new TU. Each product retains ascending atom0..3, original FP32RN operand DAG and one final Element cast, wholeK seven FP32P, fixedU and RNalpha/betaFMA. Original guards/floors/AutoDense/37earlyblock/zero globalmatrixworkspace unchanged. Fresh two-dtype machine/ABI/link/init/pending review and occupancy attributes are bound independently before launch; original instance fields remain authoritative.\n'
text+='- **Bench:** AKO quick20warm/20calls/5groups/1fresh. All original numerical FAILs and exits retained; full timing follows regardlessquick3%.\n\n| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in rows:text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Whole-call own latency evaluates the input-window schedule. Static occupancy API is a prediction, not measured residency or causal stall proof; numerical certification remains the original independent gate. Ordinary50 includes current-A encode.\n- **Next:** After this quick Git checkpoint, freeze mandatorysamebinary20 (50/49/45/39/Lt1) and separate supplemental8 originalH10 single-P input fusion/two launches andH12 S/T once/three launches before anyfull timing. All28 use20warm100calls5groups3fresh/rotate1/32Lt8GiB. Once-complete identity first-creation includes bothplanSHA; sourceGit proves7 executed sources, laterprestatus Git binds plans/identity. Strict30/serialSAN3/coverage/independent audits follow; no automatic47.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text)
Path('results/psplit_tma_bn16_k32_window_quick_archive_iter46.txt').write_text(str(archive)+'\n')
print('ITER46_LOGGER_COMPLETE',len(rows),'rows',archive)
