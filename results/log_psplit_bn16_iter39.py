from pathlib import Path
import csv
rows=list(csv.DictReader(Path('results/current.csv').open()))
archives=sorted(Path('trajectory').glob('*_iter-39'));assert archives
archive=archives[-1]
variant=int(rows[0]['variant']) if rows else 43
for suffix,src in [('csv',Path('results/current.csv')),('log',archive/'output.txt'),('exit',Path('results/psplit_bn16_bench_iter39.exit'))]:
    Path(f'results/small_psplit_bn16{variant}_launch_iter39.{suffix}').write_bytes(src.read_bytes())
proto=archive/'prototype';proto.mkdir(exist_ok=True)
for p in Path('results').glob('psplit_bn16_frozen_*_iter39.*'):
    (proto/p.name).write_bytes(p.read_bytes())
text='\n### Iter 39 — Paired BN16 RS-A and shared/shared seven-product kernels\n\n'
text+='- **Hypothesis:** Compare BN16 RS-A42 and SS43 with complete ascending-K seven P and fixed U. Source quotas/min2 alone are hypotheses; fresh per-route compiled proof precedes launch.\n'
text+='- **Changes:** Two new BM64BN16BK32/two-bank384-thread TUs. RS42 quotas48/96/96 and SS43 quotas64/80/80; both min2. New literal tail permutation preserves unique P/U coordinates. Initial42 three scalar state words spilled; equivalent PipelineState derived from the unchanged int64 K counter removes persistent phase liveness. Initial failed source and scheduling error remain archived. Four final instances have actual REG80 zero stack/local/spill; all-role/pending-A-P/init/ABI/drain and object/bench/test identity are independently checked. OccupancyAPI resource prediction is separate from measured residency. Original ST onecast/DAG/alphaRNbetaFMA/floors/AutoDense remain unchanged.\n'
text+='- **Bench:** quick20warm20calls5groups1fresh; original numerical results retained.\n\n| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in rows:
    text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Compare own complete SW latency, then same-row tuned Dense/SW. Machine safety, structural fixtures and sanitizers do not certify the original FP64 numerical gate.\n- **Next:** Mandatorysamebinary full24 for42/43/39/40/36/Lt1,100calls5groups3fresh irrespective quick3%; enabled strict30 per newroute and serialsanitizers, actualcoverage/report/profile/independentaudits.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text)
Path('.scratch/psplit_bn16_quick_archive_iter39.txt').write_text(str(archive)+'\n')
print('iter39 logger complete',len(rows),'rows',archive)
