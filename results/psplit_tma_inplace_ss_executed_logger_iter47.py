from pathlib import Path
import csv,json
rows=list(csv.DictReader(Path('results/current.csv').open()));assert len(rows)==4 and {int(x['variant']) for x in rows}=={51,52}
quick=json.load(open('results/psplit_tma_inplace_ss_combined_quick_iter47.json'));archive=Path(quick['archive'])
for variant in [51,52]:
 chosen=[x for x in rows if int(x['variant'])==variant];assert len(chosen)==2
 p=Path(f'results/small_psplit_tma_inplace_ss{variant}_launch_quick_iter47.csv');assert not p.exists()
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(chosen)
for suffix,src in [('log',archive/'output.txt'),('csv',archive/'results.csv')]:
 p=Path(f'results/small_psplit_tma_inplace_ss_combined_quick_iter47.{suffix}');assert not p.exists();p.write_bytes(src.read_bytes())
text='\n### Iter47 — cooperative in-place shared ST, SS WGMMA BN32/BN16\n\n'
text+='- **Hypothesis:** Eight raw quadrant TMA boxes load into the first four shared final planes, then all256 consumers own disjoint packed pairs and construct final S/T once per CTA/bankstage. This is not global H12 ST-once. Both current A/B maps encode inside each ordinary call. More consumer transform work/bar1 may offset avoiding redundant constructions; no source-only speed claim.\n'
text+='- **Changes:** ONLY51/52 new shared TU with original RN DAG/final onecast/wholeK sevenP/fixedU/alphaRNbetaFMA unchanged. Initial frontend failures, CPU byte-flag consumer diagnostics and BN32 min2 complete-P spills retained. Final51 min1 q40/128/128,52 min2 q48/80/80; fresh four-instance independent compiled gate binds final source/bin streams, role initialization/actual pool/pending/layout/TMA/sync. Initial-owned preINC zeros remain truthful.\n'
text+='- **Bench:** One combined serial quick51+52,20warm20calls5groups1fresh. Original gates/exit2 retained; full24 timing required regardless3% gain.\n\n| dtype | native route | Dense us | SW complete us | Dense/SW | original gate |\n|---|---|---:|---:|---:|---|\n'
for x in rows:text+=f"| {x['dtype']} | {x['actual']} v{x['variant']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Own SW complete-call latency will rank routes, not Dense denominator changes. Static occupancy API is not measured residency or causality; no precision promotion.\n- **Next:** After this quick source Git checkpoint, create immutable full24 plan and once-complete identity before first full timing:51/52/45/50/39/Lt1 ×two dtype ×ordinary/Graph;20warm100calls5groups3fresh32Lt8GiBrotate1. Six axes×four modes=24, strict30each with shared30Dense refs, SAN3each serial. Fresh H10/H12 remain iteration46 history. Close unique prestatus and independent audits; no automatic48.\n- **Archive:** '+str(archive)+'/\n'
p=Path('ITERATIONS.md');p.write_text(p.read_text()+text)
Path('results/psplit_tma_inplace_ss_quick_archive_iter47.txt').write_text(str(archive)+'\n')
print('ITER47_LOGGER_COMPLETE',len(rows),'rows',archive)
