from pathlib import Path
import csv,json,hashlib
r=list(csv.DictReader(Path('results/current.csv').open()))
archives=sorted(Path('trajectory').glob('*_iter-38'));assert archives
archive=archives[-1]
Path('results/small_psplit_hybrid_s41_launch_iter38.csv').write_bytes(Path('results/current.csv').read_bytes())
Path('results/small_psplit_hybrid_s41_launch_iter38.log').write_bytes((archive/'output.txt').read_bytes())
Path('results/small_psplit_hybrid_s41_launch_iter38.exit').write_bytes(Path('results/psplit_hybrid_s_bench_iter38.exit').read_bytes())
proto=archive/'prototype';proto.mkdir(exist_ok=True)
for f in Path('results').glob('psplit_hybrid_s_frozen_*_iter38.*'):
 (proto/f.name).write_bytes(f.read_bytes())
text='\n### Iter 38 — Hybrid S1/S2 producer and complete-stage register A\n\n'
text+='- **Hypothesis:** Use hybrid S1/S2 producer construction with original RN onecast and BK32 RS-A while preserving whole-K seven P and fixed U; each stage builds distinct packed A words before issue and retains every A operand until wait0.\n'
text+='- **Changes:** New41 uses BM64 BN32 BK32,384threads/producer64/WG192/WG192,full128/empty256/two banks; producer rawA4, final roundedS1/FP32S2 planes and finalB7, consumer complete-stage packed A24/32 plus P48/64 words. Actual initial quota/resource/full pendingA+P gate precedes any GPU launch. Original onecast/fixedU/alphaRNbetaFMA/nonaligned drain6/zeroGWS unchanged. ParentK%64 and truthful fallback. Old34CUDA preserved, registered37 disabled. Originalnumericalgate/AutoDense unchanged; onlynewfreshcompiledgatedkernels execute.\n'
text+='- **Bench:** quick20warm20calls5groups1fresh; original numericalFAIL is retained, no certification.\n\n| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |\n|---|---|---|---|---:|---:|---:|---|\n'
for x in r:
 text+=f"| {x['m']},{x['n']},{x['k']} | {x['dtype']} | {x['actual']} v{x['variant']} | {x['mode']} | {float(x['dense_event_ms'])*1000:.6f} | {float(x['solution_event_ms'])*1000:.6f} | {float(x['speedup']):.6f} | {'PASS' if x['numerical_pass']=='1' else 'FAIL'} |\n"
text+='\n- **Analysis:** Own complete SW latency determines code benefit; same-row tuned Dense/SW reports the comparison. Numerical/structural/SAN evidence stays separate. Missing unsafe routes will have explicit compiled-safety exclusions rather than invented rows.\n- **Next:** Mandatoryfull1024 measured41 plus retainedRSBK64best39/matchedrawBK32RS40/SS36/Lt1,100calls5groups3fresh; actual validation scope, independent implementation/data audits and source/raw/report/Git closure.\n- **Archive:** '+str(archive)+'/\n'
f=Path('ITERATIONS.md');f.write_text(f.read_text()+text)
Path('.scratch/psplit_hybrid_s_quick_archive_iter38.txt').write_text(str(archive)+'\n')
print('iter38 logger complete',len(r),'rows',archive)
