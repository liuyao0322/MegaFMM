from pathlib import Path
p=Path('results/opt_goal/profile131/diagnostics/generate.py')
s=p.read_text().replace('0ed59e7','d8e6261')
s=s.replace("a='''  __syncthreads();\n}\nvoid check'''", "a='''  // CTA shared storage survives until all role threads return.\n}\nvoid check'''")
s=s.replace('  __syncthreads();\n  if constexpr(Role==4)t3=stamp();','  if constexpr(Role==4)t3=stamp();')
out=Path('results/opt_goal/profile162/diagnostics/generate.py');out.write_text(s)
# Driver: Role5 only, matching asynchronous kernel launch geometry and compact maps.
s=Path('results/opt_goal/profile131/diagnostics/source/measure.cu').read_text().replace('8388608','4194304').replace('reinterpret_cast<char*>(W)+4194304','reinterpret_cast<char*>(W)+2097152')
s=s.replace('f,512,shared','f,768,shared').replace('dim3(512)','dim3(768)').replace('sw_cooperative_hilo<T,4>','sw_cooperative_hilo<T,5>')
Path('results/opt_goal/profile162/diagnostics/measure_template.cu').write_text(s)
