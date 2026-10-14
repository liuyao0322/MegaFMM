from pathlib import Path
import subprocess
base=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
g=Path('results/opt_goal/k128pdl279/generate.py').read_text();a=g.index('g=Path(');b=g.index('p=subprocess.check_output',a);g=g[a:b]
g=g.replace('base={1:0,4:2099200,5:2048,6:2097152}', 'base={1:0,4:2097152,5:2099200,6:2048}') if False else g
# The nested original static generator is loaded at execution; modify its text
# before its replacements and emission run.
g=g.replace("a=g.index(\"fn='''\");b=g.index('s=s[:a]+fn+s[b:]')", "g=g.replace('base={1:0,4:2099200,5:2048,6:2097152}', 'base={1:0,4:2097152,5:2099200,6:2048}').replace('if step in [1,5]', 'if step in [1,6]').replace('elif step==4:', 'elif step==5:')\na=g.index(\"fn='''\");b=g.index('s=s[:a]+fn+s[b:]')")
exec(g)
s=Path('solution/sw_pipeline.cu').read_text()
s=s.replace('step==4 ? 4:step==5 ? 2:3','step==4 ? 3:step==5 ? 4:2')
assert 'step==4 ? 3:step==5 ? 4:2' in s
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('product_order=1_2_6_7_5_3_4','product_order=1_2_6_7_4_5_3')
Path('solution/plan.cpp').write_text(p)
