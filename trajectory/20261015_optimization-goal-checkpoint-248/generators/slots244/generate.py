from pathlib import Path
s=Path('results/opt_goal/static218/generate.py').read_text()
s=s.replace("s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)","s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)\ns=s.replace('Stages=4','Stages=3').replace('Threads=768','Threads=672').replace('pack_ready,608','pack_ready,512').replace('local+=2*608','local+=2*512')")
s=s.replace('slot=seq%4;phase=seq//4%2','slot=seq%3;phase=seq//3%2').replace("s=s.replace('==213120','==196736')","s=s.replace('==213120','==147584')")
exec(compile(s,'slots244-generator','exec'))
p=Path('solution/plan.cpp');s=p.read_text().replace('4wgmma+19pack_only','4wgmma+16pack_only').replace('smem=196736','smem=147584').replace('operand_stages=4','operand_stages=3');p.write_text(s)
