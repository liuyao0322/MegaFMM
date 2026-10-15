"""Independent complete-call geometry control, separate from retention comparison."""
import argparse,hashlib,json,re,statistics
from pathlib import Path
from analyze import analyze
ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--new',required=True);ap.add_argument('--graph',action='store_true');args=ap.parse_args()
h=Path(__file__).resolve().parent;repo=h.parents[2]
prefix=args.new+('_shape_graph_abba' if args.graph else '_shape_abba');runs=[];files=[]
for arm in ['a0','b0','b1','a1']:
 f=h/f'{prefix}_{arm}.log';d=analyze(f)
 assert len(d['checks'])==78 and all(x['pass']=='1' and x['nonfinite']=='0' and x['graph']==str(int(args.graph)) for x in d['checks'])
 inside=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in f.read_text().splitlines() if l.startswith('INSIDE_EQUAL ')]
 assert len(inside)==78 and all(x['errors']=='0' and x['elements']=='1024' for x in inside)
 p=f.with_suffix('.json');p.write_text(json.dumps(d,indent=2)+'\n');files.extend([f,p])
 runs.append(dict(arm=arm,candidate=args.old if arm[0]=='a' else args.new,summaries=d['summaries']))
summary=[]
for dt in ['bf16','fp16']:
 groups={label:[g['corrected_us'] for x in runs if x['candidate']==label for s in x['summaries'] if s['dtype']==dt for g in s['groups']] for label in [args.old,args.new]}
 old,new=(statistics.median(groups[x]) for x in [args.old,args.new])
 summary.append(dict(dtype=dt,old_candidate=args.old,new_candidate=args.new,old_group_median_us=old,new_group_median_us=new,relative_reduction=1-new/old))
p=h/f'{prefix}.json';p.write_text(json.dumps(dict(scope='four own fresh '+('Graph replay' if args.graph else 'ordinary')+' ABBA full-call shape-control processes, separate from retained-reference comparison; same78 deterministic numerical configurations, unlocked clocks, no confidence interval',correctness_checks=312,independent_numerical_configurations=78,equal_inside_exponents=319488,runs=runs,summary=summary),indent=2)+'\n')
files.extend([p,h/f'run_{prefix}.sh',h/f'{prefix}_driver.log',h/'analyze_shape_abba.py',h/'analyze.py'])
for label in [args.old,args.new]:files.extend([h/f'{label}_manifest.json',h/'bin'/f'compare_{label}'])
(h/f'{prefix}_manifest.json').write_text(json.dumps({str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(summary,indent=2))
