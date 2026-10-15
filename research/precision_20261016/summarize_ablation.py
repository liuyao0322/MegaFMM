import json,math
from pathlib import Path
rows=json.loads(Path('research/precision_20261016/ablation.json').read_text())['rows']
summary=[]
for dt in ['float16','bfloat16']:
 for pat in sorted({r['pattern'] for r in rows}):
  group=[r for r in rows if r['dtype']==dt and r['pattern']==pat]
  s=dict(dtype=dt,pattern=pat,count=len(group),stages={k:[min(r['stages'][k] for r in group),max(r['stages'][k] for r in group)] for k in group[0]['stages']},quantization_to_gate=[r['stages']['operand_cast_only']/r['gate'] for r in group],best_seven_cross_terms=[r['minimal_subset_by_count'][7]['ef'] for r in group],first_order=[r['first_order_exact'] for r in group])
  summary.append(s)
Path('research/precision_20261016/ablation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for s in summary:
 print(s['dtype'],s['pattern'],'cast',s['stages']['operand_cast_only'],'fp32_only',s['stages']['fp32_leaf_no_operand_cast'],'7cross',s['best_seven_cross_terms'])
