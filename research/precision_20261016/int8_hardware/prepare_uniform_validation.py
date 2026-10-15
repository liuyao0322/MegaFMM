"""Prepare candidate-specific fresh stress and scoped checks, no execution."""
import argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
h=Path(__file__).resolve().parent;label=args.label
assert label.startswith('p') and label[1:].isdigit() and (h/label).is_dir()
s=(h/'stress_p31.cu').read_text().replace('"p31/corrected.cuh"',f'"{label}/corrected.cuh"')
s=s.replace('32003+trial*337+s*191+pat*53',f'{args.seed}+trial*347+s*197+pat*59')
(h/f'stress_{label}.cu').write_text(s)
for source,target in [('build_stress_p31.sh',f'build_stress_{label}.sh'),('validate_p31.sh',f'validate_{label}.sh')]:
    s=(h/source).read_text().replace('p31',label)
    (h/target).write_text(s)
s=(h/'audit_p30_validation.py').read_text().replace('P30',label.upper()).replace('p30',label)
s=s.replace("EXTRA ',84,True)","EXTRA ',108,True)").replace("set(range(7))","set(range(9))").replace("EXTRA ',28,True)","EXTRA ',36,True)")
s=s.replace('28183+trial*331+s*181+pat*47',f'{args.seed}+trial*347+s*197+pat*59')
s=s.replace('28183+trial*331+stream*181+pattern*47',f'{args.seed}+trial*347+stream*197+pattern*59')
s=s.replace('independent_stress_configurations=84,independent_stress_calls=672','independent_stress_configurations=108,independent_stress_calls=864')
s=s.replace('sanitizer_cases_per_tool=28,sanitizer_calls_per_tool=56','sanitizer_cases_per_tool=36,sanitizer_calls_per_tool=72')
s=s.replace(f"research_retention='{label.upper()} ordinary full-call research reference; independent Graph comparison against P27 recorded separately'",f"research_retention='{label.upper()} ordinary full-call research candidate; independent Graph comparison against P27 recorded separately'")
(h/f'audit_{label}_validation.py').write_text(s)
print(label, 'prepared fresh 9-pattern validation seed',args.seed)
