"""Keep every complete-call timing arm; report paired group statistics."""
import argparse
import json
import re
import statistics
from pathlib import Path

def analyze(path):
    lines=path.read_text().splitlines()
    checks=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in lines if l.startswith('CHECK ')]
    samples=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in lines if l.startswith('TIME ')]
    summaries=[]
    for dt in ('bf16','fp16'):
        groups=[]
        for g in range(5):
            times={m:statistics.mean(float(x['us']) for x in samples if x['dtype']==dt and int(x['group'])==g and int(x['method'])==m) for m in range(4)}
            assert all(sum(x['dtype']==dt and int(x['group'])==g and int(x['method'])==m for x in samples)==4 for m in range(4))
            groups.append(dict(group=g,dense_us=times[0],sw_us=times[1],direct_us=times[2],corrected_us=times[3],speedup_vs_sw=times[1]/times[3],speedup_vs_direct=times[2]/times[3]))
        cc=[x for x in checks if x['dtype']==dt]
        assert cc
        summaries.append(dict(dtype=dt,balance=int(cc[0]['balance']),graph=int(cc[0]['graph']),checks=len(cc),failures=sum(x['pass']!='1' for x in cc),max_ef=max(float(x['corrected_ef']) for x in cc),
            **{k:statistics.median(g[k] for g in groups) for k in groups[0] if k!='group'},groups=groups))
    return dict(file=str(path),checks=checks,samples=samples,summaries=summaries,components=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in lines if l.startswith('COMPONENT ')])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('log',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args();result=analyze(args.log)
    output=args.output or args.log.with_suffix('.json');output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summaries'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
