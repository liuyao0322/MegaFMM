"""Summarize complete-call paired observations, keeping every arm and group."""
import json
import re
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def parse(path):
    text=path.read_text()
    checks=[]
    for line in text.splitlines():
        if line.startswith('CHECK '):
            pairs=dict(re.findall(r'(\w+)=([^ ]+)',line))
            checks.append(dict(dtype=line.split()[1],**pairs))
    samples=[]
    for dt,graph,group,arm,method,us in re.findall(r'TIME (\w+) graph=(\d+) group=(\d+) arm=(\d+) method=(\d+) us=([\d.]+)',text):
        samples.append(dict(dtype=dt,graph=int(graph),group=int(group),arm=int(arm),method=int(method),us=float(us)))
    assert len(checks)==36 and len(samples)==120,(path,len(checks),len(samples))
    assert all(c['pass']=='1' for c in checks),path
    groups=[]
    for dt in ['bf16','fp16']:
        for g in range(5):
            values={m:statistics.mean(v['us'] for v in samples if v['dtype']==dt and v['group']==g and v['method']==m) for m in range(3)}
            assert all(len([v for v in samples if v['dtype']==dt and v['group']==g and v['method']==m])==4 for m in range(3))
            groups.append(dict(dtype=dt,graph=samples[0]['graph'],group=g,
                               dense_us=values[0],sw_us=values[1],direct_us=values[2],
                               speedup_vs_sw=values[1]/values[2],speedup_vs_dense=values[0]/values[2]))
    summary=[]
    for dt in ['bf16','fp16']:
        ds=[g for g in groups if g['dtype']==dt]
        summary.append(dict(dtype=dt,graph=samples[0]['graph'],
            **{k:statistics.median(v[k] for v in ds) for k in ['dense_us','sw_us','direct_us','speedup_vs_sw','speedup_vs_dense']},
            min_speedup_vs_sw=min(v['speedup_vs_sw'] for v in ds),
            min_speedup_vs_dense=min(v['speedup_vs_dense'] for v in ds),
            max_ef=max(float(c['direct_ef']) for c in checks if c['dtype']==dt)))
    return dict(file=path.name,checks=checks,samples=samples,groups=groups,summary=summary)


def main():
    paths=[ROOT/'direct_launch.log',ROOT/'direct_b64_w0_launch.log',ROOT/'direct_b64_w0_graph.log',ROOT/'direct_b128_w1_launch.log']
    paths += sorted(ROOT.glob('acceptance_*_*.log'))
    if (ROOT/'direct_b128_w0_launch.log').exists():paths.append(ROOT/'direct_b128_w0_launch.log')
    records=[parse(p) for p in paths]
    aggregate=[]
    for graph in (0,1):
        for dt in ('bf16','fp16'):
            groups=[g for r in records if r['file'].startswith('acceptance_')
                    for g in r['groups'] if g['graph']==graph and g['dtype']==dt]
            assert len(groups)==15
            aggregate.append(dict(dtype=dt,graph=graph,groups=len(groups),
                **{k:statistics.median(g[k] for g in groups) for k in ['dense_us','sw_us','direct_us','speedup_vs_sw','speedup_vs_dense']},
                min_speedup_vs_sw=min(g['speedup_vs_sw'] for g in groups)))
    out=dict(scope='Original-input FP64 checks and complete-call A/B/B/A timings. method0=tuned cuBLASLt; method1=current SW; method2=direct-input dense prototype. Unlocked GPU clocks; no cross-process statistical confidence claim. direct_graph.log excluded: scheduling overlap with scaling pilot.',records=records,acceptance_aggregate=aggregate)
    (ROOT/'direct_results.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in records:
        print(r['file'],json.dumps(r['summary']))
    print('ACCEPTANCE_AGGREGATE',json.dumps(aggregate))


if __name__=='__main__':main()
