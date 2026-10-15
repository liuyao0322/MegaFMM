from pathlib import Path
import json,shutil,subprocess
root=Path.cwd();src=root/'results/mnk369_acceptance';dest=root/'trajectory/20261016_mnk-validation-369/acceptance'
subprocess.run(['python3','trajectory/20261016_mnk-validation-369/audit_acceptance.py'],check=True)
a=json.loads((src/'independent_audit.json').read_text());r=json.loads((src/'registry.json').read_text())
assert a['audit_pass'] and a['precision_rows_passed']==84
assert a['stable_wins']==r['stable_wins']
dest.mkdir(exist_ok=True)
for name in ['registry.json','independent_audit.json']+[f'run{i}.{ext}' for i in range(3) for ext in ('csv','log')]:shutil.copy2(src/name,dest/name)
shutil.copy2('results/mnk369_acceptance_driver.log',dest/'driver.log')
shutil.copy2('results/archive_acceptance369.py',dest/'archive_acceptance.py')
wins=r['stable_wins'];unresolved=28-wins
report=f'''iter369 three-sweep acceptance completed: {wins}/28 stable custom wins, {unresolved} unresolved.
All84 CSV precision rows and252 fresh original-input FP64 checks pass.
All420 raw paired timing groups independently audited, full frozen-source and binary hashes unchanged.
Binary SHA256: {r['binary_sha256']}
A stable win requires a custom route in all3 independent sweeps and all15 paired groups >=1.03.
Fallback timing advantages do not count. Historical353 certification16/28 applies only to353source.
'''
for e in a['entries']:
 report+=f"{e['m']}x{e['n']}x{e['k']} {e['dtype']}: {'stable' if e['stable'] else 'unresolved'}, min15={e['min_paired_speedup']:.6f}, median15={e['median_paired_speedup']:.6f}; {','.join(e['selections'])}\n"
(dest/'REPORT.txt').write_text(report)
p=root/'trajectory/20261016_mnk-validation-369/validation.json';v=json.loads(p.read_text())
v['latest_complete_independent_certification']={'iteration':369,'stable_wins':wins,'total':28}
v['independent_certification_status']='completed, original precision84/84; raw-log and frozen-source audit PASS'
v['independent_certification_archive']='trajectory/20261016_mnk-validation-369/acceptance/'
v['historical353_stable_wins']=16
v['current369_stable_count_change_vs353']=wins-16
p.write_text(json.dumps(v,indent=2)+'\n')
p=root/'README.md';s=p.read_text()
old='最新完整独立三轮认证仍为iter353的16/28；新的112候选版本需要冻结后重新认证。'
assert old in s
new=f'''iter369 的112候选版本已完成三轮独立全套认证：**{wins}/28 稳定胜出，{unresolved} 个组合未解决**。
84/84条结果均通过原始输入 FP64 精度门槛，252次新输入检查和420组配对计时已独立核验，
源码及二进制指纹保持一致。2048³ FP16 虽然中位加速仍为正，但第二轮一组仅约1.001×，
因此未通过全部15组均至少1.03×的稳定门槛；该失败数据完整保留。
历史iter353的16/28仅认证其对应版本，不能沿用到当前候选池。
完整证据见 [iter369三轮认证与原始计时](trajectory/20261016_mnk-validation-369/acceptance/registry.json)
及 [独立核验](trajectory/20261016_mnk-validation-369/acceptance/independent_audit.json)。'''
s=s.replace(old,new)
p.write_text(s)
print(report)
