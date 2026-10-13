#!/usr/bin/env python3
"""Owner-run iteration39 report generator; actual raw archive drives numbers."""
from __future__ import annotations

import argparse
import json
import re
import types
from pathlib import Path

ROOT = Path('/share/home/liuyao/workspace/MegaFMM')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def link(path, label=None, line=None):
    suffix = '' if line is None else f':{line}'
    return f'[{label or Path(path).name}]({(ROOT / path).resolve()}{suffix})'


def us(row, field='solution_event_ms'):
    return float(row[field]) * 1000


def module():
    path = Path(__file__).with_name('psplit_bn16_measurements_generator_iter39.py')
    value = types.ModuleType('psplit_bn16_measurements_generator_iter39')
    value.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), value.__dict__)
    return value


def model_stdout(path):
    text = Path(path).read_text()
    start = text.find('{')
    require(start >= 0, 'No actual archived CPU model JSON: ' + str(path))
    result, _ = json.JSONDecoder().raw_decode(text[start:])
    return result


def generate(plan_path, identity_path, profile_path, measurements_path=None):
    raw = module()
    plan = json.loads(Path(plan_path).read_text())
    identity = json.loads(Path(identity_path).read_text())
    profile = json.loads(Path(profile_path).read_text())
    measurements_path = measurements_path or profile['full1024']['measurements_path']
    measurements = json.loads(Path(measurements_path).read_text())
    scope = measurements['measurement_scope']; exclusion = scope.get('exclusions')
    regenerated = raw.generate(plan_path, identity_path, exclusion['path'] if exclusion else None)
    for field, value in regenerated.items():
        require(measurements.get(field) == value, 'Actual raw/archive mismatch: '+field)
    require(profile['measurement_source_commit'] == identity['measurement_source_commit'], 'Profile source identity mismatch')
    require(profile['source_hashes'] == identity['frozen_sources'] and profile['binaries'] == identity['frozen_binaries'], 'Profile source/binary freeze mismatch')
    config, rows = plan['route_configurations'], measurements['rows']
    validation = profile.get('validation', {})
    def budget(variant):
        c=config.get(str(variant), {})
        if c.get('producer_quota') is None or len(c.get('consumer_quotas',[]))!=2: return '未声明配额'
        return '/'.join(map(str,[c['producer_quota'],*c['consumer_quotas']]))
    kinds={'42':'raw RS-A BN16','43':'SS BN16','39':'RS-A BK64参照（上轮已测最快）','40':'raw RS-A BN32对照','36':'SS BN32对照'}
    labels={('lt','1'):'Lt1 打包一次（完整多launch对照）'}
    for variant, kind in kinds.items():
        c=config.get(variant,{})
        labels['mega',variant]=f"Mega{variant} {kind} BM{c.get('BM','未记录')} BN{c.get('BN','未记录')} BK{c.get('BK','未记录')} {budget(variant)} min{c.get('launch_bounds_min_ctas','未记录')}"
    lines=['# 1024 BN16 raw RS-A / SS megakernel实测（iteration39）','']
    for candidate,baseline in ((42,39),(43,39),(42,40),(43,36),(42,43)):
        obs=[z for z in measurements['each_new_vs_retained_controls'] if (z['candidate_variant'],z['baseline_variant'])==(candidate,baseline)]
        if obs:
            parts=[]
            for mode in ('launch','graph'):
                own=[z for z in obs if z['mode']==mode]
                parts.append(('普通launch' if mode=='launch' else 'Graph')+'：'+'、'.join(f"{z['dtype'].upper()}耗时变化{-z['own_latency_decrease_percent']:+.3f}%" for z in own))
            lines += [f'Mega{candidate}相对同binary Mega{baseline}完整SW自身延迟：'+'；'.join(parts)+'。','']
        else:
            lines += [f'Mega{candidate}对Mega{baseline}实际配对未归档；没有用历史或其他binary补齐。','']
    lines += [f"本轮mandatory24；实际{measurements['row_count']}行、{measurements['timing_group_case_count']}条TIME、{measurements['fresh_gate_count']}次fresh CHECK；完整矩阵={'是' if scope['complete_full_matrix'] else '否'}。原始门限{measurements['fresh_gate_pass']} PASS /{measurements['fresh_gate_fail']} FAIL，nonfinite={measurements['nonfinite']}。",'']
    winner_text=[]
    for z in measurements['observed_native_sw_route_winners']:
        winner_text.append(f"{z['mode']}/{z['dtype']}为Mega{z['variant']}（{z['sw_us']:.6f}µs）")
    if winner_text:
        lines += ['本轮同binary已测native路线中，按完整SW自身延迟排序的最快结果：'+'；'.join(winner_text)+'。该排名不表示通过原始数值门限；完整多launch Lt1另列。','']
    lines += ['本报告分别记录完整调用性能、fresh gate、strict及结构测试。原始数值FAIL保留，Auto保持Dense；不宣布精度认证、策略提升或目标完成。','']
    if not scope['complete_full_matrix']:
        lines += ['完整预声明范围未完成；编译安全门限阻止的行没有速度或正确性结果。','## 实际缺失范围','','| route | dtype | mode | 分类 | 原因 | 门限证据 |','|---|---|---|---|---|---|']
        for z in exclusion['rows']:
            lines.append(f"| {z['algorithm']}{z['variant']} | {z['dtype']} | {z['mode']} | {z['classification']} | {z['reason']} | {link(z['evidence_path'])} `{z['evidence_sha256']}` |")
        lines += ['']
    lines += [f"测量源码checkpoint `{identity['measurement_source_commit']}`；bench SHA256 `{identity['frozen_binaries']['build/megafmm_bench']}`。八源、双binary及证据见{link(profile_path)}，展开数据见{link(measurements_path)}。",'',
        '## 测量协议','',
        '完整目标为新42 raw RS-A BN16、新43 SS BN16、保留39 RS-A BK64、匹配40 raw RS-A BN32、匹配36 SS BN32及Lt1 × BF16/FP16 × ordinary/Graph。每个新路线被fresh compiled门限阻止时，必须绑定该路线四个准确缺失键：单条受阻partial20、两条受阻partial16。旧结果不能代替本轮测量。',
        '每行20warmup、100calls/group、5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates、共同8GiB workspace cap。BF16/FP16输入、FP32输出。median和线性p10/p90来自五个组均值，CV为总体标准差/均值；quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优，没有Graph专用调优；Dense/SW按同行相同模式计时，比例为同行Dense event median / 完整SW event median。quick为20warmup、20calls/group、5groups、1fresh，只作筛选。','',
        '42采用raw RS-A BN16，43采用shared-A SS BN16。两者沿用原FP32 RN S/T DAG、最终onecast、whole-K七个FP32产品、固定U和RNalpha/activebetaFMA epilogue；实际source/geometry/quota/minCTAs/shared/drain/barrier/tail配置只取冻结plan与fresh compiled证据。CPU预测或静态min2 annotation不表示已测two-CTA residency。','',
        '最终route_configurations：','','```json',json.dumps(config,ensure_ascii=False,indent=2),'```','',
        '五个唯一自身延迟比较为42对39、42对40、42对43、43对39、43对36。BN、role配额、launch bounds及tail映射可同时改变，所以所有controlled_axis_claimed=false；CPU scalar bank一词/每bank不证明实际transaction grouping、bank stalls、occupancy或GPU速度。','']
    measured='/'.join('Mega'+str(v) for v in measurements['actually_measured_native_variants'])
    lines += [f'本轮实际测得的native {measured}每次读取当前原始A/B，重新构造变换，完成七个P、固定U及epilogue；完整调用为一个kernel，零native矩阵workspace，无host descriptor encode。Lt1为pack→七batch leaf GEMM→combine的完整多launch对照，不能代替单launch目标。',
        'Graph固定capture地址，replay读取当前A/B内容并重新变换；fresh检查更换内容。setup/tuning/capture不计稳态；event覆盖完整GPU工作，可能含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，再取五组median，不是独立同步单调用CPU wall。环境快照不认证组内clock。','']
    env=profile.get('environment',{})
    lines += ['环境及记录作用域：','','```json',json.dumps(env,ensure_ascii=False,indent=2),'```','']
    noisy=measurements['dense_cv_at_least_2_percent']
    if noisy:
        lines += ['## 同行Dense组均值的波动','','以下Dense总体CV达到2%；全部五组保留，没有剔除较慢组。分母波动不能单独证明代码收益。','']
        for z in noisy:
            k=z['row_key']; values=', '.join(f'{x:.6g}' for x in z['dense_event_group_means_us'])
            lines.append(f"{labels[k['algorithm'],k['variant']]} {k['mode']} {k['dtype']}：Dense CV={z['dense_cv']*100:.6f}%，SW CV={z['solution_cv']*100:.6f}%；stdout五个Dense event组均值µs=[{values}]；{link(z['path'],'CSV',z['line'])}。")
        lines += ['']
    for mode,title in (('launch','普通启动'),('graph','CUDA Graph')):
        lines += ['## '+title,'','| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |','|---|---|---:|---:|---:|---:|---:|---|---:|---|']
        for z in rows:
            row=z['csv_row']
            if row['mode']!=mode: continue
            times=[f"{us(row,p+'_event_ms'):.4f} [{us(row,p+'_p10_ms'):.4f},{us(row,p+'_p90_ms'):.4f}]" for p in ('dense','solution')]
            gate=('PASS finite' if row['numerical_pass']=='1' else 'FAIL finite') if row['nonfinite']=='0' else 'FAIL nonfinite'
            lines.append(f"| {labels[row['requested'],row['variant']]} | {row['dtype']} | {times[0]} | {times[1]} | {float(row['speedup']):.6f} | {float(row['dense_cv'])*100:.4f}/{float(row['solution_cv'])*100:.4f} | {us(row,'dense_wall_ms'):.5f}/{us(row,'solution_wall_ms'):.5f} | {gate} | {int(row['solution_workspace'])/1024**2:g} | {link(z['path'],'CSV行'+str(z['line']),z['line'])} |")
        lines += ['']
    lines += ['## 新42/43相对同binary对照的自身耗时','','| 模式 | dtype | 对照 | 新路线 | 对照 µs | 新路线 µs | 对照/新路线自身延迟比 | 新路线自身耗时变化 | 对照/新路线 batchwall µs/call |','|---|---|---|---|---:|---:|---:|---:|---:|']
    for z in measurements['each_new_vs_retained_controls']:
        lines.append(f"| {z['mode']} | {z['dtype']} | Mega{z['baseline_variant']} | Mega{z['candidate_variant']} | {z['baseline_us']:.6f} | {z['candidate_us']:.6f} | {z['baseline_over_candidate_own_latency']:.6f}× | {-z['own_latency_decrease_percent']:+.3f}% | {z['baseline_batch_wall_us']:.5f}/{z['candidate_batch_wall_us']:.5f} |")
    lines += ['',f"实际自身延迟配对{len(measurements['each_new_vs_retained_controls'])}个，未测配对{len(measurements['unavailable_comparisons'])}个；各axis配置、准确missing keys及pair contract见measurements。不重复反向添加42对43。缺失新路线不能用其他binary或历史结果补齐；收益判断使用SW自身延迟。",'',
        '## 编译资源与实际检查','','| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |','|---|---|---|---:|---:|---:|---:|---|---:|---|']
    for z in profile.get('compiled',[]):
        g=lambda k:z.get(k,'未记录')
        lines.append(f"| {g('variant')} | {g('dtype')} | {g('bm')}×{g('bn')} /{g('bk')} | {g('threads')} | {g('dynamic_shared_bytes')} | {g('effective_initial_registers_per_thread')} | {g('initial_owned_cta_register_pool')} | {budget(g('variant'))} | {g('required_role_register_pool')} | {g('STACK')}/{g('spill_loads')}/{g('spill_stores')} |")
    shared_records = [z for z in profile.get('compiled',[]) if 'static_shared_bytes' in z and 'resource_tool_reported_SHARED_bytes' in z]
    if shared_records:
        lines += ['', 'Driver function attribute的静态shared和resource tool的SHARED accounting分别记录，不互相代换：', '']
        for z in shared_records:
            lines.append(f"Mega{z['variant']} {z['dtype']}：Driver static_shared_bytes={z['static_shared_bytes']}，resource_tool_reported_SHARED_bytes={z['resource_tool_reported_SHARED_bytes']}；occupancy API predicted CTAs/SM={z.get('occupancy_API_predicted_active_CTAs_per_SM','未记录')}，measured_residency={z.get('measured_residency','未记录')}。")
        lines += ['这是实际function/cubin资源查询的静态预测，没有启动kernel测量residency；不作为实际two-CTA驻留、transactions、bank stalls或性能因果证明。', '']
    gate=identity['independent_compiled_gate']
    lines += ['',f"本轮identity绑定fresh compiled gate：status={gate['status']}，{link(gate['path'])}，SHA256 `{gate['sha256']}`。映射compiled input和原审文件分别绑定；原instances字段保持。源、object、bench/test全stream与实际resource合法性由独立review核验。",'',
        '初始与最终时序/模型作用域：','','```json',json.dumps({k:v for k,v in profile.get('prototype_provenance',{}).items() if k!='CPU_models'},ensure_ascii=False,indent=2),'```','',
        '初始pending/失败字段保留当时身份，最终配置只取fresh gate和冻结plan，不能追溯认证旧候选。','']
    cpu=validation.get('CPU_checks',{})
    for kind in ('DAG','stage'):
        if cpu.get(kind+'_path'):
            path=cpu[kind+'_path']; code=int(Path(path).with_suffix('.exit').read_text())
            lines += [f"CPU {kind}实际exit={code}，tests={cpu.get(kind+'_tests','未报')}，{link(path)}。",'']
    for name,z in validation.items():
        if name=='CPU_checks' or not name.startswith('CPU_') or not isinstance(z,dict) or not z.get('path'): continue
        value=model_stdout(z['path']); code=int(Path(z.get('exit_path',str(Path(z['path']).with_suffix('.exit')))).read_text())
        require(z.get('exit',code)==code,'Actual CPU exit/profile mismatch')
        lines += [f"CPU抽象/几何模型实际exit={code}，status={value.get('status','未报')}；{link(z['path'])}。冻结原件保护及writer剥离按真实runner核验，JSON PASS不是进程退出证明；该模型不认证CUDA transaction、liveness、配额fit、GPU精度或性能。",'']
    for name in ('default','selected42','selected43','disabled42_cli','disabled43_cli','disabled37_cli'):
        z=validation.get(name)
        if z: lines += [f"{name}：exit={z['exit']}，{z['CASE']} CASE /{z['PASS']} PASS；{link(z['path'])}。",'']
    lines += ['每个新variant的实际Graph/rebind/bitwise范围：','','```json',json.dumps(validation.get('new_by_variant',{}),ensure_ascii=False,indent=2),'```','',
        '42对43只在43 enabled且实际执行时属于N16 sameatom参照；43 blocked时42改对36，43始终对36，均明确属于N16-vs-N32 crossatom检查。严格bitwise/hostepilogue的失败如实保留，不改标为sameatom。原base K32与新增partial fixture的重复按冻结source计数，不能去重删样本。Mega37继续disabled；fallback PASS不认证native。','',
        '## 原始精度与sanitizer','']
    strict=validation.get('strict')
    if strict:
        lines += [f"本轮strict实际{strict['gates']} gates：{strict['pass']} PASS /{strict['fail']} FAIL /nonfinite {strict['nonfinite']}；{strict['native_gates']} native、{strict['Dense_range_fallback_gates']} Dense fallback。BF16/FP16原floor保持0.001/0.0001，门限3×Dense_eF+floor。",'']
        for variant,z in validation.get('strict_by_variant',{}).items():
            lines += [f"新Mega{variant}实际{z['gates']} gates、{z['pass']} PASS /{z['fail']} FAIL，nonfinite={z['nonfinite']}，exit={z['exit']}；{link(z['path'])}。",'']
    else: lines += ['本轮strict实际范围未归档，不能从源码域或预测计数认证执行。','']
    lines += ['历史strict按各source/bin身份保留；本轮full代码域为1680+每enabled新variant30（both1740/one1710/none1680），不表示该整个域已运行。原数值FAIL保留，AutoDense，目标仍active。','',
        '实际执行作用域：','','```json',json.dumps(validation.get('execution_scope'),ensure_ascii=False,indent=2),'```','']
    sans=profile.get('sanitizers',[])
    failed=[z for z in sans if any(z.get(k)!=0 for k in (('exit','errors','warnings','hazards') if z['tool']=='racecheck' else ('exit','errors')))]
    lines += [f"实际归档{len(sans)}项CUDA Compute Sanitizer，{len(failed)}项实测错误字段非零或未记录；memcheck leak bytes={[z.get('leaked_bytes') for z in sans if z['tool']=='memcheck']}。race/sync的leak字段为null；每项selected fixture multiset及variant归属由独立审计核验，不替被blocked路线或Mega37执行。",'',
        f"旧CUDA {len(profile.get('old_cuda_sha256',{}))}个，历史证据{len(profile.get('retained_old_evidence_sha256',{}))}绑定；历史onsite-only {len(profile.get('retained_onsite_only_paths',[]))}件按现场SHA保留，不冒称Git对象。受保护scratch原件与Git归档正式副本分别分类。统计、来源与最后Git绑定由独立数据审计核验。",'']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='results/psplit_bn161024_plan_iter39.json')
    parser.add_argument('--identity', default='results/psplit_bn16_measured_identity_iter39.json')
    parser.add_argument('--profile', required=True)
    parser.add_argument('--measurements')
    parser.add_argument('--output', help='Owner-selected formal output; default stdout')
    args = parser.parse_args()
    text = generate(args.plan,args.identity,args.profile,args.measurements)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text,end='')
