#!/usr/bin/env python3
"""Owner-run iteration37 report generator; actual raw archive drives numbers."""
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
    path = Path(__file__).with_name('psplit_rsa_bk32_measurements_generator_iter37.py')
    value = types.ModuleType('psplit_rsa_bk32_measurements_generator_iter37')
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
    scope = measurements['measurement_scope']
    exclusion = scope.get('exclusions')
    regenerated = raw.generate(plan_path, identity_path, exclusion['path'] if exclusion else None)
    for field, value in regenerated.items():
        require(measurements.get(field) == value, 'Actual raw/archive mismatch: ' + field)
    require(profile['measurement_source_commit'] == identity['measurement_source_commit'], 'Profile source identity mismatch')
    require(profile['source_hashes'] == identity['frozen_sources'] and
            profile['binaries'] == identity['frozen_binaries'], 'Profile frozen source/binary mismatch')
    config = plan['route_configurations']
    labels = {('lt', '1'): 'Lt1 打包一次（多launch对照）'}
    def budget(variant):
        item = config.get(str(variant), {})
        if item.get('producer_quota') is None or len(item.get('consumer_quotas', [])) != 2:
            return '未声明role配额'
        return '/'.join(map(str, [item['producer_quota'], *item['consumer_quotas']]))
    for variant, kind in (('40', 'RS-A BK32'), ('39', 'RS-A BK64 reference, prior best'), ('36', 'SS BK32 matched')):
        item = config.get(variant, {})
        labels['mega', variant] = f"Mega{variant} {kind} BK{item.get('BK', '未记录')} {budget(variant)}"
        if item.get('launch_bounds_min_ctas') is not None:
            labels['mega', variant] += f" min{item['launch_bounds_min_ctas']}"
    rows = measurements['rows']
    measured_native = '/'.join('Mega'+str(v) for v in sorted({int(z['csv_row']['variant'])
        for z in rows if z['csv_row']['requested'] == 'mega'}))
    row_pass = sum(z['csv_row']['numerical_pass'] == '1' for z in rows)
    validation = profile.get('validation', {})
    strict = validation.get('strict')
    summary = []
    for v in measurements['same_binary_megakernel_comparison']:
        if v['mode'] == 'launch':
            summary.append(f"{v['dtype'].upper()}新Mega40完整调用{v['candidate_us']:.6f}µs，"
                f"同行cuBLASLt {v['candidate_dense_us']:.6f}µs，Dense/SW {v['candidate_same_row_dense_over_sw']:.6f}×；"
                f"相对同binary保留RS-A BK64参照 Mega39自身耗时变化{-v['own_latency_decrease_percent']:+.3f}%")
    prior39_comparison = measurements.get('same_binary_megakernel_comparison', [])
    lead = []
    for mode in ('launch', 'graph'):
        observed = [entry for entry in prior39_comparison if entry['mode'] == mode]
        if observed:
            lead.append(('普通launch' if mode == 'launch' else 'Graph') + '下' + '、'.join(
                f"{entry['dtype'].upper()}耗时变化{-entry['own_latency_decrease_percent']:+.3f}%" for entry in observed))
    lead_text = '新Mega40相对同binary保留的RS-A BK64参照 Mega39（上轮已测最快）：' + '；'.join(lead) + '。' if lead else '新40与39的实际配对尚未归档。'
    matched = measurements.get('new_vs_ss_bk32_comparison', [])
    matched_lead = []
    for mode in ('launch', 'graph'):
        observed = [entry for entry in matched if entry['mode'] == mode]
        if observed:
            matched_lead.append(('普通launch' if mode == 'launch' else 'Graph') + '下' + '、'.join(
                f"{entry['dtype'].upper()}耗时变化{-entry['own_latency_decrease_percent']:+.3f}%" for entry in observed))
    matched_text = '新Mega40相对同binary matched BK32 SS Mega36：' + '；'.join(matched_lead) + '。' if matched_lead else '新40与matched36的实际配对尚未归档。'
    lines = ['# 1024 RS-A BK32 megakernel实测（iteration37）', '',
        lead_text, '', matched_text, '', '。'.join(summary) + '。' if summary else '新Mega40没有实际可用的 ordinary 测量配对。', '',
        f"本轮mandatory16；实际{measurements['row_count']}行、{measurements['timing_group_case_count']}条TIME、"
        f"{measurements['fresh_gate_count']}个fresh CHECK；完整矩阵={'是' if scope['complete_full_matrix'] else '否'}。"
        f"原始门限{measurements['fresh_gate_pass']} PASS /{measurements['fresh_gate_fail']} FAIL，"
        f"nonfinite={measurements['nonfinite']}，实际记录行{row_pass}/{len(rows)}通过。", '']
    if not scope['complete_full_matrix']:
        lines += ['完整预声明测量范围尚未完成；编译阻止的行没有速度或正确性结果。', '']
    if measurements['fresh_gate_fail'] or measurements['nonfinite'] or (strict and (strict['fail'] or strict['nonfinite'])):
        lines += ['实际原始数值门限存在FAIL或nonfinite；性能与结构检查不构成原始精度认证。', '']
    lines += ['完整调用性能、fresh gate和strict执行范围分别报告；本报告不宣布目标完成或提升Auto策略。', '']
    if scope['missing_rows']:
        lines += ['## 实际缺失范围', '', '| route | dtype | mode | 分类 | 原因 | 门限证据 |', '|---|---|---|---|---|---|']
        for item in exclusion['rows']:
            lines.append(f"| {item['algorithm']}{item['variant']} | {item['dtype']} | {item['mode']} | "
                f"{item['classification']} | {item['reason']} | {link(item['evidence_path'])} `{item['evidence_sha256']}` |")
        lines += ['']
    lines += [f"测量源码checkpoint `{identity['measurement_source_commit']}`；bench SHA256 "
        f"`{identity['frozen_binaries']['build/megafmm_bench']}`。七源、双binary与原始证据见{link(profile_path)}，"
        f"展开测量见{link(measurements_path)}。", '', '## 测量协议', '',
        '完整目标为新40 RS-A BK32/保留39 RS-A BK64（上轮已测最快）/匹配36 SS BK32/Lt1 × BF16/FP16 × ordinary/Graph。'
        '若新40编译安全门限阻止，则只有显式绑定最新gate的四个缺失key才允许partial12；旧结果不能替代本轮测量。'
        '每行20warmup、100calls/group、'
        '5groups、3fresh检查、rotate_inputs=1；32 heuristic candidates和共同8GiB workspace cap。'
        'BF16/FP16输入、FP32输出。median与线性p10/p90来自五个组均值，CV使用总体标准差/均值。'
        '每行cuBLASLt独立调优，调优发生在capture前，没有Graph专用调优；Dense和SW按同行相同模式计时。'
        'Dense/SW使用同行Dense event median / 完整SW event median。'
        'quick仅作筛选：20warmup、20calls/group、5groups、1fresh，不能替代这里的mandatory full测量。', '',
        '新40采用RS-A路线，两个独立consumer WG按whole-K计算P3+4。最终几何、名称、role配额、launch bounds、barrier与shared配置见下列冻结plan记录；'
        '初始CPU模型和候选预算只证明各自归档的抽象模型范围，不能充当最终编译fit或硬件执行证据。', '',
        '最终route_configurations：', '', '```json', json.dumps(config, ensure_ascii=False, indent=2), '```', '',
        'RS-A BK32 40对SS BK32 36，以及RS-A BK32 40对RS-A BK64 39，是两个预声明比较轴。仅共享BK或BN配置不能自动证明受控因果关系；'
        '是否只改变A操作数路径由独立源码与compiled scope审核。所有性能数字只取本轮实际完整调用。'
        'native资格、partial-K回退与sameatom reference范围见本轮实际测试coverage/profile；fallback PASS不认证native。', '',
        f'本轮实际测得的native {measured_native} 每次读取当前原始A/B，重新构造输入变换并完成七个P、固定U和epilogue，'
        '完整调用为一个kernel、零native矩阵workspace、无host descriptor encode。'
        'Lt1是pack→七batch leaf GEMM→combine的完整多launch控制，不能替代单launch目标。', '',
        'Graph捕获固定地址，replay读取当前A/B内容并重做变换；三次fresh检查更换内容。setup/tuning/capture不计入稳态。'
        'event覆盖完整GPU工作，可能包含提交空隙。batchwall为每组100调用含批末同步的host wall除以100，'
        '再取五组median；不是独立同步单调用CPU wall。clock设置按本轮环境记录，环境快照不认证组内clock。', '']
    environment = profile.get('environment', {})
    if environment:
        lines += [f"环境记录：GPU={environment.get('GPU', '见profile')}，SMs={environment.get('SMs', '见profile')}，"
            f"host={environment.get('host', '见profile')}，CUDA={environment.get('application_cuda', '见profile')}，"
            f"CUTLASS={environment.get('CUTLASS', '见profile')}；记录作用域见profile。", '']
    noisy = measurements['dense_cv_at_least_2_percent']
    if noisy:
        lines += ['## 同行Dense组均值的波动', '',
            '下列行的Dense总体CV达到2%。全部五组保留，median、p10/p90和CV均使用这些组；'
            '没有剔除较慢组。该波动会改变同行Dense/SW分母，不能据加速比变化单独认定优化收益。', '']
        for entry in noisy:
            k = entry['row_key']
            values = ', '.join(f'{v:.6g}' for v in entry['dense_event_group_means_us'])
            lines.append(f"{labels[k['algorithm'], k['variant']]} {k['mode']} {k['dtype']}："
                f"Dense CV={entry['dense_cv']*100:.6f}%，SW CV={entry['solution_cv']*100:.6f}%；"
                f"stdout五个Dense event组均值µs=[{values}]；{link(entry['path'], 'CSV', entry['line'])}。")
        lines += ['']
    for mode, title in (('launch', '普通启动'), ('graph', 'CUDA Graph')):
        lines += ['## '+title, '',
            '| 路线 | dtype | Dense µs [p10,p90] | 完整SW µs [p10,p90] | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs/call | Gate | SW WS MiB | 原始CSV |',
            '|---|---|---:|---:|---:|---:|---:|---|---:|---|']
        for entry in rows:
            row = entry['csv_row']
            if row['mode'] != mode:
                continue
            times = [f"{us(row,p+'_event_ms'):.4f} [{us(row,p+'_p10_ms'):.4f},{us(row,p+'_p90_ms'):.4f}]" for p in ('dense','solution')]
            status = ('PASS finite' if row['numerical_pass']=='1' else 'FAIL finite') if row['nonfinite']=='0' else 'FAIL nonfinite'
            lines.append(f"| {labels[row['requested'],row['variant']]} | {row['dtype']} | {times[0]} | {times[1]} | "
                f"{float(row['speedup']):.6f} | {float(row['dense_cv'])*100:.4f}/{float(row['solution_cv'])*100:.4f} | "
                f"{us(row,'dense_wall_ms'):.5f}/{us(row,'solution_wall_ms'):.5f} | {status} | "
                f"{int(row['solution_workspace'])/1024**2:g} | {link(entry['path'],'CSV行'+str(entry['line']),entry['line'])} |")
        lines += ['']
    lines += ['## 新40相对保留RS-A BK64参照39及匹配SS BK32对照36的自身耗时', '',
        '| 模式 | dtype | 对照 | 对照 µs | 新40 µs | 对照/40自身延迟比 | 40自身耗时变化 | 对照/40 batchwall µs/call |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for v in measurements['each_new_vs_retained_controls']:
        lines.append(f"| {v['mode']} | {v['dtype']} | Mega{v['baseline_variant']} | {v['baseline_us']:.6f} | "
            f"{v['candidate_us']:.6f} | {v['baseline_over_candidate_own_latency']:.6f}× | "
            f"{-v['own_latency_decrease_percent']:+.3f}% | {v['baseline_batch_wall_us']:.5f}/{v['candidate_batch_wall_us']:.5f} |")
    lines += ['', '40对39比较保留的RS-A BK64参照（上轮已测最快）；40对36比较匹配的SS BK32路线。'
        f"RS_A_vs_SS_BK32_axis_comparison实际保留{len(measurements['RS_A_vs_SS_BK32_axis_comparison'])}个轴metadata记录，"
        f"RS_A_BK_axis_comparison实际保留{len(measurements['RS_A_BK_axis_comparison'])}个轴metadata记录；"
        '两个轴的controlled_axis_claimed默认false，不重复添加性能表行。'
        '本生成器不单独归因动态occupancy、stall或traffic；收益判断使用SW自身耗时。缺失的新40不能由历史结果或其他binary补齐。', '']
    if measurements['unavailable_comparisons']:
        lines += [f"实际缺失{len(measurements['unavailable_comparisons'])}个40对照配对；准确missing keys见measurements，"
            '编译阻止原因见measurement_scope/exclusions。', '']
    lines += ['## 编译资源与实际检查', '',
        '| variant | dtype | BM×BN / BK | threads | dynamic shared bytes | ELF REG/thread | initial CTA pool | 冻结配额 | role所需pool | stack/spill load/store |',
        '|---|---|---|---:|---:|---:|---:|---|---:|---|']
    for entry in profile.get('compiled', []):
        get = lambda k: entry.get(k,'未记录')
        lines.append(f"| {get('variant')} | {get('dtype')} | {get('BM')}×{get('BN')} /{get('BK')} | {get('threads')} | "
            f"{get('dynamic_launch_shared_bytes')} | {get('elf_registers')} | {get('initial_cta_owned_register_pool')} | "
            f"{budget(get('variant'))} | {get('required_role_pool')} | {get('ptxas_stack_bytes')}/{get('ptxas_spill_load_bytes')}/{get('ptxas_spill_store_bytes')} |")
    lines += ['', '动态shared、ELF资源和spill只引用当前实际compiled metadata。候选预算或CPU预测不是actualfit；'
        '静态资源不代表实测occupancy、traffic或物理floor。最终对象和linked binary关系见独立编译审查。', '']
    gate = identity.get('independent_compiled_gate', {})
    if gate:
        lines += [f"本轮measured identity绑定的fresh compiled gate：status={gate.get('status','未记录')}，"
            f"{link(gate['path'])}，SHA256 `{gate.get('sha256','未记录')}`。"
            '生成器仅检查绑定；CAN_LAUNCH内容及source/object/linked-bin时序由独立审计核验。', '']
    provenance = profile.get('prototype_provenance', {})
    if provenance:
        lines += ['初始与最终证据时序/模型作用域按本轮实际metadata保留：', '', '```json',
                  json.dumps({k:v for k,v in provenance.items() if k != 'CPU_models'}, ensure_ascii=False, indent=2), '```', '',
                  '初始pending字段保持其初始身份，不因后续final compiled PASS而追溯认证；最终配置只取冻结plan。', '']
    else:
        lines += ['初始与最终source/build/gate时序尚未归档，不把最终compiled状态套给初始模型或初始候选。', '']
    cpu = validation.get('CPU_checks', {})
    for kind in ('DAG','stage'):
        path = cpu.get(kind+'_path')
        if path:
            text = Path(path).read_text()
            code = int(Path(path).with_suffix('.exit').read_text())
            counts = '/'.join(re.findall(r'Ran (\d+) tests', text)) or '原件未报'
            lines += [f"CPU {kind}实际归档exit={code}，tests={counts}，{link(path)}。", '']
    for name, record in validation.items():
        if name == 'CPU_checks' or not name.startswith('CPU_') or not isinstance(record,dict) or not record.get('path'):
            continue
        value = model_stdout(record['path'])
        exit_path = record.get('exit_path') or str(Path(record['path']).with_suffix('.exit'))
        code = int(Path(exit_path).read_text())
        require(record.get('exit',code)==code, 'Profile/actual CPU exit mismatch')
        counters = ', '.join(f'{k}={v}' for k,v in value.items() if isinstance(v,(int,float)) and not isinstance(v,bool))
        details = ('，' + counters) if counters else ''
        lines += [f"CPU几何与抽象模型实际归档exit={code}，status={value.get('status','未报')}{details}；{link(record['path'])}。"
            '只检查相应抽象/源码模型范围，不认证GPU memory-order、forward-progress或最终修复配额的硬件ownership。'
            '原RS-A九件保护与old main是否调用按实际runner/stdout证据核查，不能由JSON status推断真实进程exit。', '']
    for name in ('default','selected40','disabled40_cli','disabled37_cli'):
        record = validation.get(name)
        if record:
            lines.append(f"{name}：exit={record.get('exit',record.get('exit_code','未记录'))}，"
                f"{record.get('CASE',record.get('CASE_lines','未记录'))} CASE /{record.get('PASS',record.get('PASS_lines','未记录'))} PASS；{link(record['path'])}。")
    lines += ['', f"新Graph configs/replays={validation.get('new_graph_configurations','未记录')}/"
        f"{validation.get('new_graph_fresh_replays','未记录')}；全部Graph="
        f"{validation.get('total_graph_configurations','未记录')}/{validation.get('total_graph_fresh_replays','未记录')}。"
        'Mega37继续disabled；结构PASS不能替代原始FP64数值认证。', '', '## 原始精度与sanitizer', '']
    if strict:
        lines += [f"本轮strict实际{strict['gates']} gates：{strict['pass']} PASS /{strict['fail']} FAIL /"
            f"nonfinite {strict['nonfinite']}；{strict['native_gates']} native、{strict['Dense_range_fallback_gates']} Dense fallback。"
            'BF16/FP16原floor保持0.001/0.0001，门限保持3×Dense_eF+floor；回退PASS不认证native。', '']
    else:
        lines += ['本轮strict实际执行范围未归档，不能用源码域计数代替执行结果。', '']
    execution_scope = validation.get('execution_scope', {})
    lines += ['历史strict按各自source/bin身份保留；本轮执行计数只来自实际终止的stdout/exit，源码域计数不能作为执行结果。'
        '原始数值FAIL保留，Auto保持Dense，目标未宣布完成。', '']
    if execution_scope:
        lines += ['本轮实际执行作用域：', '', '```json', json.dumps(execution_scope, ensure_ascii=False, indent=2), '```', '']
    sanitizers = profile.get('sanitizers', [])
    failed = [z for z in sanitizers if any(z.get(k)!=0 for k in
        (('exit','errors','warnings','hazards') if z.get('tool')=='racecheck' else ('exit','errors')))]
    lines += [f"实际归档{len(sanitizers)}项CUDA Compute Sanitizer，其中{len(failed)}项已测作用域的exit/errors或race hazards/warnings非零或未记录；"
        f"memcheck leak bytes={[z.get('leaked_bytes') for z in sanitizers if z['tool']=='memcheck']}。"
        'race/synccheck未测泄漏字段保持null，实际summary见profile；没有替Mega37执行sanitizer。', '',
        f"旧CUDA保留{len(profile.get('old_cuda_sha256',{}))}个，历史证据{len(profile.get('retained_old_evidence_sha256',{}))}个绑定，"
        f"onsite-only {len(profile.get('retained_onsite_only_paths',[]))}个按现场SHA保留，不声称成为Git对象。"
        '本生成器填报实际原件；统计、来源、compiled gate及最终Git绑定由独立审计核验。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='results/psplit_rsa_bk321024_plan_iter37.json')
    parser.add_argument('--identity', default='results/psplit_rsa_bk32_measured_identity_iter37.json')
    parser.add_argument('--profile', required=True)
    parser.add_argument('--measurements')
    parser.add_argument('--output', help='Owner-selected formal output; default stdout')
    args = parser.parse_args()
    text = generate(args.plan,args.identity,args.profile,args.measurements)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text,end='')
