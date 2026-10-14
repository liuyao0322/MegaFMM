#!/usr/bin/env python3
"""Owner-run current47 report; source/raw counts and original resource fields."""
from pathlib import Path
import argparse, hashlib, json, types
ROOT=Path('/share/home/liuyao/workspace/MegaFMM')
def require(v,m):
    if not v:raise ValueError(m)
def link(p,label=None,line=None):
    return f"[{label or Path(p).name}]({(ROOT/p).resolve()}"+(f':{line}' if line is not None else '')+')'
def module(p):
    p=Path(p);m=types.ModuleType(p.stem);m.__file__=str(p);exec(compile(p.read_text(),str(p),'exec'),m.__dict__);return m
def us(row,k='solution_event_ms'):return float(row[k])*1000
def resource_chronology(gate, descriptor):
    lines=['本轮原编译门限：'+link(descriptor['path'])+'；原字段与编译诊断按实际身份分别保留，资源放行不认证数值或性能。','']
    for key,record in gate.items():
        if ('revision' in key or 'initial' in key or 'chronology' in key) and key not in ('instances','disabled_instances'):
            lines += [key+'：','','\x60\x60\x60json',json.dumps(record,ensure_ascii=False,indent=2),'\x60\x60\x60','']
    for c in gate.get('instances',[]):
        roles=c.get('role_CFG',[])
        role_max={x['role']:x['gpr_highest_including_implicit'] for x in roles if 'role' in x and 'gpr_highest_including_implicit' in x}
        flags={k:v for k,v in c.items() if k.startswith(('all_P_clear','legal_initial_owned','initialization_gate'))}
        lifecycle={k:v for k,v in c.items() if any(x in k for x in ('physical_windows','pending_','lifetime','inplace','TMA','tma','whole_stage'))}
        lines += [f"新Mega{c['variant']} {c['dtype']}原role最高GPR={json.dumps(role_max,sort_keys=True)}，原初始化标志={json.dumps(flags,sort_keys=True)}。这些字段只按本轮fresh门限解释，不从49/50或旧SS控制推断实际lifetime。",'']
        if lifecycle:lines += ['\x60\x60\x60json',json.dumps(lifecycle,ensure_ascii=False,indent=2),'\x60\x60\x60','']
    return lines
def generate(plan_path,identity_path,profile_path,metrics_path,measurement_generator_path):
    plan=json.loads(Path(plan_path).read_text());identity=json.loads(Path(identity_path).read_text())
    profile=json.loads(Path(profile_path).read_text());metrics=json.loads(Path(metrics_path).read_text())
    exclusion=(metrics['measurement_scope'].get('exclusions') or {}).get('path')
    require(module(measurement_generator_path).generate(plan_path,identity_path,exclusion)==metrics,'Metrics must exactly match current frozen raw regeneration')
    require(profile['source_hashes']==identity['frozen_sources'] and profile['binaries']==identity['frozen_binaries'],'Report source/binary freeze')
    scope=metrics['measurement_scope'];configs=plan['route_configurations'];validation=profile['validation'];rows=metrics['rows']
    def budget(v):
        c=configs[str(v)];return '/'.join(map(str,[c['producer_quota'],*c['consumer_quotas']]))
    pairs=metrics['same_binary_own_latency_comparison']
    lines=['# 1024 TMA原地S/T BN32与BN16实测（iteration47）','']
    for pair in metrics['own_latency_pair_contract']:
        baseline,candidate=pair['baseline_variant'],pair['candidate_variant']
        own=[p for p in pairs if (p['candidate_variant'],p['baseline_variant'])==(candidate,baseline)]
        values=[]
        for p in own:
            change=(p['candidate_us']/p['baseline_us']-1)*100
            values.append(f"{p['mode']} {p['dtype']} {p['candidate_us']:.6f}µs对{p['baseline_us']:.6f}µs，"+(f'延长{change:.3f}%' if change>=0 else f'减少{-change:.3f}%'))
        lines += [f'Mega{candidate}对同binary Mega{baseline}自身完整调用：'+('；'.join(values) if values else 'fresh compiled blocked或没有实际完整配对')+'。','']
    for w in metrics['observed_native_sw_route_winners']:
        lines.append(f"本轮{w['mode']} {w['dtype']}已测native中最快为Mega{w['variant']}，{w['sw_us']:.6f}µs，原数值门限{'PASS' if w['numerical_pass'] else 'FAIL'}。")
    lines += ['',f"实际行数{metrics['row_count']}/预声明{scope['expected_row_count']}，complete={scope['complete_full_matrix']}；fresh PASS/FAIL={metrics['fresh_gate_pass']}/{metrics['fresh_gate_fail']}，nonfinite={metrics['nonfinite']}。",'',
        '结构、sanitizer与CPU模型通过不认证原FP64数值门限。原floor与Auto选择Dense策略保持，优化目标尚未完成。','',
        f"测量源commit {identity['measurement_source_commit']}；{link(profile_path)}，{link(metrics_path)}。",'',
        '## 测量协议','',
        '51是新BN32原地S/T，52是新BN16原地S/T，均BM64/BK32；45为BN32/BK64选择性TMA rawA，50为BN16/BK64 K32窗口，39为RS-A BK64，Lt1为完整多launch。两条新路线各自fresh compiled blocked时准确保留四个缺失键，24行预声明相应剩20或16行，任何缺失不使用历史数据补齐。','',
        f"warmup={plan['warmup']}，calls/group={plan['trials']}，groups={plan['groups']}，fresh/row={plan['correct_trials']}，rotate_inputs={plan['rotate_inputs']}，heuristic candidates={plan['heuristic_candidates']}，共同workspace cap={plan['workspace_limit']}bytes。",'',
        'event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。','',
        '本轮实际enabled新51/52在每次普通完整调用中从当前A/B指针及lda/ldb编码两个tensor map，A/B原始输入由TMA搬入shared，并在原地构造S/T；完整编码与单个kernel均计时。Graph在capture时编码两个map，固定capture地址读取fresh内容。45/50普通调用编码一个被消费的A map，B仍SIMT；39没有host descriptor encode。各路线几何、quota、shared及生命周期按本轮实际源和fresh gate读取。所有实际native包含原RN变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，零矩阵workspace；Lt1计完整多launch。Graph差值不能单独当作encode成本或纯TMA收益。51 rawU参考36、52 rawU参考43的atom与共享native域只按实际source-bound coverage表述，reference runs与candidate runs分别计数；bit fixture与独立host epilogue不构成通用FP64精度认证。minCTA及API预测均不认证实际residency，六个比较的controlled_axis_claimed=false。','',
        '本轮原地S/T的一次构造范围是每个CTA、当前bank stage：每个pair task对其负责坐标的最终S/T plane各写一次；同一输入区域在不同CTA中仍可能重复搬入和变换。这是CTA内stage复用，不能称作H12的全矩阵global S/T一次构造。','',
        '巨大stride记录只按本轮实际host metadata检查与Plan setup结果表述，不能称作已分配巨大GPU输入并执行fallback；setup不可用时保留原message。此处不认证巨大strideGPU运行。','',
        f"batchwall是每组{plan['trials']}调用含批末同步的host wall除以调用数，再取组median，不是独立同步单调用CPU wall。setup/tuning/capture不计稳态；event可能含提交空隙，环境快照不认证组内clock。所有组、原FAIL/nonfinite/退出码均保留。",'',
        '冻结参数：','','\x60\x60\x60json',json.dumps(configs,ensure_ascii=False,indent=2),'\x60\x60\x60','']
    if scope['missing_rows']:lines += ['缺失范围：','','\x60\x60\x60json',json.dumps(scope,ensure_ascii=False,indent=2),'\x60\x60\x60','']
    for mode,title in (('launch','普通启动'),('graph','CUDA Graph')):
        lines += [f'## {title}完整端到端耗时','',
            '| 路线 | dtype | Dense median/p10/p90 µs | SW median/p10/p90 µs | Dense/SW | Dense/SW CV % | Dense/SW batchwall µs | 原数值门限 | SW workspace MiB | CSV |',
            '|---|---|---:|---:|---:|---:|---:|---|---:|---|']
        for item in rows:
            r=item['csv_row']
            if r['mode']!=mode:continue
            def values(prefix):return ' / '.join(f'{us(r,k):.6f}' for k in (prefix+'_event_ms',prefix+'_p10_ms',prefix+'_p90_ms'))
            name=f"Mega{r['variant']} {r['actual']}" if r['requested']=='mega' else f"Lt{r['variant']}完整多launch"
            status=('PASS finite' if r['numerical_pass']=='1' else 'FAIL finite') if r['nonfinite']=='0' else 'FAIL nonfinite'
            lines.append(f"| {name} | {r['dtype']} | {values('dense')} | {values('solution')} | {float(r['speedup']):.6f} | {float(r['dense_cv'])*100:.4f}/{float(r['solution_cv'])*100:.4f} | {us(r,'dense_wall_ms'):.5f}/{us(r,'solution_wall_ms'):.5f} | {status} | {float(r['solution_workspace'])/1024**2:g} | {link(item['path'],'CSV',item['line'])} |")
        lines += ['']
    lines += ['## 同binary自身延迟比较','',
        '| mode | dtype | baseline | candidate | baseline µs | candidate µs | baseline/candidate | candidate延迟变化 | baseline/candidate batchwall µs |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for p in pairs:lines.append(f"| {p['mode']} | {p['dtype']} | Mega{p['baseline_variant']} | Mega{p['candidate_variant']} | {p['baseline_us']:.6f} | {p['candidate_us']:.6f} | {p['baseline_over_candidate_own_latency']:.6f}× | {(p['candidate_us']/p['baseline_us']-1)*100:+.3f}% | {p['baseline_batch_wall_us']:.5f}/{p['candidate_batch_wall_us']:.5f} |")
    lines += ['','所有causal claim为false；参数比较不能单独证明bank事务、stall或驻留原因。','']
    for item in rows:
        r=item['csv_row']
        if float(r['dense_cv'])<.02:continue
        groups=sorted((v for v in item['raw_groups'] if v['case']=='dense'),key=lambda v:int(v['group']))
        vals=', '.join(format(float(g['event_ms'])*1000,'.6g') for g in groups)
        lines.append(f"Mega/Lt{r['variant']} {r['mode']} {r['dtype']}：Dense CV={float(r['dense_cv'])*100:.6f}%，SW CV={float(r['solution_cv'])*100:.6f}%；stdout全部Dense event组均值µs=[{vals}]；{link(item['path'],'CSV',item['line'])}。没有剔除较慢组；分母波动不能单独证明代码收益。")
    lines += ['','## 新51/52实际编译资源','',
        '| variant | dtype | BM×BN / BK | threads | dynamic shared | reported/effective initial REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |',
        '|---|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    gate_descriptor=identity['independent_compiled_gate']
    gate_bytes=Path(gate_descriptor['path']).read_bytes()
    require(hashlib.sha256(gate_bytes).hexdigest()==gate_descriptor['sha256'],'Original resource chronology gate SHA remains exact')
    gate=json.loads(gate_bytes)
    resource_notes=[]
    for c in profile['compiled']:
        keys=('variant','dtype','bm','bn','bk','threads','dynamic_shared_bytes','effective_initial_registers_per_thread')
        require(all(k in c for k in keys),'Actual original resource fields must render fully; unknown legacy aliases rejected')
        owned=c.get('initial_owned_CTA_pool_registers',c.get('initial_owned_cta_register_pool'))
        required=c.get('requested_role_pool_registers',c.get('required_role_register_pool'))
        stack=c.get('stack_bytes',c.get('STACK'))
        spill_loads=c.get('spill_load_bytes',c.get('spill_loads_bytes',c.get('spill_loads')))
        spill_stores=c.get('spill_store_bytes',c.get('spill_stores_bytes',c.get('spill_stores')))
        require(all(x is not None for x in (owned,required,stack,spill_loads,spill_stores)),'Original actual pool and stack/spill fields required')
        lines.append(f"| {c['variant']} | {c['dtype']} | {c['bm']}×{c['bn']} /{c['bk']} | {c['threads']} | {c['dynamic_shared_bytes']} | {c.get('actual_reported_registers',c['effective_initial_registers_per_thread'])} | {c['effective_initial_registers_per_thread']} | {owned} | {budget(c['variant'])} | {required} | {stack}/{spill_loads}/{spill_stores} |")
        conf=configs[str(c['variant'])]
        predicted=c.get('occupancy_API_predicted_active_CTAs_per_SM',conf.get('occupancy_API_predicted_active_CTAs_per_SM'))
        residency=c.get('measured_residency',c.get('actual_measured_CTA_residency',conf.get('actual_measured_CTA_residency')))
        resource_notes.append(f"Mega{c['variant']} {c['dtype']}：launch_bounds_min_ctas={conf['launch_bounds_min_ctas']}，quota={budget(c['variant'])}；Driver static_shared_bytes={c.get('Driver_static_shared_bytes',c.get('actual_driver_static_shared_bytes',c.get('static_shared_bytes')))}；resource_tool_reported_SHARED_bytes={c.get('tool_resource_SHARED_bytes',c.get('resource_tool_shared_bytes',c.get('resource_tool_reported_SHARED_bytes',conf.get('tool_resource_shared_bytes'))))}；occupancy API predicted CTAs/SM={predicted}；measured_residency={residency}。")
    lines += ['',*resource_notes,'',*resource_chronology(gate,gate_descriptor)]
    lines += ['','reported/effective initial REG在原gate无独立reported字段时取其effective_initial_registers_per_thread，不能虚构另一个ELF计数。Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。','',
        '## 实际检查范围','']
    for label in ('default','selected51','selected52','disabled37_cli'):
        r=validation.get(label)
        lines.append(f"{label}：actualexit={r['exit']}，CASE/PASS/BEGIN={r['CASE']}/{r['PASS']}/{r['BEGIN']}；{link(r['path'])}。" if r else label+'：未执行，无实际终态。')
    for v,r in validation['new_by_variant'].items():
        if r:lines.append(f"\n新{v} rawU groups={r['bitwise_rawU_host_epilogue_groups']}，reference={r['reference_variant']}（{r['reference_relationship']}）；bound total kernel runs={r['rawU_total_bound_fixture_kernel_runs']}，reference={r['rawU_bound_reference_kernel_runs']}，candidate={r['rawU_bound_candidate_kernel_runs']}。AB={r['current_pointer_binding_checks']}，epi groups/runs={r['epilogue_groups']}/{r['epilogue_kernel_runs']}，Graph configs/fresh={r['new_graph_configurations']}/{r['new_graph_fresh_replays']}。")
        if r and r.get('stride_metadata'):
            sd=r['stride_metadata'];messages=sorted({c['setup_unavailable_message'] for c in sd['case_records'] if c.get('setup_unavailable_message')})
            lines += ['',f"新{v}实际完成{sd['cases']}项主机端A/B stride边界检查（{sd['PASS_groups']}组PASS），其中native helper guard有{sd['native_guard_cases']}项。Plan设置成功{sd['setup_success_cases']}项、不可用{sd['setup_unavailable_cases']}项；input allocation bytes={sd['allocated_input_bytes']}、tensor map encode={sd['tensor_map_encodes']}、native GPU run={sd['native_runs']}。",f"原始不可用信息：{'；'.join(messages) if messages else '无'}。这些记录仅验证helper predicate与Plan setup，setup_unavailable不能被计为成功Plan路由或已执行GPU fallback。",'','\x60\x60\x60json',json.dumps(sd,ensure_ascii=False,indent=2),'\x60\x60\x60']
    strict=validation.get('strict')
    if strict:lines += ['',f"当前strict candidate gates={strict['gates']}，PASS/FAIL={strict['pass']}/{strict['fail']}，nonfinite={strict['nonfinite']}；native={strict['native_gates']}，Dense范围回退={strict['Dense_range_fallback_gates']}。full code domain={strict['full_code_scope_count']}、actually executed={strict['full_code_scope_executed']}；Dense reference CASE单独计数，不能加到candidate gate。",'','\x60\x60\x60json',json.dumps(strict['paths'],ensure_ascii=False,indent=2),'\x60\x60\x60']
    for r in profile['sanitizers']:lines.append(f"\nSAN{r['variant']} {r['tool']}：actualexit={r['exit']}，errors={r['errors']}，race hazards/warnings={r['race_hazards']}/{r['race_warnings']}，leaked bytes={r['leaked_bytes']}；{link(r['path'])}。非mem泄漏未测null。")
    for k,r in validation['CPU_checks'].items():lines.append(f"\nCPU{k} actualexit={r['exit']}，tests={r['tests']}；{link(r['path'])}。")
    for r in validation['CPU_models']:lines.append(f"\nCPU模型actualexit={r['exit']}，originalstatus={r['status']}；{link(r['path'])}。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。")
    lines += ['','环境：','','\x60\x60\x60json',json.dumps(profile['environment'],ensure_ascii=False,indent=2),'\x60\x60\x60','',
        '唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。','']
    lines += ['','## 原始单P融合与S/T一次构造的历史实测','',
        '用户要求的两条原始Hopper路线已在iteration46用同binary完整补充8行测量：H10单P输入融合为两launch，H12 S/T只构造一次为三launch。本轮不重复这8行，也不把native45或Lt1当作H10/H12。原件与原FP64 FAIL保持。',
        link('results/small_psplit_tma_bn16_k32_window_input_placement_supplement_measurements_iter46.json'),
        link('results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_K32_WINDOW_REPORT.md'),'']
    return '\n'.join(lines)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--identity',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--measurements',required=True);p.add_argument('--measurement-generator',required=True);p.add_argument('--output')
    args=p.parse_args();text=generate(args.plan,args.identity,args.profile,args.measurements,args.measurement_generator)
    if args.output:Path(args.output).write_text(text)
    else:print(text,end='')
