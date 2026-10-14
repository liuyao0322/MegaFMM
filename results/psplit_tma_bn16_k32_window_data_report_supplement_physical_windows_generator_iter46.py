#!/usr/bin/env python3
"""Owner-run current46 report; source/raw counts and original resource fields."""
from pathlib import Path
import argparse, hashlib, json, types
ROOT=Path('/share/home/liuyao/workspace/MegaFMM')
def require(v,m):
    if not v:raise ValueError(m)
def link(p,label=None,line=None):
    return f"[{label or Path(p).name}]({(ROOT/p).resolve()}"+(f':{line}' if line is not None else '')+')'
def module(p):
    p=Path(p);m=types.ModuleType(p.stem);m.__file__=str(p);exec(compile(p.read_text(),str(p),'exec'),m.__dict__);return m
def supplemental_rows(sup):return sup['row_count']
def us(row,k='solution_event_ms'):return float(row[k])*1000
def resource_chronology(gate, gate_descriptor):
    lines=['原编译门限与本轮资源修复时序：'+link(gate_descriptor['path'])+'；这是本轮编译资源放行记录，不认证数值或性能。','']
    for key in ('initial_failed_revision','revision1_failure'):
        failed=gate.get(key)
        if not failed:continue
        source=failed.get('archived_source_path',failed.get('source_path'))
        lines += [f"{key}："+(link(source) if source else '原件路径按门限记录')+'；失败身份与当前放行配置分开保留。','', '\x60\x60\x60json',json.dumps(failed,ensure_ascii=False,indent=2),'\x60\x60\x60','']
    if gate.get('revision1_source_repair'):
        lines += ['原门限记录的源码修复：'+(gate['revision1_source_repair'] if isinstance(gate['revision1_source_repair'],str) else json.dumps(gate['revision1_source_repair'],ensure_ascii=False,sort_keys=True))+'。这不构成性能收益的因果证明。','']
    for c in gate.get('instances',[]):
        role_max={r['role']:r['gpr_highest_including_implicit'] for r in c['role_CFG']}
        direct=('all_P_clear_after_successful_INC','legal_initial_owned_preINC_zero_present')
        if all(k in c for k in direct):
            init_text='、'.join(k+'='+str(c[k]).lower() for k in direct)
        else:
            init=c.get('initialization_gate',{})
            require(all(k in init for k in ('all_P_clear_after_INC','legal_initial_owned_preINC_zero')),'Original actual initialization flags must be present, not inherited or invented')
            init_text='、'.join(k+'='+str(init[k]).lower() for k in ('all_P_clear_after_INC','legal_initial_owned_preINC_zero'))
        lines += [f"最终{c['dtype']}配额 {'/'.join(map(str,[c['producer_quota'],*c['consumer_quotas']]))}，actual I={c['effective_initial_registers_per_thread']}、owned/requested pool={c['initial_owned_CTA_pool_registers']}/{c['requested_role_pool_registers']}；producer/WG0/WG1最大物理GPR={role_max['producer']}/{role_max['WG0']}/{role_max['WG1']}。实例初始化原标志 {init_text}；原字段按本轮门限逐字记录，初始化或grown标志只按当轮原件记录，不从旧控制继承。",'']
        windows=c.get('physical_windows')
        if windows:
            keys=('windows_per_stage_per_WG','K16_atoms_per_window','WG0_A_packed_words_per_window','WG1_A_packed_words_per_window','WG0_whole_P_words','WG1_whole_P_words','WG0_HGMMA_per_window','WG1_HGMMA_per_window')
            require(all(k in windows for k in keys),'Current physical window fields must come from original fresh gate')
            lines += [f"本轮{c['dtype']}实际每stage每WG为{windows['windows_per_stage_per_WG']}个窗口，每窗{windows['K16_atoms_per_window']}个K16 atom；A packed words WG0/WG1={windows['WG0_A_packed_words_per_window']}/{windows['WG1_A_packed_words_per_window']}，whole P words={windows['WG0_whole_P_words']}/{windows['WG1_whole_P_words']}，每窗HGMMA={windows['WG0_HGMMA_per_window']}/{windows['WG1_HGMMA_per_window']}。当前为K32窗口生命周期，原计数按fresh physical_windows逐项读取；不继承49的四个K16单atom窗口或A12/16。",'']
        if all(q==c['effective_initial_registers_per_thread'] for q in c['consumer_quotas']):
            lines += ['本实例两consumer配额等于actual initial I，没有新增grown寄存器；首MMA accumulator zero与pending A/P证明仍来自本轮原门限。','']
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
    lines=['# 1024 BN16 K32-window选择性TMA rawA实测（iteration46）','']
    for baseline,candidate in ((49,50),(45,50),(39,45),(39,50)):
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
        'ONLY50是新BN16 K32-window选择性四路TMA rawA路线；49为上一轮BN16 atom-local min2 TMA，45为BN32 TMA，39为原RS-A BK64，Lt1为完整多launch对照。51未注册、未运行。fresh compiled blocked50必须准确列出四个缺失键，只剩16行对照，不能用旧结果代替；45对39四个同binary配对仍可实际测得。','',
        f"warmup={plan['warmup']}，calls/group={plan['trials']}，groups={plan['groups']}，fresh/row={plan['correct_trials']}，rotate_inputs={plan['rotate_inputs']}，heuristic candidates={plan['heuristic_candidates']}，共同workspace cap={plan['workspace_limit']}bytes。",'',
        'event median、线性p10/p90和总体CV来自组均值，quantile_unit=group_mean，不是单call分位数。每行cuBLASLt在capture前独立调优；Dense/SW按同行相同普通/Graph模式计时，没有Graph专用调优。比值为同行Dense event median /完整SW event median，代码收益依据同binary自身SW延迟。','',
        '50、49和45的源码入口在普通启动时每次从当前A指针/lda编码一个A tensor map；实际enabled路线的编码与单个kernel均包含在完整调用计时内，map实际被四路rawA TMA消费，B仍SIMT；39无host descriptor encode。50最终几何/配额/lifetime由fresh gate与冻结参数读取；49为BN16/BK64，45为BN32/BK64。min2是launch-bounds配置，API预测也不等于实际两CTA驻留；所有controlled_axis_claimed=false，不认证纯BN或纯搬运因果。全部实际运行native完成原变换、wholeK七个FP32产品、固定U与RNalpha/activebetaFMA epilogue，为一个kernel、零矩阵workspace；Lt1完整多launch计时。Graph50/49/45在capture时编码，稳态replay使用固定capture地址读取当前内容，fresh检查更换内容。普通event可能含host编码/提交空隙，不能把Graph差值单独当作encode耗时。rawU的reference variant与sameatom/crossatom关系只按本轮实际coverage记录；严格bitwise fixture和独立host RN alpha/FMA不构成通用FP64精度认证，任何实际不一致/FAIL必须保留。','',
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
    lines += ['','## 新50实际编译资源','',
        '| variant | dtype | BM×BN / BK | threads | dynamic shared | ELF REG | effective initial REG | initial CTA owned pool | quota | required role pool | STACK/spill loads/stores |',
        '|---|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    gate_descriptor=identity['independent_compiled_gate']
    gate_bytes=Path(gate_descriptor['path']).read_bytes()
    require(hashlib.sha256(gate_bytes).hexdigest()==gate_descriptor['sha256'],'Original resource chronology gate SHA remains exact')
    gate=json.loads(gate_bytes)
    resource_notes=[]
    for c in profile['compiled']:
        keys=('variant','dtype','bm','bn','bk','threads','dynamic_shared_bytes','actual_reported_registers','effective_initial_registers_per_thread')
        require(all(k in c for k in keys),'Actual original resource fields must render fully; unknown legacy aliases rejected')
        owned=c.get('initial_owned_CTA_pool_registers',c.get('initial_owned_cta_register_pool'))
        required=c.get('requested_role_pool_registers',c.get('required_role_register_pool'))
        stack=c.get('stack_bytes',c.get('STACK'))
        spill_loads=c.get('spill_load_bytes',c.get('spill_loads_bytes',c.get('spill_loads')))
        spill_stores=c.get('spill_store_bytes',c.get('spill_stores_bytes',c.get('spill_stores')))
        require(all(x is not None for x in (owned,required,stack,spill_loads,spill_stores)),'Original actual pool and stack/spill fields required')
        lines.append(f"| {c['variant']} | {c['dtype']} | {c['bm']}×{c['bn']} /{c['bk']} | {c['threads']} | {c['dynamic_shared_bytes']} | {c['actual_reported_registers']} | {c['effective_initial_registers_per_thread']} | {owned} | {budget(c['variant'])} | {required} | {stack}/{spill_loads}/{spill_stores} |")
        conf=configs[str(c['variant'])]
        predicted=c.get('occupancy_API_predicted_active_CTAs_per_SM',conf.get('occupancy_API_predicted_active_CTAs_per_SM'))
        residency=c.get('measured_residency',c.get('actual_measured_CTA_residency',conf.get('actual_measured_CTA_residency')))
        resource_notes.append(f"Mega{c['variant']} {c['dtype']}：Driver static_shared_bytes={c.get('actual_driver_static_shared_bytes',c.get('static_shared_bytes'))}；resource_tool_reported_SHARED_bytes={c.get('resource_tool_shared_bytes',c.get('resource_tool_reported_SHARED_bytes',conf.get('tool_resource_shared_bytes')))}；occupancy API predicted CTAs/SM={predicted}；measured_residency={residency}。")
    lines += ['',*resource_notes,'',*resource_chronology(gate,gate_descriptor)]
    lines += ['','Driver属性、resource accounting与occupancy API预测分别记录；没有kernel residency测量，不据此宣称实际两CTA驻留、transaction grouping、bank stalls或速度因果。旧控制资源描述不是本轮fresh function query。','',
        '## 实际检查范围','']
    for label in ('default','selected50','disabled37_cli'):
        r=validation.get(label)
        lines.append(f"{label}：actualexit={r['exit']}，CASE/PASS/BEGIN={r['CASE']}/{r['PASS']}/{r['BEGIN']}；{link(r['path'])}。" if r else label+'：未执行，无实际终态。')
    for v,r in validation['new_by_variant'].items():
        if r:lines.append(f"\n新{v} rawU groups={r['bitwise_rawU_host_epilogue_groups']}，reference={r['reference_variant']}（{r['reference_relationship']}）；bound total kernel runs={r['rawU_total_bound_fixture_kernel_runs']}，reference={r['rawU_bound_reference_kernel_runs']}，candidate={r['rawU_bound_candidate_kernel_runs']}。AB={r['current_pointer_binding_checks']}，epi groups/runs={r['epilogue_groups']}/{r['epilogue_kernel_runs']}，Graph configs/fresh={r['new_graph_configurations']}/{r['new_graph_fresh_replays']}。")
        if r and r.get('stride_metadata'):
            sd=r['stride_metadata'];messages=sorted({c['setup_unavailable_message'] for c in sd['case_records'] if c.get('setup_unavailable_message')})
            lines += ['',f"新{v}实际完成{sd['cases']}项主机端A-only stride边界检查（{sd['PASS_groups']}组PASS），其中native helper guard有{sd['native_guard_cases']}项。Plan设置成功{sd['setup_success_cases']}项、不可用{sd['setup_unavailable_cases']}项；input allocation bytes={sd['allocated_input_bytes']}、tensor map encode={sd['tensor_map_encodes']}、native GPU run={sd['native_runs']}。",f"原始不可用信息：{'；'.join(messages) if messages else '无'}。这些记录仅验证helper predicate与Plan setup，setup_unavailable不能被计为成功Plan路由或已执行GPU fallback。",'','\x60\x60\x60json',json.dumps(sd,ensure_ascii=False,indent=2),'\x60\x60\x60']
    strict=validation.get('strict')
    if strict:lines += ['',f"当前strict candidate gates={strict['gates']}，PASS/FAIL={strict['pass']}/{strict['fail']}，nonfinite={strict['nonfinite']}；native={strict['native_gates']}，Dense范围回退={strict['Dense_range_fallback_gates']}。full code domain={strict['full_code_scope_count']}、actually executed={strict['full_code_scope_executed']}；Dense reference CASE单独计数，不能加到candidate gate。",'','\x60\x60\x60json',json.dumps(strict['paths'],ensure_ascii=False,indent=2),'\x60\x60\x60']
    for r in profile['sanitizers']:lines.append(f"\nSAN{r['variant']} {r['tool']}：actualexit={r['exit']}，errors={r['errors']}，race hazards/warnings={r['race_hazards']}/{r['race_warnings']}，leaked bytes={r['leaked_bytes']}；{link(r['path'])}。非mem泄漏未测null。")
    for k,r in validation['CPU_checks'].items():lines.append(f"\nCPU{k} actualexit={r['exit']}，tests={r['tests']}；{link(r['path'])}。")
    for r in validation['CPU_models']:lines.append(f"\nCPU模型actualexit={r['exit']}，originalstatus={r['status']}；{link(r['path'])}。JSON PASS不代替process exit，也不认证硬件memory order/thread convergence/quota fit/FP64精度或GPU性能。")
    lines += ['','环境：','','\x60\x60\x60json',json.dumps(profile['environment'],ensure_ascii=False,indent=2),'\x60\x60\x60','',
        '唯一prestatus供独立审查绑定；后续仅审核metadata与原字节copies，审核件不回填最终profile SHA造成hash循环。','']
    sup=profile.get('supplemental_two_routes')
    if sup:
        require(hashlib.sha256(Path(sup['measurements_path']).read_bytes()).hexdigest()==sup['measurements_sha256'] and json.loads(Path(sup['measurements_path']).read_text())=={k:v for k,v in sup.items() if k not in ('measurements_path','measurements_sha256','mandatory_rows','supplemental_rows','total_observed_rows','declared_expected_rows','supplemental_scope')},'Supplement profile embeds exact separately frozen measurements')
        lines += ['','## 原始两条路线的同binary补充实测','',f"主计划实际{metrics['row_count']}行与补充{supplemental_rows(sup)}行分别计数，总观察{metrics['row_count']+supplemental_rows(sup)}行。H10为单P输入融合、每完整调用两个kernel；H12为S/T只构造一次、每完整调用三个kernel。它们与native Mega50/49/45/39及Lt1有独立身份。原始FP64 gates/floors、五组均值与全部FAIL/NF保留。",'', '| 原始路线 / 模式 | dtype | Dense event /p10/p90 µs | SW event /p10/p90 µs | Dense/SW | CV Dense/SW % | batch wall Dense/SW µs | 原数值 | WS bytes | CSV |','|---|---|---:|---:|---:|---:|---:|---|---:|---|']
        for record in sup['rows']:
            row=record['csv_row'];status=('PASS finite' if row['numerical_pass']=='1' else 'FAIL finite') if row['nonfinite']=='0' else 'FAIL nonfinite'
            d=' / '.join(f"{float(row['dense'+suffix])*1000:.6f}" for suffix in ('_event_ms','_p10_ms','_p90_ms'));w=' / '.join(f"{float(row['solution'+suffix])*1000:.6f}" for suffix in ('_event_ms','_p10_ms','_p90_ms'))
            lines.append(f"| H{row['variant']} / {row['mode']} | {row['dtype']} | {d} | {w} | {float(row['speedup']):.6f} | {float(row['dense_cv'])*100:.4f}/{float(row['solution_cv'])*100:.4f} | {float(row['dense_wall_ms'])*1000:.5f}/{float(row['solution_wall_ms'])*1000:.5f} | {status} | {row['solution_workspace']} | {link(record['path'],line=record['line'])} |")
        lines += ['','H12相对H10的自身耗时比与下降百分比仅来自本轮补充同行SW event。比率分母变化不代表代码收益；batch wall是100call同步批次/call，非每次独立同步。','', '| 模式 | dtype | H10 µs | H12 µs | H10/H12自身耗时比 | H12自身耗时下降 % | H10/H12 batchwall µs |','|---|---|---:|---:|---:|---:|---:|']
        for c in sup['same_binary_own_latency_comparison']:
            lines.append(f"| {c['mode']} | {c['dtype']} | {c['baseline_us']:.6f} | {c['candidate_us']:.6f} | {c['baseline_over_candidate_own_latency']:.6f} | {c['own_latency_decrease_percent']:.6f} | {c['baseline_batch_wall_us']:.5f}/{c['candidate_batch_wall_us']:.5f} |")
        lines += ['',f"补充TIME={sup['timing_group_case_count']}，freshCHECK={sup['fresh_gate_count']}，PASS/FAIL={sup['fresh_gate_pass']}/{sup['fresh_gate_fail']}，nonfinite={sup['nonfinite']}；{link(sup['measurements_path'])}。编译资源CAN50不转作H10/H12数值认证。",'']
    return '\n'.join(lines)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--identity',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--measurements',required=True);p.add_argument('--measurement-generator',required=True);p.add_argument('--output')
    args=p.parse_args();text=generate(args.plan,args.identity,args.profile,args.measurements,args.measurement_generator)
    if args.output:Path(args.output).write_text(text)
    else:print(text,end='')
