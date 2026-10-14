from pathlib import Path
import json, subprocess, sys

stem = 'psplit_tma_inplace_ss'
plan = f'results/{stem}1024_plan_iter47.json'
identity = f'results/{stem}_measured_identity_iter47.json'
metrics = f'results/small_{stem}_measurements_iter47.json'
profile = f'results/small_{stem}_profile_iter47.json'
report = 'results/1024_MEGAKERNEL_PSPLIT_TMA_INPLACE_SS_REPORT.md'
profile_command = ['python', f'results/{stem}_data_profile_future_actualgate_generator_iter47.py',
    '--plan', plan, '--identity', identity, '--measurements', metrics,
    '--coverage', f'results/{stem}_test_coverage_iter47.json',
    '--compiled-json', f'results/{stem}_compiled_metadata_iter47.json',
    '--environment-json', f'results/{stem}_environment_metadata_iter47.json',
    '--ss-overlap-future-evidence', f'results/{stem}_overlap_future_evidence_iter47.json',
    '--report', report, '--output', profile]
report_command = ['python', f'results/{stem}_data_report_chronology_actualgate_generator_iter47.py',
    '--plan', plan, '--identity', identity, '--measurements', metrics,
    '--profile', profile, '--measurement-generator',
    f'results/{stem}_data_measurements_actualgate_generator_iter47.py', '--output', report]

for label, command in [('profile_initial', profile_command), ('report', report_command), ('profile_refresh', profile_command)]:
    output = Path(f'results/{stem}_{label}_generation_iter47.txt')
    assert not output.exists()
    with output.open('x') as stream:
        stream.write('COMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
    output.with_suffix('.exit').write_text(str(code) + '\n')
    print(label, 'ACTUAL_EXIT', code, flush=True)
    if code:
        print(output.read_text()[-7000:], flush=True)
        sys.exit(code)
print('PRIMARY47_GENERATION_CLOSED', flush=True)
