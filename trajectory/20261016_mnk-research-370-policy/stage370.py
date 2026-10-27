from pathlib import Path
import difflib,hashlib,json,shutil
root=Path('results/stage370');(root/'solution').mkdir(parents=True,exist_ok=True);(root/'tests').mkdir(exist_ok=True)
shutil.copy2('trajectory/20261016_mnk-research-370-policy/confirmation370.h',root/'solution/tuning_confirmation.h')
s=Path('trajectory/20261016_mnk-research-370-policy/test_confirmation370.cpp').read_text().replace('"confirmation370.h"','"tuning_confirmation.h"')
(root/'tests/test_tuning_confirmation.cpp').write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('#include "internal.h"','#include "internal.h"\n#include "tuning_confirmation.h"',1)
s=s.replace('  int confirm_candidate = -1;', '  int confirm_candidate = -1;\n  int confirm_attempts = 0, confirm_groups = 0;\n  std::string confirm_trace;')
s=s.replace('  plan.confirm_dense_ms=plan.confirm_custom_ms=NAN;', '  plan.confirm_dense_ms=plan.confirm_custom_ms=NAN;\n  plan.confirm_attempts=plan.confirm_groups=0;plan.confirm_trace.clear();',1)
a=s.index('    std::vector<size_t> shortlist;');b=s.index('\n  } catch(...) {cleanup();throw;}',a)
s=s[:a]+'''    // Short batches nominate candidates; only sustained batches accept them.
    // Do not discard a fast median for one noisy short round or a top-two cap.
    std::vector<size_t> shortlist;
    for(size_t r=lt_pool.size();r<routes.size();++r)
      if(medians[r]<medians[baseline]*0.97f)shortlist.push_back(r);
    std::sort(shortlist.begin(),shortlist.end(),[&](size_t x,size_t y){return medians[x]<medians[y];});
    auto warm=[&](size_t candidate) {
      for(int w=0;w<20;++w) {
        launch(baseline,w%plan.strategy_input_sets);
        launch(candidate,w%plan.strategy_input_sets);
      }
      cuda_check(cudaStreamSynchronize(stream),"strategy holdout warmup");
    };
    auto measure=[&](size_t candidate,int round) {
      float elapsed_ms[2]{};
      for(int order=0;order<2;++order) {
        const bool custom=round%2 ? order==0:order==1;
        const size_t route=custom?candidate:baseline;
        cuda_check(cudaEventRecord(start,stream),"strategy holdout start");
        for(int rep=0;rep<100;++rep)launch(route,rep%plan.strategy_input_sets);
        cuda_check(cudaEventRecord(stop,stream),"strategy holdout stop");
        cuda_check(cudaEventSynchronize(stop),"strategy holdout wait");
        cuda_check(cudaEventElapsedTime(&elapsed_ms[custom],start,stop),"strategy holdout elapsed");
        elapsed_ms[custom]/=100;
      }
      return std::pair<float,float>{elapsed_ms[0],elapsed_ms[1]};
    };
    const auto confirmation=detail::confirm_ranked_candidates(shortlist,warm,measure);
    plan.confirm_attempts=int(confirmation.attempts.size());
    for(const auto& attempt:confirmation.attempts) {
      const int id=routes[attempt.route];
      plan.confirm_candidate=id;plan.confirm_groups=int(attempt.groups.size());
      plan.confirm_dense_ms=attempt.dense_median;plan.confirm_custom_ms=attempt.custom_median;
      if(!plan.confirm_trace.empty())plan.confirm_trace+='|';
      plan.confirm_trace+=std::to_string(id)+":"+std::to_string(attempt.groups.size())+":"+
          (attempt.accepted?"pass":attempt.groups.size()<5?"early_fail":"median_or_pair_fail");
      for(auto pair:attempt.groups)
        plan.confirm_trace+=":"+std::to_string(pair.first)+","+std::to_string(pair.second);
      if(attempt.accepted) {
        plan.hopper_selected=id;plan.strategy_confirmed=true;
      }
    }'''+s[b:]
s=s.replace('    info+=";confirmation=5x100;confirmation_status="', '    info+=";nomination=median3percent;confirmation_coverage=ranked_until_pass;confirmation_attempts="+std::to_string(impl_->confirm_attempts)+";confirmation_groups="+std::to_string(impl_->confirm_groups);\n    info+=";confirmation_trace="+impl_->confirm_trace;\n    info+=";confirmation=5x100;confirmation_status="')
(root/p).write_text(s)
p=Path('CMakeLists.txt');s=p.read_text()+'''\n# Host-only deterministic regression tests for sustained candidate selection.
add_executable(test_tuning_confirmation tests/test_tuning_confirmation.cpp)
target_include_directories(test_tuning_confirmation PRIVATE solution)
add_test(NAME tuning_confirmation_policy COMMAND test_tuning_confirmation)
''';(root/p).write_text(s)
files=['solution/plan.cpp','solution/tuning_confirmation.h','tests/test_tuning_confirmation.cpp','CMakeLists.txt'];patch=''
for name in files:
 old=Path(name).read_text() if Path(name).exists() else ''
 patch+=''.join(difflib.unified_diff(old.splitlines(keepends=True),(root/name).read_text().splitlines(keepends=True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
Path('results/stage370.patch').write_text(patch)
(root/'manifest.json').write_text(json.dumps({'status':'staged only, not applied','base_source_commit':'b88c72f','files':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
