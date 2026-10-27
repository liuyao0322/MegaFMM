#pragma once
// Host-only policy for explicit tuning. Callbacks measure complete invocations;
// this helper never launches work, changes streams, or retains input/output data.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace megafmm::detail {
struct ConfirmationAttempt {
  std::size_t route;
  std::vector<std::pair<float,float>> groups; // dense, custom milliseconds
  float dense_median=0,custom_median=0;
  bool accepted=false;
};
struct ConfirmationResult {
  std::vector<ConfirmationAttempt> attempts;
  bool accepted() const {return !attempts.empty() && attempts.back().accepted;}
};
template<class Warm,class Measure>
ConfirmationResult confirm_ranked_candidates(const std::vector<std::size_t>& ranked,
                                             Warm&& warm,Measure&& measure) {
  ConfirmationResult result;
  for(std::size_t route:ranked) {
    warm(route);
    ConfirmationAttempt attempt;attempt.route=route;
    bool every_pair_wins=true,valid_timings=true;
    for(int round=0;round<5;++round) {
      auto pair=measure(route,round);
      attempt.groups.push_back(pair);
      const float dense=pair.first,custom=pair.second;
      // A failed pair makes the unchanged five-pair gate impossible to pass.
      // Reject immediately and give the next ranked candidate a chance.
      valid_timings=std::isfinite(dense) && std::isfinite(custom) && dense>0 && custom>0;
      if(!valid_timings || !(custom<dense)) {
        every_pair_wins=false;break;
      }
    }
    if(valid_timings) {
      std::vector<float> dense,custom;
      for(auto pair:attempt.groups){dense.push_back(pair.first);custom.push_back(pair.second);}
      std::sort(dense.begin(),dense.end());std::sort(custom.begin(),custom.end());
      attempt.dense_median=dense[dense.size()/2];
      attempt.custom_median=custom[custom.size()/2];
    } else {
      attempt.dense_median=attempt.custom_median=std::numeric_limits<float>::quiet_NaN();
    }
    // Preserve the original sustained confirmation gate exactly: five wins,
    // and custom median strictly below 97% of the dense median.
    attempt.accepted=every_pair_wins && attempt.groups.size()==5 &&
        attempt.custom_median<attempt.dense_median*0.97f;
    result.attempts.push_back(std::move(attempt));
    if(result.accepted())break;
  }
  return result;
}
} // namespace megafmm::detail
