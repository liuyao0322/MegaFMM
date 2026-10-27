#include "confirmation370.h"
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace megafmm::detail;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
  // Sustained winner is ninth in short-burst ranking. Earlier candidates fail
  // at different rounds; a fixed top-two/top-eight cap would miss this winner.
  std::vector<std::size_t> ranked={0,1,2,3,4,5,6,7,8,9};
  int warm_count=0,measure_count=0;
  auto late=confirm_ranked_candidates(ranked,[&](std::size_t){++warm_count;},
      [&](std::size_t route,int round){++measure_count;
        return std::pair<float,float>{1.f,route==8?.9f:(round==int(route%5)?1.01f:.9f)};});
  require(late.accepted() && late.attempts.size()==9 && late.attempts.back().route==8,"late sustained winner was skipped");
  require(warm_count==9,"warmup performed for a skipped or accepted successor");
  int expected=5;for(int route=0;route<8;++route)expected+=route%5+1;
  require(measure_count==expected,"failed candidate was not rejected at its first losing pair");
  require(late.attempts.back().groups.size()==5,"accepted without five complete paired groups");
  // A single losing pair invalidates an otherwise fast median.
  auto losing=confirm_ranked_candidates({0},[](std::size_t){},[](std::size_t,int round){return std::pair<float,float>{1.f,round==3?1.001f:.8f};});
  require(!losing.accepted() && losing.attempts[0].groups.size()==4,"one losing group was ignored");
  // Winning every pair by only two percent must retain the original 3% gate.
  auto weak=confirm_ranked_candidates({0},[](std::size_t){},[](std::size_t,int){return std::pair<float,float>{1.f,.98f};});
  require(!weak.accepted() && weak.attempts[0].groups.size()==5,"median improvement gate was relaxed");
  auto boundary=confirm_ranked_candidates({0},[](std::size_t){},[](std::size_t,int){return std::pair<float,float>{1.f,.97f};});
  require(!boundary.accepted(),"strict original 97% boundary was changed");
  auto after_weak=confirm_ranked_candidates({0,1},[](std::size_t){},[](std::size_t route,int){return std::pair<float,float>{1.f,route==0?.98f:.95f};});
  require(after_weak.accepted() && after_weak.attempts.size()==2 && after_weak.attempts[0].groups.size()==5,"median-only failure prevented later confirmation");
  // Median ratio, rather than median of pairwise ratios, is the frozen gate.
  auto old_median=confirm_ranked_candidates({0},[](std::size_t){},[](std::size_t,int r){
    constexpr float d[]={1,1,100,100,100},c[]={.98f,.98f,98,50,50};return std::pair<float,float>{d[r],c[r]};});
  require(old_median.accepted(),"original ratio-of-medians policy changed");
  auto median_reject=confirm_ranked_candidates({0},[](std::size_t){},[](std::size_t,int r){
    constexpr float d[]={1,1,100,10000,10000},c[]={.5f,.5f,98,5000,5000};return std::pair<float,float>{d[r],c[r]};});
  require(!median_reject.accepted(),"ratio-of-medians rejection changed to median of ratios");
  for(float invalid:{0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    auto bad_custom=confirm_ranked_candidates({0},[](std::size_t){},[&](std::size_t,int){return std::pair<float,float>{1.f,invalid};});
    auto bad_dense=confirm_ranked_candidates({0},[](std::size_t){},[&](std::size_t,int){return std::pair<float,float>{invalid,.5f};});
    require(!bad_custom.accepted() && !bad_dense.accepted(),"invalid timing accepted");
  }
  auto late_nan=confirm_ranked_candidates({0},[](std::size_t){},[](std::size_t,int round){return std::pair<float,float>{1.f,round==2?std::numeric_limits<float>::quiet_NaN():.9f};});
  require(!late_nan.accepted() && late_nan.attempts[0].groups.size()==3,"nonfinite prefix was sorted or accepted");
  int calls=0;
  auto empty=confirm_ranked_candidates({},[&](std::size_t){++calls;},[&](std::size_t,int){++calls;return std::pair<float,float>{1,.5f};});
  require(!empty.accepted() && calls==0,"empty ranking launched work");
  puts("PASS late ninth winner, failed-prefix early rejection, five complete wins, unchanged median/boundary gates, invalid/empty timing rejection");return 0;
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
