#pragma once
#include "internal.h"
namespace sw_seven_batch {
struct Config {int id=0,pack_cap=512,raster=0;};
struct Shape {int mh,nh,kh,mp,np,kp;size_t a,b,c,bytes;};
Shape shape(const megafmm::Problem& p);
megafmm::detail::SWResources setup(const megafmm::Problem& p,Config c);
void run(const megafmm::Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int sms,cudaStream_t stream);
}
