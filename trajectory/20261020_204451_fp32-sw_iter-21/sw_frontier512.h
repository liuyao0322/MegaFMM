#pragma once
#include "internal.h"
namespace sw_frontier512 {
struct Config {int id=0,pack_cap=128,raster=0;};
struct WorkspaceShape {size_t bytes;};
WorkspaceShape shape(const megafmm::Problem& p,Config c);
megafmm::detail::SWResources setup(const megafmm::Problem& p,Config c);
void run(const megafmm::Problem& p,Config c,const void* A,const void* B,float* D,void* ws,int sms,cudaStream_t stream);
}
