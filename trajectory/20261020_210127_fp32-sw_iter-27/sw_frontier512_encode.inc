void encode(CUtensorMap& map,const void* ptr,DType type,bool is_a,bool transformed,int mh,int nh,int kh,int N,int K) {
  auto dtype=type==DType::BF16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  uint64_t dims[4],strides[3];uint32_t box[4],es[4]={1,1,1,1};
  int rank;
  if(is_a) {
    dims[0]=64;dims[1]=transformed?mh:2*mh;dims[2]=(transformed?kh:2*kh)/64;dims[3]=4;
    strides[0]=uint64_t(transformed?kh:2*kh)*2;strides[1]=128;strides[2]=uint64_t(mh)*kh*2;
    box[0]=64;box[1]=N;box[2]=K/64;box[3]=1;rank=transformed?4:3;
  } else {
    dims[0]=transformed?nh:2*nh;dims[1]=transformed?kh:2*kh;dims[2]=4;
    strides[0]=dims[0]*2;strides[1]=dims[0]*dims[1]*2;
    box[0]=64;box[1]=K;box[2]=1;rank=transformed?3:2;
  }
  auto r=cuTensorMapEncodeTiled(&map,dtype,rank,const_cast<void*>(ptr),dims,strides,box,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS)throw std::runtime_error("multisize SW tensor map: "+std::to_string(int(r)));
}
