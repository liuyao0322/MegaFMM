from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','d8e6261:solution/sw_pipeline.cu'],text=True)
s=s.replace('return row*BN+(col^((row&1)|((row&6)<<2)));','return col*BM+(row^((col&6)<<2));')
s=s.replace('GMMA::Major::K,GMMA::Major::MN','GMMA::Major::MN,GMMA::Major::K')
s=s.replace('smem_desc<GMMA::Major::K>','smem_desc<GMMA::Major::MN>',1)
s=s.replace('auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>','auto db_tiles=make_tensor<GMMA::smem_desc<GMMA::Major::K>>')
s=s.replace('GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,Int<BK>>{}','GMMA::Layout_MN_SW128_Atom<T>{},Shape<_64,Int<BK>>{}')
s=s.replace('GMMA::Layout_MN_SW64_Atom<T>{},Shape<_32,Int<BK>>{}','GMMA::Layout_K_SW128_Atom<T>{},Shape<_32,Int<BK>>{}')
a=s.index('      bool alo=');b=s.index('\n    }\n  }else if',a)
s=s[:a]+'''      bool alo=p==2 || p>=4,blo=p==3 || p>=4;
      TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));
      T* op=s.operand[slot];
      // LHS is B_leaf^T, M-contiguous original B data.
      if(blo){
        int plane=p==3?3:p-4;
        SM90_TMA_LOAD_3D::copy(&tmap,&s.ready[slot],0,op,m0,k0,plane);
      }else{
        int col=p==2?512:0,row=p==1 || p==2?512:0;
        SM90_TMA_LOAD_2D::copy(&bmap,&s.ready[slot],0,op,m0+col,k0+row);
      }
      // RHS is A_leaf^T, K-contiguous original A data.
      if(alo){
        int plane=p==2?3:p-4;
        SM90_TMA_LOAD_4D::copy(&smap,&s.ready[slot],0,op+ASize,0,n0,k0/64,plane);
      }else{
        int col=p==1 || p==3?512:0,row=p==3?512:0;
        SM90_TMA_LOAD_3D::copy(&amap,&s.ready[slot],0,op+ASize,0,n0+row,(k0+col)/64);
      }
'''+s[b:]
a=s.index('    unsigned row=');b=s.index('    float first',a)
s=s[:a]+'''    unsigned row=unsigned(worker)%BM,col=unsigned(worker)/BM;
    float* out=D+(n0+col)*1024+m0+row;
'''+s[b:]
s=s.replace('int idx=((j&1) ? base_odd:base_even)+j*128;','int idx=product_index(row,col+2*j);')
s=s.replace('j*4096','j*2048')
s=s.replace('box[4]={64,BM,BK/64,1}','box[4]={64,BN,BK/64,1}')
s=s.replace('uint32_t(is_a ? 64:BN)','uint32_t(BM)')
s=s.replace('is_a ? CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B','CU_TENSOR_MAP_SWIZZLE_128B')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','d8e6261:solution/plan.cpp'],text=True).replace('product_layout=row_xor_0_3_4','product_layout=column_xor_3_4;leaf=Btrans_Atrans;lhs=MN128;rhs=K128;output=transposed_fragment')
Path('solution/plan.cpp').write_text(s)
