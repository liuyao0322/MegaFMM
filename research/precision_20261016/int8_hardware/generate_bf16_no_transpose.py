"""P44: BF16 correction B can use MN-major, removing its transpose entirely."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p43'
s=(base/'pack.cuh').read_text()
s=s.replace(' cute::bfloat16_t transpose[SideA?1:2][SideA?1:128*34];\n','')
start=s.index('  if constexpr(SideA){\n   q[QP*Plane+at]=qh;')
end=s.index('\n}\ntemplate<class T,bool SideA,int NP0',start)
s=s[:start]+'''  // BF16 supports MN-major B, so retain original row-major coordinates.
  q[QP*Plane+at]=qh;if constexpr(QP<4)q[(QP+4)*Plane+at]=ql;
 }
'''+s[end:]
s=s.replace('// Paired FP32 transforms remain shared; B correction planes are physically\n// transposed, giving K-major BF16 operands on both sides of correction MMA.\n// A 34-halfword padded row is 17 words, coprime with the 32 shared banks.',
 '// Paired FP32 transforms stay shared. BF16 correction B is MN-major and\n// correction A is K-major, so neither side needs a physical transpose.')
(h/'pack.cuh').write_text(s)
s=(base/'corrected.cuh').read_text()
s=s.replace('SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>', 'SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>')
s=s.replace('laq=tile_to_shape(GMMA::Layout_K_SW128_Atom<Q>', 'laq=tile_to_shape(GMMA::Layout_MN_SW128_Atom<Q>')
s=s.replace('daq=make_tensor<GMMA::smem_desc<GMMA::Major::K>>','daq=make_tensor<GMMA::smem_desc<GMMA::Major::MN>>')
s=s.replace('SM90_TMA_LOAD_4D::copy(&quant_b,&s.ready[slot],0,s.operand[slot],0,m0,k*2,qb_plane(term));',
 'SM90_TMA_LOAD_3D::copy(&quant_b,&s.ready[slot],0,s.operand[slot],m0,k*128,qb_plane(term));')
s=s.replace('SM90_TMA_LOAD_4D::copy(&quant_b,&s.ready[slot],0,s.operand[slot]+16384,0,m0,k*2,qb_plane(term+1));',
 'SM90_TMA_LOAD_3D::copy(&quant_b,&s.ready[slot],0,s.operand[slot]+16384,m0,k*128,qb_plane(term+1));')
s=s.replace('if(integer){rank=4;', 'if(integer&&side_a){rank=4;')
needle=' if(integer&&side_a){rank=4;dims[0]=64;dims[1]=512;dims[2]=8;dims[3]=9;strides[0]=1024;strides[1]=128;strides[2]=Plane*2;box[0]=64;box[2]=2;}'
assert s.count(needle)==1;s=s.replace(needle,needle+'\n if(integer&&!side_a){box[0]=64;box[1]=128;}')
(h/'corrected.cuh').write_text(s)
for f in ('compare.cu','pow2.cuh'):(h/f).write_text((base/f).read_text())
print('generated P44 from frozen P43; remove B correction transpose, retain all dots and slot lifetimes')
