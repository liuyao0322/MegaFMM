"""Prepared P54 shape control from frozen rejected P53; not an acceptance claim."""
from pathlib import Path
r=Path(__file__).resolve().parent
for n in ['pack.cuh','corrected.cuh','compare.cu','pow2.cuh']:
 s=(r/'p53'/n).read_text()
 if n=='corrected.cuh':
  substitutions=[
   ('operand[ConsumerStages][49152]','operand[ConsumerStages][65536]'),
   ('sizeof(Shared)==98432','sizeof(Shared)==131200'),
   ('SM90_64x32x16_F32F16F16_SS','SM90_64x64x16_F32F16F16_SS'),
   ('SM90_64x32x16_F32BF16BF16_SS','SM90_64x64x16_F32BF16BF16_SS'),
   ('Shape<_32,_128>','Shape<_64,_128>'),
   ('Shape<_32,_16>','Shape<_64,_16>'),
   ('float d[16]={},r[16]={};','float d[32]={},r[32]={};'),
   ('j<16','j<32'),
   ('slot*49152/16','slot*65536/16'),
   ('+8192/16','+16384/16'),
   ('GMMA::CLayout_64x32','GMMA::CLayout_64x64'),
   ('n0=blockIdx.x*32','n0=blockIdx.x*64'),
   ('int bytes=p<2?24576:p==2?32768:p==3?40960:49152;','int bytes=p<2?32768:p==2?49152:p==3?49152:65536;'),
   ('s.operand[slot]+40960','s.operand[slot]+49152'),
   ('box[1]=32;box[2]=2;','box[1]=64;box[2]=2;'),
   ('grid=896','grid=448'),
   ('<<<dim3(16,8,7),256,sizeof(Shared),stream>>>','<<<dim3(8,8,7),256,sizeof(Shared),stream>>>')]
  for old,new in substitutions:
   assert old in s,old
   s=s.replace(old,new)
  old=','.join(f'd[{i}]' for i in range(16))
  new=','.join(f'd[{i}]' for i in range(32))
  assert s.count(old+',scale);')==2
  s=s.replace(old+',scale);',new+',scale);')
 (r/n).write_text(s)
