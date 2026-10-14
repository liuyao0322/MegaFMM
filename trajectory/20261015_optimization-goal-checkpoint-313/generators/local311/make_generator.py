from pathlib import Path
s=Path('results/opt_goal/local309/generate.py').read_text()
s=s.replace('BK=128,Threads=384','BK=64,Threads=384')
s=s.replace('template<class T>struct alignas(128) Shared{T b[7][ASize];T a[7][BSize];uint64_t raw,pack,free;};','template<class T>struct Operands{T b[7][ASize];T a[7][BSize];};\ntemplate<class T>struct alignas(128) Shared{Operands<T> operand[2];uint64_t raw[2],pack[2],free[2];};')
s=s.replace('void transform(Shared<T>& s,int worker)','void transform(Operands<T>& s,int worker)')
s=s.replace('int r=i/BK,k=i%BK,off=lb(r,k);','int off=i;').replace('int r=i%BM,k=i/BM,off=la(r,k);','int off=i;')
s=s.replace('make_smem_ptr(s.b[0])','make_smem_ptr(s.operand[0].b[0])').replace('make_smem_ptr(s.a[0])','make_smem_ptr(s.operand[0].a[0])')
s=s.replace('for k in range(8)','for k in range(4)').replace('%38,%39','%30,%31').replace('22+j','14+j').replace('bp*1024','bp*512').replace('ap*512','ap*256')
s=s.replace('__cvta_generic_to_shared(&s.pack)','__cvta_generic_to_shared(s.pack)').replace('__cvta_generic_to_shared(&s.free)','__cvta_generic_to_shared(s.free)')
a=s.index(' for qidx in range(4):');b=s.index(' for j in range(16):\n  # original',a)
s=s[:a]+''' for qidx in range(8):
  slot=qidx%2;phase=qidx//2%2
  lines.extend([f'add.u32 rb,%2,{slot*8};',f'WAIT_PACK_{qidx}:',f'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],{phase},0x989680;',f'@!done bra WAIT_PACK_{qidx};','wgmma.fence.sync.aligned;'])
  import re
  for x in body:
   x=x.replace(',acc,',',noacc,' if qidx==0 else ',p,')
   x=re.sub(r'(add.u64 b[ab],%[01],)([0-9]+);',lambda m:m[1]+str(int(m[2])+slot*5376)+';',x)
   lines.append(x)
  lines.extend(['wgmma.wait_group.sync.aligned 0;','bar.warp.sync 0xffffffff;',f'add.u32 fb,%3,{slot*8};','@leader mbarrier.arrive.shared::cta.b64 _,[fb];'])
'''+s[b:]
s=s.replace('Barrier::init(&s.raw,1);Barrier::init(&s.pack,256);Barrier::init(&s.free,4);','for(int i=0;i<2;++i){Barrier::init(&s.raw[i],1);Barrier::init(&s.pack[i],256);Barrier::init(&s.free[i],4);}')
s=s.replace('for(int q=0;q<4;++q){\n   if(tid==128){','for(int q=0;q<8;++q){\n   int slot=q%2,phase=q/2%2;\n   auto& op=s.operand[slot];\n   if(q>=2)wait(&s.free[slot],(q/2-1)%2);\n   if(tid==128){')
s=s.replace('&s.raw,','&s.raw[slot],').replace('s.b[p],','op.b[p],').replace('s.a[p],','op.a[p],')
s=s.replace('wait(&s.raw,q%2);','wait(&s.raw[slot],phase);').replace('transform(s,tid-128);','transform(op,tid-128);').replace('arrive(&s.pack);\n   wait(&s.free,q%2);','arrive(&s.pack[slot]);')
s=s.replace('bk=128;operand_stages=1','bk=64;operand_stages=2').replace('publication=local_barriers','publication=local_double_buffer_barriers')
Path('results/opt_goal/local311/generate.py').write_text(s)
