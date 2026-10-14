#!/usr/bin/env python3
"""CPU integer/opaque geometry and protocol model. No CUDA compilation or GPU."""
from pathlib import Path
import hashlib,itertools,json,random,struct
R=Path(__file__).resolve().parent.parent;P=R/'.scratch/psplit_tma_bn64_root_future42_'
count={};neg=[]
def add(key,n=1):count[key]=count.get(key,0)+n
def require(ok,reason):
 if not ok:raise ValueError(reason)
def reject(name,fn):
 try:fn()
 except (ValueError,AssertionError):neg.append(name);return
 raise AssertionError('negative accepted: '+name)
def sw(x,bits):return x^((x&(((1<<bits)-1)<<7))>>3)
def raw_a(r,k,q):return q*4096+r*64+k
def raw_b(n,k,q):return q*4096+(k//8)*512+(n//16)*128+n%16+16*(k%8)
def a_at(o,bank,r,k,q):return sw(o+bank*90112+2*raw_a(r,k,q),3)
def b_at(o,bank,n,k,q):return sw(o+bank*90112+32768+2*raw_b(n,k,q),1)
def ref_a(o,bank,r,k,q):return o+bank*90112+q*8192+r*128+2*(k^((r&7)<<3))
def ref_b(o,bank,n,k,q):
 brick_n,inner_n=divmod(n,16);brick_k,inner_k=divmod(k,8)
 return sw(o+bank*90112+32768+q*8192+brick_k*1024+brick_n*256+inner_k*32+inner_n*2,1)
def tail(q,r,c):return q*4096+r*64+(c^((r&1)|((r&6)<<2)))
def ccoord(t,v):
 x=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4)
 return x%64,x//64

def geometry():
 origins=[0,1024,2048,8192,65536]
 for o in origins:
  for bank in range(2):
   lo=o+bank*90112;seen=set();image=bytearray(90112)
   for q in range(4):
    for r in range(64):
     for k in range(0,64,8):
      addr=a_at(o,bank,r,k,q);assert addr%16==0
      words=[(raw_a(r,k+j,q)*131+bank*16001+(o//1024)*1009)&65535 for j in range(8)]
      data=struct.pack('<8H',*words);image[addr-lo:addr-lo+16]=data
      for j in range(8):
       a=a_at(o,bank,r,k+j,q);assert a==addr+j*2==ref_a(o,bank,r,k+j,q)
       assert lo<=a<lo+32768 and a not in seen;seen.add(a)
      assert struct.unpack_from('<8H',image,addr-lo)==tuple(words);add('A_opaque_halfwords_roundtrip',8);add('A_vector8')
   for q in range(7):
    for k in range(64):
     for n in range(0,64,8):
      addr=b_at(o,bank,n,k,q);assert addr%16==0
      words=[(raw_b(n+j,k,q)*137+q*71+bank*19001+(o//1024)*1013)&65535 for j in range(8)]
      image[addr-lo:addr-lo+16]=struct.pack('<8H',*words)
      for j in range(8):
       a=b_at(o,bank,n+j,k,q);assert a==addr+j*2==ref_b(o,bank,n+j,k,q)
       assert lo+32768<=a<lo+90112 and a not in seen;seen.add(a)
      assert struct.unpack_from('<8H',image,addr-lo)==tuple(words);add('B_opaque_halfwords_roundtrip',8);add('B_vector8')
   assert seen==set(range(lo,lo+90112,2));add('full_bank_bijections')
   for q in range(7):
    for atom in range(4):
     start=lo+32768+q*8192+atom*2048
     assert start%1024==0
     desc=(start>>4)|(16<<16)|(64<<32)|(3<<62)
     assert desc&0x3fff==start>>4
     assert (desc>>16)&0x3fff==16 and (desc>>32)&0x3fff==64 and desc>>62==3
     for n in range(64):
      for ka in range(16):
       reconstructed=sw(start+(n//16)*256+(n%16)*2+(ka//8)*1024+(ka%8)*32,1)
       assert reconstructed==b_at(o,bank,n,16*atom+ka,q);add('B_descriptor_coordinates')
     add('B_descriptor_atoms')
 assert 180224+32==180256 and 180256+16==180272
 assert (180272+1023)//1024*1024==181248
 assert 7*64*64*4==114688<180224
 assert 114688<=180224<180256<181248
 # C and A partition coordinate formulas are independently instantiated.
 cseen={};aseen=set()
 for t in range(128):
  for v in range(32):
   r,c=ccoord(t,v);assert (r,c) not in cseen;cseen[r,c]=(t,v);add('C_fragment_coordinates')
  for atom in range(4):
   for v in range(8):
    x=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%2)
    r,k=x%64,x//64+16*atom;assert (r,k) not in aseen;aseen.add((r,k));add('A_RS_fragment_coordinates')
 assert len(cseen)==len(aseen)==4096
 words=set()
 for q in range(7):
  for r,c in cseen:
   a=tail(q,r,c);assert a not in words;words.add(a);add('tail_words')
 assert words==set(range(28672))
 for ldd in [1024,1025,1032]:
  out=set();pad=set()
  for bx in range(8):
   for by in range(8):
    for linear in range(4096):
     owner=linear%384;r,c=divmod(linear,64);assert 0<=owner<384
     rr=bx*64+r;cc=by*64+c
     for dr,dc in [(0,0),(0,512),(512,0),(512,512)]:
      address=(rr+dr)*ldd+cc+dc;assert address not in out;out.add(address);add('D_unique_owned_words')
  assert len(out)==1024*1024
  for r in range(1024):
   assert r*ldd+1023 in out and r*ldd in out
   for col in range(1024,ldd):assert r*ldd+col not in out;add('D_padding_words_untouched')
 # Current AB vector addresses and TMA coordinates, padded and near limit lda.
 for lda,ldb in [(1024,1024),(1032,1040),(1040,1032),((1<<39)-8,1024)]:
  for bx in range(8):
   for by in range(8):
    for stage in range(8):
     rb,cb,kb=bx*64,by*64,stage*64
     for q in range(4):
      mh,kh=(q//2)*512,(q%2)*512
      for r,k in [(0,0),(0,56),(7,8),(8,16),(63,56)]:
       row=rb+mh+r;col=kb+kh+k;address=(row*lda+col)*2
       assert 0<=row<1024 and 0<=col<=1016 and address%16==0
       assert address+16<=(1023*lda+1024)*2;add('A_global_vector_offset_checks')
     for task in range(512):
      k,col=divmod(task,8);col*=8;gk,gc=kb+k,cb+col
      for dr,dc in [(0,0),(0,512),(512,0),(512,512)]:
       row,cc=gk+dr,gc+dc;address=(row*ldb+cc)*2
       assert 0<=row<1024 and 0<=cc<=1016 and address%16==0
       assert address+16<=(1023*ldb+1024)*2;add('B_global_vector_offset_checks')
 return {'origins':origins,'bank_bytes':90112,'two_bank_payload_bytes':180224,'expected_pipeline_offset':180224,'expected_raw_done_offset':180256,'expected_logical_shared_bytes':181248,'tail_bytes':114688,'B_GMMA_descriptor_prediction':{'leading_bytes':256,'stride_bytes':1024,'layout_type':3,'base_offset':0},'actual_sizeof_and_offsets':None}

def winograd(x):
 a11,a12,a21,a22,b11,b12,b21,b22=x
 s1=a21+a22;s2=s1-a11;s3=a11-a21;s4=a12-s2
 t1=b12-b11;t2=b22-t1;t3=b22-b12;t4=t2-b21
 p=[a11*b11,a12*b21,s4*b22,a22*t4,s1*t1,s2*t2,s3*t3]
 return p,(a11*b11+a12*b21,a11*b12+a12*b22,a21*b11+a22*b21,a21*b12+a22*b22)
def fixed_u(p):
 p1,p2,p3,p4,p5,p6,p7=p;u1=p1+p2;u2=p1+p6;u3=u2+p7;u4=u2+p5;u5=u4+p3;u6=u3-p4;u7=u3+p5
 return u1,u5,u6,u7

def math_model():
 for x in itertools.product([-1,0,1],repeat=8):
  p,d=winograd(x);assert fixed_u(p)==d;add('exact_integer_scalar_DAG_cases')
 rng=random.Random(421064)
 for stages in [2,4,8,16,32]:
  for trial in range(32):
   p=[0]*7;direct=[0]*4
   for stage in range(stages):
    for atom in range(4):
     for kk in range(16):
      x=[rng.randrange(-2,3) for _ in range(8)];pp,dd=winograd(x)
      p=[a+b for a,b in zip(p,pp)];direct=[a+b for a,b in zip(direct,dd)];add('whole_K_integer_terms')
   assert fixed_u(p)==tuple(direct);add('whole_K_exact_integer_cases')
 # This bounded model uses exact integers, not the unchanged original FP64 gate.
 assert count['exact_integer_scalar_DAG_cases']==6561

class Bank:
 def __init__(self):self.stage=None;self.gen=-1;self.raw=set();self.b=set();self.f=set();self.arv=set();self.released=set();self.full=False;self.expect=0;self.done_phase=None;self.full_phase=None
 def acquire(self,stage):
  require(self.stage is None or self.released=={0,1},'both consumers must release before bank reuse')
  self.stage=stage;self.gen=stage//2;self.raw=set();self.b=set();self.f=set();self.arv=set();self.released=set();self.full=False;self.expect=0;self.done_phase=None;self.full_phase=None
 def expectation(self,tx):require(tx==32768,'four raw boxes require exact32768TX');self.expect=tx
 def complete(self,q,gen):
  require(gen==self.gen and q not in self.raw and 0<=q<4,'fresh unique raw box completion');self.raw.add(q)
  if len(self.raw)==4:require(self.expect==32768,'TX expectation before ready');self.done_phase=(self.gen&1)^1
 def write(self,tid):require(0<=tid<128 and tid not in self.b,'distinct B writer');self.b.add(tid)
 def fence(self,tid):require(tid in self.b,'writer fences own B stores');self.f.add(tid)
 def arrive(self,tid,wait_phase):
  require(tid in self.f,'writer proxy fence before commit')
  require(self.done_phase==((wait_phase&1)^1) and wait_phase==(self.gen&1),'fresh rawbar epoch before commit')
  require(len(self.raw)==4 and self.expect==32768,'all A bytes before publication')
  require(tid not in self.arv,'unique full arrival');self.arv.add(tid)
  if len(self.arv)==128:
   require(len(self.b)==len(self.f)==128,'all128 writers fenced');self.full=True;self.full_phase=(self.gen&1)^1
 def read(self,stage,phase):require(self.stage==stage and self.full and self.full_phase==(phase^1) and phase==(self.gen&1),'same stage/full phase before consume')

class Role:
 def __init__(self,g):self.g=g;self.quota=None;self.initialized=False;self.pendingA=set();self.pendingP=set();self.stage=0;self.step=0;self.issues=0;self.values=[0]*((3 if g==0 else 4)*32);self.P=(3 if g==0 else 4)*32;self.A=(3 if g==0 else 4)*16
 def alloc(self,q):require(q==[224,240][self.g],'distinct correct consumer quota');self.quota=q
 def clear(self):require(self.quota==[224,240][self.g],'INC before P initialize');self.values=[0]*self.P;self.initialized=True
 def transform(self):require(self.initialized and not self.pendingA and not self.pendingP,'stage transform after prior wait0');self.pendingA=set(range(self.A));self.issues=0
 def issue(self):
  require(len(self.pendingA)==self.A and self.initialized,'all whole A words retained');self.pendingP=set(range(self.P));self.issues+=1
  for i in range(self.P):self.values[i]+=(self.stage+1)*(self.g+1)*(i+1)
 def wait0(self):require(self.issues==(3 if self.g==0 else 4)*4,'all ascending wholeP K16 atoms before wait0');self.pendingA.clear();self.pendingP.clear()
 def release(self,b):require(not self.pendingA and not self.pendingP,'no pending A/P on release');require(self.g not in b.released,'one release per group');b.released.add(self.g)
 def overwriteA(self):require(not self.pendingA,'no A overwrite while pending')
 def readP(self):require(not self.pendingP,'no P read/store while pending')

def simulation(stages,seed):
 rng=random.Random(seed);banks=[Bank(),Bank()];roles=[Role(0),Role(1)]
 for role in roles:role.alloc([224,240][role.g]);role.clear()
 producer=0;pc=0;done=False;steps=0
 # pc:acquire0/expect1/B writers2..129/fences130..257/arrivals258..385/advance386.
 while producer<stages or any(r.stage<stages for r in roles):
  choices=[]
  if producer<stages:
   b=banks[producer%2]
   if pc==0 and (b.stage is None or b.released=={0,1}):choices.append(('p',0))
   elif pc>0 and (pc<258 or len(b.raw)==4):choices.append(('p',0))
   if pc>=2 and len(b.raw)<4:
    for q in range(4):
     if q not in b.raw:choices.append(('tma',q))
  for g,r in enumerate(roles):
   if r.stage<stages:
    b=banks[r.stage%2]
    if r.step or (b.stage==r.stage and b.full):choices.append(('r',g))
  require(bool(choices),'schedule makes progress');who,arg=rng.choice(choices);steps+=1
  require(steps<stages*1000,'bounded schedule')
  if who=='p':
   b=banks[producer%2]
   if pc==0:b.acquire(producer)
   elif pc==1:b.expectation(32768)
   elif pc<130:b.write(pc-2);add('protocol_B_writer_events')
   elif pc<258:b.fence(pc-130);add('protocol_proxy_fences')
   elif pc<386:b.arrive(pc-258,(producer//2)&1);add('protocol_full_arrivals')
   else:producer+=1;pc=-1
   pc+=1
  elif who=='tma':b=banks[producer%2];b.complete(arg,b.gen);add('protocol_raw_box_completions')
  else:
   r=roles[arg];b=banks[r.stage%2];products=3 if arg==0 else 4
   if r.step==0:b.read(r.stage,(r.stage//2)&1);r.transform();r.step=1;add('protocol_consumer_full_phase_checks')
   elif r.step<=products*4:r.issue();r.step+=1;add('protocol_K16_atom_issues')
   elif r.step==products*4+1:r.wait0();r.step+=1;add('protocol_wait0')
   else:
    r.overwriteA();r.readP();r.release(b);r.stage+=1;r.step=0;add('protocol_consumer_releases')
  add('protocol_scheduled_events')
 # Producer tail may retire only when every acquired bank has BOTH releases.
 assert all(b.stage is None or b.released=={0,1} for b in banks)
 for r in roles:
  expected=[sum((stage+1)*(r.g+1)*(i+1)*(3 if r.g==0 else 4)*4 for stage in range(stages)) for i in range(r.P)]
  assert r.values==expected and not r.pendingA and not r.pendingP
 add('protocol_both_bank_tail_drains',2);add('protocol_original_CTA_epochs',3);add('protocol_schedules')

# Native helper domain model is only the declared tile/stride/alignment overlap.
def native(m,n,k,lda,ldb,A=0x1000,B=0x2000):return m>0 and n>0 and k>0 and m%128==n%128==k%128==0 and lda>=k and ldb>=n and lda%8==ldb%8==0 and lda<(1<<39) and A!=0 and B!=0 and A%16==B%16==0

def ready_bank(gen=0):
 b=Bank();b.acquire(gen*2);b.expectation(32768)
 for q in range(4):b.complete(q,gen)
 for t in range(128):b.write(t);b.fence(t);b.arrive(t,gen&1)
 return b

def negatives():
 reject('wrong_SW32_A_counterfactual',lambda:require(all(sw(2*raw_a(r,k,0),1)==a_at(0,0,r,k,0) for r in range(64) for k in range(64)),'SW32 differs from SW128'))
 reject('wrong_B_BN32_leading_stride_counterfactual',lambda:require(all(sw(32768+(n//16)*256+(n%16)*2+(k//8)*512+(k%8)*32,1)==b_at(0,0,n,k,0) for n in range(64) for k in range(16)),'BN32 stride512 differs from BN64 stride1024'))
 reject('old_tail_2048_row32_alias',lambda:require(len({q*2048+r*32+(c^((r&1)|((r&6)<<2))) for q in range(7) for r in range(64) for c in range(64)})==28672,'old tail aliases'))
 reject('wrong_C32_coordinate_count',lambda:require(len({ccoord(t,v) for t in range(128) for v in range(16)})==4096,'old P16 incomplete'))
 reject('N64_non_native_accept',lambda:require(native(128,64,128,128,64),'new N128 required'))
 reject('A_lda_stride_limit',lambda:require(native(1024,1024,1024,1<<39,1024),'A stride <2^39'))
 reject('AB8_not16_alignment',lambda:require(native(1024,1024,1024,1024,1024,A=0x1008),'AB16'))
 reject('B_partial_vector_stride',lambda:require(native(1024,1024,1024,1024,1025),'ldb multiple8'))
 b=Bank();b.acquire(0);reject('wrong_raw_TX_expect16384',lambda:b.expectation(16384));b.expectation(32768);b.write(0);b.fence(0)
 reject('publish_before_raw_A_done',lambda:b.arrive(0,0))
 for q in range(4):b.complete(q,0)
 b.write(1);reject('publish_before_own_proxy_fence',lambda:b.arrive(1,0))
 reject('duplicate_TMA_completion',lambda:b.complete(0,0))
 reject('stale_TMA_generation',lambda:b.complete(0,1))
 b=ready_bank();b.full=False
 reject('full_before128writers',lambda:b.read(0,0))
 b=ready_bank(1);reject('wrong_raw_phase_in_epoch1',lambda:b.arrive(0,0));reject('wrong_consumer_phase_in_epoch1',lambda:b.read(2,0))
 b=ready_bank();b.released={0};reject('bank_reuse_after_only_WG0_release',lambda:b.acquire(2))
 r=Role(0);reject('P_clear_before_INC',r.clear);reject('old192_not_new224',lambda:r.alloc(192));r.alloc(224);r.clear();r.transform();r.issue()
 reject('pending_whole_A_overwrite',r.overwriteA);reject('pending_whole_P_store_or_U',r.readP);reject('release_before_wait0',lambda:r.release(ready_bank()));reject('wait0_before_all_K16_atoms',r.wait0)
 reject('tail_before_both_banks_drain',lambda:require(all(b.released=={0,1} for b in [ready_bank(),ready_bank()]),'bothbanks drain'))
 reject('U_before_28672_tail_words',lambda:require(len(set(range(28671)))==28672,'complete7P required'))
 reject('initial_pool_167_insufficient',lambda:require(384*167>=128*(40+224+240),'new owned pool'))
 reject('role_highest_equal_quota_not_below',lambda:require(all(h<q for h,q in zip([40,223,239],[40,224,240])),'highest strictly below quota'))
 reject('wrong_descriptor_encoded_leading16B',lambda:require(1*16==256,'encoded leading16'))
 reject('tail_over_barrier_payload_boundary',lambda:require(7*64*64*4<=90112,'onebank insufficient for P tail'))
 # Independent negative fixed U controls, including an incomplete whole-K P.
 p,d=winograd([1,2,1,1,-1,1,2,1]);assert p[4]!=0;bad=list(fixed_u(p));bad[3]=p[0]+p[5]+p[6]-p[4]
 reject('fixed_U_wrong_U7_sign',lambda:require(tuple(bad)==d,'fixedU sign'))
 reject('partial_whole_P_dropped_K_term',lambda:require(fixed_u(p)==tuple(2*x for x in d),'wholeK accumulation'))

assert Path(str(P)+'candidate.cu').is_file()
g=geometry();math_model()
for stages in [2,4,8,16,32]:
 for seed in range(8):simulation(stages,421064+stages*101+seed)
negatives()
assert native(1024,1024,1024,1024,1024) and native(128,128,128,(1<<39)-8,128)
result={'status':'PASS_CPU_BN64_GEOMETRY_AND_PROTOCOL','model_scope':'Opaque integer layout/source DAG/state assertions only. No GPU simulation score or precision certification.',
 'candidate_sha256':hashlib.sha256(Path(str(P)+'candidate.cu').read_bytes()).hexdigest(),'geometry_prediction':g,'counts':count,
 'negative_control_count':len(neg),'rejected_negative_controls':neg,
 'whole_A_words_per_thread':[48,64],'whole_P_words_per_thread':[96,128],'source_quotas_only':[40,224,240],
 'tile_and_grid_hypothesis':{'BM':64,'BN':64,'BK':64,'banks':2,'threads':384,'grid_for_1024':[8,8],'CTA_count':64,'parent45_CTA_count':128,'source_only_A_reuse_change':'N extent per same rawA doubles; launch CTA count halves. More P/B/tail per CTA; performance tradeoff unknown.'},
 'original_FP64_gate_unchanged':True,'actual_CUDA_compiled':None,'actual_CAN':None,'actual_Driver_resources':None,'actual_GPR':None,'actual_initial_pool':None,'actual_sizeof':None,'actual_pending_lifetime':None,'actual_SAN':None,'actual_precision':None,'actual_performance':None}
print(json.dumps(result,indent=2))
