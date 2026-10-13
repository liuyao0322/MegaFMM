"""CPU-only explicit CuTe layout/DAG/register-lifetime hypothesis checks."""
from pathlib import Path
import hashlib,json,collections
import numpy as np
R=Path('/share/home/liuyao/workspace/MegaFMM')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sw(a):return a^((a&128)>>3)
def raw_a(r,k,q,bk):return (q*8*(bk//16)+(k//16)*8+r//8)*128+(r%8)*16+k%16
def raw_b(n,k,q,bk):return (q*2*(bk//8)+(k//8)*2+n//16)*128+n%16+(k%8)*16
def rs_coord(t,v,atom):
 x=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%2)
 return x%64,x//64+16*atom
def counts(bk):
 a=64*bk;b=32*bk;s=512//bk
 c={'BK':bk,'stages_1024':s,'CTA_grid_1024':[8,16],'CTAs':128,
    'logical_A_cells':a,'logical_B_cells':b,
    'producer_original_A_read_bytes':4*a*2,'producer_original_B_read_bytes':4*b*2,
    'SS_producer_A_write_bytes':7*a*2,'RS_producer_A_write_bytes':4*a*2,
    'producer_final_B_write_bytes':7*b*2,'RS_consumer_A_raw_read_bytes':7*a*2,
    'RS_consumer_WG0_A_quadrant_reads':4,'RS_consumer_WG1_A_quadrant_reads':3,
    'SS_logical_A_operand_cells_consumed':7*a,'RS_generic_raw_A_cells_read':7*a,
    'SS_A_RN_operations':4*a,'RS_A_RN_operations':6*a,
    'RS_A_RN_WG0':3*a,'RS_A_RN_WG1':3*a,'B_RN_operations':4*b,
    'A_final_onecasts':4*a,'B_final_onecasts':4*b,
    'RS_A_onecasts_WG0':a,'RS_A_onecasts_WG1':3*a,
    'SS_total_ST_RN':4*a+4*b,'RS_total_ST_RN':6*a+4*b,
    'SS_producer_ST_RN':4*a+4*b,'RS_producer_ST_RN':4*b,
    'SS_producer_ST_onecasts':4*a+4*b,'RS_producer_ST_onecasts':4*b,
    'raw_operand_bank_bytes':(4*a+7*b)*2,'raw_operand_two_banks_bytes':2*(4*a+7*b)*2,
    'SS_operand_two_banks_bytes':2*(7*a+7*b)*2,
    'tail_bytes':7*64*32*4,'barrier_expected_bytes':32,
    'WG0_full_stage_packed_A_words':3*(bk//16)*4,
    'WG1_full_stage_packed_A_words':4*(bk//16)*4,
    'WG0_FP32_P_registers':48,'WG1_FP32_P_registers':64}
 c['expected_shared_align128']=((c['raw_operand_two_banks_bytes']+32+127)//128)*128
 c['tail_overlaps_bank1_bytes']=max(0,c['tail_bytes']-c['raw_operand_bank_bytes'])
 c['tail_read_write_bytes_per_CTA']=2*c['tail_bytes']
 for k in ['producer_original_A_read_bytes','producer_original_B_read_bytes','SS_producer_A_write_bytes','RS_producer_A_write_bytes','producer_final_B_write_bytes','RS_consumer_A_raw_read_bytes','SS_total_ST_RN','RS_total_ST_RN','SS_producer_ST_RN','RS_producer_ST_RN','A_final_onecasts','B_final_onecasts']:
  c['wholeCTA_'+k]=c[k]*s
 c['wholeCTA_SS_operand_shared_writes']=(7*a+7*b)*2*s
 c['wholeCTA_RS_operand_shared_writes']=(4*a+7*b)*2*s
 return c
def geometry(bk):
 c=counts(bk);a=64*bk;b=32*bk;bank=c['raw_operand_bank_bytes'];vector_checks=0;pair_checks=0
 assert {raw_a(r,k,q,bk) for q in range(4) for r in range(64) for k in range(bk)}==set(range(4*a))
 assert {raw_b(n,k,q,bk) for q in range(7) for n in range(32) for k in range(bk)}==set(range(7*b))
 for base in [0,128,256,384]:
  all_addr=set()
  for stage in range(2):
   origin=base+stage*bank
   for q in range(4):
    for r in range(64):
     for k in range(0,bk,8):
      ad=[sw(origin+2*raw_a(r,k+j,q,bk)) for j in range(8)]
      assert ad[0]%16==0 and ad==list(range(ad[0],ad[0]+16,2))
      assert not all_addr.intersection(ad);all_addr.update(ad);vector_checks+=1
   for q in range(7):
    for k in range(bk):
     for n in range(0,32,8):
      ad=[sw(origin+2*4*a+2*raw_b(n+j,k,q,bk)) for j in range(8)]
      assert ad[0]%16==0 and ad==list(range(ad[0],ad[0]+16,2))
      assert not all_addr.intersection(ad);all_addr.update(ad);vector_checks+=1
   for t in range(128):
    for atom in range(bk//16):
     for pair in range(4):
      r,k=rs_coord(t,2*pair,atom);r1,k1=rs_coord(t,2*pair+1,atom)
      assert r1==r and k1==k+1
      for q in range(4):
       ad=sw(origin+2*raw_a(r,k,q,bk));ad1=sw(origin+2*raw_a(r1,k1,q,bk))
       assert ad%4==0 and ad1==ad+2;pair_checks+=1
  assert all_addr==set(range(base,base+2*bank,2))
 coords=[rs_coord(t,v,atom) for t in range(128) for atom in range(bk//16) for v in range(8)]
 assert len(coords)==a and len(set(coords))==a
 max_bank=0;bank_instructions=0
 for base in [0,128,256,384]:
  for stage in range(2):
   for warp in range(4):
    for atom in range(bk//16):
     for pair in range(4):
      for q in range(4):
       addrs=[]
       for lane in range(32):
        r,k=rs_coord(32*warp+lane,2*pair,atom)
        addrs.append(sw(base+stage*bank+2*raw_a(r,k,q,bk)))
       hist=collections.Counter((a//4)%32 for a in set(addrs))
       max_bank=max(max_bank,max(hist.values()));bank_instructions+=1
 assert max_bank==1
 return {'status':'PASS_EXPLICIT_LITERAL_LAYOUT_HYPOTHESIS_NOT_CUDA_INSTANTIATION','vector8_physical_checks':vector_checks,'aligned_contiguous_RS_pair_checks':pair_checks,'RS_partition_unique_A_cells':a,'generic_raw_A_u32_warp_load_bank_checks':bank_instructions,'generic_raw_A_worst_distinct_words_per_bank':max_bank,'bank_scope':'Only scalar generic rawA layout loads; no inference about actual compiled instruction grouping or WGMMA hardware bank traffic','barrier_union_intersection_bytes':0}
def numerical():
 out=[]
 with np.errstate(over='ignore',invalid='ignore',under='ignore'):
  for dtype in ['BF16','FP16']:
   all_bits=np.arange(65536,dtype=np.uint16)
   pairs=all_bits.astype(np.uint32)|(all_bits[::-1].astype(np.uint32)<<16)
   assert np.array_equal((pairs&65535).astype(np.uint16),all_bits)
   assert np.array_equal((pairs>>16).astype(np.uint16),all_bits[::-1])
   finite=all_bits[(all_bits& (0x7f80 if dtype=='BF16' else 0x7c00))!=(0x7f80 if dtype=='BF16' else 0x7c00)]
   n=len(finite);ix=np.arange(n,dtype=np.uint32)
   w=[finite[ix],finite[(ix*17+31)%n],finite[(ix*37+7)%n],finite[(ix*53+101)%n]]
   def unpack(v):return (v.astype(np.uint32)<<16).view(np.float32) if dtype=='BF16' else v.view(np.float16).astype(np.float32)
   def cast(v):
    if dtype=='FP16':return v.astype(np.float16).view(np.uint16)
    u=v.view(np.uint32);assert not np.isnan(v).any()
    return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
   f11,f12,f21,f22=[unpack(v) for v in w]
   s1=np.add(f21,f22,dtype=np.float32);s2=np.subtract(s1,f11,dtype=np.float32)
   s3=np.subtract(f11,f21,dtype=np.float32);s4=np.subtract(f12,s2,dtype=np.float32)
   ss=[w[0],w[1],cast(s4),w[3],cast(s1),cast(s2),cast(s3)]
   g0s1=np.add(f21,f22,dtype=np.float32);g0s2=np.subtract(g0s1,f11,dtype=np.float32)
   g0s4=np.subtract(f12,g0s2,dtype=np.float32)
   g1s1=np.add(f21,f22,dtype=np.float32);g1s2=np.subtract(g1s1,f11,dtype=np.float32)
   g1s3=np.subtract(f11,f21,dtype=np.float32)
   rs=[w[0],w[1],cast(g0s4),w[3],cast(g1s1),cast(g1s2),cast(g1s3)]
   mismatch=sum(np.count_nonzero(a!=b) for a,b in zip(ss,rs));assert mismatch==0
   out.append({'dtype':dtype,'finite_quadruples':n,'seven_product_A16bit_comparisons':7*n,'mismatches':int(mismatch),'all_original_raw16bit_patterns_bitcopy_checks':65536,'limitations':'NumPy FP32/RNE analytic emulation of ordered source; no CUDA converter/FTZ/ABI/NaN payload/FP64 full-GEMM gate proof'})
 return out
def output_geometry():
 def qcoord(t,v):
  x=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%4)
  return x%64,x//64
 def tail(p,r,c):return p*2048+r*32+(c^((r&1)|((r&6)<<2)))
 q=[qcoord(t,v) for t in range(128) for v in range(16)]
 assert len(set(q))==2048
 pwords={tail(p,r,c) for p in range(7) for r,c in q};assert pwords==set(range(14336))
 owners=[(t,l//32,l%32) for t in range(384) for l in range(t,2048,384)]
 assert len(owners)==len({(r,c) for t,r,c in owners})==2048
 reads=collections.Counter(tail(p,r,c) for t,r,c in owners for p in range(7))
 assert set(reads)==pwords and set(reads.values())=={1}
 outs={((64*x+r+512*qr)*1024+32*y+c+512*qc) for x in range(8) for y in range(16) for t,r,c in owners for qr in range(2) for qc in range(2)}
 assert outs==set(range(1024*1024))
 return {'P_unique_words':14336,'U_unique_owners':2048,'each_P_word_read_count':1,'full1024_unique_outputs':len(outs),'exact_C0_D_alias':'Source four C0 loads precede four D stores; unique four-quadrant owner. Existing equalldc/ldd, overlap/span guards still required in future host integration.'}
def lifetime(np_,stages=8):
 labels={(p,k,w) for p in range(np_) for k in range(4) for w in range(4)}
 trace=[];pending=set();writes=0;MMA=0;maxpending=0
 def access(regs,why):
  assert not pending.intersection(regs),(why,pending.intersection(regs))
 for stage in range(stages):
  for label in sorted(labels):access({label},'prebuild A word');writes+=1
  access(labels,'preissue +r tensor fences');trace.append([stage,'built/fenced_all_A',len(labels),len(pending)])
  for p in range(np_):
   for k in range(4):
    # Access by WGMMA itself; accumulated same-shapeP is allowed. No generic A read/write.
    ar={(p,k,w) for w in range(4)};assert len(ar)==4 and not pending.intersection(ar)
    pending.update(ar);MMA+=1;maxpending=max(maxpending,len(pending));trace.append([stage,'issue',p,k,len(pending)])
  trace.append([stage,'commit_one_group',len(pending)])
  pending.clear();trace.append([stage,'wait0_complete',0])
  access(labels,'postwait +r tensor fences');trace.append([stage,'postwait_fence_then_release',0])
 assert not pending
 # Negative controls catch a temptingfour-word reuse and accessing (+r) a pending A source.
 bad={(0,0,w) for w in range(4)};pending.update(bad)
 rejects={name:bool(pending.intersection(regs)) for name,regs in [('four_word_rewrite_after_first_MMA',bad),('generic_read_or_fence_before_wait',bad)]}
 assert all(rejects.values());pending.clear()
 return {'products':np_,'K16_atoms_per_P':4,'packed_A_words_per_thread':len(labels),'FP32_P_words_per_thread':16*np_,'payload_lower_bound_registers_per_thread':len(labels)+16*np_,'stages':stages,'A_word_writes':writes,'MMA_issues':MMA,'peak_pending_A_words':maxpending,'source_after_wait_liveness_fences_required':True,'generic_A_access_pending_intersections':0,'negative_controls_rejected':rejects,'trace':trace,'scope':'Conservative source virtual-register model; actual allocation, spills, hidden copies/widths/PTX/SASS still pending'}
candidate=R/'.scratch/psplit_rsA_future35_candidate.cu';src=candidate.read_text()
assert src.count('barrier.sync 6,384;')==2 and src.count('warpgroup_wait<0>();')==2
assert src.count('alignas(16) Element a')==2
assert 'GMMA::rs_op_selector' in src and 'Element a0_data' in src and 'warpgroup_reg_alloc<192>()' in src
assert 'tCrA' not in src and 'p.k%128 == 0' in src
for s in ['fence.proxy.async.shared::cta','__fadd_rn(p1,p2)','__fmul_rn(p.alpha,u1)','__fmaf_rn(p.beta,c11,d11)','float* tail = storage.tail_products;']:assert s in src
baseline=(R/'results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu').read_text()
def Utext(x):return x[x.index('    const float p1 ='):x.index('\ntemplate <class Element>\nvoid prepare_')]
assert Utext(src).replace('psplit_rsA','psplit')==Utext(baseline)
headers=['third_party/cutlass/include/cute/arch/mma_sm90.hpp','third_party/cutlass/include/cute/arch/mma_sm90_gmma.hpp','third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp','third_party/cutlass/include/cute/atom/mma_atom.hpp','third_party/cutlass/include/cute/algorithm/gemm.hpp','third_party/cutlass/include/cute/layout.hpp','third_party/cutlass/include/cute/pointer_flagged.hpp','third_party/cutlass/include/cute/swizzle.hpp','third_party/cutlass/include/cutlass/half.h','third_party/cutlass/include/cutlass/bfloat16.h','third_party/cutlass/include/cutlass/numeric_conversion.h','third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp']
traits=(R/headers[2]).read_text();assert 'using ALayout_64x16 = CLayout_64x16;' in traits
assert 'using CLayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,' in traits
assert 'Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;' in traits
assert (R/headers[1]).read_text().count('using ARegisters = uint32_t[4];')>0
cs=[counts(32),counts(64)]
for k in cs[0]:
 if k.startswith('wholeCTA_'):assert cs[0][k]==cs[1][k],k
report={'status':'PASS_CPU_MODEL_ONLY_UNCOMPILED_RS_A_FUTURE','evidence_SHA256':{str(p.relative_to(R)):sha(p) for p in [candidate,R/'results/psplit_revision6_megakernel_wgmma_psplit_iter34.cu',R/'results/psplit_bk64_frozen_psplit_bk64_candidate_iter35.cu',*[R/h for h in headers]]},'counts_are_analytic_not_measured_traffic_stalls_or_performance':True,'counts_BK32_BK64':cs,'geometry':[geometry(32),geometry(64)],'numeric_source_equivalence':numerical(),'A_pending_lifetimes':[lifetime(3),lifetime(4)],'source_budget_hypothesis':{'producer':64,'consumer0':192,'consumer1':192,'final_pool_registers':57344,'minimum_initial_gran8':152,'maximum_initial_hardware_gran8_384threads_65536regs':168,'CPU_candidates_initial':[152,160,168],'pool_directions':{i:{'initial_pool':384*i,'producer_release':128*(i-64),'consumer_each_request':128*(192-i)} for i in [152,160,168]},'source_minblocks':1,'actual_allocation_widths_spills_shared_occupancy':None},'PTX_primary':{'url':'https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group','version':'ISA8.4/CUDA12.4','sections':['9.7.14.7.1 wgmma.fence','9.7.14.7.3 wgmma.wait_group'],'rule':'Generic A fragment register accesses require completion of their own MMA group; writes/fences on pending A undefined. BeforefirstMMA and aftergenericaccess, wgmma.fence orders inputs. Candidateprebuildsallstage and postwait +r liveness fences.'},'not_claimed':['CUDA compilation/API instantiation','physical live register allocations beyond sourcevirtual labels','fullSASS CFG or asynchronous input overlap proof','FP64 originalnumerical gate resolved','CUDA sanitizer/forwardprogress','net faster/cuBLAS speedup','actual WGMMA bank traffic']}
report['output_geometry']=output_geometry()
report['packed_A_alignment_source']='Explicit alignas(16) plain Element arrays, no initializer; natural2byte half/BF16 alignment alone does not guarantee uint32 recast.'
report['PTX_primary']['url']='https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html'
report['fixed_U_alpha_beta_normalized_source_byte_equal_to_fdaae75a']=True
out=R/'.scratch/psplit_rsA_future35_review.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'review':str(out),'sha256':sha(out),'numeric':report['numeric_source_equivalence'],'geometry':report['geometry'],'budget':report['source_budget_hypothesis']},indent=2))
