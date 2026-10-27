#!/usr/bin/env python3
"""CPU authoring only; reads frozen prior model as a template, writes new prefix only."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent.parent;p='.scratch/psplit_tma_bn16_root_future43_'
prior=R/'.scratch/psplit_tma_bn64_root_future42_model.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='04f26e951f684481e42c069a765460215a2f19eacf3626c0e0b2e2c3d3731b36'
s=prior.read_text().replace('psplit_tma_bn64_root_future42','psplit_tma_bn16_root_future43').replace('BN64','BN16').replace('90112','47104')
def replace(a,b):
 global s
 assert s.count(a)==1,(a[:90],s.count(a));s=s.replace(a,b)
replace('def raw_b(n,k,q):return q*4096+(k//8)*512+(n//16)*128+n%16+16*(k%8)','def raw_b(n,k,q):return q*1024+(k//8)*128+n%16+16*(k%8)')
replace('def ref_b(o,bank,n,k,q):\n brick_n,inner_n=divmod(n,16);brick_k,inner_k=divmod(k,8)\n return sw(o+bank*47104+32768+q*8192+brick_k*1024+brick_n*256+inner_k*32+inner_n*2,1)','def ref_b(o,bank,n,k,q):\n brick_k,inner_k=divmod(k,8)\n return sw(o+bank*47104+32768+q*2048+brick_k*256+inner_k*32+n*2,1)')
replace('def tail(q,r,c):return q*4096+r*64+(c^((r&1)|((r&6)<<2)))','def tail(q,r,c):return q*1024+r*16+(c^(((r&1)|((r&6)<<2))&15))')
a=s.index('def geometry():');b=s.index('\ndef winograd(x):',a);s=s[:a]+(R/(p+'geometry_fragment.py')).read_text().split('\n',1)[1]+s[b:]
replace('self.values=[0]*((3 if g==0 else 4)*32);self.P=(3 if g==0 else 4)*32','self.values=[0]*((3 if g==0 else 4)*8);self.P=(3 if g==0 else 4)*8')
s=s.replace('[224,240]','[128,128]').replace('m%128 or n%128 or k%128','m%128 or n%32 or k%128')
s=s.replace('wrong_B_BN32_leading_stride_counterfactual','wrong_B_BN32_stride_counterfactual')
replace("all(sw(32768+(n//16)*256+(n%16)*2+(k//8)*512+(k%8)*32,1)==b_at(0,0,n,k,0) for n in range(64) for k in range(16)),'BN32 stride512 differs from BN16 stride1024'","all(sw(32768+n*2+(k//8)*512+(k%8)*32,1)==b_at(0,0,n,k,0) for n in range(16) for k in range(16)),'BN32 stride512 differs from BN16 stride256'")
replace("len({q*2048+r*32+(c^((r&1)|((r&6)<<2))) for q in range(7) for r in range(64) for c in range(64)})==28672,'old tail aliases'","{q*2048+r*32+(c^((r&1)|((r&6)<<2))) for q in range(7) for r in range(64) for c in range(16)}==set(range(7168)),'old BN32 tail has wrong footprint'")
s=s.replace('old_tail_2048_row32_alias','old_tail_2048_row32_footprint')
replace("len({ccoord(t,v) for t in range(128) for v in range(16)})==4096,'old P16 incomplete'","len({ccoord(t,v) for t in range(128) for v in range(16)})==1024,'old P16 has excess columns'")
replace("'N64_non_native_accept',lambda:require(native(128,64,128,128,64),'new N128 required')","'N16_non_native_accept',lambda:require(native(128,16,128,128,16),'new N32 required')")
s=s.replace('old192_not_new224','old192_not_new128').replace('r.alloc(224)','r.alloc(128)')
replace("'U_before_28672_tail_words',lambda:require(len(set(range(28671)))==28672","'U_before_7168_tail_words',lambda:require(len(set(range(7167)))==7168")
replace("'initial_pool_167_insufficient',lambda:require(384*167>=128*(40+224+240)","'initial_pool_98_insufficient',lambda:require(384*98>=128*(40+128+128)")
replace('zip([40,223,239],[40,224,240])','zip([40,127,127],[40,128,128])')
replace("'wrong_descriptor_encoded_leading16B',lambda:require(1*16==256,'encoded leading16')","'wrong_descriptor_nonzero_degenerate_N_leading',lambda:require(16==0,'pinned CuTE extent1 leading field0; source conformance, unused coordinate')")
replace("'tail_over_barrier_payload_boundary',lambda:require(7*64*64*4<=47104,'onebank insufficient for P tail')","'tail_over_barrier_payload_boundary',lambda:require(94212<=94208,'tail payload cannot alias barrier')")
replace("'whole_A_words_per_thread':[48,64],'whole_P_words_per_thread':[96,128],'source_quotas_only':[40,224,240]","'whole_A_words_per_thread':[48,64],'whole_P_words_per_thread':[24,32],'source_quotas_only':[40,128,128], 'source_quota_pool_words':37888, 'two_CTA_quota_pool_words':75776, 'two_CTA_with_these_quotas_fits_65536':False")
s=s.replace("'BN':64","'BN':16").replace("'grid_for_1024':[8,8],'CTA_count':64","'grid_for_1024':[8,32],'CTA_count':256")
replace('N extent per same rawA doubles; launch CTA count halves. More P/B/tail per CTA; performance tradeoff unknown.','N extent per same rawA halves; launch CTA count doubles. More A transport/transform repetition, smaller P/B/tail per CTA; performance tradeoff unknown. Min1 only; current quota pool excludes two simultaneous full requests.')
anchor=" reject('wrong_C32_coordinate_count'";idx=s.index(anchor)
s=s[:idx]+" reject('unclipped_original_tail_not_row_local',lambda:require(all(0<= (c^((r&1)|((r&6)<<2)))<16 for r in range(64) for c in range(16)),'N16 row-local tail uses low4 mask; unmasked original may still be whole-tile bijective'))\n reject('two_CTA_quota_pool_exceeds_SM_registers',lambda:require(2*128*(40+128+128)<=65536,'single candidate min1; quota pool75776'))\n"+s[idx:]
assert 'future42' not in s and '64512' not in s and 'source_quotas_only\':[40,224,240]' not in s
assert 'warpgroup_reg_alloc<240>' not in s and 'warpgroup_reg_alloc<224>' not in s
(R/(p+'model.py')).write_text(s)
print(json.dumps({'status':'CPU_MODEL_ASSEMBLED_NEW_PREFIX','model_sha256':hashlib.sha256(s.encode()).hexdigest(),'prior_frozen_model_unchanged':True,'actual_CUDA_compiled':None,'actual_GPU':None},indent=2))
