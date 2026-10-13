#!/usr/bin/env python3
"""Independent CPU raw-copy/collective-order model, never a hardware simulator."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import struct

ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'.scratch/psplit_sw128_simt_control_root_future41_'
BM,BN,BK=64,32,64
AEL,BEL=4*BM*BK,7*BN*BK
BANK=2*(AEL+BEL)
COUNTS=Counter();NEGATIVE=[]

class Violation(ValueError):pass
def need(ok,message):
    if not ok:raise Violation(message)
def rejected(name,fn,tag):
    try:fn()
    except Violation as e:
        assert tag in str(e),(name,str(e),tag);NEGATIVE.append({'name':name,'violation':str(e)})
    else:raise AssertionError('negative control accepted: '+name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sw(byte,bits=3):return byte^((byte&(((1<<bits)-1)<<7))>>3)
def raw_A(row,k,q):return q*4096+row*64+k
def raw_A32(row,k,q):return (q*8*4+(k//16)*8+row//8)*128+(row%8)*16+k%16
def raw_B(n,k,q):return (q*2*8+(k//8)*2+n//16)*128+n%16+(k%8)*16
def opaque_halfword(row,k):return ((row*40503)^(k*61937)^(row>>6)^((k>>3)*17))&65535
def rs(t,v,atom):
    linear=128*(t%4)+(t//4%8)+16*(t//32%4)+64*(v%2)+8*(v//2%2)+512*(v//4%2)
    return linear%64,linear//64+16*atom

# Exhaustive literal CuTe flagged-pointer vs independent row/chunk permutation.
for base in [0,1024,2048,8192,65536]:
    need(base%1024==0,'CONTROL_ABSOLUTE_ALIGNMENT')
    occupied=set()
    for bank in range(2):
        origin=base+bank*BANK
        for q in range(4):
            for row in range(64):
                for k in range(0,64,8):
                    vector=[sw(origin+2*raw_A(row,k+j,q)) for j in range(8)]
                    independent=[origin+q*8192+row*128+2*((k+j)^((row&7)<<3)) for j in range(8)]
                    need(vector==independent,'SW128_DENSE_COORDINATE_MISMATCH')
                    need(vector[0]%16==0 and vector==list(range(vector[0],vector[0]+16,2)),'SIMT_VECTOR8_ALIGNMENT_OR_CONTIGUITY')
                    need(not occupied.intersection(vector),'RAW_A_WRITE_ALIAS')
                    occupied.update(vector)
                    COUNTS['A_vector8_absolute_geometry']+=1
                    COUNTS['A_halfword_absolute_geometry']+=8
            for t in range(128):
                for atom in range(4):
                    for pair in range(4):
                        row,k=rs(t,2*pair,atom);row1,k1=rs(t,2*pair+1,atom)
                        a=sw(origin+2*raw_A(row,k,q));b=sw(origin+2*raw_A(row1,k1,q))
                        need(row==row1 and k1==k+1 and a%4==0 and b==a+2,'RS_PACKED_PAIR_MISMATCH')
                        COUNTS['consumer_RS_pair_absolute_geometry']+=1
        for q in range(7):
            for k in range(64):
                for n in range(0,32,8):
                    vector=[sw(origin+2*AEL+2*raw_B(n+j,k,q),1) for j in range(8)]
                    need(vector[0]%16==0 and vector==list(range(vector[0],vector[0]+16,2)),'UNCHANGED_B_VECTOR_GEOMETRY')
                    need(not occupied.intersection(vector),'A_B_BANK_ALIAS')
                    occupied.update(vector);COUNTS['B_unchanged_vector8_geometry']+=1
    need(occupied==set(range(base,base+2*BANK,2)),'TWO_BANK_FULL_PAYLOAD_BIJECTION')
    COUNTS['complete_two_bank_absolute_bases']+=1

# Raw opaque16-bit payload checks in native1024, all8 A-row CTAs/all8 K stages.
# Column CTAs use identical A addressing; there is no col_base in fill_a.
for lda in [1024,1032,1040,(1<<32),(1<<39)-8]:
    need(lda%8==0 and lda>=1024 and lda<(1<<39),'CONTROL_A_STRIDE_GUARD')
    global_positions=set() if lda==1024 else None
    for x in range(8):
        for stage in range(8):
            seen=set();payload={}
            for tid in range(128):
                own_tasks=list(range(tid,512,128));need(len(own_tasks)==4,'WRITER_A_TASK_COUNT')
                for task in own_tasks:
                    row=task//8;k=(task%8)*8
                    for q in range(4):
                        gr=x*64+row+(512 if q>=2 else 0)
                        gk=stage*64+k+(512 if q&1 else 0)
                        gbase=gr*lda+gk
                        need((2*gbase)%16==0 and 0<=gr<1024 and 0<=gk<=1016,'CURRENT_A_VECTOR_BOUNDS_ALIGNMENT')
                        physical=sw(2*raw_A(row,k,q))
                        # Independent load.v4.u32/store.v4.b32 opaque payload.
                        words=struct.pack('<8H',*[opaque_halfword(gr,gk+j) for j in range(8)])
                        lane_words=struct.unpack('<8H',words)
                        for j,value in enumerate(lane_words):
                            logical=(q,row,k+j);need(logical not in seen,'SIMT_A_DUPLICATE_WRITER')
                            seen.add(logical);payload[physical+2*j]=value
                            if global_positions is not None:global_positions.add((gr,gk+j))
                        COUNTS['opaque_uint4_raw_A_vectors']+=1
                        COUNTS['global_A_halfword_address_checks']+=8
            need(len(seen)==16384 and set(payload)==set(range(0,32768,2)),'STAGE_RAW_A_COVERAGE')
            for q in range(4):
                for row in range(64):
                    for k in range(64):
                        gr=x*64+row+(512 if q>=2 else 0)
                        gk=stage*64+k+(512 if q&1 else 0)
                        need(payload[sw(2*raw_A(row,k,q))]==opaque_halfword(gr,gk),'RAW_A_ROUNDTRIP_BITS')
                        COUNTS['raw_A_halfword_roundtrip']+=1
    if global_positions is not None:need(len(global_positions)==1024*1024,'PACKED_A_ENTIRE_MATRIX_COVERAGE')
    COUNTS['A_stride_domains']+=1

def mixed_layout(use_old_raw,use_old_swizzle):
    payload={}
    for q in range(4):
        for row in range(64):
            for k in range(64):
                raw=raw_A32(row,k,q) if use_old_raw else raw_A(row,k,q)
                payload[sw(2*raw,1 if use_old_swizzle else 3)]=q*4096+row*64+k
    wrong=sum(payload[sw(2*raw_A(row,k,q))]!=(q*4096+row*64+k)
      for q in range(4) for row in range(64) for k in range(64))
    need(wrong==0,'WRONG_SW32_WRITER_WITH_SW128_CONSUMER')
rejected('counterfactual_full_SW32_writer_SW128_reader',lambda:mixed_layout(True,True),'WRONG_SW32')
rejected('counterfactual_dense_writer_SW32_swizzle_only',lambda:mixed_layout(False,True),'WRONG_SW32')
rejected('misaligned_raw_A_vector',lambda:need(sw(2*raw_A(0,4,0))%16==0,'SIMT_VECTOR8_ALIGNMENT'),'SIMT_VECTOR8_ALIGNMENT')
rejected('A_stride_at_exclusive_2pow39',lambda:need((1<<39)<(1<<39),'CONTROL_A_STRIDE_GUARD'),'CONTROL_A_STRIDE_GUARD')
rejected('shared_control_only512_aligned',lambda:need(512%1024==0,'CONTROL_ABSOLUTE_ALIGNMENT'),'CONTROL_ABSOLUTE_ALIGNMENT')

class State:
    """Each producer lane modeled separately; WGs are abstract collective actors."""
    WRITER=('acquire','A','B','fence','arrive')
    CONSUMER=('wait','issue','retire','release')
    def __init__(self,n):
        self.n=n;self.w=defaultdict(int);self.c=defaultdict(int);self.arrivals=defaultdict(set)
        self.released=defaultdict(set);self.pending=set();self.tail=set();self.drain=False
        self.stores=set();self.Uready=False;self.done=False;self.issued=defaultdict(list)
        self.full_phase=[0,0];self.empty_phase=[0,0];self.phase_assertions=0
    def apply(self,event):
        kind,s,actor=event
        if kind=='init0':return
        if kind in self.WRITER:
            need(0<=s<self.n and 0<=actor<128,'WRITER_EVENT_DOMAIN')
            need(self.w[s,actor]<5 and self.WRITER[self.w[s,actor]]==kind,'WRITER_PROGRAM_ORDER')
            if kind=='acquire':
                need(s==0 or self.w[s-1,actor]==5,'PER_LANE_STAGE_ORDER')
                need(s<2 or self.released[s-2]=={0,1},'BANK_REUSE_BEFORE_BOTH_WG_RELEASE')
                need(self.empty_phase[s&1]!=(1^((s//2)&1)),'PRODUCER_EMPTY_PHASE_MISMATCH')
                self.phase_assertions+=1
            if kind=='arrive':
                self.arrivals[s].add(actor)
                if len(self.arrivals[s])==128:
                    self.full_phase[s&1]^=1
                    need(self.full_phase[s&1]!=(s//2&1),'COLLECTIVE_FULL_PHASE_MISMATCH')
                    self.phase_assertions+=1
            self.w[s,actor]+=1
        elif kind in self.CONSUMER:
            need(0<=s<self.n and actor in (0,1),'CONSUMER_EVENT_DOMAIN')
            need(self.c[s,actor]<4 and self.CONSUMER[self.c[s,actor]]==kind,'CONSUMER_PROGRAM_ORDER')
            if kind=='wait':
                need(self.arrivals[s]==set(range(128)),'PUBLICATION_BEFORE_ALL_128_A_B_WRITERS_FENCE_ARRIVE')
                need(s==0 or actor in self.released[s-1],'WHOLE_K_ASCENDING_STAGE_ORDER')
                need(self.full_phase[s&1]!=(s//2&1),'CONSUMER_FULL_PHASE_MISMATCH')
                self.phase_assertions+=1
            if kind=='issue':
                self.pending.add((s,actor))
                for product in ((0,1,2) if actor==0 else (3,4,5,6)):
                    self.issued[product].extend((s,atom) for atom in range(4))
            if kind=='retire':
                need((s,actor) in self.pending,'RETIRE_WITHOUT_WGMMA')
                self.pending.remove((s,actor))
            if kind=='release':
                need((s,actor) not in self.pending,'RELEASE_WITH_PENDING_A_P')
                self.released[s].add(actor)
                if self.released[s]=={0,1}:
                    self.empty_phase[s&1]^=1
                    need(self.empty_phase[s&1]!=(1^(((s+2)//2)&1)),'COLLECTIVE_EMPTY_PHASE_MISMATCH')
                    self.phase_assertions+=1
            self.c[s,actor]+=1
        elif kind=='tail':
            need(self.w[self.n-1,actor]==5,'TAIL_BEFORE_OWN_FINAL_ARRIVAL')
            need(all(self.released[s]=={0,1} for s in range(max(0,self.n-2),self.n)),'TAIL_BEFORE_BOTH_BANKS_DRAINED')
            self.tail.add(actor)
        elif kind=='drain6':
            need(self.tail==set(range(128)) and self.released[self.n-1]=={0,1} and not self.pending,'CTA6_BEFORE_PRODUCER_TAIL_CONSUMER_DRAIN')
            self.drain=True
        elif kind=='store_tail':
            need(self.drain,'TAIL_ALIAS_BEFORE_CTA6');self.stores.add(actor)
        elif kind=='Ubarrier0':
            need(self.drain and self.stores=={0,1},'U_BARRIER_BEFORE_ALL_SEVEN_P_STORES');self.Uready=True
        elif kind=='U':
            need(self.Uready,'U_LOAD_BEFORE_FINAL_CTA0');self.done=True
        else:raise Violation('UNKNOWN_MODEL_EVENT')
    def finish(self):
        need(self.done and len(self.tail)==128,'INCOMPLETE_MAIN_TAIL_U_MODEL')
        expected=[(s,a) for s in range(self.n) for a in range(4)]
        need(all(self.issued[p]==expected for p in range(7)),'SEVEN_WHOLE_P_K16_ORDER')
        need(all(self.arrivals[s]==set(range(128)) and self.released[s]=={0,1} for s in range(self.n)),'INCOMPLETE_STAGE_COLLECTIVES')

def dependencies(n):
    graph={('init0',-1,-1):set()}
    for s in range(n):
        for tid in range(128):
            prior={('init0',-1,-1)} if s==0 else {('arrive',s-1,tid)}
            if s>=2:prior|={('release',s-2,wg) for wg in (0,1)}
            for kind in State.WRITER:
                node=(kind,s,tid);graph[node]=set(prior);prior={node}
        for wg in (0,1):
            prior={('arrive',s,tid) for tid in range(128)}
            if s:prior.add(('release',s-1,wg))
            for kind in State.CONSUMER:
                node=(kind,s,wg);graph[node]=set(prior);prior={node}
    for tid in range(128):
        graph['tail',-1,tid]={('arrive',n-1,tid)}|{('release',s,wg) for s in range(max(0,n-2),n) for wg in (0,1)}
    graph['drain6',-1,-1]={('tail',-1,tid) for tid in range(128)}|{('release',n-1,wg) for wg in (0,1)}
    for wg in (0,1):graph['store_tail',-1,wg]={('drain6',-1,-1)}
    graph['Ubarrier0',-1,-1]={('store_tail',-1,wg) for wg in (0,1)}
    graph['U',-1,-1]={('Ubarrier0',-1,-1)}
    return graph

for n in (1,2,3,4,8,9,16):
    graph=dependencies(n)
    for repeat in range(16):
        rng=random.Random(7919*n+repeat);degree={event:len(deps) for event,deps in graph.items()};successors=defaultdict(list)
        for event,deps in graph.items():
            for dep in deps:successors[dep].append(event)
        ready=[e for e,d in degree.items() if d==0];state=State(n);visited=0
        while ready:
            index=rng.randrange(len(ready));event=ready[index];ready[index]=ready[-1];ready.pop()
            state.apply(event);visited+=1
            for successor in successors[event]:
                degree[successor]-=1
                if degree[successor]==0:ready.append(successor)
        need(visited==len(graph),'DEADLOCK_OR_CYCLE');state.finish()
        COUNTS['valid_per_writer_stage_schedules']+=1
        COUNTS['validated_stage_schedule_events']+=visited
        COUNTS['validated_producer_lane_phase_steps']+=128*n
        COUNTS['explicit_full_empty_phase_assertions']+=state.phase_assertions
        COUNTS['validated_whole_P_K16_atoms']+=7*n*4

def one_lane_arrived(state,s=0,tid=0):
    for kind in State.WRITER:state.apply((kind,s,tid))
early=State(8);one_lane_arrived(early)
rejected('publish_after_only_one_producer_lane',lambda:early.apply(('wait',0,0)),'PUBLICATION_BEFORE_ALL_128')
missing=State(8)
for tid in range(127):one_lane_arrived(missing,tid=tid)
rejected('publish_missing_writer127',lambda:missing.apply(('wait',0,1)),'PUBLICATION_BEFORE_ALL_128')
unfenced=State(8)
for kind in ('acquire','A','B'):unfenced.apply((kind,0,0))
rejected('arrival_without_proxy_fence',lambda:unfenced.apply(('arrive',0,0)),'WRITER_PROGRAM_ORDER')
noA=State(8);noA.apply(('acquire',0,0))
rejected('B_before_raw_A_completion',lambda:noA.apply(('B',0,0)),'WRITER_PROGRAM_ORDER')
reuse=State(8)
one_lane_arrived(reuse,0,0);one_lane_arrived(reuse,1,0);reuse.released[0]={0}
rejected('bank_reuse_after_WG0_only',lambda:reuse.apply(('acquire',2,0)),'BANK_REUSE_BEFORE_BOTH')
tail=State(8);tail.w[7,0]=5;tail.released[6]={0,1};tail.released[7]={0}
rejected('tail_before_last_WG1_release',lambda:tail.apply(('tail',-1,0)),'TAIL_BEFORE_BOTH_BANKS')
pending=State(8);pending.arrivals[0]=set(range(128));pending.full_phase[0]=1;pending.apply(('wait',0,0));pending.apply(('issue',0,0))
rejected('release_before_wait0_postfences',lambda:pending.apply(('release',0,0)),'CONSUMER_PROGRAM_ORDER')
bad=State(8)
rejected('typed_tail_alias_before_bank_drain_barrier',lambda:bad.apply(('store_tail',-1,0)),'TAIL_ALIAS_BEFORE_CTA6')
rejected('U_before_both_tail_store_roles',lambda:bad.apply(('Ubarrier0',-1,-1)),'U_BARRIER_BEFORE_ALL_SEVEN')
rejected('U_load_before_final_CTA0',lambda:bad.apply(('U',-1,-1)),'U_LOAD_BEFORE_FINAL_CTA0')
bad=State(8);bad.tail=set(range(128));bad.released[7]={0,1};bad.pending={(7,1)}
rejected('drain6_with_pending_A_P',lambda:bad.apply(('drain6',-1,-1)),'CTA6_BEFORE_PRODUCER_TAIL_CONSUMER_DRAIN')
need(BANK==61440 and 2*BANK==122880,'BANK_BYTES')
need(((122880+32+16+1023)//1024)*1024==123904,'SHARED_CONTROL_SIZE_LAYOUT')
rejected('raw_done_aliases_operand_tail_union',lambda:need(122872>=122880+32,'RAW_DONE_ALIASES_OPERANDS_OR_PIPELINE'),'RAW_DONE_ALIASES')
bad=State(8);bad.empty_phase[0]=1
rejected('incorrect_initial_empty_phase',lambda:bad.apply(('acquire',0,0)),'PRODUCER_EMPTY_PHASE_MISMATCH')
bad=State(8);bad.arrivals[0]=set(range(128))
rejected('incorrect_full_phase_at_consumer_wait',lambda:bad.apply(('wait',0,0)),'CONSUMER_FULL_PHASE_MISMATCH')

binding=json.loads(Path(str(P)+'bindings.json').read_text())
for key in ('protected_future_before_sha256','library_source_before_sha256','pinned_header_before_sha256'):
    need(all(sha(ROOT/name)==want for name,want in binding[key].items()),'PROTECTED_BYTES_DRIFT')
print(json.dumps({'status':'PASS_CPU_ONLY','counts':dict(COUNTS),'negative_controls_rejected':len(NEGATIVE),
 'negative_controls':NEGATIVE,'old_future_files_unchanged':71,'library_source_files_unchanged':len(binding['library_source_before_sha256']),
 'new_shared_storage_source_layout_expected_bytes':123904,'new_actual_sizeof':None,'new_actual_register_resource':None,
 'CUDA_compiled':False,'GPU_launched':False,'independent_CAN':None,'actual_GPU_precision':None,'actual_GPU_performance':None,
 'scope':'Raw opaque A bits/absolute SW128 geometry and logical lane/collective order only; not hardware scheduling, compiled ABI, precision or performance.'},indent=2))
