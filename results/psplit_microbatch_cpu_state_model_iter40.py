"""Bounded scalar RN DAG and abstract per-product A reuse model for Mega44.

This CPU-only program does not prove CUDA memory ordering, register allocation,
lane convergence or FP64 accuracy. It never writes a file or runs frozen mains.
"""
import hashlib
import json
import random
import struct
from pathlib import Path

SOURCE = Path('solution/megakernel_wgmma_psplit_rsa_microbatch.cu')
STAGES = [1,2,3,4,6,7,8,9,10,16,17,18,33,34]

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def rounded(x, dtype):
    if dtype == 'fp16':
        return struct.pack('<e', x)
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return struct.pack('<H', ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff)

def decode(bits, dtype):
    if dtype == 'fp16':
        return struct.unpack('<e', bits)[0]
    return struct.unpack('<f', b'\0\0'+bits)[0]

def reference(a):
    a11,a12,a21,a22=a
    s1=f32(a21+a22)
    s2=f32(s1-a11)
    s3=f32(a11-a21)
    s4=f32(a12-s2)
    return [a11,a12,s4,a22,s1,s2,s3]

def product(a,p):
    a11,a12,a21,a22=a
    if p==0:return a11
    if p==1:return a12
    if p==3:return a22
    if p==2:return f32(a12-f32(f32(a21+a22)-a11))
    if p==4:return f32(a21+a22)
    if p==5:return f32(f32(a21+a22)-a11)
    return f32(a11-a21)

def run():
    source=SOURCE.read_text()
    assert source.count('alignas(16) Element a_data[')==1
    assert 'a0_data' not in source and 'build_stage' not in source
    assert source.count('psplit_rsa_microbatch_consume_product<Element,')==7
    helper=source[source.index('CUTE_DEVICE void psplit_rsa_microbatch_consume_product'):source.index('// BK64/BN32')]
    markers=['build_product<Element,Product,BK>','warpgroup_fence_operand(q);','warpgroup_fence_operand(a);','warpgroup_arrive();','gemm(mma,a,b(_,_,_,Int<Product>{}),q);','warpgroup_commit_batch();','warpgroup_wait<0>();','warpgroup_fence_operand(a);','warpgroup_fence_operand(q);']
    offset=0
    for marker in markers:offset=helper.index(marker,offset)+len(marker)
    rng=random.Random(401024)
    comparisons=0
    for dtype in ['bf16','fp16']:
        for _ in range(10000):
            a=[decode(rounded(rng.uniform(-32,32),dtype),dtype) for _ in range(4)]
            expected=reference(a)
            for p in range(7):
                assert rounded(expected[p],dtype)==rounded(product(a,p),dtype)
                comparisons+=1
    events=0
    products=0
    for stages in STAGES:
        for owned in [(0,1,2),(3,4,5,6)]:
            p_atoms={p:[] for p in owned}
            for stage in range(stages):
                bank=stage%2
                phase=(stage//2)%2
                assert (bank,phase)==(stage&1,(stage>>1)&1)
                pending=False
                for p in owned:
                    assert not pending # generic writes to A occur only here
                    pending=True
                    p_atoms[p].extend(stage*4+atom for atom in range(4))
                    events+=4
                    pending=False # covered wait0, then post A/P operand fences
                    products+=1
                assert not pending # only this boundary can release shared stage
            for p in owned:assert p_atoms[p]==list(range(stages*4))
    def reject(pending, operation):
        assert not (pending and operation in ['overwriteA','genericReadA','genericReadP','releaseBank'])
    rejected=0
    for op in ['overwriteA','genericReadA','genericReadP','releaseBank']:
        try:reject(True,op)
        except AssertionError:rejected+=1
    assert rejected==4
    print(json.dumps({'status':'PASS_CPU_BOUNDED_RN_DAG_AND_MICROBATCH_REUSE_ONLY','source_path':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'geometry':[64,32,64],'k16_atoms_per_P_stage':4,'stage_domains':STAGES,'bounded_final_A_bit_comparisons':comparisons,'abstract_product_retirements':products,'abstract_K16_events':events,'negative_pending_controls_rejected':rejected,'actual_GPU_memory_order_or_thread_convergence_proven':False,'register_pool_or_quota_fit_proven':False,'FP64_numerical_certification_proven':False},indent=2))

if __name__=='__main__':run()
