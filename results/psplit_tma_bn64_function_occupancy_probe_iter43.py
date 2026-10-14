"""Query final feeding cubins via Driver API; no kernel launch.

The occupancy API predicts a resource bound, not measured CTA residency.
Dynamic shared sizes are also checked against actual host object immediates.
"""
from pathlib import Path
import ctypes as C
import hashlib,json,re

root=Path.cwd()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lib=C.CDLL('libcuda.so.1')
def call(name,*args):
    code=getattr(lib,name)(*args)
    if code: raise RuntimeError((name,code))
call('cuInit',0)
dev=C.c_int();call('cuDeviceGet',C.byref(dev),0)
ctx=C.c_void_p();call('cuDevicePrimaryCtxRetain',C.byref(ctx),dev)
call('cuCtxSetCurrent',ctx)
host=Path('results/psplit_tma_bn64_final_host_objdump_iter43.txt').read_text()
sass=Path('results/psplit_tma_bn64_final_object_sass_iter43.txt').read_text()
records=[]
for stem,dyn,imm in [('tma_bn64',181248,'2c400')]:
    assert host.count('$0x'+imm+',%r8d')==2
    assert host.count('$0x'+imm+',%edx')>=2
    cubin=Path(f'results/psplit_tma_bn64_final_megakernel_wgmma_psplit_{stem}_iter43.cubin')
    mod=C.c_void_p();call('cuModuleLoad',C.byref(mod),str(root/cubin).encode())
    names=[n for n in re.findall(r'^Function : (.+)$',sass,re.M) if f'winograd_wgmma_psplit_{stem}' in n]
    assert len(names)==2
    for name in names:
        fn=C.c_void_p();call('cuModuleGetFunction',C.byref(fn),mod,name.encode())
        attrs={}
        for key,num in [('max_threads_per_block',0),('static_shared_bytes',1),('local_bytes',3),('registers_per_thread',4),('max_dynamic_shared_bytes_before_optin',8)]:
            value=C.c_int();call('cuFuncGetAttribute',C.byref(value),num,fn);attrs[key]=value.value
        call('cuFuncSetAttribute',fn,8,dyn)
        value=C.c_int();call('cuFuncGetAttribute',C.byref(value),8,fn)
        attrs['max_dynamic_shared_bytes_after_optin']=value.value
        blocks=C.c_int();call('cuOccupancyMaxActiveBlocksPerMultiprocessor',C.byref(blocks),fn,384,C.c_size_t(dyn))
        records.append({'variant':47,'dtype':'fp16' if '6half_t' in name else 'bf16','symbol':name,'cubin_path':str(cubin),'cubin_sha256':sha(cubin),'queried_dynamic_shared_bytes':dyn,'queried_threads':384,'function_attributes':attrs,'occupancy_API_predicted_active_CTAs_per_SM':blocks.value,'measured_residency':None})
    call('cuModuleUnload',mod)
call('cuDevicePrimaryCtxRelease_v2',dev)
out={'schema':'megafmm.iter43.actual-function-attributes-occupancy-api.v1','no_kernel_launches':True,'scope':'Actual final feeding cubin attributes and static occupancy API resource bound; not measured residency, bank transactions or stall evidence.','host_object_dump':{'path':'results/psplit_tma_bn64_final_host_objdump_iter43.txt','sha256':sha('results/psplit_tma_bn64_final_host_objdump_iter43.txt')},'records':records}
print(json.dumps(out,indent=2))
