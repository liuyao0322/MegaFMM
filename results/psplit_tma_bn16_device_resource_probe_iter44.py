import ctypes as C,json
lib=C.CDLL('libcuda.so.1')
def call(name,*a):
 c=getattr(lib,name)(*a)
 if c:raise RuntimeError((name,c))
call('cuInit',0);d=C.c_int();call('cuDeviceGet',C.byref(d),0);attrs={}
for k,n in [('max_shared_memory_per_SM',81),('max_optin_shared_memory_per_block',97),('reserved_shared_memory_per_block',111),('max_registers_per_SM',82),('max_threads_per_SM',39)]:
 v=C.c_int();call('cuDeviceGetAttribute',C.byref(v),n,d);attrs[k]=v.value
print(json.dumps({'actual_device_attributes':attrs,'source':'pinned CUDA12.4 cuda.h Driver API attribute enums','no_kernel_launches':True},indent=2))
