#!/usr/bin/env python3
"""Read-only iteration37 device/toolchain snapshot after performance and serial SAN; no kernel launch."""
import ctypes
import json
import socket
import subprocess
from datetime import datetime, timezone

print('POST_PERFORMANCE_SAN_SNAPSHOT_UTC', datetime.now(timezone.utc).isoformat(), flush=True)
print('HOST', socket.gethostname(), flush=True)
print('SCOPE read-only snapshot after full performance and serial SAN work; no device kernel launched; idle clocks do not prove timing-group clocks', flush=True)
for command in [
    ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,compute_cap,clocks.current.sm,clocks.current.memory,temperature.gpu,pstate', '--format=csv,noheader'],
    ['nvcc', '--version'],
    ['/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer', '--version'],
    ['git', '-C', 'third_party/cutlass', 'rev-parse', 'HEAD'],
]:
    print('COMMAND', json.dumps(command), flush=True)
    subprocess.run(command, check=True)
driver = ctypes.CDLL('libcuda.so.1')
def checked(name, *args):
    code = getattr(driver, name)(*args)
    if code:
        raise RuntimeError(f'{name} failed: {code}')
checked('cuInit', 0)
device = ctypes.c_int()
checked('cuDeviceGet', ctypes.byref(device), 0)
sms = ctypes.c_int()
checked('cuDeviceGetAttribute', ctypes.byref(sms), 16, device)
print('DRIVER_API_MULTIPROCESSOR_COUNT', sms.value, flush=True)

for name,attribute in [("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",97),("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",81),("RESERVED_SHARED_MEMORY_PER_BLOCK",111),("MAX_REGISTERS_PER_MULTIPROCESSOR",82),("MAX_REGISTERS_PER_BLOCK",12)]:
    value=ctypes.c_int()
    checked("cuDeviceGetAttribute",ctypes.byref(value),attribute,device)
    print("ACTUAL_DEVICE_ATTRIBUTE",name,value.value,flush=True)
