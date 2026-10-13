#!/usr/bin/env python3
"""Read-only postmeasurement device/toolchain snapshot; no kernel launch."""
import ctypes
import json
import socket
import subprocess
from datetime import datetime, timezone

print('POST_MEASUREMENT_SNAPSHOT_UTC', datetime.now(timezone.utc).isoformat(), flush=True)
print('HOST', socket.gethostname(), flush=True)
print('SCOPE read-only snapshot after all performance and serial SAN work; no device kernel launched', flush=True)
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
