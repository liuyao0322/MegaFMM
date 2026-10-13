"""Execute frozen RS-A CPU computations without its historical output writer."""
import sys, os, hashlib, json
from pathlib import Path
sys.dont_write_bytecode = True
root = Path('/share/home/liuyao/workspace/MegaFMM')
original = root / '.scratch/psplit_rsA_future35_model.py'
expected = 'aab2e1940470681b51d83596440ae49c5f2369e6abf2a1ff33d7f3c5ab7644ee'
raw = original.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
marker = "\nout=R/'.scratch/psplit_rsA_future35_review.json'"
source = raw.decode()
assert source.count(marker) == 1
prefix = source[:source.index(marker)]
def refuse_writes(event, args):
    if event != 'open':
        return
    mode, flags = args[1], args[2]
    if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
        raise PermissionError('Iteration36 CPU runner refuses all filesystem writes')
sys.addaudithook(refuse_writes)
scope = {'__name__': '__iteration36_frozen_computations__', '__file__': str(original)}
exec(compile(prefix, str(original), 'exec'), scope)
report = scope['report']
report['owner_execution'] = {
    'frozen_original_path': str(original.relative_to(root)),
    'frozen_original_sha256': expected,
    'execution_scope': 'All original computations executed; only historical output-file writer and its print tail removed.',
    'file_write_audit_hook': 'All write opens refused; no old model main/output writer invoked.',
    'hardware_memory_order_compilation_precision_performance_proven': False,
}
print(json.dumps(report, indent=2))
