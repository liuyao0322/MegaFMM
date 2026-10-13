"""Execute frozen BK32 geometry/model computations with every file write refused."""
import hashlib
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
root = Path('/share/home/liuyao/workspace/MegaFMM')
original = root / '.scratch/psplit_rsA_bk32_future36_model.py'
expected = 'dcd63a2e77f76b4304d5341c33c89b1ffd54897bf9863a542492295481d1c411'
raw = original.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
marker = "\npath = ROOT / '.scratch/psplit_rsA_bk32_future36_review.json'"
source = raw.decode()
assert source.count(marker) == 1
prefix = source[:source.index(marker)]
def refuse_writes(event, args):
    if event != 'open':
        return
    mode, flags = args[1], args[2]
    if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
        raise PermissionError('Iteration37 CPU runner refuses all filesystem writes')
sys.addaudithook(refuse_writes)
scope = {'__name__': '__iteration37_frozen_computations__', '__file__': str(original)}
exec(compile(prefix, str(original), 'exec'), scope)
report = scope['report']
report['owner_execution'] = {
    'frozen_original_path': str(original.relative_to(root)),
    'frozen_original_sha256': expected,
    'execution_scope': 'All frozen BK32 computations executed; only old output-file writer and its brief print tail removed. Parent definitions loaded by the frozen model remain definitions-only.',
    'file_write_audit_hook': 'All filesystem write opens refused; no old model output writer invoked.',
    'actual_compilation_hardware_order_liveness_precision_performance_certified': False,
}
print(json.dumps(report, indent=2))
