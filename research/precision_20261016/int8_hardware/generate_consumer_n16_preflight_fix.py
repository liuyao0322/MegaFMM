"""Retry original P52 with a spaced NVCC host-only ternary; preserve preflight."""
from pathlib import Path
import runpy
r=Path(__file__).resolve().parent
runpy.run_path(str(r/'generate_consumer_n16.py'))
p=r/'corrected.cuh';s=p.read_text();old='(std::is_same_v<T,cute::half_t>?"fp16":"bf16")'
assert s.count(old)==1
p.write_text(s.replace(old,'(std::is_same_v<T,cute::half_t> ? "fp16" : "bf16")'))
