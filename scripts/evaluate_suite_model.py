#!/usr/bin/env python3
"""Evaluate a local causal LM with original, cuBLASLt and MegaFMM linears."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.suite.model_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
