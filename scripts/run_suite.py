#!/usr/bin/env python3
"""Run the versioned MegaFMM experimental suite (metadata commands need no torch)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.suite.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
