#!/usr/bin/env python3
"""Freeze an isolated scientific benchmark source tree, build, and validate it.

Run inside the GPU allocation, for example:
  bash scripts/on_gpu.sh python3 scripts/prepare_scientific.py \
    --snapshot .scratch/scientific-new --kernel-ref 14d5781e

The optional kernel ref selects solution/include and their compile-time type
test from Git. Benchmark Python/configuration always comes from the working
tree. No existing source, build or result directory is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="new directory for source/build/logs")
    parser.add_argument("--kernel-ref", help="optional committed kernel; default is working tree")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not os.environ.get("SLURM_STEP_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run within an allocated GPU step, e.g. bash scripts/on_gpu.sh python3 scripts/prepare_scientific.py ...")
    target = Path(args.snapshot).expanduser().resolve()
    if target.exists():
        parser.error(f"snapshot directory already exists: {target}")
    commit = git("rev-parse", args.kernel_ref).decode().strip() if args.kernel_ref else None
    source, build = target / "source", target / "build"
    source.mkdir(parents=True)
    for folder in ("solution", "include", "bench", "tests", "scripts"):
        shutil.copytree(ROOT / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "CMakeLists.txt", source / "CMakeLists.txt")
    if commit:
        for folder in ("solution", "include"):
            shutil.rmtree(source / folder)
        names = git("ls-tree", "-r", "--name-only", commit, "solution", "include",
                    "tests/check_fp32_policy_types.py").decode().splitlines()
        for name in names:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(git("show", f"{commit}:{name}"))
    cutlass = ROOT / "third_party/cutlass"
    if not (cutlass / "include/cute/tensor.hpp").is_file():
        parser.error(f"missing existing CUTLASS checkout: {cutlass}")
    (source / "third_party").mkdir()
    (source / "third_party/cutlass").symlink_to(cutlass, target_is_directory=True)
    identity = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kernel_commit": commit, "kernel_source": "git_ref" if commit else "working_tree",
                "origin_root": str(ROOT), "root_git_commit": git("rev-parse", "HEAD").decode().strip(),
                "root_git_status": git("status", "--porcelain").decode(),
                "source_sha256": {str(p.relative_to(source)): digest(p)
                                  for folder in ("solution", "include", "bench", "scripts", "tests")
                                  for p in sorted((source / folder).rglob("*")) if p.is_file()},
                "cutlass_commit": subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True).strip(),
                "cuda_compiler": subprocess.check_output(["nvcc", "--version"], text=True)}
    identity["source_sha256"]["CMakeLists.txt"] = digest(source / "CMakeLists.txt")
    (target / "snapshot.json").write_text(json.dumps(identity, indent=2) + "\n")
    with (target / "build.log").open("w") as log:
        for command in (["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                        ["cmake", "--build", str(build), "--clean-first", "-j", str(args.jobs)]):
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with (target / "ctest.log").open("w") as log:
        subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    identity["artifact_sha256"] = {p.name: digest(p) for p in build.iterdir()
                                   if p.is_file() and (p.suffix in (".a", ".so") or p.name in
                                   ("megafmm_bench", "test_correctness", "test_tuning_precision", "test_sw_paths"))}
    identity["ctest_sha256"] = digest(target / "ctest.log")
    (build / "build_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"source": str(source), "library": str(build / "libmegafmm_suite.so"),
                      "runner": str(source / "scripts/run_suite.py"),
                      "config": str(source / "bench/configs/scientific_core.json"),
                      "build_manifest": str(build / "build_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
