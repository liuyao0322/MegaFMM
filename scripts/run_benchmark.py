#!/usr/bin/env python3
"""Read bench/shapes.json and execute the existing GEMM benchmark."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys


def command(config_path, binary, overrides):
    config = json.loads(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a JSON object")
    if config.get("tuple_order", ["M", "N", "K"]) != ["M", "N", "K"]:
        raise ValueError("tuple_order must be [M,N,K]")
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    defaults = []
    if "--shapes" not in supplied:
        shapes = []
        for group in ("squares", "rectangles", "boundaries"):
            entries = config.get(group, [])
            if not isinstance(entries, list):
                raise ValueError(f"{group} must be an array of [M,N,K]")
            for index, shape in enumerate(entries):
                if (not isinstance(shape, list) or len(shape) != 3 or
                        any(type(d) is not int or not 0 < d <= 2147483647 for d in shape)):
                    raise ValueError(f"{group}[{index}] must contain three positive int dimensions")
                shapes.append(",".join(map(str, shape)))
        if not shapes:
            raise ValueError("no shapes in squares/rectangles/boundaries")
        defaults += ["--shapes", ";".join(shapes)]
    if "--dtype" not in supplied and "dtypes" in config:
        modes = {("bf16",): "bf16", ("fp16",): "fp16", ("fp32",): "fp32",
                 ("bf16", "fp16"): "both", ("bf16", "fp16", "fp32"): "all"}
        values = config["dtypes"]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("dtypes must be an array of dtype names")
        dtype = modes.get(tuple(sorted(set(values))))
        if dtype is None:
            raise ValueError("unsupported dtypes combination; specify --dtype explicitly")
        defaults += ["--dtype", dtype]
    for key, flag, minimum in [("workspace_limit_bytes", "--workspace-limit", 0),
                               ("warmup", "--warmup", 0), ("trials", "--trials", 1),
                               ("groups", "--groups", 1), ("correctness_trials", "--correct-trials", 0)]:
        if flag not in supplied and key in config:
            value = config[key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {key}")
            defaults += [flag, str(value)]
    return [binary, *defaults, *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="bench/shapes.json")
    parser.add_argument("--binary", default="build/megafmm_bench")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    try:
        argv = command(args.config, args.binary, overrides)
        if args.dry_run:
            print(shlex.join(argv))
        else:
            os.execv(argv[0], argv)
    except (OSError, ValueError, TypeError) as error:
        print(f"ERROR {args.config}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
