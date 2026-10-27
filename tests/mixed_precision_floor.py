#!/usr/bin/env python3
"""Independent exact-leaf diagnostic for the fixed one-cast SW policy.

This is a numerical limitation witness, not a relaxed correctness test. Each
2x2 coefficient matrix can be tensored with an identity to construct aligned
large matrices with the same relative transform-cast error. Inputs are already
rounded to the target dtype; the dense reference uses those original values.
"""
import json
import math
import random
import struct
from pathlib import Path


def bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def fp16(x):
    return struct.unpack("e", struct.pack("e", x))[0]


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def evaluate(a, b, cast):
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    # The binary input sums in this range are exactly representable in FP32.
    s1 = f32(a21 + a22)
    s2 = f32(s1 - a11)
    s3 = f32(a11 - a21)
    s4 = f32(a12 - s2)
    t1 = f32(b12 - b11)
    t2 = f32(b22 - t1)
    t3 = f32(b22 - b12)
    t4 = f32(t2 - b21)
    # FP64 products isolate operand cast loss; FP32 accumulation cannot
    # generally repair the information removed by this cast.
    p1 = a11 * b11
    p2 = a12 * b21
    p3 = cast(s4) * b22
    p4 = a22 * cast(t4)
    p5 = cast(s1) * cast(t1)
    p6 = cast(s2) * cast(t2)
    p7 = cast(s3) * cast(t3)
    u2 = p1 + p6
    u3 = u2 + p7
    sw = [p1 + p2, u2 + p5 + p3, u3 - p4, u3 + p5]
    ref = [a11 * b11 + a12 * b21, a11 * b12 + a12 * b22,
           a21 * b11 + a22 * b21, a21 * b12 + a22 * b22]
    err = math.sqrt(sum((x-y)**2 for x, y in zip(sw, ref)))
    norm = math.sqrt(sum(x*x for x in ref))
    pp = [f32(x) for x in (p1, p2, p3, p4, p5, p6, p7)]
    uu2 = f32(pp[0] + pp[5])
    uu3 = f32(uu2 + pp[6])
    uu4 = f32(uu2 + pp[4])
    sw32 = [f32(pp[0] + pp[1]), f32(uu4 + pp[2]),
            f32(uu3 - pp[3]), f32(uu3 + pp[4])]
    sw32_ef = math.sqrt(sum((x-y)**2 for x, y in zip(sw32, ref))) / norm
    dense32_ef = math.sqrt(sum((f32(x)-x)**2 for x in ref)) / norm
    return sw, ref, err / norm, sw32, sw32_ef, dense32_ef


def main():
    witnesses = []
    for name, cast, floor in (("bf16", bf16, 1e-3), ("fp16", fp16, 1e-4)):
        rng = random.Random(20261013)
        for trial in range(10000):
            a = [cast(rng.uniform(-1, 1)) for _ in range(4)]
            b = [cast(rng.uniform(-1, 1)) for _ in range(4)]
            sw, ref, ef, sw32, sw32_ef, dense32_ef = evaluate(a, b, cast)
            if ef > 4 * floor:
                assert sw32_ef > 3 * dense32_ef + floor
                witnesses.append(dict(dtype=name, trial=trial, A=a, B=b,
                                      exact_dense=ref, exact_leaf_sw=sw,
                                      sw_ef=ef, frozen_floor=floor,
                                      fp32_leaf_sw=sw32, fp32_sw_ef=sw32_ef,
                                      fp32_dense_ef=dense32_ef,
                                      gate_limit=3*dense32_ef+floor,
                                      final_operand_policy="FP32 DAG then one target-dtype cast"))
                break
        else:
            raise RuntimeError("no numerical witness found")
    print(json.dumps(witnesses, indent=2))
    if Path("results").exists():
        Path("results/mixed_precision_witness.json").write_text(json.dumps(witnesses, indent=2) + "\n")


if __name__ == "__main__":
    main()
