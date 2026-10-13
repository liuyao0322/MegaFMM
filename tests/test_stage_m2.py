#!/usr/bin/env python3
"""Independent CPU checks for PLAN §6.5's M2 accumulation contract.

M2 computes seven *fresh* K16 product scratches, in P1..P7 order, then
applies 14 separately rounded FP32 contributions to four long-lived C blocks.
The scalar dot below deliberately specifies sequential RN FP32 multiply-add.
It is an arithmetic model, NOT a bitwise oracle for SM80 MMA's internal K16
reduction. Exact integer/bilinear checks and contribution-order witnesses do
not grant GPU numerical acceptance: the frozen original-input FP64 AB gate
in test_correctness.cu remains independent and unchanged.

Only the Python standard library is required. Run: python3 tests/test_stage_m2.py
"""

import math
import random
import struct
import unittest


def fp32(value):
    """IEEE binary32 round-to-nearest, ties-to-even."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def low_round(value, dtype):
    value = fp32(value)
    if dtype == "fp16":
        try:
            return struct.unpack("<e", struct.pack("<e", value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)
    if dtype != "bf16":
        raise ValueError("dtype must be bf16 or fp16")
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7F800000 == 0x7F800000:
        return value  # The fixtures are finite; preserve exceptional inputs.
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def add32(a, b):
    return fp32(a + b)


def fma32(a, b, accumulator):
    # All fixtures use low-precision operands with modest exponent spans, so
    # binary64 represents this product plus FP32 accumulator exactly before RN.
    return fp32(a * b + accumulator)


# The normative contribution map is intentionally separate from M1's U DAG.
# Coordinates are C11,C12,C21,C22; entries are (coordinate, sign), in P order.
FANOUT = (
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((0, 1),),
    ((1, 1),),
    ((2, -1),),
    ((1, 1), (3, 1)),
    ((1, 1), (2, 1), (3, 1)),
    ((2, 1), (3, 1)),
)


def transformed_operands(a, b, dtype):
    """One original-input coordinate; keep S/T dependencies in FP32."""
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    s1 = add32(a21, a22)
    s2 = add32(s1, -a11)
    s3 = add32(a11, -a21)
    s4 = add32(a12, -s2)
    t1 = add32(b12, -b11)
    t2 = add32(b22, -t1)
    t3 = add32(b22, -b12)
    t4 = add32(t2, -b21)
    left = (a11, a12, s4, a22, s1, s2, s3)
    right = (b11, b21, b22, t4, t1, t2, t3)
    return tuple((low_round(x, dtype), low_round(y, dtype))
                 for x, y in zip(left, right))


def m2_coordinate(operands, stage_width=16):
    c = [0.0] * 4
    for begin in range(0, len(operands), stage_width):
        for product, targets in enumerate(FANOUT):
            scratch = 0.0
            for terms in operands[begin:begin + stage_width]:
                left, right = terms[product]
                scratch = fma32(left, right, scratch)
            for coordinate, sign in targets:
                c[coordinate] = add32(c[coordinate], sign * scratch)
    return tuple(c)


def m1_coordinate(operands):
    products = [0.0] * 7
    for terms in operands:
        for product, (left, right) in enumerate(terms):
            products[product] = fma32(left, right, products[product])
    p1, p2, p3, p4, p5, p6, p7 = products
    u1 = add32(p1, p2)
    u2 = add32(p1, p6)
    u3 = add32(u2, p7)
    u4 = add32(u2, p5)
    u5 = add32(u4, p3)
    u6 = add32(u3, -p4)
    u7 = add32(u3, p5)
    return u1, u5, u6, u7


def rounded_inputs(a, b, dtype):
    return ([low_round(x, dtype) for x in a],
            [low_round(x, dtype) for x in b])


def multiply_model(a, b, m, n, k, dtype, mode="m2"):
    """Model block arithmetic, not host routing or the GPU tile guard.

    Even non-tile dimensions are supported here to test the algebra; a short
    final conceptual stage is equivalent to zero padding. Mega18's GPU fast
    path itself is restricted to complete tiles and falls back outside them.
    """
    if min(m, n, k) <= 0 or any(x % 2 for x in (m, n, k)):
        raise ValueError("positive even matrix dimensions required")
    if len(a) != m * k or len(b) != k * n:
        raise ValueError("matrix storage size mismatch")
    if mode not in ("m1", "m2"):
        raise ValueError("mode must be m1 or m2")
    a, b = rounded_inputs(a, b, dtype)
    mh, nh, kh = m // 2, n // 2, k // 2
    result = [0.0] * (m * n)
    for row in range(mh):
        for col in range(nh):
            operands = []
            for inner in range(kh):
                av = tuple(a[(row + dr * mh) * k + inner + dc * kh]
                           for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)))
                bv = tuple(b[(inner + dr * kh) * n + col + dc * nh]
                           for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)))
                operands.append(transformed_operands(av, bv, dtype))
            c = m2_coordinate(operands) if mode == "m2" else m1_coordinate(operands)
            for value, (dr, dc) in zip(c, ((0, 0), (0, 1), (1, 0), (1, 1))):
                result[(row + dr * mh) * n + col + dc * nh] = value
    return result


def original_ab_reference(a, b, m, n, k, dtype):
    """Binary64 AB from rounded ORIGINAL inputs, without S/T or U."""
    a, b = rounded_inputs(a, b, dtype)
    return [math.fsum(a[row * k + inner] * b[inner * n + col]
                      for inner in range(k))
            for row in range(m) for col in range(n)]


def dense_sequential(a, b, m, n, k, dtype):
    a, b = rounded_inputs(a, b, dtype)
    result = []
    for row in range(m):
        for col in range(n):
            accumulator = 0.0
            for inner in range(k):
                accumulator = fma32(a[row * k + inner], b[inner * n + col], accumulator)
            result.append(accumulator)
    return result


def relative_frobenius(actual, reference):
    error = math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(actual, reference)))
    norm = math.sqrt(math.fsum(x * x for x in reference))
    return error / norm if norm else error


class StageM2(unittest.TestCase):
    def test_fourteen_ordered_bilinear_contributions(self):
        # Expanded linear forms use ordered A_i B_j keys. Treating the two
        # operand spaces separately prevents scalar commutation hiding swaps.
        left = ({0: 1}, {1: 1}, {0: 1, 1: 1, 2: -1, 3: -1}, {3: 1},
                {2: 1, 3: 1}, {0: -1, 2: 1, 3: 1}, {0: 1, 2: -1})
        right = ({0: 1}, {2: 1}, {3: 1}, {0: 1, 1: -1, 2: -1, 3: 1},
                 {0: -1, 1: 1}, {0: 1, 1: -1, 3: 1}, {1: -1, 3: 1})
        expanded = [dict() for _ in range(4)]
        for av, bv, targets in zip(left, right, FANOUT):
            for coordinate, sign in targets:
                for ai, ac in av.items():
                    for bi, bc in bv.items():
                        key = (ai, bi)
                        expanded[coordinate][key] = expanded[coordinate].get(key, 0) + sign * ac * bc
        expanded = [{key: value for key, value in block.items() if value} for block in expanded]
        self.assertEqual(sum(len(targets) for targets in FANOUT), 14)
        self.assertEqual(expanded, [
            {(0, 0): 1, (1, 2): 1}, {(0, 1): 1, (1, 3): 1},
            {(2, 0): 1, (3, 2): 1}, {(2, 1): 1, (3, 3): 1},
        ])

    def test_all_noncommuting_two_by_two_block_basis_pairs(self):
        # Each quadrant is a 2x2 matrix. Exhaust all 16 A-entry x 16 B-entry
        # units: this also checks inner-index orientation, beyond scalar blocks.
        for dtype in ("bf16", "fp16"):
            for ai in range(16):
                for bi in range(16):
                    a, b = [0.0] * 16, [0.0] * 16
                    a[ai] = b[bi] = 1.0
                    with self.subTest(dtype=dtype, a_entry=ai, b_entry=bi):
                        self.assertEqual(multiply_model(a, b, 4, 4, 4, dtype),
                                         original_ab_reference(a, b, 4, 4, 4, dtype))
        # E01 E10 = E00, while E10 E01 = E11: an explicit noncommuting witness.
        a, b = [0.0] * 16, [0.0] * 16
        a[1], b[4] = 1.0, 1.0
        self.assertNotEqual(original_ab_reference(a, b, 4, 4, 4, "fp16"),
                            original_ab_reference(b, a, 4, 4, 4, "fp16"))

    def test_integer_multistage_rectangles_and_short_final_stage(self):
        rng = random.Random(2026101322)
        for dtype in ("bf16", "fp16"):
            for k in (32, 64, 96, 48, 34):
                m, n = 4, 6
                a = [rng.randrange(-3, 4) for _ in range(m * k)]
                b = [rng.randrange(-3, 4) for _ in range(k * n)]
                with self.subTest(dtype=dtype, k=k):
                    expected = original_ab_reference(a, b, m, n, k, dtype)
                    self.assertEqual(multiply_model(a, b, m, n, k, dtype), expected)
                    self.assertTrue(all(math.isfinite(x) for x in expected))

    def test_rn_ties_and_unrounded_transform_dependencies(self):
        self.assertEqual(fp32(1.0 + 2.0 ** -24), 1.0)
        self.assertEqual(fp32(1.0 + 3 * 2.0 ** -24), 1.0 + 2.0 ** -22)
        for dtype, halfway in (("bf16", 2.0 ** -8), ("fp16", 2.0 ** -11)):
            with self.subTest(dtype=dtype):
                self.assertEqual(low_round(1 + halfway, dtype), 1.0)
                self.assertEqual(low_round(1 + 3 * halfway, dtype), 1 + 4 * halfway)
                self.assertEqual(low_round(-1 - halfway, dtype), -1.0)
                operands = transformed_operands((1.0, 0.0, 1.0, halfway), (0.0,) * 4, dtype)
                self.assertEqual(operands[4][0], 1.0)  # rounded S1
                self.assertEqual(operands[5][0], halfway)  # S2 from FP32 S1
                self.assertEqual(operands[2][0], -halfway)  # S4 from FP32 S2
                self.assertNotEqual(operands[5][0], low_round(operands[4][0] - 1.0, dtype))

    def assert_fixed_model_gate(self, a, b, m, n, k, dtype, m2):
        reference = original_ab_reference(a, b, m, n, k, dtype)
        dense = dense_sequential(a, b, m, n, k, dtype)
        self.assertTrue(all(math.isfinite(x) for x in m2))
        floor = 1e-3 if dtype == "bf16" else 1e-4
        self.assertLessEqual(relative_frobenius(m2, reference),
                             3 * relative_frobenius(dense, reference) + floor)
        # This fixture check uses the frozen gate formula; it is not a claim
        # that this scalar Dense model reproduces cuBLASLt or accepts GPU M2.
        return reference

    def test_contribution_order_differs_from_m1_for_original_inputs(self):
        # Half-K16: P1=1, P3=RN(-1-epsilon)=-1, P5=epsilon, P6=0.
        # M1 C12=RN(RN(1+epsilon)-1)=0; M2 C12=RN(RN(1-1)+epsilon).
        k, epsilon = 32, 2.0 ** -25
        a, b = [0.0] * (2 * k), [0.0] * (2 * k)
        a[0], a[k // 2], a[k] = 1.0, -1.0, 1.0
        a[k + 1] = 2.0 ** -12
        b[0], b[1], b[(k // 2) * 2 + 1] = 1.0, 1.0, 1.0
        b[3], b[(k // 2 + 1) * 2 + 1] = 2.0 ** -13, 2.0 ** -13
        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype):
                m1 = multiply_model(a, b, 2, 2, k, dtype, "m1")
                m2 = multiply_model(a, b, 2, 2, k, dtype)
                self.assertEqual(m1[1], 0.0)
                self.assertEqual(m2[1], epsilon)
                reference = self.assert_fixed_model_gate(a, b, 2, 2, k, dtype, m2)
                self.assertEqual(reference, [1.0, 0.0, 1.0, 1.0 + epsilon])
                self.assertNotEqual(m1, m2)

    def test_fresh_stage_dot_differs_from_complete_product_accumulation(self):
        # Stage0 contributes 1. Stage1's sixteen epsilon products add up in a
        # zero scratch; adding them one-by-one to 1 would round each away.
        k, epsilon = 64, 2.0 ** -25
        a, b = [0.0] * (2 * k), [0.0] * (2 * k)
        a[0] = b[0] = 1.0
        for inner in range(16, 32):
            a[inner], b[inner * 2] = 2.0 ** -12, 2.0 ** -13
        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype):
                m1 = multiply_model(a, b, 2, 2, k, dtype, "m1")
                m2 = multiply_model(a, b, 2, 2, k, dtype)
                self.assertEqual(m1[0], 1.0)
                self.assertEqual(m2[0], 1.0 + 16 * epsilon)
                reference = self.assert_fixed_model_gate(a, b, 2, 2, k, dtype, m2)
                self.assertEqual(m2, reference)
                self.assertNotEqual(m1, m2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
