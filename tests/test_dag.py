#!/usr/bin/env python3
"""Check the fixed SW DAG with exact, noncommuting bilinear coefficients."""
import random
import unittest


def add(a, b, sign=1):
    return tuple(x + sign * y for x, y in zip(a, b))


def product(a, b):
    return tuple(x * y for x in a for y in b)


def winograd(a, b, multiply=product):
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    s1 = add(a21, a22)
    s2 = add(s1, a11, -1)
    s3 = add(a11, a21, -1)
    s4 = add(a12, s2, -1)
    t1 = add(b12, b11, -1)
    t2 = add(b22, t1, -1)
    t3 = add(b22, b12, -1)
    t4 = add(t2, b21, -1)
    p1 = multiply(a11, b11)
    p2 = multiply(a12, b21)
    p3 = multiply(s4, b22)
    p4 = multiply(a22, t4)
    p5 = multiply(s1, t1)
    p6 = multiply(s2, t2)
    p7 = multiply(s3, t3)
    u1 = add(p1, p2)
    u2 = add(p1, p6)
    u3 = add(u2, p7)
    u4 = add(u2, p5)
    u5 = add(u4, p3)
    u6 = add(u3, p4, -1)
    u7 = add(u3, p5)
    return u1, u5, u6, u7


def matmul_flat(a, b, m, n, k):
    return tuple(sum(a[i * k + p] * b[p * n + j] for p in range(k))
                 for i in range(m) for j in range(n))


class WinogradDAG(unittest.TestCase):
    def test_all_sixteen_bilinear_terms(self):
        units = [tuple(int(i == j) for i in range(4)) for j in range(4)]
        actual = winograd(units, units)
        expected = (
            add(product(units[0], units[0]), product(units[1], units[2])),
            add(product(units[0], units[1]), product(units[1], units[3])),
            add(product(units[2], units[0]), product(units[3], units[2])),
            add(product(units[2], units[1]), product(units[3], units[3])),
        )
        self.assertEqual(actual, expected)

    def test_exact_integer_block_matrices(self):
        # Rectangular blocks prevent accidentally relying on commutativity.
        rng = random.Random(20261013)
        for m, n, k in ((1, 1, 1), (2, 3, 4), (3, 2, 5), (4, 4, 2)):
            for _ in range(20):
                a = [tuple(rng.randrange(-9, 10) for _ in range(m * k)) for _ in range(4)]
                b = [tuple(rng.randrange(-9, 10) for _ in range(k * n)) for _ in range(4)]
                multiply = lambda x, y: matmul_flat(x, y, m, n, k)
                actual = winograd(a, b, multiply)
                expected = (
                    add(multiply(a[0], b[0]), multiply(a[1], b[2])),
                    add(multiply(a[0], b[1]), multiply(a[1], b[3])),
                    add(multiply(a[2], b[0]), multiply(a[3], b[2])),
                    add(multiply(a[2], b[1]), multiply(a[3], b[3])),
                )
                self.assertEqual(actual, expected)

    def test_p4_sign_is_paired(self):
        # A22*T4 is subtracted in C21; swapping only one sign is observable.
        units = [tuple(int(i == j) for i in range(4)) for j in range(4)]
        actual = winograd(units, units)[2]
        p4 = product(units[3], add(add(units[3], add(units[1], units[0], -1), -1), units[2], -1))
        wrong = add(actual, p4, 2)
        self.assertNotEqual(actual, wrong)


if __name__ == "__main__":
    unittest.main()
