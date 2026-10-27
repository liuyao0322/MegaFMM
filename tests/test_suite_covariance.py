"""Covariance scientific definition, stored-input reference and quality checks."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from bench.suite.covariance import CovariancePropagation, _orthogonal_circulant


class FrameworkBackend:
    def __init__(self):
        self.calls = 0
        self.input_dtypes = []

    def gemm(self, a, b):
        self.calls += 1
        self.input_dtypes.append((a.dtype, b.dtype))
        return a @ b


@unittest.skipIf(torch is None, "PyTorch is required")
class CovarianceTests(unittest.TestCase):
    def test_dense_transition_has_declared_norm_without_qr_or_eigendecomposition(self):
        for n in (2, 7, 8, 17):
            with self.subTest(n=n):
                transition = _orthogonal_circulant(n, 0.95, torch.Generator().manual_seed(11))
                torch.testing.assert_close(transition @ transition.T,
                                           0.95 ** 2 * torch.eye(n, dtype=torch.float64),
                                           rtol=1e-13, atol=1e-13)
                self.assertGreater(torch.linalg.vector_norm(transition - torch.diag(transition.diag())).item(), 0.5)

    def test_recurrence_matches_independent_covariance_sum_and_counts_gemms(self):
        app = CovariancePropagation({"n": 9, "iterations": 4}, "cpu", 17)
        originals = [value.clone() for value in
                     (app.transition, app.initial_covariance, app.process_covariance)]
        backend = FrameworkBackend()
        output = app.execute(backend)
        self.assertEqual(backend.calls, 8)
        self.assertEqual(output["result"].dtype, torch.float32)
        self.assertTrue(all(pair == (torch.float32, torch.float32) for pair in backend.input_dtypes))
        # Independent closed finite-horizon covariance sum, rather than the
        # implementation's two-product recurrence.
        a, initial, noise = [value.double() for value in originals]
        power = torch.linalg.matrix_power(a, 4)
        expected = power @ initial @ power.T
        for step in range(4):
            power = torch.linalg.matrix_power(a, step)
            expected += power @ noise @ power.T
        torch.testing.assert_close(app.make_reference()["result"], expected, rtol=1e-13, atol=1e-13)
        metrics = app.validate(output)
        self.assertTrue(metrics["quality_pass"], metrics)
        self.assertLess(metrics["relative_error"], 1e-6)
        for actual, original in zip((app.transition, app.initial_covariance, app.process_covariance), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        torch.testing.assert_close(output["result"], app.execute(FrameworkBackend())["result"], rtol=0, atol=0)

    def test_corrupt_covariance_and_nonfinite_output_fail_quality(self):
        app = CovariancePropagation({"n": 8, "iterations": 3}, "cpu", 4)
        correct = app.execute(FrameworkBackend())["result"]
        self.assertFalse(app.validate({"result": correct * 1.1})["quality_pass"])
        asymmetric = correct.clone()
        asymmetric[0, 1] += 1
        metrics = app.validate({"result": asymmetric})
        self.assertGreater(metrics["symmetry_error"], 1e-5)
        self.assertFalse(metrics["quality_pass"])
        # A relative-error threshold cannot make a negative covariance valid.
        loose = CovariancePropagation({"n": 8, "quality": {"relative_error": 10}}, "cpu", 4)
        metrics = loose.validate({"result": -torch.eye(8)})
        self.assertEqual(metrics["cholesky_failure"], 1)
        self.assertFalse(metrics["quality_pass"])
        bad = correct.clone()
        bad[0, 0] = float("nan")
        self.assertFalse(app.validate({"result": bad})["quality_pass"])

    def test_stored_inputs_hashes_and_flop_contract_are_reproducible(self):
        case = {"n": 11, "iterations": 6, "transition_radius": 0.995}
        first = CovariancePropagation(case, "cpu", 25)
        second = CovariancePropagation(case, "cpu", 25)
        other = CovariancePropagation(case, "cpu", 26)
        self.assertEqual(first.metadata()["stored_inputs"], second.metadata()["stored_inputs"])
        self.assertNotEqual(first.metadata()["stored_inputs"], other.metadata()["stored_inputs"])
        self.assertEqual(first.metadata()["nominal_flops"], 4 * 6 * 11 ** 3)
        self.assertLess(first.metadata()["transition_2norm_bound_from_generator_and_fp32_rounding"], 1)
        self.assertEqual(len(first.metadata()["stored_inputs"]), 3)

    def test_fp32_only_and_invalid_settings_are_rejected(self):
        for case in ({"dtype": "bf16"}, {"n": 0}, {"n": 2.0},
                     {"transition_radius": 0}, {"transition_radius": 1},
                     {"transition_radius": float("nan")}):
            with self.subTest(case=case), self.assertRaises(ValueError):
                CovariancePropagation(case, "cpu", 0)


if __name__ == "__main__":
    unittest.main()
