"""CPU checks for scientific definitions, precision boundaries, and provenance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

if torch is not None:
    from bench.suite.applications import create_application
    from bench.suite.data import load_input_data


class FrameworkBackend:
    """Real stored-dtype matmul, with the suite's FP32 output interface."""

    def __init__(self):
        self.calls = 0

    def gemm(self, a, b):
        self.calls += 1
        return (a @ b).float()


@unittest.skipIf(torch is None, "PyTorch and NumPy are required")
class ApplicationTests(unittest.TestCase):
    def test_spd_sqrt_and_lowdin_inverse_sqrt(self):
        for function in ("sqrt", "inverse_sqrt"):
            with self.subTest(function=function):
                app = create_application({"app": "matrix_function", "n": 12,
                                          "condition_number": 100, "iterations": 20,
                                          "function": function}, "cpu", 19)
                before = app.matrix.clone()
                backend = FrameworkBackend()
                result = app.execute(backend)
                self.assertEqual(backend.calls, 60)
                metrics = app.validate(result)
                self.assertTrue(metrics["quality_pass"], metrics)
                self.assertLess(metrics["relative_error"], 2e-5)
                torch.testing.assert_close(app.matrix, before, rtol=0, atol=0)
                torch.testing.assert_close(result["result"], app.execute(FrameworkBackend())["result"], rtol=0, atol=0)

    def test_scientific_reference_retains_imported_double_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "overlap.pt"
            original = torch.diag(torch.tensor([1.000000035, 0.100000005], dtype=torch.float64))
            torch.save({"S": original}, path)
            app = create_application({"app": "matrix_function", "input_path": str(path),
                                      "iterations": 15}, "cpu", 0)
            exact = app.make_reference()["result"]
            torch.testing.assert_close(exact, torch.diag(torch.sqrt(original.diag())), rtol=0, atol=1e-15)
            self.assertGreater(torch.linalg.vector_norm(exact - torch.diag(torch.sqrt(app.matrix.double().diag()))).item(), 0)

    def test_muon_matches_official_rounding_and_rectangular_orientation(self):
        for dtype in ("bf16", "fp16", "fp32"):
            with self.subTest(dtype=dtype):
                app = create_application({"app": "muon", "dtype": dtype, "m": 11, "n": 7}, "cpu", 3)
                gradient = app.gradient.clone()
                x = gradient.T
                x = x / (x.norm() + 1e-7)
                for _ in range(5):
                    a = x @ x.T
                    b = -4.775 * a + 2.0315 * (a @ a)
                    x = 3.4445 * x + b @ x
                backend = FrameworkBackend()
                result = app.execute(backend)
                self.assertEqual(backend.calls, 15)
                self.assertEqual(result["result"].dtype, app.dtype)
                # Materializing the public backend's contiguous operands can
                # change FP32 CPU BLAS reduction order by a few ulps.
                tolerance = 2e-6 if dtype == "fp32" else 0
                torch.testing.assert_close(result["result"], x.T, rtol=1e-5 if tolerance else 0, atol=tolerance)
                torch.testing.assert_close(app.gradient, gradient, rtol=0, atol=0)
                self.assertTrue(app.validate(result)["quality_pass"])
                self.assertIn("polar_relative_error_diagnostic", app.validate(result))

    def test_muon_zero_input_is_finite(self):
        app = create_application({"app": "muon", "dtype": "bf16", "n": 8, "input_scale": 0}, "cpu", 0)
        result = app.execute(FrameworkBackend())
        self.assertEqual(result["result"].count_nonzero().item(), 0)
        self.assertTrue(app.validate(result)["quality_pass"])

    def test_imported_mlp_respects_weight_layout_bias_and_swiglu(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mlp.npz"
            rng = np.random.default_rng(2)
            arrays = {"X": rng.normal(size=(7, 5)).astype(np.float32),
                      "W1": rng.normal(size=(5, 9)).astype(np.float32),
                      "W2": rng.normal(size=(9, 5)).astype(np.float32),
                      "W3": rng.normal(size=(5, 9)).astype(np.float32),
                      "b1": rng.normal(size=9).astype(np.float32),
                      "b2": rng.normal(size=5).astype(np.float32),
                      "b3": rng.normal(size=9).astype(np.float32)}
            np.savez(path, **arrays, metadata_json=json.dumps({"mlp_kind": "swiglu", "checkpoint": "fixture"}))
            app = create_application({"app": "mlp", "dtype": "fp32", "input_path": str(path)}, "cpu", 0)
            data = {key: torch.from_numpy(value) for key, value in arrays.items()}
            expected = (torch.nn.functional.silu(data["X"] @ data["W1"] + data["b1"]) *
                        (data["X"] @ data["W3"] + data["b3"])) @ data["W2"] + data["b2"]
            backend = FrameworkBackend()
            output = app.execute(backend)
            self.assertEqual(backend.calls, 3)
            torch.testing.assert_close(output["result"], expected, rtol=0, atol=0)
            self.assertTrue(app.validate(output)["quality_pass"])
            self.assertEqual(app.metadata()["provenance"]["scope"], "imported_input")

    def test_gelu_modes_match_the_declared_module(self):
        for approximation in ("none", "tanh"):
            app = create_application({"app": "mlp", "tokens": 9, "hidden": 7, "intermediate": 13,
                                      "gelu_approximate": approximation, "dtype": "bf16"}, "cpu", 5)
            expected = (torch.nn.functional.gelu(app.x @ app.w1 + app.biases[0], approximate=approximation)
                        @ app.w2 + app.biases[1])
            output = app.execute(FrameworkBackend())
            torch.testing.assert_close(output["result"], expected, rtol=0, atol=0)
            self.assertTrue(app.validate(output)["quality_pass"])

    def test_sp2_constructs_the_low_energy_projector(self):
        app = create_application({"app": "purification", "n": 16, "occupied": 5,
                                  "gap": 0.4, "iterations": 20}, "cpu", 22)
        before = app.h.clone()
        backend = FrameworkBackend()
        output = app.execute(backend)
        self.assertEqual(backend.calls, 20)
        metrics = app.validate(output)
        self.assertTrue(metrics["quality_pass"], metrics)
        self.assertLess(metrics["trace_error"], 1e-4)
        self.assertLess(metrics["band_energy_absolute_error"], 1e-4)
        torch.testing.assert_close(app.h, before, rtol=0, atol=0)

    def test_orthonormal_import_does_not_transform_twice_and_uses_nocc(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fock.pt"
            h = torch.diag(torch.tensor([-3.0000000001, -2.0, 1.0, 2.0], dtype=torch.float64))
            s = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64))
            torch.save({"tensors": {"H": h, "S": s},
                        "metadata": {"H_basis": "orthonormal", "nocc": 1}}, path)
            app = create_application({"app": "purification", "input_path": str(path), "iterations": 30})
            self.assertEqual(app.occupied, 1)
            torch.testing.assert_close(app.original_h, h, rtol=0, atol=0)
            torch.testing.assert_close(app.make_reference()["result"], torch.diag(torch.tensor([1., 0., 0., 0.], dtype=torch.float64)))
            self.assertTrue(app.validate(app.execute(FrameworkBackend()))["quality_pass"])

    def test_imported_ao_hamiltonian_uses_lowdin_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ao.pt"
            h = torch.diag(torch.tensor([-6., -6., 4., 10.], dtype=torch.float64))
            s = torch.diag(torch.tensor([2., 3., 4., 5.], dtype=torch.float64))
            torch.save({"tensors": {"H": h, "S": s}, "metadata": {"nocc": 2}}, path)
            app = create_application({"app": "purification", "input_path": str(path), "iterations": 30})
            torch.testing.assert_close(app.original_h.diag(), torch.tensor([-3., -2., 1., 2.], dtype=torch.float64))
            self.assertIn("outside timing", app.metadata()["preprocessing"])
            self.assertTrue(app.validate(app.execute(FrameworkBackend()))["quality_pass"])

    def test_dynamic_stop_counts_iterations_and_preserves_precision_setting(self):
        old_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            for kind in ("matrix_function", "purification"):
                with self.subTest(kind=kind):
                    app = create_application({"app": kind, "n": 8, "iterations": 1,
                                              "max_iterations": 40, "stop_tolerance": 1e-4}, "cpu", 1)
                    result = app.execute_to_quality(FrameworkBackend())
                    self.assertTrue(result["converged"].item())
                    self.assertGreater(result["iterations"].item(), 1)
                    self.assertLessEqual(result["iterations"].item(), 40)
                    self.assertTrue(app.validate(result)["quality_pass"])
                    self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.set_float32_matmul_precision(old_precision)

    def test_nonconvergence_fails_even_if_fixed_work_quality_would_pass(self):
        app = create_application({"app": "matrix_function", "n": 8, "iterations": 30,
                                  "max_iterations": 30, "stop_tolerance": 1e-15}, "cpu", 0)
        result = app.execute_to_quality(FrameworkBackend())
        self.assertFalse(result["converged"].item())
        self.assertLess(app.validate(result)["residual"], 5e-3)
        self.assertFalse(app.validate(result)["quality_pass"])

    def test_adversarial_output_fails_quality(self):
        app = create_application({"app": "matrix_function", "n": 8, "iterations": 20})
        output = app.execute(FrameworkBackend())
        output["result"] = output["result"] * 1.5
        self.assertFalse(app.validate(output)["quality_pass"])
        output["result"][0, 0] = float("nan")
        self.assertFalse(app.validate(output)["quality_pass"])

    def test_stored_input_hashes_and_scale_are_reproducible(self):
        case = {"app": "muon", "dtype": "bf16", "n": 8, "dynamic_range": 1e4,
                "distribution": "uniform", "input_scale": 0.1}
        first, second = create_application(case, "cpu", 6), create_application(case, "cpu", 6)
        self.assertEqual(first.metadata()["stored_inputs"], second.metadata()["stored_inputs"])
        other = create_application({**case, "input_scale": 0.2}, "cpu", 6)
        self.assertNotEqual(first.metadata()["stored_inputs"][0]["sha256"], other.metadata()["stored_inputs"][0]["sha256"])
        self.assertEqual(first.metadata()["provenance"]["scope"], "synthetic_algorithm")

    def test_input_quantization_can_be_part_of_the_quality_contract(self):
        app = create_application({"app": "muon", "dtype": "bf16", "n": 8,
                                  "quality": {"source_relative_error": 0}}, "cpu", 3)
        metrics = app.validate(app.execute(FrameworkBackend()))
        self.assertGreater(metrics["input_quantization_reference_error"], 0)
        self.assertGreater(metrics["source_relative_error"], 0)
        self.assertFalse(metrics["quality_pass"])

    def test_safe_loader_records_file_digest_and_rejects_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.npz"
            np.savez(path, G=np.eye(3), metadata_json=json.dumps({"dataset": "fixture"}))
            tensors, info = load_input_data(path)
            self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(info["source_metadata"]["dataset"], "fixture")
            self.assertEqual(tuple(tensors["G"].shape), (3, 3))
            unsafe = Path(folder) / "object.npz"
            np.savez(unsafe, G=np.array([{"unexpected": "object"}], dtype=object))
            with self.assertRaises(ValueError):
                load_input_data(unsafe)

    def test_invalid_inputs_fail_before_measurement(self):
        with self.assertRaises(ValueError):
            create_application({"app": "matrix_function", "condition_number": 0})
        with self.assertRaises(ValueError):
            create_application({"app": "purification", "n": 8, "occupied": 9})
        with self.assertRaises(ValueError):
            create_application({"app": "muon", "quality": {"unused": 1}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing_occupancy.pt"
            torch.save({"H": torch.diag(torch.tensor([-1., 1.]))}, path)
            with self.assertRaisesRegex(ValueError, "requires case occupied"):
                create_application({"app": "purification", "input_path": str(path)})
        with self.assertRaises(FileNotFoundError):
            create_application({"app": "mlp", "input_path": "/missing/application/input.npz"})


if __name__ == "__main__":
    unittest.main()
