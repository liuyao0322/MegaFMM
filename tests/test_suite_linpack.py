"""Blocked LU correctness, pivot semantics, residual gates and runner integration."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bench.suite import runner, report

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from bench.suite.applications import create_application


class CountingBackend:
    def __init__(self):
        self.shapes = []

    def gemm(self, a, b):
        self.shapes.append([a.shape[0], b.shape[1], a.shape[1]])
        return a @ b


class LinpackConfigTests(unittest.TestCase):
    def config(self):
        return runner.load_config(Path(__file__).resolve().parents[1] / "bench/configs/linpack_smoke.json")

    def test_invalid_contracts_rejected_without_torch(self):
        for field, value in (("dtype", "bf16"), ("block_size", 0), ("block_size", True),
                             ("iterations", 3), ("modes", ["graph"])):
            config = self.config()
            config["cases"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(runner.ConfigurationError):
                runner.validate_config(config)

    def test_host_work_uses_wall_time_for_speedup(self):
        records = [{"kind": "linpack", "backend": name, "status": "ok", "implementation_pass": True,
                    "application_quality_pass": True, "steady_event_ms": event, "steady_wall_ms": wall}
                   for name, event, wall in (("cublaslt", 1, 4), ("megafmm", 2, 2))]
        runner._finalize_pairs({"runs": records}, "cublaslt")
        self.assertEqual(records[1]["speedup"], 2)
        self.assertEqual(records[1]["speedup_time_basis"], "steady_wall_ms")


@unittest.skipUnless(HAS_TORCH, "PyTorch is required")
class LinpackApplicationTests(unittest.TestCase):
    def test_factorization_solution_tails_and_immutable_inputs(self):
        for n, block in ((1, 8), (9, 1), (17, 4), (32, 8), (13, 13), (13, 30)):
            with self.subTest(n=n, block=block):
                app = create_application({"kind": "linpack", "n": n, "block_size": block}, "cpu", 18)
                matrix, rhs = app.matrix.clone(), app.rhs.clone()
                backend = CountingBackend()
                output = app.execute(backend)
                torch.testing.assert_close(app.matrix, matrix, rtol=0, atol=0)
                torch.testing.assert_close(app.rhs, rhs, rtol=0, atol=0)
                torch.testing.assert_close(output["result"].double(), torch.linalg.solve(matrix.double(), rhs.double()), rtol=1e-4, atol=1e-4)
                lower = torch.tril(output["lu"], diagonal=-1) + torch.eye(n)
                upper = torch.triu(output["lu"])
                torch.testing.assert_close(lower @ upper, matrix[output["permutation"]], rtol=1e-4, atol=2e-6)
                self.assertEqual(backend.shapes, app.metadata()["gemm_shapes_mnk"])
                self.assertEqual(app.metadata()["distribution"], "uniform")
                self.assertTrue(app.validate(output)["quality_pass"])
                torch.testing.assert_close(output["result"], app.execute(CountingBackend())["result"], rtol=0, atol=0)

    def test_pivots_cross_panel_boundaries_and_vector_rhs_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "system.pt"
            a = torch.tensor([[0., 1., 2., 3.], [0., 0., 4., 5.],
                              [0., 6., 7., 8.], [9., 10., 11., 12.]])
            b = a @ torch.tensor([1., -2., 3., -4.])
            torch.save({"A": a, "b": b}, path)
            app = create_application({"kind": "linpack", "input_path": str(path), "block_size": 2})
            output = app.execute(CountingBackend())
            self.assertEqual(output["permutation"][0].item(), 3)
            self.assertTrue(app.validate(output)["quality_pass"])
            torch.testing.assert_close(output["result"].flatten(), torch.tensor([1., -2., 3., -4.]), rtol=1e-5, atol=1e-5)
            self.assertEqual(app.metadata()["provenance"]["scope"], "imported_input")
            self.assertEqual(app.metadata()["distribution"], "imported")

    def test_scaled_residual_definition_and_bad_answers(self):
        app = create_application({"kind": "linpack", "n": 12, "block_size": 5})
        output = app.execute(CountingBackend())
        values = app.validate(output)
        self.assertAlmostEqual(values["scaled_residual"], values["backward_error"] / (12 * torch.finfo(torch.float32).eps))
        output["result"] += 0.5
        self.assertFalse(app.validate(output)["quality_pass"])
        output["result"][0] = float("nan")
        self.assertFalse(app.validate(output)["quality_pass"])

    def test_singular_and_invalid_inputs_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "system.pt"
            for a, b in ((torch.zeros(4, 4), torch.ones(4)),
                         (torch.ones(3, 4), torch.ones(3)),
                         (torch.eye(4), torch.ones(3)),
                         (torch.eye(4), torch.ones(4, 2))):
                torch.save({"A": a, "b": b}, path)
                with self.subTest(shape=tuple(a.shape), rhs=tuple(b.shape)), self.assertRaises((ValueError, RuntimeError)):
                    create_application({"kind": "linpack", "input_path": str(path)}).execute(CountingBackend())
            torch.save({"A": torch.eye(4), "b": torch.zeros(4)}, path)
            with self.assertRaisesRegex(ValueError, "dimension"):
                create_application({"kind": "linpack", "n": 5, "input_path": str(path)})
            app = create_application({"kind": "linpack", "input_path": str(path)})
            self.assertTrue(app.validate(app.execute(CountingBackend()))["quality_pass"])

    def test_cpu_cli_process_report_and_throughput(self):
        config = LinpackConfigTests().config()
        config.update(backends=["torch"], baseline="torch")
        config["cases"] = [{"id": "tiny", "kind": "linpack", "n": 17, "block_size": 4}]
        config["measurement"].update(groups=2, calls_per_group=1, warmup=0, rotate_inputs=1,
                                      fresh_quality_trials=1)
        with tempfile.TemporaryDirectory() as folder:
            path, output = Path(folder) / "config.json", Path(folder) / "run"
            path.write_text(json.dumps(config))
            process = subprocess.run([sys.executable, "scripts/run_suite.py", "--config", str(path),
                                      "--device", "cpu", "--output", str(output)],
                                     cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            records = report.read_jsonl(output / "runs.jsonl")
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertTrue(record["application_quality_pass"])
            self.assertAlmostEqual(record["effective_gflops"], record["nominal_flops"] / (record["steady_wall_ms"] * 1e6))
            self.assertGreater(record["effective_gflops"], 0)
            self.assertIn("Equivalent FP32 GFLOPS", (output / "report.html").read_text())
            self.assertIn("not FP64 HPL", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
