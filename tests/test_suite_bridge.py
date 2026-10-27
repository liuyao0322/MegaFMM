"""Backend contracts; optional native CUDA integration uses explicit opt-in."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite.backends import Backend, BackendError, _library_path, fp64_gemm_diagnostics

try:
    import torch
except ImportError:
    torch = None


class ImportAndDiscoveryTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import bench.suite.backends; assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_missing_library_is_actionable(self):
        with self.assertRaisesRegex(BackendError, "does not exist"):
            _library_path(ROOT / "nonexistent-suite-test-library.so")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchBackendTests(unittest.TestCase):
    def test_noncontiguous_inputs_and_fp32_contract(self):
        a = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
        b = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        with Backend("torch", "cpu", "fp32") as backend:
            actual = backend.gemm(a, b)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, a @ b)
            metadata = backend.metadata()
            self.assertEqual(metadata["implementation"], "torch.mm_framework_dispatch")
            self.assertEqual(metadata["output_rounding"], "fp32")
            json.dumps(metadata, allow_nan=False)

    def test_low_precision_rounding_and_recording(self):
        generator = torch.Generator().manual_seed(99)
        a = torch.randn((4, 8), generator=generator)
        b = torch.randn((8, 3), generator=generator)
        for dtype, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            with self.subTest(dtype=name), Backend("torch", "cpu", name,
                    {"record_inputs": True, "record_limit": 1}) as backend:
                expected = (a.to(dtype) @ b.to(dtype)).float()
                torch.testing.assert_close(backend.gemm(a, b), expected, rtol=0, atol=0)
                self.assertEqual(backend.recorded[0]["a"].dtype, dtype)
                self.assertEqual(backend.recorded[0]["out"].dtype, torch.float32)
                self.assertEqual(backend.metadata()["output_rounding"], name + "_then_fp32")
                backend.gemm(a, b)
                self.assertEqual(backend.recorded_dropped, 1)
                backend.clear_recording()
                self.assertEqual(backend.recorded, [])
                self.assertEqual(backend.recorded_dropped, 0)

    def test_setup_and_freeze_reject_unseen_shape(self):
        with Backend("torch", "cpu", "fp32") as backend:
            backend.setup((torch.eye(3), torch.eye(3))).freeze()
            torch.testing.assert_close(backend.gemm(torch.eye(3), torch.eye(3)), torch.eye(3))
            with self.assertRaisesRegex(BackendError, "Unprepared GEMM shape"):
                backend.gemm(torch.eye(4), torch.eye(4))

    def test_closed_backend_rejects_execution(self):
        backend = Backend("torch", "cpu", "fp32")
        backend.close()
        backend.close()
        self.assertTrue(backend.metadata()["closed"])
        with self.assertRaisesRegex(BackendError, "closed"):
            backend.gemm(torch.eye(2), torch.eye(2))

    def test_invalid_contracts_fail_without_native_fallback(self):
        with self.assertRaises(ValueError):
            Backend("cublaslt", "cpu", "fp32")
        with self.assertRaises(ValueError):
            Backend("torch", "cpu", "fp64")
        with self.assertRaises(ValueError):
            Backend("torch", "cpu", "fp32", {"allow_tf32": True})
        for options in ({"workspace_limit": -1}, {"heuristic_candidates": 129},
                        {"tuning_input_sets": 0}, {"sw_tuning_candidates": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Backend("torch", "cpu", "fp32", options)
        with Backend("torch", "cpu", "fp32") as backend:
            with self.assertRaisesRegex(ValueError, "compatible 2D"):
                backend.gemm(torch.zeros(2, 3), torch.zeros(4, 2))

    def test_local_diagnostics_do_not_invent_an_unmeasured_gate(self):
        with Backend("torch", "cpu", "fp32", {"record_inputs": True}) as backend:
            self.assertIsNone(backend.local_gemm_diagnostics()["gemm_gate_pass"])
            backend.gemm(torch.eye(3), torch.eye(3))
            diagnostic = backend.local_gemm_diagnostics()
            self.assertEqual(diagnostic["records"][0]["relative_error"], 0)
            self.assertIsNone(diagnostic["gemm_gate_pass"])
            self.assertEqual(diagnostic["count"], 1)

    def test_local_gate_uses_same_input_dense_error_and_preserves_failure(self):
        a, b = torch.eye(3), torch.eye(3)
        dense = a @ b
        record = {"a": a, "b": b, "out": dense.clone()}
        self.assertTrue(fp64_gemm_diagnostics([record], [dense])["gemm_gate_pass"])
        record["out"] = dense + .01
        self.assertFalse(fp64_gemm_diagnostics([record], [dense])["gemm_gate_pass"])
        record["out"][0, 0] = float("nan")
        diagnostic = fp64_gemm_diagnostics([record], [dense])
        self.assertFalse(diagnostic["finite_pass"])
        self.assertEqual(diagnostic["records"][0]["relative_error"], "+inf")
        json.dumps(diagnostic, allow_nan=False)


NATIVE_TESTS = bool(os.environ.get("MEGAFMM_RUN_GPU_TESTS")) and torch is not None


@unittest.skipUnless(NATIVE_TESTS, "Set MEGAFMM_RUN_GPU_TESTS=1 to run native GPU integration")
class NativeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        if torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("The production kernel requires Hopper")

    def test_dtype_contract_graph_and_changed_binding(self):
        for dtype in ("fp32", "bf16", "fp16"):
            # Fixed SW tests bridge semantics without running a long tuner for
            # every unit test.  The runner separately exercises SWTuned.
            for name in ("cublaslt", "megafmm_fixed"):
                with self.subTest(dtype=dtype, backend=name), Backend(name, "cuda", dtype,
                        {"workspace_limit": 64 << 20}) as backend:
                    # This is an ABI/layout contract test, not an application
                    # accuracy gate.  Small integer data and a signed
                    # permutation keep the seven-product transforms and
                    # reconstruction exactly representable in every dtype.
                    # Random rounding-error behavior is measured separately by
                    # the suite's original-input FP64 and application gates.
                    index = torch.arange(32, device="cuda")
                    values = torch.arange(32 * 32, device="cuda").reshape(32, 32)
                    a = ((7 * values + values // 32) % 5 - 2).float().T
                    permutation = torch.zeros((32, 32), device="cuda")
                    permutation[index, (5 * index + 3) % 32] = torch.where(
                        index % 2 == 0, 1., -1.)
                    b = permutation.T
                    backend.setup((a, b)).freeze()
                    actual = backend.gemm(a, b)
                    reference = a.to(backend.dtype).double() @ b.to(backend.dtype).double()
                    torch.testing.assert_close(actual.double(), reference, rtol=0, atol=0)
                    # Rebind the cached plan to new storage and retain the
                    # earlier output, which must not alias a reused buffer.
                    a2, b2 = a.clone(), (-b).clone()
                    rebound = backend.gemm(a2, b2)
                    torch.testing.assert_close(rebound.double(), -reference, rtol=0, atol=0)
                    torch.testing.assert_close(actual.double(), reference, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = backend.gemm(a, b)
                    a.fill_(1)
                    b.fill_(2)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(captured, torch.full_like(captured, 64), rtol=0, atol=0)
                    metadata = backend.metadata()
                    self.assertEqual(metadata["input_dtype"], dtype)
                    self.assertEqual(metadata["output_rounding"], "fp32")
                    self.assertGreater(metadata["loaded_native_versions"]["cublas_version"], 0)
                    self.assertEqual(metadata["plan_count"], 1)
                    json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
