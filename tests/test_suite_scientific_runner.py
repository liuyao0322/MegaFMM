"""CPU regressions for scientific input sharing, numerical gates and pairing."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite import runner


def scientific_config():
    return {
        "schema_version": 1, "suite": "scientific-runner-test",
        "backends": ["torch", "practical_torch"], "baseline": "torch",
        "modes": ["launch"],
        "measurement": {"independent_processes": 1, "warmup": 1, "groups": 2,
                        "calls_per_group": 2, "rotate_inputs": 2,
                        "fresh_quality_trials": 2, "timeout_seconds": 120},
        "cases": [{"id": "covariance", "kind": "covariance", "dtype": "fp32",
                   "n": 7, "iterations": 3, "transition_radius": 0.95}],
    }


def covariance_payload():
    config = runner.validate_config(scientific_config())
    return {"case": config["cases"][0], "mode": "launch", "process_index": 0,
            "seed": 23, "device": "cpu", "backends": config["backends"],
            "baseline": config["baseline"], "measurement": config["measurement"],
            "backend_options": config["backend_options"]}


def paired_records():
    common = {"kind": "covariance", "status": "ok", "implementation_pass": True,
              "application_quality_pass": True}
    # Event-only timing would report a loss; application wall time reports a win.
    baseline = {**common, "backend": "torch", "steady_wall_ms": 10.,
                "steady_event_ms": 2., "cold_setup_ms": 3.}
    candidate = {**common, "backend": "megafmm", "steady_wall_ms": 5.,
                 "steady_event_ms": 4., "cold_setup_ms": 15.}
    return {"runs": [baseline, candidate]}


class ScientificConfigTests(unittest.TestCase):
    def test_transition_radius_is_finite_open_interval_and_covariance_only(self):
        for value in (0, 1, -0.1, 1.1, True, None, "0.95",
                      float("nan"), float("inf"), -float("inf")):
            config = scientific_config()
            config["cases"][0]["transition_radius"] = value
            with self.subTest(value=value), self.assertRaises(runner.ConfigurationError):
                runner.validate_config(config)
        config = scientific_config()
        config["cases"][0]["kind"] = "matrix_function"
        with self.assertRaisesRegex(runner.ConfigurationError, "transition_radius"):
            runner.validate_config(config)
        for value in (0.001, 0.95, 0.999):
            config = scientific_config()
            config["cases"][0]["transition_radius"] = value
            with self.subTest(valid=value):
                self.assertEqual(runner.validate_config(config)["cases"][0]["transition_radius"], value)

    def test_covariance_requires_fp32_and_defaults_to_fp32(self):
        for dtype in ("fp16", "bf16"):
            config = scientific_config()
            config["cases"][0]["dtype"] = dtype
            with self.subTest(dtype=dtype), self.assertRaisesRegex(runner.ConfigurationError, "covariance requires fp32"):
                runner.validate_config(config)
        config = scientific_config()
        del config["cases"][0]["dtype"]
        runner.validate_config(config)


class ScientificPairingTests(unittest.TestCase):
    def test_application_speedup_uses_wall_and_preserves_baseline_evidence(self):
        result = paired_records()
        runner._finalize_pairs(result, "torch")
        candidate = result["runs"][1]
        self.assertEqual(candidate["speedup"], 2.)
        self.assertEqual(candidate["speedup_time_basis"], "steady_wall_ms")
        self.assertIs(candidate["performance_win"], True)
        self.assertIs(candidate["qualified_win"], True)
        self.assertEqual(candidate["baseline_backend"], "torch")
        self.assertEqual(candidate["baseline_status"], "ok")
        self.assertIs(candidate["baseline_quality_pass"], True)
        self.assertIs(candidate["baseline_implementation_pass"], True)
        self.assertEqual(candidate["break_even_calls"], 3)

    def test_baseline_quality_or_implementation_failure_disqualifies_speedup(self):
        for field in ("application_quality_pass", "implementation_pass"):
            result = paired_records()
            result["runs"][0][field] = False
            with self.subTest(field=field):
                runner._finalize_pairs(result, "torch")
                candidate = result["runs"][1]
                self.assertEqual(candidate["speedup"], 2.)
                self.assertIs(candidate["performance_win"], True)
                self.assertIs(candidate["qualified_win"], False)
                self.assertIs(candidate["baseline_quality_pass" if field == "application_quality_pass"
                                        else "baseline_implementation_pass"], False)

    def test_native_gemm_retains_event_timing(self):
        result = paired_records()
        for row in result["runs"]:
            row.update(kind="gemm", gemm_gate_pass=True)
        runner._finalize_pairs(result, "torch")
        self.assertEqual(result["runs"][1]["speedup"], 0.5)
        self.assertEqual(result["runs"][1]["speedup_time_basis"], "steady_event_ms")
        self.assertIs(result["runs"][1]["performance_win"], False)

    def test_nonpositive_or_nonfinite_timings_cannot_create_speedups(self):
        for backend_index in (0, 1):
            for invalid in (True, False, 0., -1., float("nan"), float("inf"), -float("inf"), "5"):
                result = paired_records()
                result["runs"][backend_index]["steady_wall_ms"] = invalid
                with self.subTest(backend=backend_index, invalid=invalid):
                    runner._finalize_pairs(result, "torch")
                    self.assertNotIn("speedup", result["runs"][1])
                    self.assertNotIn("qualified_win", result["runs"][1])


class ScientificResumeTests(unittest.TestCase):
    def test_changed_runtime_rejects_resume_before_worker_rerun(self):
        config = runner.validate_config(scientific_config())
        source_hashes = {"bench/suite/runner.py": "same-source"}
        original_runtime = {"runtime": {"torch": "original-build", "cuda_runtime": None},
                            "visible_devices": None, "driver": None}
        changed_runtime = copy.deepcopy(original_runtime)
        changed_runtime["runtime"]["torch"] = "different-build"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = runner.parser().parse_args(["--output", directory, "--device", "cpu", "--resume"])
            runner.write_json(out / "resolved_config.json", config)
            runner.write_json(out / "manifest.json", {
                "device": "cpu", "source_hashes": source_hashes, "input_asset_hashes": {},
                "runtime_fingerprint": original_runtime,
            })
            prior = out / "workers" / "covariance__launch__p0" / "result.json"
            runner.write_json(prior, {"runs": [{"backend": "torch", "status": "ok"}],
                                      "samples": [], "metrics": []})
            saved_result = prior.read_bytes()
            saved_manifest = (out / "manifest.json").read_bytes()
            with mock.patch.object(runner, "source_hashes", return_value=source_hashes), \
                 mock.patch.object(runner, "runtime_fingerprint", return_value=changed_runtime) as fingerprint, \
                 mock.patch.object(runner, "_execute_process") as launch:
                with self.assertRaisesRegex(runner.ConfigurationError, "runtime differs"):
                    runner.run_suite(config, args)
                launch.assert_not_called()
                fingerprint.assert_called_once_with("cpu")
            self.assertEqual(prior.read_bytes(), saved_result)
            self.assertEqual((out / "manifest.json").read_bytes(), saved_manifest)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
class ScientificCPUWorkerTests(unittest.TestCase):
    def test_covariance_backends_share_slots_and_keep_all_quality_checks(self):
        from bench.suite.applications import create_application

        payload = covariance_payload()
        with mock.patch("bench.suite.applications.create_application", wraps=create_application) as construct:
            result = runner.application_worker(payload)
        measure = payload["measurement"]
        self.assertEqual(construct.call_count, measure["rotate_inputs"] + measure["fresh_quality_trials"])
        self.assertEqual([call.args[2] for call in construct.call_args_list],
                         [payload["seed"] + 1000 + i for i in range(measure["rotate_inputs"])] +
                         [payload["seed"] + 100000 + i for i in range(measure["fresh_quality_trials"])])
        runs = {row["backend"]: row for row in result["runs"]}
        self.assertEqual(set(runs), set(payload["backends"]))
        self.assertEqual(runs["torch"]["input_sets"], runs["practical_torch"]["input_sets"])
        self.assertEqual(runs["torch"]["input_hash"], runs["practical_torch"]["input_hash"])
        self.assertNotEqual(runs["torch"]["input_sets"][0]["stored_inputs"],
                            runs["torch"]["input_sets"][1]["stored_inputs"])
        expected_flops = 4 * payload["case"]["iterations"] * payload["case"]["n"] ** 3
        quality_by_backend = {}
        for name, run in runs.items():
            self.assertEqual(run["status"], "ok", run.get("traceback"))
            self.assertIs(run["implementation_pass"], True)
            self.assertIs(run["application_quality_pass"], True)
            self.assertEqual(run["primary_time_basis"], "steady_wall_ms")
            self.assertEqual(run["throughput_time_basis"], "steady_wall_ms")
            self.assertEqual(run["nominal_flops"], expected_flops)
            self.assertAlmostEqual(run["effective_gflops"], expected_flops / (run["steady_wall_ms"] * 1e6))
            self.assertIsNone(run["steady_event_ms"])
            metrics = [row for row in result["metrics"] if row["backend"] == name]
            fresh = [row for row in metrics if "trial" in row]
            self.assertEqual([row["trial"] for row in fresh], list(range(measure["fresh_quality_trials"])))
            self.assertEqual([row["input"]["seed"] for row in fresh], run["validation_seeds"])
            for phase in ("timed_path_before", "timed_path_after"):
                checks = [row for row in metrics if row.get("phase") == phase]
                self.assertEqual([row["input_slot"] for row in checks], list(range(measure["rotate_inputs"])))
                self.assertEqual([row["input"] for row in checks], run["input_sets"])
            self.assertTrue(all(row["values"]["quality_pass"] for row in metrics))
            quality_by_backend[name] = [row["values"] for row in fresh]
            samples = [row for row in result["samples"] if row["backend"] == name]
            self.assertEqual(len(samples), measure["groups"])
            self.assertTrue(all(row["wall_ms"] > 0 and row["event_ms"] is None for row in samples))
        # Both CPU framework backends execute the same stored FP32 problem.
        self.assertEqual(quality_by_backend["torch"], quality_by_backend["practical_torch"])
        self.assertEqual(runs["practical_torch"]["speedup_time_basis"], "steady_wall_ms")

    def test_fresh_and_persistent_slot_quality_failures_reach_run_gate(self):
        from bench.suite.applications import create_application

        for failing_phase in ("fresh", "persistent"):
            payload = covariance_payload()
            failing_seed = payload["seed"] + (100000 if failing_phase == "fresh" else 1001)

            def construct(case, device, seed):
                app = create_application(case, device, seed)
                if seed == failing_seed:
                    execute = app.execute

                    def corrupt_candidate(backend):
                        output = execute(backend)
                        if backend.name == "practical_torch":
                            output = {**output, "result": output["result"] + 1}
                        return output

                    app.execute = corrupt_candidate
                return app

            with self.subTest(phase=failing_phase), \
                 mock.patch("bench.suite.applications.create_application", side_effect=construct):
                result = runner.application_worker(payload)
                runs = {row["backend"]: row for row in result["runs"]}
                candidate = runs["practical_torch"]
                self.assertEqual(candidate["status"], "ok", candidate.get("traceback"))
                self.assertIs(candidate["implementation_pass"], True)
                self.assertIs(candidate["application_quality_pass"], False)
                self.assertIs(runs["torch"]["application_quality_pass"], True)
                self.assertIs(candidate["qualified_win"], False)
                fresh = [row for row in result["metrics"] if row["backend"] == "practical_torch" and "trial" in row]
                persistent = [row for row in result["metrics"] if row["backend"] == "practical_torch" and "input_slot" in row]
                self.assertEqual(all(row["values"]["quality_pass"] for row in fresh), failing_phase != "fresh")
                self.assertEqual(all(row["values"]["quality_pass"] for row in persistent), failing_phase != "persistent")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class SharedDynamicStateTests(unittest.TestCase):
    def test_validation_iterations_belong_to_saved_output(self):
        from bench.suite.applications import create_application
        from bench.suite.backends import Backend

        for kind in ("matrix_function", "purification"):
            with self.subTest(kind=kind):
                app = create_application({"kind": kind, "n": 8, "iterations": 25,
                    "max_iterations": 45, "stop_tolerance": 1e-3}, "cpu", 17)
                backend = Backend("torch", "cpu", "fp32")
                output = app.execute_to_quality(backend)
                saved_count = int(output["iterations"].item())
                app.last_iterations = saved_count + 99
                self.assertEqual(app.validate(output)["iterations"], saved_count)
                backend.close()


if __name__ == "__main__":
    unittest.main()
