"""Contracts for configuration, recovery, failures and process-level statistics."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite import runner, report


def minimal_config():
    return {"schema_version": 1, "suite": "test", "backends": ["torch"], "baseline": "torch",
            "modes": ["launch"], "measurement": {"independent_processes": 1, "warmup": 0, "groups": 2,
            "calls_per_group": 1, "rotate_inputs": 2, "fresh_quality_trials": 1, "timeout_seconds": 120},
            "cases": [{"id": "matrix", "kind": "matrix_function", "n": 8, "iterations": 15, "dtype": "fp32"}]}


class ConfigTests(unittest.TestCase):
    def test_all_checked_in_configs_are_valid_without_torch(self):
        for path in (ROOT / "bench" / "configs").glob("*.json"):
            with self.subTest(path=path):
                runner.load_config(path)

    def test_unknown_fields_and_versions_fail(self):
        for change in ({"schema_version": 2}, {"typo": 7}, {"measurement": {"group": 2}}):
            with self.subTest(change=change), self.assertRaises(runner.ConfigurationError):
                runner.validate_config({**minimal_config(), **change})

    def test_bool_and_invalid_quality_do_not_pass_as_numbers(self):
        for value in (True, -1, float("inf"), "0.1"):
            cfg = minimal_config()
            cfg["cases"][0]["quality"] = {"residual": value}
            with self.subTest(value=value), self.assertRaises(runner.ConfigurationError):
                runner.validate_config(cfg)
        cfg = minimal_config()
        cfg["measurement"]["groups"] = True
        with self.assertRaises(runner.ConfigurationError):
            runner.validate_config(cfg)

    def test_duplicate_ids_and_bad_backend_option_fail(self):
        cfg = minimal_config()
        cfg["cases"].append(dict(cfg["cases"][0]))
        with self.assertRaises(runner.ConfigurationError):
            runner.validate_config(cfg)
        cfg = minimal_config()
        cfg["backend_options"] = {"record_inputs": "true"}
        with self.assertRaises(runner.ConfigurationError):
            runner.validate_config(cfg)

    def test_aliases_do_not_evaluate_absent_kind(self):
        for alias in ("app", "application"):
            cfg = minimal_config()
            cfg["cases"][0][alias] = cfg["cases"][0].pop("kind")
            checked = runner.validate_config(cfg)
            payload = {"case": checked["cases"][0], "mode": "launch", "process_index": 0, "seed": 1, "device": "cpu"}
            self.assertEqual(runner._base_record(payload, "torch")["kind"], "matrix_function")

    def test_list_and_dry_run_require_only_stdlib(self):
        for flag in ("--list", "--dry-run"):
            # -S removes installed site packages, including torch.
            done = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / "run_suite.py"),
                "--config", str(ROOT / "bench" / "configs" / "smoke.json"), flag], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("muon_bf16_64", done.stdout)

    def test_relative_assets_are_resolved_at_repo_root(self):
        cfg = minimal_config()
        cfg["cases"][0]["input_path"] = "data/suite/missing.pt"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(cfg))
            self.assertEqual(runner.load_config(path)["cases"][0]["input_path"], str(ROOT / "data/suite/missing.pt"))

    def test_dynamic_graph_is_recorded_as_unsupported(self):
        cfg = minimal_config()
        cfg["cases"][0]["evaluation"] = "time_to_quality"
        checked = runner.validate_config(cfg)
        self.assertEqual(checked["cases"][0]["evaluation"], "time_to_quality")
        # Dynamic convergence in other applications is a configuration error.
        cfg["cases"][0]["kind"] = "muon"
        with self.assertRaises(runner.ConfigurationError):
            runner.validate_config(cfg)


class StatisticsTests(unittest.TestCase):
    def run_record(self, process, speed=1.2):
        return {"case_id": "a", "mode": "launch", "backend": "megafmm", "process_index": process,
                "status": "ok", "speedup": speed, "kind": "matrix_function", "application_quality_pass": True,
                "implementation_pass": True, "qualified_win": True}

    def test_one_process_has_no_confidence_interval(self):
        s = report.process_summary([self.run_record(0)])[0]
        self.assertEqual(s["paired_process_count"], 1)
        self.assertIsNone(s["speedup_ci95_low"])
        self.assertIn("insufficient", s["statistical_note"])
        self.assertFalse(s["statistically_supported_speedup"])

    def test_duplicate_observations_cannot_inflate_process_count(self):
        s = report.process_summary([self.run_record(0)] * 20)[0]
        self.assertEqual(s["paired_process_count"], 1)

    def test_failures_remain_in_coverage_and_block_supported_claim(self):
        rows = [self.run_record(i) for i in range(3)]
        rows += [{**self.run_record(3), "status": "oom", "speedup": None}]
        s = report.process_summary(rows)[0]
        self.assertEqual(s["statuses"], {"ok": 3, "oom": 1})
        self.assertEqual(s["process_count"], 4)
        self.assertFalse(s["statistically_supported_speedup"])

    def test_baseline_must_pass_quality_for_qualified_win(self):
        base = {"backend": "cublaslt", "status": "ok", "kind": "matrix_function", "steady_event_ms": 2.,
                "application_quality_pass": False, "implementation_pass": True}
        candidate = {"backend": "megafmm", "status": "ok", "kind": "matrix_function", "steady_event_ms": 1.,
                     "application_quality_pass": True, "implementation_pass": True}
        data = {"runs": [base, candidate]}
        runner._finalize_pairs(data, "cublaslt")
        self.assertTrue(candidate["performance_win"])
        self.assertFalse(candidate["qualified_win"])

    def test_numeric_failure_is_separate_from_implementation(self):
        r = {"kind": "gemm", "backend": "megafmm", "status": "ok", "implementation_pass": True,
             "gemm_gate_pass": False, "native_accepted": True}
        self.assertTrue(runner._quality_failed(r))


class ArchiveTests(unittest.TestCase):
    def test_gemm_exit_codes_keep_numeric_rejection_separate_from_execution_failure(self):
        for code, expected_status, expected_implementation in ((0, "ok", True), (2, "ok", True), (1, "failed", False)):
            with self.subTest(returncode=code), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                binary = folder / "megafmm_bench"
                binary.touch()
                artifact_dir = folder / "artifacts"
                artifact_dir.mkdir()
                row = {"nonfinite": "0", "numerical_pass": "0", "accepted": "0", "precision_required": "1",
                       "dense_event_ms": "1", "solution_event_ms": ".5", "actual": "SW"}
                with (artifact_dir / "gemm.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                payload = {"case": {"id": "gemm", "kind": "gemm", "m": 64, "n": 64, "k": 64},
                    "mode": "launch", "process_index": 0, "seed": 1, "device": "cuda:0",
                    "backends": ["cublaslt", "megafmm"], "baseline": "cublaslt", "backend_options": {},
                    "measurement": runner.MEASUREMENT_DEFAULTS, "gemm_binary": str(binary), "artifact_dir": str(artifact_dir)}
                with mock.patch("bench.suite.runner.subprocess.run", return_value=subprocess.CompletedProcess([], code, "", "")):
                    result = runner.gemm_worker(payload)
                candidate = next(r for r in result["runs"] if r["backend"] == "megafmm")
                self.assertEqual(candidate["status"], expected_status)
                self.assertIs(candidate["implementation_pass"], expected_implementation)
                self.assertIs(candidate["gemm_gate_pass"], False)
                self.assertIs(candidate["native_accepted"], False)
                self.assertEqual(runner._quality_failed(candidate), expected_status == "ok")

    def test_recovery_uses_worker_result_after_parent_was_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            result = folder / "workers" / "matrix__launch__p0" / "result.json"
            row = {"case_id": "matrix", "backend": "torch", "mode": "launch", "process_index": 0,
                   "status": "ok", "implementation_pass": True}
            runner.write_json(result, {"runs": [row], "samples": [], "metrics": [], "environment": {"torch": "test"}})
            for _ in range(2):
                runner.rebuild_collections(folder)
            records = report.read_jsonl(folder / "runs.jsonl")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["worker_key"], "matrix__launch__p0")
            self.assertEqual(records[0]["process_environment"]["torch"], "test")

    def test_reports_preserve_failure_and_escape_html(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            runner.write_json(directory / "resolved_config.json", minimal_config())
            runner._append_jsonl(directory / "runs.jsonl", [{"case_id": "matrix", "backend": "torch",
                "mode": "launch", "process_index": 0, "status": "failed", "error": "<script>bad</script>"}])
            report.build_report(directory)
            rendered = (directory / "report.html").read_text()
            self.assertIn("&lt;script&gt;", rendered)
            self.assertIn("failed", rendered)
            self.assertTrue((directory / "summary.csv").exists())

    def test_resume_rejects_changed_source_or_binary(self):
        cfg = runner.validate_config(minimal_config())
        with tempfile.TemporaryDirectory() as temp:
            args = runner.parser().parse_args(["--output", temp, "--device", "cpu", "--resume"])
            runner.write_json(Path(temp) / "resolved_config.json", cfg)
            runner.write_json(Path(temp) / "manifest.json", {"device": "cpu", "source_hashes": {"old": "hash"}})
            with self.assertRaisesRegex(runner.ConfigurationError, "source hashes"):
                runner.run_suite(cfg, args)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional for runner metadata tests")
class CPUIntegrationTests(unittest.TestCase):
    def test_worker_passes_measurement_mode_to_tuner_over_conflicting_config(self):
        from bench.suite.backends import Backend as RealBackend
        for mode, requested, expected in (("launch", True, False), ("graph", False, True)):
            with self.subTest(mode=mode):
                calls = []
                def construct(name, device, dtype, options):
                    calls.append(dict(options))
                    return RealBackend(name, device, dtype, options)
                case = minimal_config()["cases"][0]
                payload = {"case": case, "mode": mode, "process_index": 0, "seed": 1,
                    "device": "cpu", "backends": ["torch"], "baseline": "torch",
                    "backend_options": {"sw_tuning_graph": requested},
                    "measurement": {**runner.MEASUREMENT_DEFAULTS, "warmup": 0, "groups": 1,
                        "calls_per_group": 1, "fresh_quality_trials": 1, "rotate_inputs": 1}}
                with mock.patch("bench.suite.backends.Backend", side_effect=construct):
                    result = runner.application_worker(payload)
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0]["sw_tuning_graph"], expected)
                record = result["runs"][0]
                self.assertIs(record["requested_backend_options"]["sw_tuning_graph"], requested)
                self.assertIs(record["resolved_backend_options"]["sw_tuning_graph"], expected)
                self.assertEqual(record["status"], "ok" if mode == "launch" else "unsupported")

    def test_real_cpu_subprocess_is_torch_and_roundtrip_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            path, out = temp / "config.json", temp / "out"
            path.write_text(json.dumps(minimal_config()))
            command = [sys.executable, str(ROOT / "scripts" / "run_suite.py"), "--config", str(path),
                       "--output", str(out), "--device", "cpu"]
            first = subprocess.run(command, capture_output=True, text=True, timeout=150)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            runs = report.read_jsonl(out / "runs.jsonl")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["backend"], "torch")
            self.assertNotEqual(runs[0]["worker_pid"], 0)
            self.assertTrue(runs[0]["application_quality_pass"])
            self.assertIsNone(runs[0]["steady_event_ms"])
            self.assertGreater(runs[0]["steady_wall_ms"], 0)
            self.assertEqual(len(report.read_jsonl(out / "samples.jsonl")), 2)
            before = (out / "runs.jsonl").read_text()
            second = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=30)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(before, (out / "runs.jsonl").read_text())

    def test_external_missing_input_is_explicit_skip(self):
        payload = {"case": {"id": "missing", "kind": "muon", "dtype": "bf16", "input_path": "/nonexistent/suite/input.pt"},
                   "mode": "launch", "measurement": runner.MEASUREMENT_DEFAULTS, "process_index": 0,
                   "seed": 3, "device": "cpu", "backends": ["torch"], "baseline": "torch", "backend_options": {}}
        result = runner.worker(payload)
        self.assertEqual(result["runs"][0]["status"], "skipped")
        self.assertFalse(result["runs"][0]["implementation_pass"])


if __name__ == "__main__":
    unittest.main()
