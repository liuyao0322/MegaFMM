"""Scientific conclusions require complete, numerically valid process pairs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite import report


def records(count=3, ratio=1.4):
    rows = []
    for process in range(count):
        common = {"case_id": "science", "kind": "matrix_function", "mode": "launch",
                  "process_index": process, "status": "ok", "implementation_pass": True,
                  "application_quality_pass": True, "steady_event_ms": 10.0,
                  "steady_wall_ms": 12.0, "effective_gflops": 100.0}
        rows.append({**common, "backend": "cublaslt"})
        rows.append({**common, "backend": "megafmm", "speedup": ratio,
                     "qualified_win": ratio > 1})
    return rows


def candidate_summary(rows, expected=3):
    summaries = report.process_summary(rows, baseline="cublaslt", expected_processes=expected)
    return next(s for s in summaries if s["backend"] == "megafmm")


class ScientificQualificationTests(unittest.TestCase):
    def test_complete_valid_pairs_support_speedup(self):
        summary = candidate_summary(records())
        self.assertEqual(summary["qualified_paired_process_count"], 3)
        self.assertAlmostEqual(summary["qualified_speedup_geomean"], 1.4)
        self.assertTrue(summary["statistically_supported_speedup"])
        self.assertEqual(summary["missing_process_count"], 0)

    def test_rejected_candidate_keeps_timing_without_qualified_claim(self):
        rows = records()
        for row in rows:
            if row["backend"] == "megafmm":
                row["application_quality_pass"] = False
        summary = candidate_summary(rows)
        self.assertEqual(summary["paired_process_count"], 3)
        self.assertAlmostEqual(summary["speedup_geomean"], 1.4)
        self.assertTrue(summary["statistically_supported_timing_speedup"])
        self.assertFalse(summary["statistically_supported_speedup"])
        self.assertEqual(summary["qualified_paired_process_count"], 0)
        self.assertIsNone(summary["qualified_speedup_geomean"])
        self.assertEqual(summary["quality_failed_process_count"], 3)
        self.assertEqual(summary["unqualified_timing_pair_count"], 3)
        self.assertEqual(summary["qualified_win_count"], 0)

    def test_baseline_quality_is_required_even_with_candidate_qualified_win(self):
        rows = records()
        for row in rows:
            if row["backend"] == "cublaslt":
                row["application_quality_pass"] = False
        summary = candidate_summary(rows)
        self.assertEqual(summary["paired_process_count"], 3)
        self.assertEqual(summary["qualified_paired_process_count"], 0)
        self.assertEqual(summary["baseline_quality_failed_process_count"], 3)
        self.assertFalse(summary["statistically_supported_speedup"])

    def test_baseline_execution_and_implementation_are_required(self):
        for changed in ({"status": "failed"}, {"implementation_pass": False}):
            with self.subTest(changed=changed):
                rows = records()
                for row in rows:
                    if row["backend"] == "cublaslt":
                        row.update(changed)
                summary = candidate_summary(rows)
                self.assertEqual(summary["paired_process_count"], 0)
                self.assertEqual(summary["qualified_paired_process_count"], 0)
                self.assertFalse(summary["statistically_supported_speedup"])

    def test_unchecked_gates_are_counted_and_do_not_pass(self):
        rows = records()
        for row in rows:
            row.pop("application_quality_pass")
        summary = candidate_summary(rows)
        self.assertEqual(summary["quality_unchecked_process_count"], 3)
        self.assertEqual(summary["baseline_quality_unchecked_process_count"], 3)
        self.assertEqual(summary["quality_checked_count"], 0)
        self.assertEqual(summary["qualified_paired_process_count"], 0)
        self.assertFalse(summary["statistically_supported_speedup"])

    def test_missing_candidate_or_baseline_blocks_complete_coverage_claim(self):
        summary = candidate_summary(records(), expected=4)
        self.assertEqual(summary["qualified_paired_process_count"], 3)
        self.assertEqual(summary["missing_process_count"], 1)
        self.assertFalse(summary["statistically_supported_speedup"])
        rows = [r for r in records() if not (r["backend"] == "cublaslt" and r["process_index"] == 1)]
        summary = candidate_summary(rows)
        self.assertEqual(summary["baseline_missing_process_count"], 1)
        self.assertEqual(summary["qualified_paired_process_count"], 2)
        self.assertFalse(summary["statistically_supported_speedup"])

    def test_valid_slowdowns_remain_in_qualified_ratios(self):
        rows = records()
        next(r for r in rows if r["backend"] == "megafmm")["speedup"] = 0.5
        summary = candidate_summary(rows)
        self.assertEqual(summary["qualified_paired_process_count"], 3)
        self.assertAlmostEqual(summary["qualified_speedup_geomean"], (0.5 * 1.4 * 1.4) ** (1 / 3))
        self.assertEqual(summary["qualified_win_count"], 2)
        self.assertFalse(summary["statistically_supported_speedup"])

    def test_native_gemm_uses_local_gate_and_finite_dense_contract(self):
        rows = records()
        for row in rows:
            row["kind"] = "gemm"
            row.pop("application_quality_pass")
            if row["backend"] == "megafmm":
                row["gemm_gate_pass"] = True
        self.assertTrue(candidate_summary(rows)["statistically_supported_speedup"])
        next(r for r in rows if r["backend"] == "megafmm")["gemm_gate_pass"] = False
        summary = candidate_summary(rows)
        self.assertEqual(summary["qualified_paired_process_count"], 2)
        self.assertFalse(summary["statistically_supported_speedup"])

    def test_standalone_records_require_explicit_baseline_evidence(self):
        rows = [r for r in records() if r["backend"] == "megafmm"]
        summary = report.process_summary(rows)[0]
        self.assertEqual(summary["paired_process_count"], 3)
        self.assertEqual(summary["qualified_paired_process_count"], 0)
        self.assertEqual(summary["baseline_missing_process_count"], 3)
        for row in rows:
            row.update(baseline_status="ok", baseline_implementation_pass=True, baseline_quality_pass=True)
        self.assertTrue(report.process_summary(rows)[0]["statistically_supported_speedup"])


class ScientificTimingTests(unittest.TestCase):
    def test_failed_partial_and_implementation_rejected_times_do_not_affect_medians(self):
        rows = records(count=5)
        for row in rows:
            if row["backend"] == "megafmm" and row["process_index"] >= 2:
                row.update(status="failed", steady_event_ms=0.1, steady_wall_ms=0.1,
                           effective_gflops=10000.0)
        summary = candidate_summary(rows, expected=5)
        self.assertEqual(summary["median_event_ms"], 10.0)
        self.assertEqual(summary["median_wall_ms"], 12.0)
        self.assertEqual(summary["median_effective_gflops"], 100.0)
        self.assertEqual(summary["successful_process_count"], 2)
        self.assertFalse(summary["statistically_supported_speedup"])
        for row in rows:
            if row["backend"] == "megafmm" and row["process_index"] >= 2:
                row.update(status="ok", implementation_pass=False)
        self.assertEqual(candidate_summary(rows, expected=5)["median_event_ms"], 10.0)

    def test_nonfinite_nonpositive_and_boolean_ratios_are_excluded(self):
        for invalid in (float("nan"), float("inf"), -1, 0, True):
            with self.subTest(invalid=invalid):
                rows = records()
                for row in rows:
                    if row["backend"] == "megafmm":
                        row.update(speedup=invalid, steady_event_ms=invalid, steady_wall_ms=invalid)
                summary = candidate_summary(rows)
                self.assertEqual(summary["invalid_ratio_process_count"], 3)
                self.assertEqual(summary["paired_process_count"], 0)
                self.assertIsNone(summary["median_event_ms"])
                self.assertIsNone(summary["median_wall_ms"])
                json.dumps(summary, allow_nan=False)

    def test_duplicate_observations_do_not_weight_statistics_and_conflicts_are_excluded(self):
        rows = records()
        duplicate = next(r for r in rows if r["backend"] == "megafmm")
        rows.extend(copy.deepcopy(duplicate) for _ in range(20))
        summary = candidate_summary(rows)
        self.assertEqual(summary["qualified_paired_process_count"], 3)
        self.assertEqual(summary["application_quality_pass_count"], 3)
        rows.append({**duplicate, "steady_event_ms": 0.1, "speedup": 99.0})
        summary = candidate_summary(rows)
        self.assertEqual(summary["conflicting_process_count"], 1)
        self.assertEqual(summary["qualified_paired_process_count"], 2)
        self.assertEqual(summary["median_event_ms"], 10.0)
        self.assertFalse(summary["statistically_supported_speedup"])

    def test_report_labels_qualified_and_timing_evidence_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = {"suite": "science", "baseline": "cublaslt",
                      "measurement": {"independent_processes": 3},
                      "cases": [{"id": "science", "kind": "matrix_function", "n": 32}]}
            (directory / "resolved_config.json").write_text(json.dumps(config))
            rows = records()
            for row in rows:
                if row["backend"] == "megafmm":
                    row["application_quality_pass"] = False
            (directory / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            summaries = report.build_report(directory)
            candidate = next(s for s in summaries if s["backend"] == "megafmm")
            self.assertEqual(candidate["qualified_paired_process_count"], 0)
            document = (directory / "report.html").read_text()
            self.assertIn("No quality-qualified paired performance measurements", document)
            self.assertIn("Timing evidence before quality qualification", document)
            self.assertIn("Qualified speedup", document)
            markdown = (directory / "report.md").read_text()
            self.assertIn("Timing pairs", markdown)
            self.assertIn("Qualified pairs", markdown)
            self.assertIn("Quality failed / unchecked", markdown)
            json.loads((directory / "summary.json").read_text())

    def test_covariance_throughput_is_not_described_as_linpack_solve_work(self):
        for kind in ("covariance", "linpack"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config = {"suite": "throughput", "baseline": "cublaslt",
                          "measurement": {"independent_processes": 3},
                          "cases": [{"id": "science", "kind": kind, "n": 32}]}
                (directory / "resolved_config.json").write_text(json.dumps(config))
                rows = records()
                for row in rows:
                    row["kind"] = kind
                (directory / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                report.build_report(directory)
                for filename in ("report.md", "report.html"):
                    document = (directory / filename).read_text()
                    self.assertIn("Equivalent FP32 GFLOPS", document)
                    self.assertIn("each application's declared classical work convention", document)
                    self.assertIn("rather than SW executed instructions", document)
                    if kind == "covariance":
                        self.assertNotIn("LINPACK", document)
                        self.assertNotIn("classical solve FLOPs", document)
                    else:
                        self.assertIn("LINPACK-style FP32 solve rows", document)
                        self.assertIn("equivalent classical solve FLOPs", document)


if __name__ == "__main__":
    unittest.main()
