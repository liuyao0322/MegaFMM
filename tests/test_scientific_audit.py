"""CPU-only corruption regressions for scientific evidence integrity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_scientific import audit


class ScientificAuditMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.case = {"id": "covariance_small", "kind": "covariance"}
        self.backends = ["torch", "practical_torch"]
        self.measurement = {"seed": 7, "independent_processes": 1, "groups": 2,
                            "calls_per_group": 2, "fresh_quality_trials": 2, "rotate_inputs": 2}
        self.write_json("resolved_config.json", {"cases": [self.case], "modes": ["launch"],
            "backends": self.backends, "baseline": "torch", "measurement": self.measurement})
        self.write_json("manifest.json", {"selected_case_ids": [self.case["id"]]})
        self.worker_key = self.case["id"] + "__launch__p0"
        self.worker_path = self.directory / "workers" / self.worker_key
        seed = 7 + int(hashlib.sha256(self.case["id"].encode()).hexdigest()[:8], 16)
        self.write_json(self.worker_path / "payload.json", {"case": self.case, "mode": "launch",
            "process_index": 0, "backends": self.backends, "seed": seed})
        input_sets = [{"stored_inputs": [{"sha256": str(slot) * 64}], "seed": seed + 1000 + slot}
                      for slot in range(2)]
        input_hash = hashlib.sha256(json.dumps([s["stored_inputs"] for s in input_sets], sort_keys=True).encode()).hexdigest()
        self.worker = {"runs": [], "samples": [], "metrics": []}
        for backend, times in (("torch", [4.0, 6.0]), ("practical_torch", [2.0, 3.0])):
            common = {"case_id": self.case["id"], "mode": "launch", "process_index": 0, "backend": backend}
            row = {**common, "kind": "covariance", "worker_pid": 123, "seed": seed,
                "status": "ok", "implementation_pass": True, "group_count": 2,
                "steady_wall_ms": sum(times) / 2, "application_quality_pass": backend == "torch",
                "input_hash": input_hash, "input_sets": input_sets,
                "validation_seeds": [seed + 100000 + trial for trial in range(2)]}
            if backend != "torch":
                row.update(speedup=2.0, speedup_time_basis="steady_wall_ms")
            self.worker["runs"].append(row)
            self.worker["samples"].extend({**common, "group": group, "calls": 2, "wall_ms": value}
                                           for group, value in enumerate(times))
            for trial in range(2):
                rejected = backend != "torch" and trial == 0
                self.worker["metrics"].append({**common, "trial": trial, "input": {"seed": seed + 100000 + trial},
                    "values": {"quality_pass": not rejected, "relative_error": "+inf" if rejected else 1e-7,
                               "nonfinite_diagnostics": ["nan", "+inf", "-inf"] if rejected else []}})
            for phase in ("timed_path_before", "timed_path_after"):
                for slot in range(2):
                    self.worker["metrics"].append({**common, "phase": phase, "input_slot": slot,
                        "input": input_sets[slot], "values": {"quality_pass": True, "relative_error": 1e-7}})
        self.save_worker_and_aggregates()

    def write_json(self, name, value):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))

    def write_collection(self, name, rows):
        (self.directory / (name + ".jsonl")).write_text("".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))

    def save_worker_and_aggregates(self):
        self.write_json(self.worker_path / "result.json", self.worker)
        for name, rows in self.worker.items():
            self.write_collection(name, [{**row, "worker_key": self.worker_key} for row in rows])

    def aggregate_metrics(self):
        return [json.loads(line) for line in (self.directory / "metrics.jsonl").read_text().splitlines()]

    def codes(self, result):
        return {issue["code"] for issue in result["issues"]}

    def test_numerical_rejection_and_nonfinite_strings_keep_integrity(self):
        before = (self.directory / "metrics.jsonl").read_bytes()
        result = audit(self.directory)
        self.assertTrue(result["integrity_pass"], result["issues"])
        self.assertTrue(result["complete"])
        self.assertTrue(result["execution_pass"])
        self.assertFalse(result["quality_all_pass"])
        self.assertEqual(result["counts"]["observed_quality_observations"], 12)
        self.assertEqual(result["counts"]["expected_quality_observations"], 12)
        self.assertEqual((self.directory / "metrics.jsonl").read_bytes(), before)

    def test_deleted_aggregate_metric_is_missing(self):
        self.write_collection("metrics", self.aggregate_metrics()[1:])
        result = audit(self.directory)
        self.assertFalse(result["integrity_pass"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"]["missing_aggregate_metrics"], 1)
        self.assertIn("aggregate_metrics_mismatch", self.codes(result))

    def test_changed_nested_payload_or_worker_identity_is_rejected(self):
        original = self.aggregate_metrics()
        for field in ("values", "input", "worker_key", "backend"):
            with self.subTest(field=field):
                rows = deepcopy(original)
                if field == "values":
                    rows[0][field]["relative_error"] = "nan"
                elif field == "input":
                    rows[0][field]["seed"] += 1
                else:
                    rows[0][field] = "corrupted"
                self.write_collection("metrics", rows)
                result = audit(self.directory)
                self.assertIn("aggregate_metrics_mismatch", self.codes(result))
                self.assertEqual(result["counts"]["missing_aggregate_metrics"], 1)
                self.assertEqual(result["counts"]["unexpected_aggregate_metrics"], 1)

    def test_extra_and_duplicated_aggregate_metrics_are_rejected(self):
        original = self.aggregate_metrics()
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                extra = deepcopy(original[0])
                if not duplicate:
                    extra["values"]["extra_diagnostic"] = 1
                self.write_collection("metrics", [*original, extra])
                result = audit(self.directory)
                self.assertIn("aggregate_metrics_mismatch", self.codes(result))
                self.assertEqual(result["counts"]["unexpected_aggregate_metrics"], 1)
                self.assertEqual(result["counts"]["duplicate_aggregate_metrics"], int(duplicate))

    def test_missing_metrics_collection_is_incomplete(self):
        (self.directory / "metrics.jsonl").unlink()
        result = audit(self.directory)
        self.assertFalse(result["complete"])
        self.assertFalse(result["checks"]["aggregate_metrics"])
        self.assertEqual(result["counts"]["missing_aggregate_metrics"], 12)

    def test_missing_authoritative_observation_detected_even_when_mirrored(self):
        self.worker["metrics"].pop(0)
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertIn("missing_quality_observations", self.codes(result))
        self.assertEqual(result["counts"]["missing_quality_observations"], 1)
        self.assertFalse(result["complete"])

    def test_duplicated_authoritative_observation_detected_even_when_mirrored(self):
        self.worker["metrics"].append(deepcopy(self.worker["metrics"][0]))
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertIn("duplicate_quality_observation", self.codes(result))
        self.assertEqual(result["counts"]["duplicate_quality_observations"], 1)

    def test_unexpected_rotation_slot_and_boolean_trial_are_rejected(self):
        for replacement in ({"phase": "timed_path_after", "input_slot": 2}, {"trial": False}):
            with self.subTest(replacement=replacement):
                original = deepcopy(self.worker["metrics"][0])
                self.worker["metrics"][0] = {k: v for k, v in original.items() if k not in {"trial", "phase", "input_slot"}}
                self.worker["metrics"][0].update(replacement)
                self.save_worker_and_aggregates()
                result = audit(self.directory)
                self.assertIn("unexpected_quality_observation", self.codes(result))
                self.worker["metrics"][0] = original

    def test_null_or_unhashable_phase_is_rejected_without_crashing(self):
        for phase in (None, [], {}):
            with self.subTest(phase=phase):
                self.worker["metrics"][0]["phase"] = phase
                self.save_worker_and_aggregates()
                result = audit(self.directory)
                self.assertIn("unexpected_quality_observation", self.codes(result))

    def test_application_gate_cannot_hide_a_failed_observation(self):
        self.worker["runs"][1]["application_quality_pass"] = True
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertIn("application_quality_aggregation_mismatch", self.codes(result))
        self.assertEqual(result["counts"]["quality_aggregation_mismatches"], 1)
        self.assertFalse(result["integrity_pass"])
        self.assertFalse(result["quality_all_pass"])

    def test_quality_observation_requires_boolean_gate(self):
        self.worker["metrics"][0]["values"]["quality_pass"] = 1
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertIn("invalid_quality_observation", self.codes(result))

    def test_optional_local_gemm_diagnostics_do_not_change_application_gate(self):
        row = self.worker["metrics"][0]
        self.worker["metrics"].append({k: row[k] for k in ("case_id", "mode", "process_index", "backend")})
        self.worker["metrics"][-1].update(phase="local_gemm_diagnostics", values={"gemm_gate_pass": False,
                                                                                  "relative_error": "+inf"})
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertTrue(result["integrity_pass"], result["issues"])
        self.assertEqual(result["counts"]["observed_metrics"], 13)
        self.assertEqual(result["counts"]["observed_quality_observations"], 12)

    def test_failed_application_may_retain_partial_validation(self):
        self.worker["runs"][1].update(status="failed", implementation_pass=False)
        self.worker["metrics"] = [m for m in self.worker["metrics"]
                                  if m["backend"] == "torch" or m.get("trial") == 0]
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertTrue(result["integrity_pass"], result["issues"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["execution_pass"])
        self.assertEqual(result["counts"]["expected_quality_observations"], 6)

    def test_native_gemm_metrics_have_no_application_observation_schema(self):
        for row in self.worker["runs"]:
            row.update(kind="gemm", gemm_gate_pass=True, steady_event_ms=row["steady_wall_ms"])
            if "speedup" in row:
                row["speedup_time_basis"] = "steady_event_ms"
        self.worker["metrics"] = [{**{k: row[k] for k in ("case_id", "mode", "process_index", "backend")},
            "values": {"relative_frobenius": 1e-7, "max_absolute": 1e-6,
                       "gemm_gate_pass": True, "native_accepted": True}} for row in self.worker["runs"]]
        self.save_worker_and_aggregates()
        result = audit(self.directory)
        self.assertTrue(result["integrity_pass"], result["issues"])
        self.assertTrue(result["quality_all_pass"])
        self.assertEqual(result["counts"]["expected_quality_observations"], 0)
        self.assertEqual(result["counts"]["observed_metrics"], 2)


if __name__ == "__main__":
    unittest.main()
