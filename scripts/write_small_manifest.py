#!/usr/bin/env python3
"""Validate a complete small campaign and write its independent JSON manifest.

This uses only CPU file reads and read-only git object queries. It never runs a
GPU command, compiler, benchmark, or test. The measured source commit comes from
small_final_runs.csv and small_verify_runs.csv, never from a later document HEAD.
All declared cases, both precisions, every shape, and the full verification must
be present. A partial archive or mixed binary/source identity is rejected.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from summarize_small import FLOORS, Record, WORKSPACE_LIMIT, read_records


VERIFY_LABELS = ("dag", "correctness", "memcheck", "racecheck", "synccheck", "strict_numerics")
PATTERN_NAMES = {"Uniform": "uniform", "Positive": "positive", "Range": "range",
                 "Cancellation": "cancellation", "HighHalf": "fp16_upper_range"}
ALGORITHM_NAMES = {"WinogradLt": "lt", "WinogradHopper": "hopper", "Megakernel": "mega"}
METADATA_FIELDS = {"label", "source_commit", "binary_sha256", "exit_code"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact(path: Path, root: Path, **extra) -> dict:
    value = {"path": path_label(path,root), "sha256": digest(path), "bytes": path.stat().st_size}
    value.update(extra)
    return value


def declared_cases(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^all_labels=\((.*?)^\)",text,re.M|re.S)
    require(match is not None, f"cannot read predeclared all_labels from {path}")
    labels = shlex.split(match.group(1))
    require(labels and len(labels) == len(set(labels)), "closure requires unique predeclared labels")
    cases = {}
    for line in text.splitlines():
        if not line.startswith("run_case "):
            continue
        args = shlex.split(line)
        label = args[1]
        require(label not in cases, f"duplicate run_case: {label}")
        tokens = args[2:]
        require(len(tokens) in (4,5) and tokens[0] == "--algorithm", f"unsupported command layout: {line}")
        algorithm = tokens[1]
        flags = {"lt": "--lt-variant", "hopper": "--hopper-variant", "mega": "--mega-variant"}
        require(algorithm in flags and tokens[2] == flags[algorithm], f"invalid variant flag: {label}")
        mode = "graph" if "--graph" in tokens else "launch"
        require(label.endswith("_"+mode), f"mode/label mismatch: {label}")
        require(tokens[4:] == (["--graph"] if mode == "graph" else []), f"extra case flags: {label}")
        cases[label] = {"algorithm": algorithm, "variant": int(tokens[3]), "mode": mode}
    require(set(cases) == set(labels), "all_labels and run_case declarations disagree")
    modes=defaultdict(set)
    for case in cases.values():
        modes[case["algorithm"],case["variant"]].add(case["mode"])
    require(all(value=={"launch","graph"} for value in modes.values()),
            "each declared algorithm/variant needs separate launch and Graph cases")
    return {label:cases[label] for label in labels}


def read_suite(path: Path) -> tuple[dict, list[tuple[int,int,int]]]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    shapes = [tuple(shape) for shape in suite["squares"]+suite["rectangles"]]
    require(len(shapes) == len(set(shapes)) == suite["shape_count"] == 9,
            "small suite must contain exactly nine unique shapes")
    require(all(len(shape)==3 and all(isinstance(n,int) and n>0 for n in shape) for shape in shapes),
            "invalid declared dimensions")
    require(suite["tuple_order"] == ["M","N","K"], "unexpected tuple order")
    require(suite["cli_shapes"] == ";".join(",".join(map(str,s)) for s in shapes), "CLI shapes differ from declaration")
    require(set(suite["dtypes"]) == set(FLOORS), "both original target precisions are required")
    require(suite["output_dtype"] == "fp32" and suite["alpha"] == 1 and suite["beta"] == 0,
            "output/scalar contract differs from the campaign")
    require((suite["full"]["warmup"],suite["full"]["trials"],suite["full"]["groups"],
             suite["full"]["correctness_trials"]) == (20,100,5,3), "full 20/100/5/3 protocol changed")
    gate = suite["numeric_gate"]
    require(gate["frozen"] and not gate["nonfinite_allowed"] and
            gate["bf16_floor"] == FLOORS["bf16"] and gate["fp16_floor"] == FLOORS["fp16"],
            "frozen numerical contract changed")
    return suite, shapes


def read_run_metadata(path: Path, labels: set[str], verification: bool = False) -> tuple[list[dict],str,str]:
    require(path.is_file() and path.stat().st_size > 0, f"missing run metadata: {path}")
    with path.open(newline="",encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(METADATA_FIELDS.issubset(reader.fieldnames or []), f"incomplete metadata columns: {path}")
        rows = list(reader)
    require(len(rows) == len(labels) and {r.get("label") for r in rows} == labels,
            f"partial/duplicate metadata: {path}; require exactly {len(labels)} labels")
    commits = {r["source_commit"] for r in rows}
    binaries = {r["binary_sha256"] for r in rows}
    require(len(commits) == 1, f"mixed measured source commits: {path}")
    require(len(binaries) == 1, f"mixed measured binaries: {path}")
    commit, binary = next(iter(commits)), next(iter(binaries))
    require(re.fullmatch(r"[0-9a-f]{40,64}",commit) is not None, f"invalid source commit in {path}")
    require(re.fullmatch(r"[0-9a-f]{64}",binary) is not None, f"invalid binary SHA256 in {path}")
    for row in rows:
        status = int(row["exit_code"])
        allowed = {0,2} if not verification or row["label"] == "strict_numerics" else {0}
        require(status in allowed, f"runtime/verification failure: {row['label']} exit={status}")
    return rows, commit, binary


def log_fields(text: str, prefix: str) -> list[dict[str,str]]:
    return [dict(re.findall(r"(\w+)=(\S+)",line)) for line in text.splitlines() if line.startswith(prefix+" ")]


def check_benchmark_log(path: Path, records: list[Record]) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"missing benchmark log: {path}")
    text = path.read_text(encoding="utf-8",errors="replace")
    pattern = re.compile(r"^CHECK (\d+),(\d+),(\d+) (bf16|fp16) trial=(\d+) "
                         r"dense_eF=(\S+) solution_eF=(\S+) gate=(\S+) pass=([01])",re.M)
    trials = {}
    for match in pattern.finditer(text):
        m,n,k,dtype,trial,_,_,_,passed = match.groups()
        key = (dtype,(int(m),int(n),int(k)),int(trial))
        require(key not in trials, f"duplicate fresh-input correctness trial: {path}: {key}")
        trials[key] = int(passed)
    expected = {(r.values["dtype"],r.shape,i) for r in records for i in range(r.integer("correct_trials"))}
    require(set(trials) == expected, f"partial correctness trial log: {path}; expected {len(expected)} CHECK lines")
    for r in records:
        passed = all(trials[r.values["dtype"],r.shape,i] for i in range(r.integer("correct_trials")))
        require(int(r.values["numerical_pass"]) == int(passed), f"CSV/trial conjunction disagrees: {path}: {r.shape}")
    return {"fresh_input_trials":len(trials),"pass_trials":sum(trials.values()),
            "failed_trials":len(trials)-sum(trials.values())}


def collect_dataset(path: Path, case: dict, status: int, shapes: list[tuple[int,int,int]]) -> tuple[list[Record],dict]:
    records, warnings = read_records([path],"candidate")
    require(not warnings, f"CSV inconsistencies in {path}: {'; '.join(warnings)}")
    expected = {(dtype,shape) for dtype in FLOORS for shape in shapes}
    observed = [(r.values["dtype"],r.shape) for r in records]
    require(len(records)==len(expected) and len(set(observed))==len(expected) and set(observed)==expected,
            f"partial/duplicate shape rows: {path}; require all {len(expected)} precision/shape rows")
    for r in records:
        require(r.values["requested"]==case["algorithm"] and r.integer("variant")==case["variant"]
                and r.values["mode"]==case["mode"], f"case identity mismatch: {path}:{r.line}")
        require((r.integer("trials"),r.integer("groups"),r.integer("correct_trials"))==(100,5,3),
                f"diagnostic or partial sampling configuration: {path}:{r.line}")
        require(r.values["quantile_unit"]=="group_mean" and r.integer("workspace_limit")==WORKSPACE_LIMIT,
                f"timing/workspace contract mismatch: {path}:{r.line}")
        require(r.values["cache"]=="warm_cache" and r.integer("rotate_inputs")==1,
                f"cache scenario mismatch: {path}:{r.line}")
        m,n,k=r.shape
        require(r.integer("input_working_set_bytes")==2*(m*k+k*n), f"input workspace contract mismatch: {path}:{r.line}")
        require(r.checked and r.frozen_contract, f"unchecked or unfrozen correctness: {path}:{r.line}")
        require(r.values["numerical_pass"]!="1" or r.numerical_pass,
                f"recorded numerical pass contradicts metrics: {path}:{r.line}")
        require(max(r.integer("dense_workspace"),r.integer("solution_workspace"))<=WORKSPACE_LIMIT,
                f"workspace exceeds equal limit: {path}:{r.line}")
    expected_status=0 if all(r.numerical_pass for r in records) else 2
    require(status==expected_status, f"benchmark exit/CSV gate status mismatch: {path}")
    counts=Counter((r.backend,"pass" if r.numerical_pass else "fail") for r in records)
    return records, {"rows":len(records), "backend_counts":dict(Counter(r.actual for r in records)),
                     "classification_counts":{f"{backend}:{verdict}":count for (backend,verdict),count in sorted(counts.items())},
                     "numerical_pass_rows":sum(r.numerical_pass for r in records),
                     "sw_numerical_pass_rows":sum(r.backend=="sw" and r.numerical_pass for r in records),
                     "dense_fallback_pass_rows":sum(r.backend=="dense_fallback" and r.numerical_pass for r in records),
                     "nonfinite_elements":sum(r.integer("nonfinite") for r in records)}


def source_test_profile(text: str) -> tuple[set[tuple[str,int]],list[str],int,set[tuple[str,int]]]:
    start=text.find("bool strict_numerics(")
    end=text.find("int main(",start)
    require(start>=0 and end>start,"cannot identify strict_numerics source contract")
    block=text[start:end]
    candidate_end=block.find("for(auto pat:")
    require(candidate_end>0,"cannot identify strict candidate list")
    candidates={(ALGORITHM_NAMES[a],int(v)) for a,v in re.findall(
        r"Algorithm::(WinogradLt|WinogradHopper|Megakernel)\s*,\s*(\d+)",block[:candidate_end])}
    loops=re.findall(r"for\s*\(\s*int\s+variant\s*=\s*0\s*;\s*variant\s*<=\s*(\d+)\s*;\s*\+\+variant\s*\)\s*"
                     r"candidates\.push_back\(\{megafmm::Algorithm::(WinogradHopper|Megakernel),\s*variant\}\)",block[:candidate_end])
    for maximum,algorithm in loops:
        candidates.update((ALGORITHM_NAMES[algorithm],v) for v in range(int(maximum)+1))
    pattern_match=re.search(r"for\(auto pat:\{([^}]+)\}\)",block)
    require(pattern_match is not None,"cannot identify strict patterns")
    enums=re.findall(r"Pattern::(\w+)",pattern_match.group(1))
    require(all(p in PATTERN_NAMES for p in enums),"unknown strict pattern; update manifest parser")
    patterns=[PATTERN_NAMES[p] for p in enums]
    trials_match=re.search(r"for\(int trial=0;trial<(\d+);\+\+trial\)",block)
    require(trials_match is not None,"cannot identify strict trial count")
    trials=int(trials_match.group(1))
    require(candidates and len(patterns)==5 and trials==3,"strict source contract is incomplete")
    smoke_start=text.find("if(candidates_only)",end)
    smoke_end=text.find("structural_suite<",smoke_start)
    require(smoke_start>=0 and smoke_end>smoke_start,"cannot identify all-candidate smoke source")
    smoke={(ALGORITHM_NAMES[a],int(v)) for a,v in re.findall(
        r"std::pair<megafmm::Algorithm,int>\{megafmm::Algorithm::(Megakernel|WinogradHopper|WinogradLt),\s*(\d+)\}",
        text[smoke_start:smoke_end])}
    require(smoke,"empty all-candidate sanitizer source list")
    return candidates,patterns,trials,smoke


def collect_strict(text: str, status: int, candidates: set[tuple[str,int]],patterns: list[str],trials: int) -> dict:
    gates=log_fields(text,"NUMERICAL_GATE")
    distributions=log_fields(text,"NUMERICAL_DISTRIBUTION")
    expected=set(itertools.product(FLOORS,candidates,patterns,range(trials)))
    observed={}
    for gate in gates:
        key=(gate["dtype"],(gate["algorithm"],int(gate["variant"])),gate["pattern"],int(gate["trial"]))
        require(key not in observed,f"duplicate strict trial: {key}")
        require(gate["pass"] in {"0","1"},f"invalid strict pass flag: {key}")
        floor=FLOORS[gate["dtype"]]
        require(math.isclose(float(gate["floor"]),floor,rel_tol=0,abs_tol=1e-12),f"strict floor changed: {key}")
        require(math.isclose(float(gate["limit"]),floor+3*float(gate["dense_ef"]),
                             rel_tol=1e-5,abs_tol=floor*1e-5),f"strict gate formula changed: {key}")
        nonfinite=int(gate["nonfinite"])
        require(nonfinite>=0,f"invalid strict nonfinite count: {key}")
        if gate["pass"]=="1":
            error=float(gate["solution_ef"])
            require(gate["finite"]=="1" and nonfinite==0 and math.isfinite(error)
                    and error<=float(gate["limit"])*(1+1e-5),f"strict pass contradicts metrics: {key}")
        observed[key]=gate
    require(set(observed)==expected,f"partial strict coverage: expected {len(expected)} unique trials, got {len(observed)}")
    expected_distributions={(dtype,candidate,pattern) for dtype,candidate,pattern,_ in expected}
    seen=set()
    for distribution in distributions:
        key=(distribution["dtype"],(distribution["algorithm"],int(distribution["variant"])),distribution["pattern"])
        require(key not in seen,f"duplicate strict distribution: {key}")
        seen.add(key)
        group=[observed[(*key,i)] for i in range(trials)]
        require(int(distribution["trials"])==trials and
                int(distribution["failures"])==sum(g["pass"]=="0" for g in group) and
                int(distribution["nonfinite"])==sum(int(g["nonfinite"]) for g in group),
                f"strict distribution counts disagree: {key}")
    require(seen==expected_distributions,"partial strict distribution summaries")
    failures=sum(g["pass"]=="0" for g in gates)
    require(status==(2 if failures else 0),"strict exit disagrees with parsed gate failures")
    by_pattern={}
    for pattern in patterns:
        group=[g for g in gates if g["pattern"]==pattern]
        by_pattern[pattern]={"trials":len(group),"passes":sum(g["pass"]=="1" for g in group),
                             "failures":sum(g["pass"]=="0" for g in group),
                             "nonfinite_elements":sum(int(g["nonfinite"]) for g in group),
                             "range_unknown_trials":sum(g.get("certified_range")=="0" for g in group),
                             "range_unknown_pass_trials":sum(g.get("certified_range")=="0" and g["pass"]=="1" for g in group)}
    return {"trials":len(gates),"passes":len(gates)-failures,"failures":failures,
            "nonfinite_elements":sum(int(g["nonfinite"]) for g in gates),
            "candidate_algorithm_variants":sorted([{"algorithm":a,"variant":v} for a,v in candidates],key=lambda x:(x["algorithm"],x["variant"])),
            "patterns":by_pattern,"distribution_summaries":len(distributions),
            "unknown_range_passes_are_not_sw_certification":True}


def collect_verification(results: Path, rows: list[dict],root: Path,test_text: str,cases: dict) -> tuple[list[dict],dict]:
    candidates,patterns,trials,smoke=source_test_profile(test_text)
    measured={(case["algorithm"],case["variant"]) for case in cases.values()}
    require(measured.issubset(candidates),f"strict source omits measured candidates: {sorted(measured-candidates)}")
    require({p for p in measured if p[0]!="lt"}.issubset(smoke),"sanitizer smoke source omits measured non-Lt candidates")
    metadata={row["label"]:row for row in rows}
    evidence=[];strict={}
    for label in VERIFY_LABELS:
        path=results/f"small_{label}.txt";exit_path=results/f"small_{label}.exit"
        require(path.is_file() and path.stat().st_size>0,f"missing verification log: {path}")
        require(exit_path.is_file() and exit_path.stat().st_size>0,f"missing verification exit: {exit_path}")
        status=int(metadata[label]["exit_code"])
        require(int(exit_path.read_text().strip())==status,f"exit file/metadata disagree: {label}")
        text=path.read_text(encoding="utf-8",errors="replace")
        item=artifact(path,root,label=label,exit_code=status,exit_artifact=artifact(exit_path,root))
        if label in {"memcheck","racecheck","synccheck"}:
            executable=re.search(r"^SANITIZER_EXECUTABLE=(.+)$",text,re.M)
            version=re.search(r"^Version (.+)$",text,re.M)
            item["sanitizer_executable"]=executable.group(1) if executable else "not recorded"
            item["sanitizer_version"]=version.group(1) if version else "not recorded"
        if label=="dag":
            require(re.search(r"^OK\s*$",text,re.M) is not None and "FAILED" not in text,"DAG log lacks successful completion")
        elif label in {"correctness","memcheck","strict_numerics"}:
            require("PASS structural DAG" in text and "FAIL " not in text,f"structural completion absent: {label}")
        if label in {"memcheck","synccheck"}:
            summaries=re.findall(r"ERROR SUMMARY:\s*(\d+) errors",text)
            require(summaries and all(int(x)==0 for x in summaries),f"sanitizer errors/incomplete summary: {label}")
            item["sanitizer_errors"]=0
        if label=="racecheck":
            summaries=re.findall(r"RACECHECK SUMMARY:\s*(\d+) hazards displayed \((\d+) errors, (\d+) warnings\)",text)
            require(summaries and all(all(int(x)==0 for x in summary) for summary in summaries),"racecheck errors/hazards/incomplete summary")
            item.update(hazards=0,errors=0,warnings=0)
        if label in {"racecheck","synccheck"}:
            require("PASS selected candidate kernel paths" in text,"candidate sanitizer completion absent")
            probes=log_fields(text,"BEGIN_CANDIDATE")
            observed={(p["dtype"],(p["algorithm"],int(p["variant"]))) for p in probes}
            expected=set(itertools.product(FLOORS,smoke))
            require(observed==expected,f"partial candidate sanitizer coverage: {label}")
            item.update(candidate_cases=len(probes),dtype_algorithm_variant_combinations=len(observed))
        if label=="strict_numerics":
            strict=collect_strict(text,status,candidates,patterns,trials)
        evidence.append(item)
    return evidence,strict


def git_bytes(root: Path,*args: str) -> bytes:
    result=subprocess.run(["git","-C",str(root),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    require(result.returncode==0,f"read-only git query failed: {' '.join(args)}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def collect_source_identity(root: Path,measured_commit: str) -> dict:
    commit=git_bytes(root,"rev-parse","--verify",measured_commit+"^{commit}").decode().strip()
    solution_tree=git_bytes(root,"rev-parse",commit+":solution").decode().strip()
    names=git_bytes(root,"ls-tree","-r","--name-only",commit,"--","solution","include","bench","tests","CMakeLists.txt").decode().splitlines()
    def relevant(name: str) -> bool:
        return name=="CMakeLists.txt" or name=="bench/small_shapes.json" or Path(name).suffix in {".cpp",".cu",".h",".cuh",".py"}
    names={name for name in names if relevant(name)}
    current={"CMakeLists.txt","bench/small_shapes.json"}
    for directory in ("solution","include","bench","tests"):
        current.update(path.relative_to(root).as_posix() for path in (root/directory).rglob("*")
                       if path.is_file() and relevant(path.relative_to(root).as_posix()))
    require(names==current,f"current source set differs from measured commit; added={sorted(current-names)}, removed={sorted(names-current)}")
    hashes={}
    for name in sorted(names):
        path=root/name
        committed=hashlib.sha256(git_bytes(root,"show",f"{commit}:{name}")).hexdigest()
        actual=digest(path)
        require(actual==committed,f"current kernel/benchmark/test source differs from measured commit: {name}")
        hashes[name]=actual
    return {"measurement_source_commit":commit,"source_commit_authority":"both measured run metadata CSVs",
            "solution_tree_git_object":solution_tree,"source_files_sha256":hashes,
            "current_source_matches_measured_commit":True,
            "later_document_head_is_not_used_as_measurement_source":True}


def collect_telemetry(paths: list[Path],root: Path) -> tuple[list[dict],dict]:
    artifacts=[];uuids=set();values=defaultdict(list);timestamps=[]
    for path in paths:
        require(path.is_file() and path.stat().st_size>0,f"missing telemetry: {path}")
        with path.open(newline="",encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        require(len(rows)==1,f"expected one allocated GPU telemetry row: {path}")
        row={key.strip():value.strip() for key,value in rows[0].items()}
        require(row.get("uuid","").startswith("GPU-"),f"missing telemetry GPU UUID: {path}")
        uuids.add(row["uuid"]);timestamps.append(row.get("timestamp",""))
        for key,value in row.items():
            if key.startswith(("clocks.","power.draw","temperature.gpu","utilization.gpu")):
                number=re.search(r"[-+]?\d+(?:\.\d+)?",value)
                if number:
                    values[key].append(float(number.group()))
        artifacts.append(artifact(path,root,gpu_uuid=row["uuid"],snapshot=row))
    require(len(uuids)==1,"small final measurements contain multiple GPU UUIDs")
    device={"gpu_uuid":next(iter(uuids)),"telemetry_snapshot_count":len(paths),
            "timestamp_range":[min(timestamps),max(timestamps)],
            "snapshot_ranges":{key:{"min":min(v),"max":max(v),"snapshots":len(v)} for key,v in sorted(values.items())},
            "sampling_scope":"before/after snapshots, not continuous kernel sampling",
            "clocks_locked":"not established by these snapshots"}
    return artifacts,device


def build_environment(root: Path,results: Path,device_info: Path|None,device: dict) -> dict:
    environment={"device":device,"optional_artifacts":[]}
    if device_info is not None:
        supplied=json.loads(device_info.read_text(encoding="utf-8"))
        uuid=supplied.get("gpu_uuid",supplied.get("uuid"))
        require(uuid is None or uuid==device["gpu_uuid"],"explicit device info conflicts with measured telemetry UUID")
        environment["explicit_device_info"]=supplied
        environment["optional_artifacts"].append(artifact(device_info,root))
    cache=root/"build/CMakeCache.txt"
    if cache.is_file():
        fields={}
        keys={"CMAKE_BUILD_TYPE","CMAKE_CUDA_COMPILER","CMAKE_CXX_COMPILER","CMAKE_CUDA_FLAGS",
              "CMAKE_CUDA_FLAGS_RELEASE","CMAKE_CXX_FLAGS_RELEASE","CUTLASS_ROOT"}
        for line in cache.read_text(encoding="utf-8",errors="replace").splitlines():
            match=re.match(r"([^:#]+):[^=]+=(.*)",line)
            if match and match.group(1) in keys:
                fields[match.group(1)]=match.group(2)
        environment["cmake_cache_fields"]=fields
        environment["optional_artifacts"].append(artifact(cache,root))
    for name in ("small_environment.txt","small_dependency.txt","environment.txt","dependency.txt"):
        path=results/name
        if path.is_file():
            environment["optional_artifacts"].append(artifact(path,root,
                provenance_scope="small campaign supplemental log" if name.startswith("small_") else
                "existing environment/dependency log; not fresh small verification"))
    return environment


def build_manifest(root: Path,results: Path,suite_path: Path,bench_binary: Path,test_binary: Path,
                   device_info: Path|None) -> dict:
    suite,shapes=read_suite(suite_path)
    cases=declared_cases(root/"scripts/small_suite.sh")
    bench_rows,source_commit,bench_sha=read_run_metadata(results/"small_final_runs.csv",set(cases))
    verify_rows,verify_commit,test_sha=read_run_metadata(results/"small_verify_runs.csv",set(VERIFY_LABELS),True)
    require(source_commit==verify_commit,"benchmark and verification used different measured source commits")
    require(digest(bench_binary)==bench_sha,"current benchmark binary differs from measured binary SHA256")
    require(digest(test_binary)==test_sha,"current correctness binary differs from verified binary SHA256")
    source=collect_source_identity(root,source_commit)
    metadata={row["label"]:row for row in bench_rows}
    datasets=[];records=[];telemetry_paths=[]
    for label,case in cases.items():
        path=results/f"small_final_{label}.csv"
        status=int(metadata[label]["exit_code"])
        samples,summary=collect_dataset(path,case,status,shapes)
        log=results/f"small_final_{label}.log"
        log_summary=check_benchmark_log(log,samples)
        entry=artifact(path,root,label=label,source_commit=source_commit,binary_sha256=bench_sha,
                       exit_code=status,case=case,**summary)
        entry["log"]=artifact(log,root,**log_summary)
        datasets.append(entry);records.extend(samples)
        telemetry_paths.extend(results/f"small_final_{label}_telemetry_{phase}.csv" for phase in ("before","after"))
    telemetry,device=collect_telemetry(telemetry_paths,root)
    test_text=(root/"tests/test_correctness.cu").read_text(encoding="utf-8")
    verification,strict=collect_verification(results,verify_rows,root,test_text,cases)
    artifacts=[]
    for path in (root/"scripts/small_suite.sh",root/"scripts/verify_small.sh",root/"scripts/summarize_small.py",
                 root/"scripts/write_small_manifest.py",results/"SMALL_REPORT.md"):
        if path.is_file():
            artifacts.append(artifact(path,root))
    artifacts.extend(artifact(path,root) for path in sorted((results/"small_plots").glob("*"))
                     if path.is_file() and path.suffix.lower() in {".svg",".png"})
    counts=Counter((r.backend,"pass" if r.numerical_pass else "fail") for r in records)
    aggregate={"total_rows":len(records),"datasets":len(datasets),"expected_rows_per_dataset":len(shapes)*2,
               "classification_counts":{f"{backend}:{verdict}":count for (backend,verdict),count in sorted(counts.items())},
               "sw_checked_rows":sum(r.backend=="sw" for r in records),
               "sw_numerical_pass_rows":sum(r.backend=="sw" and r.numerical_pass for r in records),
               "dense_fallback_pass_rows":sum(r.backend=="dense_fallback" and r.numerical_pass for r in records),
               "unchecked_rows":sum(not r.checked for r in records),
               "nonfinite_elements":sum(r.integer("nonfinite") for r in records),
               "csv_nonfinite_scope":"sum of dense and solution nonfinite elements across fresh correctness trials"}
    protocol={"warmup":20,"calls_per_group":100,"groups":5,"fresh_correctness_trials":3,
              "quantile_unit":"group_mean","workspace_limit_bytes":WORKSPACE_LIMIT,
              "reference":"FP64 ordinary GEMM of original rounded BF16/FP16 inputs","output_dtype":"fp32",
              "frozen_floors":FLOORS,"per_trial_gate":"eF_solution <= 3*eF_dense + floor; no nonfinite",
              "cache":"warm_cache","rotate_inputs":1,"shapes":[list(s) for s in shapes],
              "warmup_evidence":"declared suite/script; warmup is not stored per CSV row",
              "input_transforms":"recomputed in every full invocation, including Graph replay"}
    return {"schema_version":1,"campaign":"predeclared small-matrix optimization",
            "generated_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),**source,
            "binary_sha256":{path_label(bench_binary,root):bench_sha,path_label(test_binary,root):test_sha},
            "predeclared_suite":artifact(suite_path,root),"protocol":protocol,
            "acceptance":suite["acceptance"],"aggregate_evidence_counts":aggregate,
            "run_metadata":[artifact(results/"small_final_runs.csv",root),artifact(results/"small_verify_runs.csv",root)],
            "datasets":datasets,"verification":verification,"strict_numerical_checks":strict,
            "telemetry":telemetry,"environment":build_environment(root,results,device_info,device),
            "report_and_reproduction_artifacts":artifacts,
            "acceptance_status":"manifest validates evidence completeness and identity; it does not enable Auto",
            "uncertified_paths_remain_experimental":True,
            "historical_large_manifest_is_unchanged":True}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--results-dir",type=Path,default=Path("results"))
    parser.add_argument("--suite",type=Path,default=Path("bench/small_shapes.json"))
    parser.add_argument("--bench-binary",type=Path,default=Path("build/megafmm_bench"))
    parser.add_argument("--test-binary",type=Path,default=Path("build/test_correctness"))
    parser.add_argument("--device-info",type=Path,help="Optional existing JSON hardware/toolkit info; UUID checked against telemetry")
    parser.add_argument("--output",type=Path,default=Path("results/small_manifest.json"))
    args=parser.parse_args()
    root=args.root.resolve()
    resolve=lambda path:path.resolve() if path.is_absolute() else (root/path).resolve()
    output=resolve(args.output)
    try:
        require(output != resolve(Path("results/final_manifest.json")),"refusing to overwrite historical final_manifest.json")
        value=build_manifest(root,resolve(args.results_dir),resolve(args.suite),resolve(args.bench_binary),
                             resolve(args.test_binary),resolve(args.device_info) if args.device_info else None)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    except (ValueError,OSError,KeyError,TypeError) as error:
        parser.error(str(error))
    print(f"Wrote {output}: {len(value['datasets'])} complete datasets, "
          f"{value['aggregate_evidence_counts']['total_rows']} rows, measured source {value['measurement_source_commit']}")


if __name__=="__main__":
    main()
