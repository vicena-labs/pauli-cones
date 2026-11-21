"""Run optional CPU comparisons in isolated, timeout-bounded subprocesses.

Example (from repository root):
    python -m benchmarks.compare --output benchmarks/results/local.json

Every backend gets the same circuit, output, parameter vectors and tolerance.
Exact per-gate parameter shift is labeled separately from analytic adjoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

try:
    import resource
except ImportError:  # Windows has no getrusage; do not relabel live RSS as peak.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
NAMES = ("native", "graph", "dense", "dense_adjoint", "qulacs", "aer", "pauli_prop")


def package_versions():
    versions = {}
    for name in ("pauli-cones", "numpy", "scipy", "qulacs", "qiskit", "qiskit-aer", "pauli-prop"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import pauli_cones
        versions["pauli_cones_module"] = pauli_cones.__version__
    except ImportError:
        versions["pauli_cones_module"] = None
    return versions


def cpu_details():
    details = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                details["model_name"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        details["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    return details


def source_hashes():
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "src").rglob("*.cpp"))
    paths += [ROOT / "benchmarks" / name for name in ("__init__.py", "models.py", "adapters.py", "compare.py")]
    paths += [ROOT / "pyproject.toml", ROOT / "setup.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def worker(case_name, backend, gradients, repeats, cold_only):
    # Imports are deliberately inside the subprocess, charged to first setup and
    # to the separate cold-process measurement. They are not warm evaluation work.
    setup_start = time.perf_counter()
    from .models import make_cases, timing_vectors
    from .adapters import prepare
    case = next(case for case in make_cases() if case.name == case_name)
    try:
        adapter = prepare(backend, case)
        setup = time.perf_counter() - setup_start
        start = time.perf_counter()
        first = adapter.evaluate(case.vectors[0], gradients)
        first_seconds = time.perf_counter() - start
        result = {"status": "ok", "first_setup_seconds_including_imports": setup,
                  "first_evaluation_seconds": first_seconds,
                  "derivative_method": adapter.derivative_method if gradients else "none",
                  "results": [first]}
        if not cold_only:
            outputs = [adapter.evaluate(row, gradients) for row in case.vectors]
            samples, timed_outputs = [], []
            for rows in timing_vectors(case, repeats):
                start = time.perf_counter()
                values = [adapter.evaluate(row, gradients) for row in rows]
                samples.append((time.perf_counter() - start) / len(rows))
                timed_outputs.extend(values)
            setup_samples = []
            for _ in range(2):
                start = time.perf_counter()
                replacement = prepare(backend, case)
                setup_samples.append(time.perf_counter() - start)
                del replacement
            result.update(results=outputs, warm_seconds_per_vector_samples=samples,
                          warm_results=timed_outputs,
                          warm_seconds_per_vector_median=statistics.median(samples),
                          prepare_seconds_samples=setup_samples,
                          prepare_seconds_median=statistics.median(setup_samples))
        diagnostics = getattr(adapter, "diagnostics", None)
        if diagnostics is not None:
            result["diagnostics"] = diagnostics() if callable(diagnostics) else diagnostics
        # Linux ru_maxrss is KiB; macOS reports bytes. This is total worker RSS,
        # including imports and preparation, not a live-state allocation claim.
        if resource is not None:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_process_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        else:
            result["peak_process_rss_bytes"] = None
            result["peak_process_rss_unavailable"] = "platform has no resource.getrusage"
        return result
    except (ImportError, NotImplementedError) as exc:
        return {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        status = "resource_limit" if type(exc).__name__ in ("ConeOverflow", "GraphLimitExceeded", "DenseLimitExceeded") else "error"
        return {"status": status, "reason": f"{type(exc).__name__}: {exc}"}


def launch(case, backend, gradients, args, cold=False):
    command = [sys.executable, "-m", "benchmarks.compare", "--worker", "--case", case.name,
               "--backend", backend, "--repeats", str(args.repeats)]
    if gradients:
        command.append("--gradients")
    if cold:
        command.append("--cold-only")
    start = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, env={**os.environ, **THREADS},
                                 text=True, capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "limit_seconds": args.timeout}
    elapsed = time.perf_counter() - start
    if process.returncode:
        return {"status": "process_error", "exit_code": process.returncode,
                "reason": process.stderr[-2000:]}
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"status": "invalid_worker_output", "reason": process.stdout[-2000:]}
    if cold:
        result["cold_process_seconds"] = elapsed
    if process.stderr.strip():
        result["warnings"] = process.stderr.strip().replace(str(ROOT), "<repository>").replace(sys.prefix, "<python-environment>")[-3000:]
    return result


def accuracy(record, expected, gradients, warm_expected=None):
    if record.get("status") != "ok":
        record["comparison_eligible"] = False
        return
    values, derivatives = [], []
    actual_results, target_results = list(record["results"]), list(expected)
    if warm_expected is not None:
        actual_results.extend(record.get("warm_results", []))
        target_results.extend(warm_expected)
    if len(actual_results) != len(target_results):
        record.update(status="accuracy_failed", reason="wrong result count", comparison_eligible=False)
        return
    for (actual, gradient), (target, target_gradient) in zip(actual_results, target_results):
        values.append(abs(actual - target))
        if gradients:
            if len(gradient) != len(target_gradient):
                record.update(status="accuracy_failed", reason="wrong gradient length", comparison_eligible=False)
                return
            derivatives.extend(abs(a - b) for a, b in zip(gradient, target_gradient))
    finite = all(math.isfinite(x) for x in values + derivatives)
    record["max_value_absolute_error"] = max(values, default=0.)
    record["max_gradient_absolute_error"] = max(derivatives, default=0.) if gradients else None
    passed = finite and max(values, default=0.) <= 1e-10 and max(derivatives, default=0.) <= 1e-9
    diagnostic = record.get("diagnostics") or {}
    truncated = diagnostic.get("max_discarded_coefficient_l1", 0.) != 0.
    record["accuracy_passed"] = passed
    record["comparison_eligible"] = passed and not truncated
    if not passed:
        record["status"] = "accuracy_failed"
    elif truncated:
        record["status"] = "truncated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/local.json")
    parser.add_argument("--backends", default=",".join(NAMES))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cold-repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=40.)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--backend", choices=NAMES)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--cold-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or not 0 <= args.cold_repeats <= 10 or not 1 <= args.timeout <= 600:
        parser.error("repeats must be 1..100, cold-repeats 0..10, timeout 1..600 seconds")
    if args.worker:
        print(json.dumps(worker(args.case, args.backend, args.gradients, args.repeats, args.cold_only), allow_nan=False))
        return
    from .models import make_cases, oracle, timing_vectors
    cases = make_cases()
    if args.cases != "all":
        selected = set(args.cases.split(","))
        if selected - {case.name for case in cases}:
            parser.error("unknown case name")
        cases = [case for case in cases if case.name in selected]
    backends = args.backends.split(",")
    if set(backends) - set(NAMES):
        parser.error("unknown backend")
    hashes = source_hashes()
    report = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "bounded comparison, not a universal speed or release qualification claim",
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version(), "logical_cpu_count": os.cpu_count(),
                    "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "thread_environment_for_workers": THREADS, "gpu_tested": False,
                    **cpu_details()},
        "versions": package_versions(), "source_sha256": hashes,
        "settings": {"repeats": args.repeats, "cold_repeats": args.cold_repeats,
                     "per_worker_timeout_seconds": args.timeout, "ordering_seed": 61119,
                     "warm_parameter_policy": "new deterministic shifts every timed round; all checked against oracle",
                     "value_absolute_tolerance": 1e-10, "gradient_absolute_tolerance": 1e-9},
        "not_executed": [
            {"backend": "PauliPropagation.jl", "reason": "Julia is not installed; no timing or rank inferred"},
            {"backend": "cuPauliProp", "reason": "no CUDA GPU is available; no timing or rank inferred"},
            {"backend": "MPDO/TN/trajectory", "reason": "no accuracy-matched adapter validated in this CPU suite"}],
        "cases": [asdict(case) for case in cases], "records": []}
    expected = {case.name: oracle(case) for case in cases}
    schedules = {case.name: timing_vectors(case, args.repeats) for case in cases}
    warm_expected = {case.name: oracle(case, [row for rows in schedules[case.name] for row in rows]) for case in cases}
    report["timed_parameter_vectors_by_round"] = schedules
    report["oracle"] = {name: {"method": "analytic product formula" if name.startswith("contractible")
                             else "independent dense Kraus channel adjoint", "results": values,
                             "timed_results": warm_expected[name]}
                        for name, values in expected.items()}
    tasks = [(case, backend, gradients) for case in cases for backend in backends for gradients in (False, True)]
    random.Random(61119).shuffle(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case, backend, gradients in tasks:
        record = {"case": case.name, "backend": backend, "output": "value_gradient" if gradients else "value"}
        record.update(launch(case, backend, gradients, args))
        accuracy(record, expected[case.name], gradients, warm_expected[case.name])
        if record.get("status") in ("ok", "truncated", "accuracy_failed"):
            cold = [launch(case, backend, gradients, args, cold=True) for _ in range(args.cold_repeats)]
            for sample in cold:
                accuracy(sample, expected[case.name][:1], gradients)
            record["cold_process_samples"] = cold
            times = [sample["cold_process_seconds"] for sample in cold if sample.get("status") == "ok"]
            if times:
                record["cold_process_seconds_median"] = statistics.median(times)
        report["records"].append(record)
        print(f"{case.name:28s} {backend:14s} {record['output']:14s} {record['status']}", flush=True)
        # Save each complete result so a long optional comparison can be resumed/inspected.
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["source_unchanged_during_run"] = hashes == source_hashes()
    report["all_requested_comparisons_eligible"] = all(record.get("comparison_eligible", False) for record in report["records"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
