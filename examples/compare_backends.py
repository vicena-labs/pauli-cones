"""Compare supported Quantum Circuit Simulator representations on precisely one circuit.

This is a reproducible *within-package* diagnostic, not a comparison to an
external specialist. Preparation and evaluation are timed separately.
"""
from __future__ import annotations

import json
import math
import time

from quantum_circuit_simulator import Topology
from architecture_playground import build_probe


def compare():
    circuit, obs = build_probe(Topology.line(2), layers=1, noise=.005)
    vectors = [[.01], [-.04], [.07]]
    records = []
    reference = None
    for method in ("native", "python", "graph", "dense", "auto"):
        start = time.perf_counter()
        plan = circuit.prepare(obs, backend=method)
        prepare_ms = 1e3 * (time.perf_counter() - start)
        start = time.perf_counter()
        out = plan.evaluate_many(vectors)
        eval_ms = 1e3 * (time.perf_counter() - start)
        pairs = [(r.value, list(r.gradient)) for r in out]
        if reference is None:
            reference = pairs
        if any(abs(a[0]-b[0]) > 1e-9 or any(abs(x-y) > 1e-8 for x,y in zip(a[1], b[1]))
               for a,b in zip(reference,pairs)):
            raise AssertionError(f"{method} disagrees with reference")
        records.append({"requested_backend": method, "actual_backend": plan.backend,
                        "prepare_ms": prepare_ms, "evaluate_three_ms": eval_ms,
                        "diagnostics": plan.diagnostics})
    return {"reference_value_first_vector": reference[0][0],
            "reference_gradient_first_vector": reference[0][1],
            "same_circuit_and_parameter_vectors": True,
            "times_are_local_descriptive_not_sota_comparison": True,
            "records": records}


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
