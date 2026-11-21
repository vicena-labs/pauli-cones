"""Choose a solver automatically, and sweep circuits over chips, noise and size.

    from pauli_cones.experiments import solve, sweep
    result = solve(circuit, terms)            # value, method, exact or not, and why
    table = sweep("qaoa-ring", sizes=[8, 16, 32], devices=["heavy-hex-127", "all-to-all-ions:n=32"],
                  noise_scales=[0, 0.5, 1, 2])
    print(table.markdown())

solve() tries the fastest exact method first and degrades visibly: exact
sparse propagation; then exact local density matrices when every cone has at
most 6 wires; then truncation at 1e-4 and 1e-5, reporting the change between
the two as the convergence check alongside the rigorous bound. Every result
says which method produced it and whether it is exact.

sweep() compiles the circuit onto each chip at each noise scale (and each size,
for a circuit family or a function of n) and solves it, with the ideal
noise-free value beside every noisy one.
"""
from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field

from .api import ConeOverflow, Truncation


def solve(circuit, terms, shifts=None, *, tolerances=(1e-4, 1e-5)):
    """Evaluate observable terms on a Pauli Cones Circuit with the best available method.

    Returns a dict: value, method, exact, and for truncated answers the
    convergence change between the two tolerances and the rigorous error bound;
    `reasons` records each method tried and why it was set aside.
    """
    reasons = []
    try:
        result = circuit.prepare(terms).evaluate(shifts, gradients=False)
        return {"value": result.value, "method": "native", "exact": True, "reasons": reasons}
    except ConeOverflow as exc:
        reasons.append(f"native exact: {exc}")
    try:
        result = circuit.prepare(terms, backend="dense").evaluate(shifts, gradients=False)
        return {"value": result.value, "method": "dense", "exact": True, "reasons": reasons}
    except ConeOverflow as exc:
        reasons.append(f"dense exact: {exc}")
    values = []
    for tolerance in tolerances:
        try:
            result = circuit.prepare(terms, truncation=Truncation(tolerance)).evaluate(shifts, gradients=False)
            values.append((tolerance, result.value, result.error_bound))
        except ConeOverflow as exc:
            reasons.append(f"truncated at {tolerance:g}: {exc}")
            break
    if not values:
        raise ConeOverflow("no method fits: " + "; ".join(reasons))
    tolerance, value, bound = values[-1]
    out = {"value": value, "method": f"native truncated at {tolerance:g}", "exact": False,
           "error_bound": bound, "reasons": reasons}
    if len(values) > 1:
        out["convergence_change"] = abs(values[-1][1] - values[-2][1])
    else:
        out["convergence_change"] = None
        reasons.append("only one tolerance fitted the term cap; convergence is unchecked")
    return out


@dataclass
class SweepResult:
    """Rows of a sweep, with table, CSV and JSON views."""
    rows: list = field(default_factory=list)

    COLUMNS = ("size", "device", "noise_scale", "value", "ideal", "deviation", "method", "exact",
               "swaps", "two_qubit_gates", "depth", "duration_s", "seconds", "error")

    def to_json(self):
        return json.dumps(self.rows, indent=1)

    def csv(self):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.rows)
        return buffer.getvalue()

    def markdown(self):
        def cell(value):
            if isinstance(value, float):
                return f"{value:.6g}"
            return "" if value is None else str(value)
        lines = ["| " + " | ".join(self.COLUMNS) + " |", "|" + "---|" * len(self.COLUMNS)]
        lines += ["| " + " | ".join(cell(row.get(c)) for c in self.COLUMNS) + " |" for row in self.rows]
        return "\n".join(lines)


def sweep(circuit, observable=None, *, devices=None, noise_scales=(1.0,), sizes=None, values=None,
          router="greedy", layout="greedy", idle_noise=True, family_parameters=None):
    """Compile and solve `circuit` on each device at each noise scale (and size).

    circuit is OpenQASM text, a Qiskit circuit, a circuit-family name from
    pauli_cones.circuits (with `sizes`), or a function n -> circuit.
    observable is a string, or a function n -> string; a family supplies its
    own when it is None. devices are Device objects, catalogue names, or
    "name:k=v,k=v" strings; None evaluates only the ideal circuit.
    """
    from . import circuits, devices as catalogue
    from .compiler import compile_circuit
    from .interop import convert, parse_observable
    chips = [_device(d, catalogue) for d in (devices or [])]
    rows = []
    for n in (sizes or [None]):
        source, obs = _instance(circuit, observable, n, circuits, family_parameters or {})
        ideal_conv = convert(source)
        shifts = ideal_conv.shifts(values) if ideal_conv.parameters else None
        start = time.perf_counter()
        ideal = solve(ideal_conv.circuit, parse_observable(obs, ideal_conv.circuit.n_qubits), shifts)
        rows.append({"size": n, "device": "ideal", "noise_scale": 0.0, "value": ideal["value"],
                     "ideal": ideal["value"], "deviation": 0.0, "method": ideal["method"], "exact": ideal["exact"],
                     "seconds": time.perf_counter() - start})
        for chip in chips:
            for scale in noise_scales:
                row = {"size": n, "device": chip.name, "noise_scale": scale, "ideal": ideal["value"]}
                start = time.perf_counter()
                try:
                    compiled = compile_circuit(source, chip, router=router, layout=layout, noise=True,
                                               idle_noise=idle_noise, noise_scale=scale)
                    result = solve(compiled.circuit, compiled.observable(obs),
                                   compiled.shifts(values) if compiled.parameters else None)
                    row.update(value=result["value"], deviation=result["value"] - ideal["value"],
                               method=result["method"], exact=result["exact"],
                               **{k: compiled.report[k] for k in ("swaps", "two_qubit_gates", "depth", "duration_s")})
                    if not result["exact"]:
                        row.update(error_bound=result["error_bound"],
                                   convergence_change=result["convergence_change"])
                except (ConeOverflow, ValueError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
    return SweepResult(rows)


def _instance(circuit, observable, n, circuits, parameters):
    if isinstance(circuit, str) and circuit in circuits.FAMILIES:
        if n is None:
            raise ValueError("a circuit family needs sizes")
        source, default = circuits.family(circuit, n=n, **parameters)
        obs = observable(n) if callable(observable) else (observable or default)
        return source, obs
    if callable(circuit):
        source = circuit(n)
    else:
        source = circuit
    if observable is None:
        raise ValueError("give an observable")
    return source, observable(n) if callable(observable) else observable


def _device(spec, catalogue):
    if isinstance(spec, dict):
        return catalogue.Device.from_dict(spec)
    if not isinstance(spec, str):
        return spec
    name, _, rest = spec.partition(":")
    parameters = {}
    for pair in filter(None, rest.split(",")):
        key, _, value = pair.partition("=")
        for kind in (int, float):
            try:
                parameters[key] = kind(value)
                break
            except ValueError:
                continue
        else:
            parameters[key] = value
    return catalogue.get(name, **parameters)
