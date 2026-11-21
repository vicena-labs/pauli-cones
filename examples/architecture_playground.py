"""Explore a new undirected chip coupling graph with the Quantum Circuit Simulator engine.

Run from the source root after installing the package:
    python examples/architecture_playground.py --topology grid --rows 2 --cols 2
    python examples/architecture_playground.py --topology custom --file examples/custom_topology.json

A topology only restricts the entanglers *built by this example*: it is not a
physical chip simulation, a transpiler, or a calibrated noise model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from quantum_circuit_simulator import Circuit, Topology


def load_topology(kind: str, *, qubits: int = 4, rows: int = 2, cols: int = 2, path: str | None = None) -> Topology:
    if kind == "line":
        return Topology.line(qubits)
    if kind == "grid":
        return Topology.grid(rows, cols)
    if kind == "custom":
        if path is None:
            raise ValueError("custom topology requires --file")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"n_qubits", "couplers"}:
            raise ValueError("custom topology requires only n_qubits and couplers")
        return Topology(data["n_qubits"], data["couplers"])
    raise ValueError("topology must be line, grid or custom")


def build_probe(topology: Topology, *, layers: int = 2, noise: float = 0.01) -> tuple[Circuit, list]:
    if not 1 <= layers <= 16:
        raise ValueError("layers must be in [1,16] because they share one parameter per layer")
    circuit = Circuit(topology.n_qubits, n_parameters=layers)
    for wire in range(topology.n_qubits):
        circuit.gate("H", [wire])
    for depth in range(layers):
        for wire in range(topology.n_qubits):
            circuit.gate("RY", [wire], theta=0.10 + .07 * (depth + wire % 3), param=depth)
        topology.add_cz_layer(circuit, depolarizing_probability=noise)
    observable = [(1.0/topology.n_qubits, {wire: "Z"}) for wire in range(topology.n_qubits)]
    return circuit, observable


def run_probe(topology: Topology, *, layers: int = 2, noise: float = .01, backend: str = "native") -> dict:
    circuit, obs = build_probe(topology, layers=layers, noise=noise)
    t0 = time.perf_counter()
    plan = circuit.prepare(obs, backend=backend)
    prepared_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    shifts = [0.05] * layers
    result = plan.evaluate(shifts)
    evaluated_ms = (time.perf_counter() - t1) * 1000
    return {
        "qubits": topology.n_qubits, "couplers": [list(e) for e in topology.edges],
        "matching_layers": len(topology.matchings()), "circuit_layers": layers,
        "backend": result.backend, "value": result.value,
        "gradient": list(result.gradient), "prepare_ms": prepared_ms,
        "evaluate_ms": evaluated_ms, "diagnostics": plan.diagnostics,
        "not_a_hardware_calibration": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["line", "grid", "custom"], default="grid")
    parser.add_argument("--qubits", type=int, default=4, help="Line size")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--file", help="JSON with n_qubits and couplers")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--noise", type=float, default=.01)
    parser.add_argument("--backend", choices=["native", "python", "graph", "dense", "auto"], default="native")
    args = parser.parse_args()
    topology = load_topology(args.topology, qubits=args.qubits, rows=args.rows, cols=args.cols, path=args.file)
    print(json.dumps(run_probe(topology, layers=args.layers, noise=args.noise, backend=args.backend), indent=2))


if __name__ == "__main__":
    main()
