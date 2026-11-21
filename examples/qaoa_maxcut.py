"""One-layer MaxCut QAOA with exact analytic gradients on a small graph.

All gates are lowered to supported H, CZ, RX, RZ primitives. This is an
educational small-graph implementation, not a production QAOA optimizer or a
claim of advantage over specialist QAOA software.
"""
from __future__ import annotations

import argparse
import json
import math

from pauli_cones import Circuit, Topology


def cnot(circuit: Circuit, control: int, target: int, topology: Topology) -> None:
    """CNOT = H(target) CZ(control,target) H(target), on an allowed edge."""
    topology.require_edge(control, target)
    circuit.gate("H", [target]).gate("CZ", [control, target]).gate("H", [target])


def build_qaoa(topology: Topology, *, noise: float = 0.0) -> tuple[Circuit, list]:
    """Return cost expectation and shared parameters (-gamma, beta).

    The cost Hamiltonian is sum_{(u,v)} (I-Z_u Z_v)/2. Preparing all |+>
    wires and applying CNOT RZ(-gamma) CNOT gives the standard cost layer
    up to a global phase. The library shifts RZ by its *first* parameter, so
    requests pass -gamma. Two RX(beta) gates per wire implement RX(2*beta).
    """
    if not topology.edges:
        raise ValueError("QAOA demonstration requires at least one edge")
    circuit = Circuit(topology.n_qubits, n_parameters=2)
    for q in range(topology.n_qubits):
        circuit.gate("H", [q])
    for control, target in topology.edges:
        cnot(circuit, control, target, topology)
        circuit.gate("RZ", [target], param=0)
        cnot(circuit, control, target, topology)
        if noise:
            circuit.depolarize([control, target], noise)
    for q in range(topology.n_qubits):
        circuit.gate("RX", [q], param=1).gate("RX", [q], param=1)
    obs = [(len(topology.edges)/2, {})] + [(-.5, {u: "Z", v: "Z"}) for u, v in topology.edges]
    return circuit, obs


def evaluate_qaoa(topology: Topology, *, gamma: float, beta: float, noise: float = 0.0,
                  backend: str = "native") -> dict:
    circuit, obs = build_qaoa(topology, noise=noise)
    plan = circuit.prepare(obs, backend=backend)
    result = plan.evaluate([-gamma, beta])
    return {
        "qubits": topology.n_qubits, "edges": [list(e) for e in topology.edges],
        "gamma": gamma, "beta": beta, "noise": noise,
        "expected_cut": result.value,
        "gradient_gamma": -result.gradient[0],
        "gradient_beta": result.gradient[1],
        "backend": result.backend,
        "convention": "MaxCut cost = sum((I - Zi Zj)/2) over supplied graph",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4, help="Line graph size")
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--backend", choices=["native", "python", "graph", "dense", "auto"], default="native")
    args = parser.parse_args()
    print(json.dumps(evaluate_qaoa(Topology.line(args.qubits), gamma=args.gamma, beta=args.beta,
                                   noise=args.noise, backend=args.backend), indent=2))


if __name__ == "__main__":
    main()
