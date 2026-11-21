"""Adapt a three-qubit variational energy calculation using exact gradients.

The illustrative Hamiltonian is not a measured molecule. This is an example
of building a new algorithm on top of the expectation/gradient API, rather
than a production VQE chemistry workflow.
"""
from __future__ import annotations

import argparse
import json

from pauli_cones import Circuit


def make_energy_problem(*, noise: float = .005):
    circuit = Circuit(3, n_parameters=3)
    for q in range(3):
        circuit.gate("RY", [q], theta=.1 * (q + 1), param=q)
    circuit.gate("CZ", [0, 1]).gate("CZ", [1, 2])
    if noise:
        circuit.depolarize([0, 1], noise).depolarize([1, 2], noise)
    hamiltonian = [(0.7, {0: "Z", 1: "Z"}), (-0.4, {1: "X"}), (0.2, {2: "Z"})]
    return circuit, hamiltonian


def descend(*, backend: str = "native", steps: int = 6, rate: float = .15):
    if not 0 <= steps <= 100:
        raise ValueError("steps must be in [0,100]")
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0,1]")
    circuit, h = make_energy_problem()
    prepared = circuit.prepare(h, backend=backend)
    shifts = [0.1, -0.15, 0.20]
    history = []
    for step in range(steps + 1):
        result = prepared.evaluate(shifts)
        history.append({"step": step, "energy": result.value,
                        "parameters": list(shifts), "gradient": list(result.gradient),
                        "backend": result.backend})
        if step < steps:
            shifts = [value - rate * gradient
                      for value, gradient in zip(shifts, result.gradient)]
    return {"hamiltonian": ["0.7 Z0 Z1", "-0.4 X1", "0.2 Z2"],
            "toy_variational_problem": True, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--rate", type=float, default=.15)
    parser.add_argument("--backend", choices=["native", "python", "dense", "graph", "auto"], default="native")
    args = parser.parse_args()
    print(json.dumps(descend(backend=args.backend, steps=args.steps, rate=args.rate), indent=2))


if __name__ == "__main__":
    main()
