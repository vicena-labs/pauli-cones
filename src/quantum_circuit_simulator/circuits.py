"""Standard circuit families as OpenQASM 2, for scaling and noise studies.

Each family returns (qasm, observable): dependency-free OpenQASM 2 text and a
natural observable string for it, so sweeps over size need no other tools.

    from quantum_circuit_simulator.circuits import FAMILIES, family
    qasm, observable = family("qaoa-ring", n=12, p=2)
"""
from __future__ import annotations


import numpy as np

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\n'


def ghz(n=8):
    """GHZ state by a CNOT chain; <Z0 Z(n-1)> is 1 without noise."""
    body = "h q[0];\n" + "".join(f"cx q[{i}],q[{i + 1}];\n" for i in range(n - 1))
    return HEADER.format(n=n) + body, f"Z0*Z{n - 1}"


def hardware_efficient(n=8, layers=2, seed=0):
    """RY-RZ layers with a CZ brickwork; the observable is the mean Z."""
    rng = np.random.default_rng(seed)
    body = []
    for layer in range(layers):
        for q in range(n):
            body.append(f"ry({rng.uniform(-1, 1):.12g}) q[{q}];\nrz({rng.uniform(-1, 1):.12g}) q[{q}];\n")
        body += [f"cz q[{q}],q[{q + 1}];\n" for q in range(layer % 2, n - 1, 2)]
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.12g}*Z{q}" for q in range(n))


def qaoa_ring(n=8, p=1, gamma=0.4, beta=0.3):
    """QAOA for MaxCut on a ring; the observable is the mean ZZ over ring edges
    (the expected cut fraction is (1 - <mean ZZ>) / 2)."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    body = ["h q;\n"]
    for _ in range(p):
        body += [f"rzz({2 * gamma:.12g}) q[{a}],q[{b}];\n" for a, b in edges]
        body.append(f"rx({2 * beta:.12g}) q;\n")
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.12g}*Z{a}*Z{b}" for a, b in edges)


def ising_trotter(n=8, steps=4, dt=0.1, coupling=1.0, field=1.0):
    """First-order Trotter evolution of the transverse-field Ising chain from |0...0>;
    the observable is the mean Z magnetisation."""
    body = []
    for _ in range(steps):
        body += [f"rzz({-2 * coupling * dt:.12g}) q[{i}],q[{i + 1}];\n" for i in range(n - 1)]
        body.append(f"rx({-2 * field * dt:.12g}) q;\n")
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.12g}*Z{q}" for q in range(n))


FAMILIES = {
    "ghz": (ghz, "GHZ state by a CNOT chain; observable Z0*Z(n-1), ideally 1"),
    "hardware-efficient": (hardware_efficient, "RY-RZ layers with CZ brickwork; observable mean Z"),
    "qaoa-ring": (qaoa_ring, "QAOA MaxCut on a ring; observable mean ZZ over edges"),
    "ising-trotter": (ising_trotter, "Trotterized transverse-field Ising chain; observable mean Z"),
}


def family(name, **parameters):
    """(qasm, observable) for a named family; parameters as in its function."""
    if name not in FAMILIES:
        raise KeyError(f"unknown circuit family {name!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[name][0](**parameters)
