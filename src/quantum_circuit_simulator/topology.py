"""Small, dependency-free topology helpers for research circuit construction.

An edge describes *allowed* two-wire connectivity. Topology is not a hardware
transpiler, a calibrated device model, or a promise of native physical gates.
Callers must use these helpers (or validate manually) to enforce connectivity;
Circuit.gate itself intentionally remains hardware-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import math
from typing import Iterable


@dataclass(frozen=True)
class Topology:
    """Undirected coupler graph; nodes are wires ``0..n_qubits-1``.

    Edge orientation is normalized because the included CZ helper is symmetric.
    Hardware directionality, routing, timing, crosstalk and calibrations are
    outside this helper's contract.
    """

    n_qubits: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        n = self.n_qubits
        if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= 16384:
            raise ValueError("n_qubits must be an integer from 1 to 16384")
        canonical: set[tuple[int, int]] = set()
        try:
            supplied = tuple(self.edges)
        except TypeError as exc:
            raise ValueError("edges must be an iterable of wire pairs") from exc
        for pair in supplied:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("every edge must contain two wire indices")
            u, v = pair
            if any(isinstance(x, bool) or not isinstance(x, Integral) or not 0 <= x < n for x in (u, v)):
                raise ValueError("edge endpoint is not a valid wire index")
            if u == v:
                raise ValueError("self-loop is not a coupler")
            edge = (int(min(u, v)), int(max(u, v)))
            if edge in canonical:
                raise ValueError("duplicate coupler")
            canonical.add(edge)
        object.__setattr__(self, "n_qubits", int(n))
        object.__setattr__(self, "edges", tuple(sorted(canonical)))

    @classmethod
    def line(cls, n_qubits: int) -> Topology:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
            raise ValueError("line size must be an integer")
        return cls(n_qubits, tuple((i, i + 1) for i in range(max(0, int(n_qubits) - 1))))

    @classmethod
    def grid(cls, rows: int, columns: int) -> Topology:
        if any(isinstance(v, bool) or not isinstance(v, Integral) or not 1 <= v <= 16384 for v in (rows, columns)):
            raise ValueError("rows and columns must be positive integers")
        if rows * columns > 16384:
            raise ValueError("grid exceeds the supported wire limit")
        edges = []
        for r in range(rows):
            for c in range(columns):
                a = r * columns + c
                if c + 1 < columns:
                    edges.append((a, a + 1))
                if r + 1 < rows:
                    edges.append((a, a + columns))
        return cls(rows * columns, tuple(edges))

    def has_edge(self, first: int, second: int) -> bool:
        return tuple(sorted((first, second))) in self.edges

    def require_edge(self, first: int, second: int) -> None:
        if not self.has_edge(first, second):
            raise ValueError(f"coupler ({first}, {second}) is absent from the topology")

    def matchings(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Greedy deterministic partition into disjoint-edge layers.

        These are not guaranteed to be the minimum number of layers, and they
        do not represent hardware timing or an optimal edge-coloring.
        """
        remaining = list(self.edges)
        result: list[tuple[tuple[int, int], ...]] = []
        while remaining:
            busy: set[int] = set()
            layer: list[tuple[int, int]] = []
            next_remaining: list[tuple[int, int]] = []
            for u, v in remaining:
                if u in busy or v in busy:
                    next_remaining.append((u, v))
                else:
                    busy.update((u, v))
                    layer.append((u, v))
            result.append(tuple(layer))
            remaining = next_remaining
        return tuple(result)

    def add_cz_layer(self, circuit, *, depolarizing_probability: float = 0.0):
        """Apply each allowed CZ once, using disjoint matching order.

        The noise parameter has the Quantum Circuit Simulator probability convention.
        Independent gate noise is a user-supplied *example*, not calibration.
        """
        from .api import Circuit
        if not isinstance(circuit, Circuit) or circuit.n_qubits != self.n_qubits:
            raise ValueError("circuit size must match topology")
        p = depolarizing_probability
        if isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(float(p)) or not 0 <= p <= 1:
            raise ValueError("depolarizing_probability must be in [0, 1]")
        for matching in self.matchings():
            for u, v in matching:
                circuit.gate("CZ", [u, v])
                if p:
                    circuit.depolarize([u, v], p)
        return circuit
