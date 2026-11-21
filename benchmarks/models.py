"""Small explicit benchmark models, independent of Pauli Cones internals.

Wire zero is the least significant bit. Angles are base angles plus shared
parameter shifts. Readout is an affine transformation of the measured Z only.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


@dataclass
class Case:
    name: str
    n_qubits: int
    n_parameters: int
    operations: list[dict]
    observables: list[tuple[float, dict[int, str]]]
    readout: dict[int, tuple[float, float]]
    vectors: list[list[float]]
    max_terms: int = 50000
    description: str = ""

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["observables"] = [(c, {int(q): a for q, a in ops.items()})
                               for c, ops in data["observables"]]
        data["readout"] = {int(q): tuple(p) for q, p in data["readout"].items()}
        return cls(**data)


def measurement_terms(case):
    """Expand independent classical asymmetric readout exactly on each Z."""
    merged = {}
    for weight, operators in case.observables:
        terms = [(weight, {})]
        for q, axis in sorted(operators.items()):
            if q in case.readout and axis not in ("I", "Z"):
                raise ValueError("readout is defined only for Z")
            choices = [(1.0, axis)]
            if axis == "Z" and q in case.readout:
                p01, p10 = case.readout[q]
                choices = [(p10 - p01, "I"), (1 - p01 - p10, "Z")]
            terms = [(coef * scale, {**mapping, **({q: out} if out != "I" else {})})
                     for coef, mapping in terms for scale, out in choices if scale]
        for coef, mapping in terms:
            key = tuple(sorted(mapping.items()))
            merged.setdefault(key, []).append(coef)
    return [(math.fsum(weights), dict(key)) for key, weights in merged.items()
            if math.fsum(weights)]


def local_unitary(operation, shifts):
    name = operation["kind"]
    if name in ("RX", "RY", "RZ"):
        theta = operation["theta"]
        if operation.get("param", -1) >= 0:
            theta += shifts[operation["param"]]
        p = PAULIS[name[-1]]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return c * I - 1j * s * p, -.5 * s * I - .5j * c * p
    if name == "H":
        return (X + Z) / math.sqrt(2), None
    if name == "SX":
        return ((1 + 1j) * I + (1 - 1j) * X) / 2, None
    if name in ("X", "Y", "Z"):
        return PAULIS[name], None
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(complex), None
    raise ValueError(f"unsupported benchmark gate {name}")


def local_kraus(operation):
    name = operation["kind"]
    if name == "depolarize":
        n = len(operation["wires"])
        p = operation["probability"]
        matrices = []
        for axes in product("IXYZ", repeat=n):
            m = np.ones((1, 1), complex)
            for axis in reversed(axes):
                m = np.kron(m, PAULIS[axis])
            probability = 1 - p if all(a == "I" for a in axes) else p / (4**n - 1)
            matrices.append(math.sqrt(probability) * m)
        return matrices
    if name == "pauli_error":
        p = operation["probability"]
        return [math.sqrt(1 - p) * I, math.sqrt(p) * PAULIS[operation["axis"]]]
    if name == "relax":
        t, t1, t2 = (operation[x] for x in ("duration", "t1", "t2"))
        if t2 > 2 * t1:
            raise ValueError("unphysical T2")
        decay = math.exp(-t / t1)
        coherence = math.exp(-t / t2 + t / (2 * t1))
        amplitude = [np.diag([1, math.sqrt(decay)]),
                     np.array([[0, math.sqrt(1 - decay)], [0, 0]], complex)]
        phase = [math.sqrt((1 + coherence) / 2) * I,
                 math.sqrt((1 - coherence) / 2) * Z]
        return [p @ a for p in phase for a in amplitude]
    raise ValueError(f"unsupported benchmark channel {name}")


def embed(local, wires, n):
    """Explicit basis embedding, intentionally independent of PTM machinery."""
    dim = 1 << n
    matrix = np.zeros((dim, dim), complex)
    clear = sum(1 << q for q in wires)
    for column in range(dim):
        local_column = sum(((column >> q) & 1) << j for j, q in enumerate(wires))
        base = column & ~clear
        for local_row in range(1 << len(wires)):
            row = base | sum(((local_row >> j) & 1) << q for j, q in enumerate(wires))
            matrix[row, column] = local[local_row, local_column]
    return matrix


class DenseAdjoint:
    """Independent exact dense Kraus evolution and reverse differentiation.

    This is a transparent correctness/algorithm comparator, not an optimized
    third-party simulator. No finite differences and no inverse noisy channels.
    Its full density matrices deliberately restrict this oracle to <=8 wires.
    """
    derivative_method = "analytic dense channel adjoint"

    def __init__(self, case):
        if case.n_qubits > 8:
            raise NotImplementedError("dense oracle limited to eight qubits")
        self.case = case
        n = case.n_qubits
        self.observable = np.zeros((1 << n, 1 << n), complex)
        for coef, mapping in measurement_terms(case):
            term = np.ones((1, 1), complex)
            for q in reversed(range(n)):
                term = np.kron(term, PAULIS[mapping.get(q, "I")])
            self.observable += coef * term
        self.fixed = []
        for op in case.operations:
            if op["kind"] in ("depolarize", "pauli_error", "relax"):
                self.fixed.append([embed(k, op["wires"], n) for k in local_kraus(op)])
            elif op.get("param", -1) < 0:
                unitary, _ = local_unitary(op, [0.0] * case.n_parameters)
                self.fixed.append([embed(unitary, op["wires"], n)])
            else:
                # Embed I and P once; rebinding is a pair of scaled additions.
                self.fixed.append((embed(I, op["wires"], n),
                                   embed(PAULIS[op["kind"][-1]], op["wires"], n)))

    def evaluate(self, shifts, gradients=False):
        n = self.case.n_qubits
        rho = np.zeros((1 << n, 1 << n), complex)
        rho[0, 0] = 1
        states, channels = [], []
        for op, fixed in zip(self.case.operations, self.fixed):
            derivative = None
            if isinstance(fixed, tuple):
                theta = op["theta"] + shifts[op["param"]]
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                ident, p = fixed
                kraus = [c * ident - 1j * s * p]
                if gradients:
                    derivative = -.5 * s * ident - .5j * c * p
            else:
                kraus = fixed
            if gradients:
                states.append(rho)
                channels.append((kraus, derivative, op.get("param", -1)))
            rho = sum((k @ rho @ k.conj().T for k in kraus), np.zeros_like(rho))
        value = float(np.trace(self.observable @ rho).real)
        gradient = np.zeros(self.case.n_parameters) if gradients else np.empty(0)
        adjoint = self.observable
        if gradients:
            for before, (kraus, derivative, parameter) in reversed(list(zip(states, channels))):
                if derivative is not None:
                    tangent = derivative @ before @ kraus[0].conj().T
                    tangent += kraus[0] @ before @ derivative.conj().T
                    gradient[parameter] += float(np.trace(adjoint @ tangent).real)
                adjoint = sum((k.conj().T @ adjoint @ k for k in kraus), np.zeros_like(adjoint))
        return value, gradient.tolist()


def make_cases():
    """Deterministic cases contain nonzero shared gradients and two noise classes."""
    cases = []
    for n, depth, realistic in ((2, 2, False), (4, 3, False), (4, 3, True), (6, 2, True)):
        rng = np.random.default_rng(12000 + n * 100 + depth * 10 + realistic)
        operations = []
        for layer in range(depth):
            for q in range(n):
                for name in ("RY", "RZ"):
                    operations.append({"kind": name, "wires": [q],
                                       "theta": float(rng.uniform(-.8, .8)),
                                       "param": (q + layer + (name == "RZ")) % 3})
                operations.append({"kind": "depolarize", "wires": [q], "probability": .013})
            for q in range(layer % 2, n - 1, 2):
                operations.extend([{"kind": "CZ", "wires": [q, q + 1]},
                                   {"kind": "depolarize", "wires": [q, q + 1], "probability": .021}])
            if realistic:
                for q in range(n):
                    operations.append({"kind": "relax", "wires": [q], "duration": 2e-7,
                                       "t1": 65e-6 + q * 1e-6, "t2": 80e-6 + q * 2e-6})
        readout = {q: (.021 + .001 * q, .012) for q in range(n)} if realistic else {}
        observables = [(1 / n, {q: "Z"}) for q in range(n)] + [(.17, {0: "Z", n - 1: "Z"})]
        vectors = [[0., 0., 0.], [.11, -.07, .13], [-.23, .19, -.17]]
        cases.append(Case(f"{'relaxation' if realistic else 'pauli_noise'}_{n}q_d{depth}",
                          n, 3, operations, observables, readout, vectors,
                          description="Seeded nearest-neighbor rotations/CZ; shared parameters."))
    # A moderately dense cone with more shared parameters tests when a reusable
    # graph amortizes its larger preparation cost. Not only graph-friendly cases
    # are present: the preceding small cones often favor direct execution.
    n, depth, n_parameters = 4, 5, 8
    rng = np.random.default_rng(61119)
    operations = []
    for layer in range(depth):
        for q in range(n):
            for offset, kind in enumerate(("RX", "RY")):
                operations.append({"kind": kind, "wires": [q],
                                   "theta": float(rng.uniform(-.6, .6)),
                                   "param": (q * 2 + offset + layer) % n_parameters})
        for q in range(n - 1):
            operations += [{"kind": "CZ", "wires": [q, q + 1]},
                           {"kind": "depolarize", "wires": [q, q + 1], "probability": .01}]
        for q in range(n):
            operations.append({"kind": "relax", "wires": [q], "duration": .03, "t1": 1., "t2": 1.4})
    vectors = [[0.] * n_parameters, [.07 * math.sin(q) for q in range(n_parameters)],
               [-.11 * math.cos(q) for q in range(n_parameters)]]
    cases.append(Case("repeated_gradient_4q_d5", n, n_parameters, operations,
                      [(1., {q: "Z" for q in range(n)})], {}, vectors,
                      description="Eight shared parameters; a denser cone tests graph preparation amortization."))
    # Exact solution remains easy while uncontracted Pauli support grows as 2**n.
    n = 32
    operations = [{"kind": "RY", "wires": [q], "theta": .37, "param": 0} for q in range(n)]
    operations += [{"kind": "depolarize", "wires": [q], "probability": .02} for q in range(n)]
    for q in range(1, n):
        operations += [{"kind": "H", "wires": [0]}, {"kind": "CZ", "wires": [q, 0]},
                       {"kind": "H", "wires": [0]}]
    cases.append(Case("contractible_parity_32q", n, 1, operations, [(1., {0: "Z"})], {},
                      [[0.], [.13], [-.21]], description="Constructed boundary-contraction stress case; not a general speed claim."))
    return cases


def timing_vectors(case, repeats):
    """Fresh deterministic shifts each round, outside the timed evaluation block."""
    schedule = []
    for repeat in range(repeats):
        offset = repeat % len(case.vectors)
        ordered = case.vectors[offset:] + case.vectors[:offset]
        schedule.append([[float(value + (repeat + 1) * .003 * math.cos((row + 1) * (parameter + 1)))
                          for parameter, value in enumerate(vector)]
                         for row, vector in enumerate(ordered)])
    return schedule


def oracle(case, vectors=None):
    vectors = case.vectors if vectors is None else vectors
    if case.name == "contractible_parity_32q":
        n, scale = case.n_qubits, 1 - 4 * .02 / 3
        return [(float((scale * math.cos(.37 + shifts[0]))**n),
                 [float(-n * scale**n * math.sin(.37 + shifts[0]) * math.cos(.37 + shifts[0])**(n - 1))])
                for shifts in vectors]
    solver = DenseAdjoint(case)
    return [solver.evaluate(shifts, True) for shifts in vectors]
