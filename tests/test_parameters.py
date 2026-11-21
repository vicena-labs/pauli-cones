"""Many shared parameters, linear angles k*p + b, fixed unitaries and controlled gates."""
import math

import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, Truncation
from quantum_circuit_simulator.interop import expectation, from_qasm


def ansatz(n, layers, scales=True):
    """A hardware-efficient ansatz with one parameter per rotation, some shared with weights."""
    params = n * layers * 2
    c = Circuit(n, params, max_terms=200_000)
    k = 0
    for layer in range(layers):
        for q in range(n):
            c.gate("RY", [q], 0.1 * q + 0.2, k, 1.5 if scales and k % 3 == 0 else 1.0)
            c.gate("RZ", [q], -0.3 + 0.05 * layer, k + 1, -0.5 if scales and k % 5 == 0 else 1.0)
            k += 2
        for q in range(layer % 2, n - 1, 2):
            c.gate("CZ", [q, q + 1]).depolarize([q, q + 1], 0.01)
    return c, params


OBSERVABLE = [(1.0, {2: "Z"}), (0.5, {1: "X", 2: "X"}), (-0.25, {0: "Y", 3: "Z"})]


@pytest.mark.parametrize("backend", ["native", "graph", "dense", "python"])
def test_gradients_beyond_sixteen_parameters_agree_with_finite_differences(backend):
    c, params = ansatz(5, 4)          # 40 parameters
    assert params > 16
    shifts = np.linspace(-0.4, 0.6, params)
    plan = c.prepare(OBSERVABLE, backend=backend)
    result = plan.evaluate(shifts)
    assert len(result.gradient) == params
    h = 1e-6
    for j in (0, 7, 16, 17, 33, params - 1):
        up, down = shifts.copy(), shifts.copy()
        up[j] += h
        down[j] -= h
        numeric = (plan.evaluate_value(up) - plan.evaluate_value(down)) / (2 * h)
        assert result.gradient[j] == pytest.approx(numeric, abs=1e-7)


def test_backends_agree_with_many_parameters_and_scales():
    c, params = ansatz(6, 5)          # 60 parameters
    shifts = np.sin(np.arange(params))
    results = {b: c.prepare(OBSERVABLE, backend=b).evaluate(shifts) for b in ("native", "graph", "dense")}
    for b in ("graph", "dense"):
        assert results[b].value == pytest.approx(results["native"].value, abs=1e-12)
        assert np.allclose(results[b].gradient, results["native"].gradient, atol=1e-12)
    facts = c.prepare(OBSERVABLE).evaluate(shifts)
    assert facts.backend == "native"


def test_truncated_bounds_hold_beyond_sixteen_parameters():
    c, params = ansatz(8, 6)
    shifts = 0.3 * np.cos(np.arange(params))
    exact = c.prepare(OBSERVABLE).evaluate(shifts)
    for backend in ("native", "graph"):
        cut = c.prepare(OBSERVABLE, backend=backend, truncation=Truncation(min_coefficient=1e-3)).evaluate(shifts)
        assert abs(cut.value - exact.value) <= cut.error_bound + 1e-12
        assert len(cut.gradient_error_bound) == params
        assert np.all(np.abs(np.array(cut.gradient) - exact.gradient) <= np.array(cut.gradient_error_bound) + 1e-12)


def test_scale_is_the_chain_rule():
    # RY(0.2 + 3 p) on |0>: <Z> = cos(0.2 + 3p), d/dp = -3 sin(0.2 + 3p).
    c = Circuit(1, 1).gate("RY", [0], 0.2, 0, 3.0)
    r = c.prepare([(1.0, {0: "Z"})]).evaluate([0.1])
    assert r.value == pytest.approx(math.cos(0.5), abs=1e-14)
    assert r.gradient[0] == pytest.approx(-3 * math.sin(0.5), abs=1e-14)
    with pytest.raises(ValueError):
        Circuit(1, 1).gate("RY", [0], 0.2, 0, 0.0)


def test_fixed_unitary_matches_its_rotation():
    theta = 0.7
    u = np.array([[math.cos(theta / 2), -math.sin(theta / 2)], [math.sin(theta / 2), math.cos(theta / 2)]])
    a = Circuit(1).unitary([0], u).prepare([(1.0, {0: "Z"}), (1.0, {0: "X"})]).evaluate_value()
    b = Circuit(1).gate("RY", [0], theta).prepare([(1.0, {0: "Z"}), (1.0, {0: "X"})]).evaluate_value()
    assert a == pytest.approx(b, abs=1e-14)
    with pytest.raises(ValueError):
        Circuit(1).unitary([0], [[1, 1], [0, 1]])


def test_qiskit_parameters_with_weights_share_one_plan():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp, Statevector
    gamma, beta = Parameter("g"), Parameter("b")
    qc = qiskit.QuantumCircuit(4)
    qc.h(range(4))
    for (a, b), w in zip([(0, 1), (1, 2), (2, 3), (3, 0)], [0.5, 1.0, 1.5, 2.0]):
        qc.rzz(2 * w * gamma, a, b)
    qc.rx(2 * beta, range(4))
    from quantum_circuit_simulator.interop import from_qiskit
    converted = from_qiskit(qc)
    assert converted.circuit.n_parameters == 2
    values = {"g": 0.37, "b": -0.21}
    want = Statevector(qc.assign_parameters({gamma: 0.37, beta: -0.21})).expectation_value(
        SparsePauliOp("IIZZ")).real
    got = converted.circuit.prepare([(1.0, {0: "Z", 1: "Z"})]).evaluate(converted.shifts(values))
    assert got.value == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("gate, arity, angle", [
    ("crx", 2, 0.9), ("cry", 2, -1.1), ("crz", 2, 0.4), ("ch", 2, None), ("ccx", 3, None), ("cswap", 3, None)])
def test_controlled_gates_match_qiskit(gate, arity, angle):
    pytest.importorskip("qiskit")
    from qiskit import qasm2
    from qiskit.quantum_info import SparsePauliOp, Statevector
    prep = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; ry(0.7) q[0]; ry(1.9) q[1]; rx(0.4) q[2]; h q[2];'
    args = ",".join(f"q[{i}]" for i in range(arity))
    text = prep + f" {gate}{f'({angle})' if angle is not None else ''} {args};"
    reference = Statevector(qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
    for label in ("ZII", "IZI", "IIZ", "XXI", "IYX", "ZXY"):
        want = reference.expectation_value(SparsePauliOp(label)).real
        assert expectation(text, "1*" + label).value == pytest.approx(want, abs=1e-12), label
    assert from_qasm(text).circuit.n_parameters == 0
