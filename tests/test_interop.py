"""OpenQASM, Qiskit and observable conversion, and the Qiskit Estimator, against Qiskit's statevector."""
import math

import numpy as np
import pytest

from pauli_cones import Truncation
from pauli_cones.interop import ConversionError, expectation, from_qasm, parse_observable

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

ALL_GATES_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg r[1];
creg c[4];
gate mix(a) x, y { rzz(a) x, y; h x; cp(a/2) y, x; }
h q; x q[0]; y q[1]; z q[2]; s q[0]; sdg q[1]; t q[2]; tdg q[0]; sx q[1]; sxdg q[2];
rx(0.3) q[0]; ry(-0.7) q[1]; rz(pi/5) q[2]; u1(0.2) q[0]; u2(0.1, -0.4) q[1]; u3(0.5, 0.6, -0.2) q[2];
u(0.4, -0.3, 0.9) r[0]; p(1.1) r[0];
cx q[0], q[1]; cy q[1], q[2]; cz q[2], r[0]; swap q[0], r[0];
rzz(0.37) q[0], q[1]; rxx(-0.41) q[1], q[2]; cp(0.8) q[1], r[0];
mix(0.9) q[2], q[0];
barrier q;
measure q[0] -> c[0];
measure r[0] -> c[3];
"""

OBSERVABLES = ["Z0", "0.5*X1*Y2 - 0.25*Z3 + Y0", "IXYZ", "0.3*ZZII - 1.2*XIXI"]


def statevector_expectation(circuit, observable):
    terms = parse_observable(observable)
    n = circuit.num_qubits
    labels = [("".join(p.get(q, "I") for q in reversed(range(n))), c) for c, p in terms]
    return Statevector(circuit.remove_final_measurements(inplace=False)).expectation_value(
        SparsePauliOp.from_list(labels)).real


def test_dependency_free_qasm2_reader_matches_qiskit_for_every_supported_gate():
    reference = qasm2.loads(ALL_GATES_QASM, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    converted = from_qasm(ALL_GATES_QASM)
    assert converted.circuit.n_qubits == 4 and converted.parameters == []
    for observable in OBSERVABLES:
        got = converted.circuit.prepare(parse_observable(observable)).evaluate().value
        assert got == pytest.approx(statevector_expectation(reference, observable), abs=1e-12)


def test_qiskit_circuit_with_ecr_and_transpiled_gates():
    qc = QuantumCircuit(3)
    qc.h([0, 1, 2]); qc.ecr(0, 1); qc.ecr(2, 1); qc.ryy(0.23, 2, 0); qc.rzx(0.6, 0, 2)
    qc.crx(0.7, 0, 2); qc.ccx(0, 1, 2); qc.cswap(2, 0, 1)
    qc.ry(0.4, 1)
    for observable in ["Z0*Z1", "X2 + 0.5*Y1", "Z0*X1*Y2"]:
        assert expectation(qc, observable).value == pytest.approx(statevector_expectation(qc, observable), abs=1e-12)


def test_shared_parameters_including_negative_slopes_give_values_and_gradients():
    a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
    qc = QuantumCircuit(3)
    qc.h(0); qc.ry(a, 0); qc.rz(-b + 0.3, 1); qc.rx(0.2 - a, 2); qc.cx(0, 1); qc.rzz(c, 1, 2)
    qc.ry(b, 2); qc.cx(2, 0); qc.rx(-c, 0)
    values = {"a": 0.31, "b": -0.47, "c": 0.9}
    observable = "Z0*Z1 + 0.5*X2 - 0.3*Y0*Z2"
    result = expectation(qc, observable, values=values, gradients=True)
    bound = qc.assign_parameters({p: values[p.name] for p in qc.parameters})
    assert result.value == pytest.approx(statevector_expectation(bound, observable), abs=1e-12)
    # Gradient order follows the converted parameters; compare by name.
    from pauli_cones.interop import convert
    names = [p.name for p in convert(qc).parameters]
    for name, g in zip(names, result.gradient):
        plus, minus = dict(values), dict(values)
        plus[name] += 1e-6; minus[name] -= 1e-6
        fd = (statevector_expectation(qc.assign_parameters({p: plus[p.name] for p in qc.parameters}), observable)
              - statevector_expectation(qc.assign_parameters({p: minus[p.name] for p in qc.parameters}), observable)) / 2e-6
        assert g == pytest.approx(fd, abs=1e-7)
    # A sequence of values follows Qiskit's own parameter order.
    in_order = [values[p.name] for p in qc.parameters]
    assert expectation(qc, observable, values=in_order).value == pytest.approx(result.value, abs=1e-14)


def test_unshareable_parameters_need_values_and_then_bind():
    a = Parameter("a")
    qc = QuantumCircuit(1); qc.ry(2 * a, 0)
    with pytest.raises(ConversionError) as info:
        expectation(qc, "Z0")
    assert "values" in info.value.hint
    assert expectation(qc, "Z0", values=[0.3]).value == pytest.approx(math.cos(0.6), abs=1e-14)


def test_observable_forms_agree():
    forms = ["0.5*Z0*Z1 - X2", "0.5 Z0 Z1 - 1.0*X2", {"IZZ": 0.5, "XII": -1.0},
             SparsePauliOp.from_list([("IZZ", 0.5), ("XII", -1.0)]), [(0.5, {0: "Z", 1: "Z"}), (-1.0, {2: "X"})]]
    canonical = sorted((c, tuple(sorted(p.items()))) for c, p in parse_observable(forms[0]))
    for form in forms[1:]:
        assert sorted((c, tuple(sorted(p.items()))) for c, p in parse_observable(form)) == canonical
    assert parse_observable("1e-3*Z0 + 2.5e+1*X1") == [(1e-3, {0: "Z"}), (25.0, {1: "X"})]
    for bad in ["0.5*Q0", "Z0*Z0", {"XZ": 1j}]:
        with pytest.raises(ConversionError):
            parse_observable(bad)


@pytest.mark.parametrize("bad, hint", [
    ("OPENQASM 2.0; qreg q[1]; reset q[0];", "|0"),
    ("OPENQASM 2.0; qreg q[1]; creg c[1]; measure q[0] -> c[0]; h q[0];", "end"),
    ("OPENQASM 2.0; qreg q[1]; foo q[0];", "Qiskit"),
])
def test_unsupported_qasm_explains_what_to_do(bad, hint):
    with pytest.raises(ConversionError) as info:
        from_qasm(bad)
    assert hint in info.value.hint


def test_estimator_matches_statevector_estimator_with_broadcasting():
    from qiskit.primitives import StatevectorEstimator
    from pauli_cones.estimator import Estimator
    a, b = Parameter("a"), Parameter("b")
    qc = QuantumCircuit(3)
    qc.h(0); qc.ry(a, 1); qc.cx(0, 1); qc.rz(-b, 2); qc.cx(1, 2); qc.rx(a + 0.1, 0)
    observables = [[SparsePauliOp(["ZZI", "IIX"], [1, 0.5])], [SparsePauliOp("XYZ")]]
    values = np.random.default_rng(3).normal(size=(4, 2))
    ours = Estimator().run([(qc, observables, values)]).result()[0]
    ref = StatevectorEstimator().run([(qc, observables, values)]).result()[0]
    assert ours.data.evs.shape == ref.data.evs.shape == (2, 4)
    assert np.allclose(ours.data.evs, ref.data.evs, atol=1e-12)
    assert np.all(ours.data.stds == 0)
    assert ours.metadata["pauli_cones"]["shared_parameters"] is True
    assert ours.metadata["pauli_cones"]["plans"] == 2  # one per observable, reused across 4 bindings
    # A circuit whose parameters cannot be shared is rebuilt per binding.
    q2 = QuantumCircuit(1); q2.ry(2 * a, 0)
    vals = np.array([[0.1], [0.4]])
    res = Estimator().run([(q2, "Z", vals)]).result()[0]
    assert np.allclose(res.data.evs, np.cos(2 * vals[:, 0]), atol=1e-14)
    assert res.metadata["pauli_cones"]["shared_parameters"] is False


def test_estimator_truncation_and_noise():
    from pauli_cones.estimator import Estimator
    qc = QuantumCircuit(4)
    for layer in range(4):
        for q in range(4):
            qc.ry(0.3 + 0.1 * q + layer, q)
        for q in range(layer % 2, 3, 2):
            qc.cx(q, q + 1)
    obs = SparsePauliOp(["ZIII", "IIZZ"], [1.0, 0.5])
    exact = Estimator().run([(qc, obs)]).result()[0].data.evs
    approx = Estimator(truncation=Truncation(1e-2)).run([(qc, obs)]).result()[0].data
    assert abs(approx.evs - exact) <= approx.error_bounds * (1 + 1e-12) + 1e-14
    noisy = Estimator(noise={"depolarizing_1q": 0.01, "depolarizing_2q": 0.02}).run([(qc, obs)]).result()[0]
    assert abs(noisy.data.evs) < abs(exact)  # depolarizing noise shrinks this expectation
