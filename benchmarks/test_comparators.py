"""Meaningful checks of adapter conventions, not timing assertions.

Run explicitly: python -m pytest benchmarks/test_comparators.py
Optional competitors are skipped if not installed.
"""
import math
import numpy as np
import pytest

from .models import Case, DenseAdjoint, make_cases, measurement_terms, oracle
from .adapters import prepare


@pytest.mark.parametrize("name,module", [("qulacs", "qulacs"), ("aer", "qiskit_aer"),
                                         ("pauli_prop", "pauli_prop")])
@pytest.mark.parametrize("realistic", [False, True])
def test_adapter_matches_independent_density_and_shared_gradient(name, module, realistic):
    pytest.importorskip(module)
    case = make_cases()[2 if realistic else 1]
    if realistic and name == "pauli_prop":
        with pytest.raises(NotImplementedError, match="T1 relaxation"):
            prepare(name, case)
        return
    adapter = prepare(name, case)
    for shifts, expected in zip(case.vectors, oracle(case)):
        value, gradient = adapter.evaluate(shifts, gradients=True)
        assert value == pytest.approx(expected[0], abs=1e-11)
        assert gradient == pytest.approx(expected[1], abs=1e-10)


@pytest.mark.parametrize("name,module", [("qulacs", "qulacs"), ("aer", "qiskit_aer"),
                                         ("pauli_prop", "pauli_prop"), ("dense_adjoint", "numpy")])
def test_zero_angle_paths_survive_rebinding_and_shared_parameters(name, module):
    pytest.importorskip(module)
    case = Case("zero", 1, 1,
                [{"kind": "RY", "wires": [0], "theta": 0., "param": 0},
                 {"kind": "RY", "wires": [0], "theta": 0., "param": 0}],
                [(1., {0: "X"})], {}, [[0.], [.3], [-.6]])
    adapter = prepare(name, case)
    for shifts in case.vectors:
        value, gradient = adapter.evaluate(shifts, gradients=True)
        assert value == pytest.approx(math.sin(2 * shifts[0]), abs=1e-12)
        assert gradient == pytest.approx([2 * math.cos(2 * shifts[0])], abs=1e-12)


def test_readout_is_affine_on_each_measured_z():
    case = Case("readout", 2, 0, [], [(2., {0: "Z", 1: "Z"})],
                {0: (.2, .1), 1: (.3, .05)}, [[]])
    terms = dict((tuple(sorted(ops.items())), coef) for coef, ops in measurement_terms(case))
    assert terms[()] == pytest.approx(.05)
    assert terms[((0, "Z"),)] == pytest.approx(-.35)
    assert terms[((1, "Z"),)] == pytest.approx(-.13)
    assert terms[((0, "Z"), (1, "Z"))] == pytest.approx(.91)
    value, gradient = DenseAdjoint(case).evaluate([], True)
    assert value == pytest.approx(.48)
    assert gradient == []


def test_independent_adjoint_matches_finite_difference_as_crosscheck_only():
    case = make_cases()[2]
    solver = DenseAdjoint(case)
    shifts = np.array([.13, -.27, .09])
    _, derivative = solver.evaluate(shifts, True)
    h = 1e-5
    differences = []
    for parameter in range(case.n_parameters):
        step = np.zeros(case.n_parameters)
        step[parameter] = h
        plus = solver.evaluate(shifts + step)[0]
        minus = solver.evaluate(shifts - step)[0]
        differences.append((plus - minus) / (2 * h))
    assert derivative == pytest.approx(differences, abs=2e-8)


def test_accuracy_failures_and_truncation_cannot_be_ranked():
    from .compare import accuracy
    record = {"status": "ok", "results": [(2., [])]}
    accuracy(record, [(1., [])], False)
    assert record["status"] == "accuracy_failed"
    assert not record["comparison_eligible"]
    record = {"status": "ok", "results": [(1., [])],
              "diagnostics": {"max_discarded_coefficient_l1": .01}}
    accuracy(record, [(1., [])], False)
    assert record["status"] == "truncated"
    assert not record["comparison_eligible"]


def test_timing_schedule_uses_fresh_vectors_and_missing_results_fail():
    from .models import timing_vectors
    from .compare import accuracy
    case = make_cases()[0]
    rows = [tuple(vector) for group in timing_vectors(case, 5) for vector in group]
    assert len(set(rows)) == len(rows)
    assert not set(rows) & set(tuple(vector) for vector in case.vectors)
    record = {"status": "ok", "results": []}
    accuracy(record, [(1., [])], False)
    assert record["status"] == "accuracy_failed"
    assert not record["comparison_eligible"]
