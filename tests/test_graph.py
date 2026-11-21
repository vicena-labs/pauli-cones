"""Reusable graph qualifications independent of the native sparse recurrence."""
import math

import numpy as np
import pytest

from pauli_cones._core._graph import GraphLimitExceeded, GraphPrepared, _fourier
from pauli_cones._core._reference import NoisyPauliCircuit, fsim, transfer
from test_reference_density import build_case, oracle


@pytest.mark.parametrize("name", ["RX", "RY", "RZ", "FSIM"])
@pytest.mark.parametrize("theta", [0., math.pi / 2, math.pi, -.37, 1.92])
def test_structural_fourier_coefficients_cover_values_and_tangents(name, theta):
    circuit = NoisyPauliCircuit(2, 1)
    if name == "FSIM":
        circuit.fsim((0, 1), theta, .19, param=0)
    else:
        circuit.gate(name, (0,), theta, param=0)
    stage = circuit.stages[0]
    coeff = _fourier(stage)
    weights = [1., math.cos(theta), math.sin(theta), math.cos(2*theta), math.sin(2*theta)]
    tangent = [0., -math.sin(theta), math.cos(theta), -2*math.sin(2*theta), 2*math.cos(2*theta)]
    assert np.einsum("f,fij->ij", weights, coeff) == pytest.approx(stage.matrix, abs=2e-15)
    assert np.einsum("f,fij->ij", tangent, coeff) == pytest.approx(stage.derivative, abs=2e-15)


@pytest.mark.parametrize("initial", ["zero", "plus"])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_graph_against_independent_kraus_density_with_shared_parameters(n, initial):
    circuit, steps = build_case(n, 0., .0)
    # A second parameterized gate family shares the first parameter. Its
    # derivative must contribute to the same reverse-mode accumulator.
    circuit.fsim((0, 1), .0, .27, param=0)
    steps.append(("FSIM", (0, 1), (.0, .27)))
    plan = GraphPrepared(circuit, [(1., {0: "Z"})], initial=initial)
    fingerprint = (plan.stats["nodes"], plan.stats["edges"])
    for shifts in ([0., 0.], [.31, -.17], [math.pi / 2, .19], [-.83, .71]):
        def expected(delta):
            bound = []
            for position, (op, wires, value) in enumerate(steps):
                if op == "RY":
                    value += delta[0]
                elif op == "RZ":
                    value += delta[1]
                elif position == len(steps) - 1:
                    value = (value[0] + delta[0], value[1])
                bound.append((op, wires, value))
            return oracle(n, bound, 0, initial, readout=(.026, .017))[0]

        value, gradient = plan.bind(shifts).evaluate()
        assert value == pytest.approx(expected(shifts), abs=2e-12)
        for parameter in range(2):
            plus, minus = list(shifts), list(shifts)
            plus[parameter] += 2e-5
            minus[parameter] -= 2e-5
            assert gradient[parameter] == pytest.approx(
                (expected(plus) - expected(minus)) / 4e-5, abs=2e-7)
        scalar, empty = plan.evaluate(with_gradients=False)
        assert scalar == value
        assert empty.shape == (0,)
        assert fingerprint == (plan.stats["nodes"], plan.stats["edges"])


def test_zero_binding_retains_paths_needed_at_later_angles_and_by_derivatives():
    circuit = NoisyPauliCircuit(1, 1).gate("RY", (0,), 0., param=0)
    plan = GraphPrepared(circuit, [(1., {0: "X"})])
    for angle in (0., .3, math.pi / 2, -1.7, 0.):
        value, gradient = plan.bind([angle]).evaluate()
        assert value == pytest.approx(math.sin(angle), abs=1e-15)
        assert gradient == pytest.approx([math.cos(angle)], abs=1e-15)


def test_shared_parameter_gradient_sums_multiple_gate_occurrences():
    circuit = NoisyPauliCircuit(1, 1)
    for _ in range(5):
        circuit.gate("RY", (0,), 0., param=0)
    plan = GraphPrepared(circuit, [(1., {0: "Z"})])
    for angle in (0., .29, -.3):
        value, gradient = plan.bind([angle]).evaluate()
        assert value == pytest.approx(math.cos(5*angle), abs=2e-15)
        assert gradient == pytest.approx([-5*math.sin(5*angle)], abs=4e-15)


@pytest.mark.parametrize("initial,axis", [("zero", "Z"), ("plus", "X")])
def test_no_stage_product_boundaries_and_unobserved_wires(initial, axis):
    circuit = NoisyPauliCircuit(200)
    assert GraphPrepared(circuit, [(2., {199: axis})], initial=initial).evaluate()[0] == 2.
    forbidden = "X" if initial == "zero" else "Z"
    assert GraphPrepared(circuit, [(2., {73: forbidden})], initial=initial).evaluate()[0] == 0.
    assert GraphPrepared(circuit, [], initial=initial).evaluate()[0] == 0.


def test_early_input_contraction_prevents_affine_relaxation_explosion():
    circuit = NoisyPauliCircuit(128, max_terms=1)
    for q in range(128):
        circuit.channel_relax(q, -math.log(.75), 1., 1.)
    plan = GraphPrepared(circuit, [(1., {q: "Z" for q in range(128)})])
    value, gradient, stats = plan.evaluate(include_stats=True)
    assert value == pytest.approx(1., abs=1e-14)
    assert gradient.shape == (0,)
    assert stats["max_single_cone_terms"] == 1
    assert stats["edges"] == 256


def test_idle_readout_is_contracted_before_expanding_a_large_product():
    circuit = NoisyPauliCircuit(128, max_terms=1)
    for q in range(128):
        circuit.set_readout(q, .03, .01)
    plan = GraphPrepared(circuit, [(1., {q: "Z" for q in range(128)})])
    value, _ = plan.evaluate()
    assert value == pytest.approx(.94 ** 128, rel=2e-13)
    assert plan.stats["max_single_cone_terms"] <= 1


def test_readout_initialization_preserves_small_observable_coefficients():
    circuit = NoisyPauliCircuit(1, 1).gate("RY", (0,), .3, param=0)
    circuit.set_readout(0, .03, .01)
    coefficient = 1e-18
    plan = GraphPrepared(circuit, [(coefficient, {0: "Z"})])
    value, gradient = plan.evaluate()
    assert value == pytest.approx(coefficient * (-.02 + .96*math.cos(.3)), rel=2e-13, abs=0.)
    assert gradient[0] == pytest.approx(-coefficient * .96*math.sin(.3), rel=2e-13, abs=0.)
    assert plan.stats["coefficient_pruning"] is False


def test_graph_budgets_fail_before_returning_partial_answers():
    circuit = NoisyPauliCircuit(1, 1).gate("RY", (0,), param=0)
    with pytest.raises(GraphLimitExceeded, match="max_nodes"):
        GraphPrepared(circuit, [(1., {0: "Z"})], max_nodes=1)
    circuit.gate("RY", (0,), param=0)
    with pytest.raises(GraphLimitExceeded, match="max_edges"):
        GraphPrepared(circuit, [(1., {0: "Z"})], max_edges=1)
    circuit.max_terms = 1
    with pytest.raises(GraphLimitExceeded, match="max_terms"):
        GraphPrepared(circuit, [(1., {0: "Z"})])


def test_readout_initialization_overflow_is_a_graph_budget_failure():
    circuit = NoisyPauliCircuit(3, max_terms=1)
    for q in range(3):
        circuit.gate("H", (q,)).set_readout(q, .02, .03)
    with pytest.raises(GraphLimitExceeded, match="observable initialization"):
        GraphPrepared(circuit, [(1., {q: "Z" for q in range(3)})])


@pytest.mark.parametrize("invalid", [0, -1, True, 1.3])
def test_invalid_graph_budgets(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        GraphPrepared(NoisyPauliCircuit(1), [], max_edges=invalid)


def test_rebinding_validates_width_and_finiteness():
    plan = GraphPrepared(NoisyPauliCircuit(1, 1), [(1., {})])
    for shifts in ([], [0., 0.], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            plan.bind(shifts)


def test_all_fsim_fourier_paths_remain_valid_at_special_phi():
    # The graph must remain valid when an initially zero value and derivative
    # later become nonzero, and when fixed phi causes exact path cancellation.
    for phi in (0., math.pi / 2, math.pi, 1e-14):
        circuit = NoisyPauliCircuit(2, 1).fsim((0, 1), 0., phi, param=0)
        coefficients = _fourier(circuit.stages[0])
        for angle in (0., .19, .57, -1.31):
            unitary, derivative, _ = fsim(angle, phi)
            expected, tangent = transfer(unitary, derivative)
            weights = [1, math.cos(angle), math.sin(angle), math.cos(2*angle), math.sin(2*angle)]
            derivs = [0, -math.sin(angle), math.cos(angle), -2*math.sin(2*angle), 2*math.cos(2*angle)]
            assert np.einsum("f,fij->ij", weights, coefficients) == pytest.approx(expected, abs=1e-15)
            assert np.einsum("f,fij->ij", derivs, coefficients) == pytest.approx(tangent, abs=1e-15)


def _wide_noisy_brickwork(n=600):
    from pauli_cones import Circuit
    c = Circuit(n, 2)
    for q in range(n):
        c.gate('RY', [q], theta=.01 * q, param=q % 2)
    for q in range(0, n - 1, 2):
        c.gate('CZ', [q, q + 1]).depolarize([q, q + 1], .01)
    for q in range(n):
        c.gate('RX', [q], theta=-.02 * q, param=(q + 1) % 2)
    obs = [(1. / n, {q: 'Z'}) for q in range(n)] + [(.1, {q: 'X', q + 1: 'X'}) for q in range(0, n - 1, 7)]
    return c, obs


def test_wide_graph_matches_native_for_any_thread_count():
    c, obs = _wide_noisy_brickwork()
    native = c.prepare(obs).evaluate([.1, -.1])
    results = [c.prepare(obs, backend='graph', threads=t).evaluate([.1, -.1]) for t in (1, 4)]
    assert results[0] == results[1]
    assert results[0].backend == 'wide-graph'
    assert results[0].value == pytest.approx(native.value, abs=1e-14)
    assert results[0].gradient == pytest.approx(native.gradient, abs=1e-14)


def test_wide_graph_over_budget_falls_back_to_wide_native():
    c, obs = _wide_noisy_brickwork()
    plan = c.prepare(obs, backend='graph', graph_max_edges=10)
    assert plan.evaluate([.1, -.1]).backend == 'wide-native'
    assert 'max_edges' in plan.diagnostics['fallback_reason']


def test_joint_graph_group_over_the_cap_splits_into_single_cones():
    # Jointly X0, X1 and RY's Z1 image need 3 terms; each cone alone needs 2.
    from pauli_cones import Circuit
    c = Circuit(2, 2, max_terms=2).gate('H', [0]).gate('H', [1])
    c.gate('RY', [0], theta=.4, param=0).gate('RY', [1], theta=-.3, param=1)
    plan = c.prepare([(1., {0: 'X'}), (1., {1: 'X'})], backend='graph')
    result = plan.evaluate([0., 0.])
    assert result.backend == 'graph'
    assert result.value == pytest.approx(math.cos(.4) + math.cos(.3))
    assert result.gradient == pytest.approx((-math.sin(.4), math.sin(.3)))
    assert plan.diagnostics['cone_groups'] == 2
