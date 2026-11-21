"""Regressions: observable-scale invariance and max_cone_qubits at every width."""
import pytest

from quantum_circuit_simulator import Circuit, ConeOverflow

SPARSE_AND_ALTERNATES = ['native', 'python', 'graph', 'dense', 'auto']


def entangled():
    c = Circuit(2, 1)
    c.gate('RY', [0], theta=.7, param=0).gate('H', [1]).gate('CZ', [0, 1])
    c.depolarize([0, 1], .01).readout(0, .02, .03)
    return c


@pytest.mark.parametrize('backend', SPARSE_AND_ALTERNATES)
@pytest.mark.parametrize('scale', [1e-12, 1e-17, 1e-30, 1e12])
def test_results_scale_linearly_with_observable_weights(backend, scale):
    # Weights of 1e-17 were silently returned as exactly 0 by native/python
    # (absolute 1e-16 coefficient cutoff); e.g. Hamiltonians in SI units.
    c = entangled()
    obs = [(1., {0: 'Z'}), (.5, {0: 'Z', 1: 'X'})]
    base = c.prepare(obs, backend=backend).evaluate([.1])
    small = c.prepare([(scale * w, p) for w, p in obs], backend=backend).evaluate([.1])
    assert small.value / scale == pytest.approx(base.value, rel=1e-12)
    assert small.gradient[0] / scale == pytest.approx(base.gradient[0], rel=1e-12)


@pytest.mark.parametrize('backend', SPARSE_AND_ALTERNATES)
def test_small_term_survives_noise_damping(backend):
    # 1e-14 * X damped by 20 depolarizing layers (factor 2/3 each) is ~3e-18:
    # below the old absolute cutoff, but the whole observable.
    c = Circuit(1).gate('H', [0])
    for _ in range(20):
        c.depolarize([0], .25)
    value = c.prepare([(1e-14, {0: 'X'})], backend=backend).evaluate().value
    assert value == pytest.approx(1e-14 * (2 / 3) ** 20, rel=1e-12)


@pytest.mark.parametrize('backend', SPARSE_AND_ALTERNATES)
@pytest.mark.parametrize('n_qubits', [8, 256, 300])
def test_max_cone_qubits_is_enforced_at_every_register_width(backend, n_qubits):
    # The limit used to be read only by the >256-wire planner.
    c = Circuit(n_qubits)
    for q in range(7):
        c.gate('CZ', [q, q + 1])
    with pytest.raises(ConeOverflow):
        c.prepare([(1., {7: 'Z'})], backend=backend, max_cone_qubits=7)
    plan = c.prepare([(1., {7: 'Z'})], backend=backend, max_cone_qubits=8, dense_max_qubits=8)
    assert plan.evaluate().value == pytest.approx(1.)


def test_max_cone_qubits_counts_each_observable_separately():
    c = Circuit(6).gate('CZ', [0, 1]).gate('CZ', [2, 3]).gate('CZ', [4, 5])
    # Three disjoint 2-wire cones: a 2-wire limit is enough.
    obs = [(1., {0: 'Z'}), (1., {2: 'Z'}), (1., {4: 'Z'})]
    assert c.prepare(obs, max_cone_qubits=2).evaluate().value == pytest.approx(3.)
    with pytest.raises(ConeOverflow):
        c.prepare([(1., {0: 'Z', 2: 'Z'})], max_cone_qubits=3)
