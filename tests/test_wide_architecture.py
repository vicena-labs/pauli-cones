"""Large logical-register routing, correctness, and explicit failure boundaries."""
import math
import pytest
import numpy as np
from pauli_cones import Circuit, ConeOverflow, register_backend, unregister_backend, available_backends
from pauli_cones.topology import Topology


@pytest.mark.parametrize('backend', ['native', 'python', 'auto', 'graph'])
def test_4096_single_high_wire_with_gradient(backend):
    c = Circuit(4096, 1).gate('RY', [4095], theta=.4, param=0)
    plan = c.prepare([(1., {4095:'Z'})], backend=backend)
    result = plan.evaluate([.1])
    assert result.value == pytest.approx(math.cos(.5), abs=2e-12)
    assert result.gradient == pytest.approx((-math.sin(.5),), abs=2e-12)
    assert plan.diagnostics['logical_qubits'] == 4096
    assert plan.diagnostics['max_cone_qubits'] == 1
    assert result.backend.startswith('wide-')


def test_large_logical_grid_exact_local_cones_and_readout():
    topology = Topology.grid(64, 64)
    c = Circuit(topology.n_qubits, 1, max_terms=1000)
    # Independent connected pairs in a huge physical register.
    for q in range(0, 64, 2):
        c.gate('RY', [q], theta=.2, param=0).gate('CZ', [q,q+1])
        c.depolarize([q,q+1], .01)
    c.readout(4095, .03, .04)
    obs = [(1., {q:'Z'}) for q in range(0, 64, 2)]
    obs.append((2., {4095:'Z'}))
    plan = c.prepare(obs, backend='native')
    got = plan.evaluate([.3]); ref = plan.evaluate([.3], gradients=False)
    # CZ on control |0> and two-wire depolarization are calibrated by the
    # library's convention: every nonidentity Pauli is scaled by 1-16p/15.
    expected = 32 * (1 - 16*.01/15) * math.cos(.5) + 2*(1-2*.03)
    derivative = -32 * (1 - 16*.01/15) * math.sin(.5)
    assert got.value == pytest.approx(expected, abs=3e-10)
    assert got.gradient == pytest.approx((derivative,), abs=3e-10)
    assert ref.value == pytest.approx(expected, abs=3e-10)
    assert ref.gradient == ()
    assert plan.diagnostics['max_cone_qubits'] <= 2


def test_wide_entangled_bell_pair_against_compact():
    wide = Circuit(4096, 1).gate('RY',[1023],.4,0).gate('CZ',[1023,1024])
    wide.gate('RY',[1024],-.2,0).relax(1024,1e-6,50e-6,70e-6)
    tiny = Circuit(2,1).gate('RY',[0],.4,0).gate('CZ',[0,1])
    tiny.gate('RY',[1],-.2,0).relax(1,1e-6,50e-6,70e-6)
    for shifts in ([.05],[-.09]):
        a=wide.prepare([(.7,{1023:'X',1024:'Z'}),(-.3,{1024:'Y'})]).evaluate(shifts)
        b=tiny.prepare([(.7,{0:'X',1:'Z'}),(-.3,{1:'Y'})]).evaluate(shifts)
        assert a.value == pytest.approx(b.value, abs=3e-12)
        assert a.gradient == pytest.approx(b.gradient, abs=3e-12)


def test_wide_connected_chain_uses_backward_causal_cone():
    c = Circuit(1024,1)
    # All wires are connected in the *hardware graph* / full circuit,
    # but only a small cone influences Z on the last wire.
    for q in reversed(range(1023)):
        c.gate('CZ',[q,q+1])
    c.gate('RY',[1023],theta=.2,param=0)
    result=c.prepare([(1.,{1023:'Z'})]).evaluate([.1])
    assert result.value == pytest.approx(math.cos(.3), abs=1e-12)


def test_wide_over_limit_fails_before_native_allocation():
    c=Circuit(512)
    # Backward CZ gates are diagonal and can be algebraically simple, but a
    # structural cone can still exceed the declared 256-wire boundary.
    for q in range(300):
        c.gate('CZ',[q,q+1])
    with pytest.raises(ConeOverflow, match='backward cone'):
        c.prepare([(1.,{300:'X'})],max_cone_qubits=64)


def test_wide_topology_and_bounds():
    assert len(Topology.line(4096).edges)==4095
    assert len(Topology.grid(64,64).edges)==8064
    for n in (0,16385,True,1.2):
        with pytest.raises(ValueError):Circuit(n)
    with pytest.raises(ValueError):Circuit(4096).prepare([(1.,{0:'Z'})],max_cone_qubits=257)


def test_external_backend_registration_exact_input_and_repeat():
    class CustomPlan:
        def __init__(self, c):self.c=c
        def bind(self, shifts):self.shifts=tuple(shifts)
        def evaluate(self, *, with_gradients, include_stats):
            return (math.cos(self.shifts[0]),
                    np.array([-math.sin(self.shifts[0])] if with_gradients else []),
                    {'custom': True})
    register_backend('ext-sample', lambda circuit, observables, *, options: CustomPlan(circuit))
    try:
        assert 'ext-sample' in available_backends()
        c=Circuit(1,1).gate('RY',[0],0.,0)
        p=c.prepare([(1.,{0:'Z'})],backend='ext-sample')
        assert p.evaluate([.2]).gradient == pytest.approx((-math.sin(.2),))
        assert p.evaluate([.1],gradients=False).value == pytest.approx(math.cos(.1))
        assert p.numerics['precision']=='backend-defined'
    finally:
        unregister_backend('ext-sample')
    assert 'ext-sample' not in available_backends()


def test_registered_backend_can_receive_full_4096_qubit_model():
    seen = []
    class ConstantBackend:
        def bind(self, shifts):
            self.shifts=tuple(shifts)
        def evaluate(self, *, with_gradients, include_stats):
            return 1., np.zeros(1 if with_gradients else 0), {'custom_large': True}
    def factory(circuit, observables, *, options):
        seen.append(circuit.nqubits)
        return ConstantBackend()
    register_backend('ext-large', factory)
    try:
        c=Circuit(4096,1)
        r=c.prepare([(1.,{4095:'Z'})],backend='ext-large').evaluate([.2])
        assert seen == [4096]
        assert r.value == 1.
    finally:
        unregister_backend('ext-large')


def test_maximum_logical_wires_with_single_local_observable():
    n=16_384
    result=Circuit(n,1).gate('RX',[n-1],.2,0).prepare([(1.,{n-1:'Z'})]).evaluate([.1])
    assert result.value == pytest.approx(math.cos(.3),abs=2e-12)
    assert result.gradient == pytest.approx((-math.sin(.3),),abs=2e-12)


def test_2d_horizontal_vertical_wide_matches_small_reference():
    wide = Circuit(4096,1)
    tiny = Circuit(4,1)
    coordinates = [0,1,64,65]
    for local, physical in enumerate(coordinates):
        wide.gate('RY',[physical],.1 + .03*local,0)
        tiny.gate('RY',[local],.1 + .03*local,0)
    for old_a,old_b in ((0,1),(64,65),(0,64),(1,65)):
        wide.gate('CZ',[old_a,old_b]).depolarize([old_a,old_b],.005)
        a,b=coordinates.index(old_a),coordinates.index(old_b)
        tiny.gate('CZ',[a,b]).depolarize([a,b],.005)
    for theta in ([0.0],[.03],[-.21]):
        a=wide.prepare([(1.,{0:'X',65:'Z'})]).evaluate(theta)
        b=tiny.prepare([(1.,{0:'X',3:'Z'})]).evaluate(theta)
        assert a.value == pytest.approx(b.value, abs=2e-12)
        assert a.gradient == pytest.approx(b.gradient, abs=2e-12)


def test_example_large_grid_vertical_matching_has_finite_result():
    from examples.large_qubit_grid import run
    result=run(4, 4, observable_count=4, vertical=True)
    assert result['horizontal_pairs'] == 8
    assert result['vertical_pairs'] == 8
    assert abs(result['mean_z']) <= 1.000001
    assert math.isfinite(result['shared_gradient'][0])
