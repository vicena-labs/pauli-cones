"""Developer examples must stay executable, not merely illustrative."""
import math
from pathlib import Path
import sys

import pytest
from quantum_circuit_simulator import Circuit, Topology

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
from architecture_playground import build_probe, load_topology, run_probe  # noqa: E402
from qaoa_maxcut import build_qaoa, evaluate_qaoa  # noqa: E402


@pytest.mark.parametrize("topology", [Topology.line(5), Topology.grid(2, 3),
                                        Topology(5, ((0, 2), (2, 4), (1, 4)))])
def test_matchings_are_complete_non_overlapping_and_deterministic(topology):
    matched = []
    for layer in topology.matchings():
        used = set()
        for u, v in layer:
            assert u not in used and v not in used
            used.update((u, v))
            matched.append((u, v))
    assert sorted(matched) == list(topology.edges)
    assert topology.matchings() == topology.matchings()


@pytest.mark.parametrize("n,edges", [
    (0, ()), (16385, ()), (2, ((0, 0),)),
    (2, ((0, 1), (1, 0))), (2, ((0, 2),)),
    (2, ((0, 1, 2),)), (True, ())])
def test_invalid_topologies_fail(n, edges):
    with pytest.raises(ValueError):
        Topology(n, edges)


def test_custom_topology_example_loaded_and_guarded():
    t = load_topology("custom", path=str(EXAMPLES / "custom_topology.json"))
    assert t.n_qubits == 5
    assert (0, 4) in t.edges
    t.require_edge(0, 4)
    with pytest.raises(ValueError, match="coupler"):
        t.require_edge(0, 3)
    with pytest.raises(ValueError, match="size"):
        t.add_cz_layer(Circuit(4))


def test_supported_chip_probe_matches_independent_backends():
    t = Topology.grid(2, 2)
    n = run_probe(t, layers=2, noise=.01, backend="native")
    p = run_probe(t, layers=2, noise=.01, backend="python")
    assert n["value"] == pytest.approx(p["value"], abs=1e-10)
    assert n["gradient"] == pytest.approx(p["gradient"], abs=1e-10)
    assert n["not_a_hardware_calibration"]


def test_qaoa_zero_parameters_uniform_reference_and_gradient():
    t = Topology.line(3)
    for backend in ("native", "python", "dense"):
        out = evaluate_qaoa(t, gamma=0.0, beta=0.0, backend=backend)
        assert out["expected_cut"] == pytest.approx(len(t.edges) / 2, abs=1e-11)
        assert out["gradient_gamma"] == pytest.approx(0., abs=1e-10)
        assert out["gradient_beta"] == pytest.approx(0., abs=1e-10)


def test_qaoa_gradients_match_independent_finite_difference():
    t = Topology.line(3)
    gamma, beta = .32, .19
    out = evaluate_qaoa(t, gamma=gamma, beta=beta, noise=.015, backend="native")
    ref = evaluate_qaoa(t, gamma=gamma, beta=beta, noise=.015, backend="dense")
    assert out["expected_cut"] == pytest.approx(ref["expected_cut"], abs=2e-10)
    assert out["gradient_gamma"] == pytest.approx(ref["gradient_gamma"], abs=2e-10)
    assert out["gradient_beta"] == pytest.approx(ref["gradient_beta"], abs=2e-10)
    h = 1e-5
    for field, target in (("gamma", "gradient_gamma"), ("beta", "gradient_beta")):
        low = {"gamma": gamma, "beta": beta}
        high = dict(low)
        low[field] -= h
        high[field] += h
        lo = evaluate_qaoa(t, **low, noise=.015, backend="dense")["expected_cut"]
        hi = evaluate_qaoa(t, **high, noise=.015, backend="dense")["expected_cut"]
        assert out[target] == pytest.approx((hi-lo) / (2*h), abs=3e-8)


def test_variational_energy_example_descends_against_reference():
    from variational_energy import descend
    native = descend(backend="native", steps=4)
    reference = descend(backend="dense", steps=4)
    assert native["history"][-1]["energy"] < native["history"][0]["energy"]
    for a,b in zip(native["history"],reference["history"]):
        assert a["energy"] == pytest.approx(b["energy"], abs=1e-10)
        assert a["gradient"] == pytest.approx(b["gradient"], abs=1e-10)


def test_small_backend_comparison_example_runs():
    from compare_backends import compare
    result = compare()
    assert len(result["records"]) == 5
    assert all(math.isfinite(r["prepare_ms"]) for r in result["records"])
