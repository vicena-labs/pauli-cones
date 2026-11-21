"""State simulation and sampling against Pauli propagation; optimizers, VQE, QAOA, ZNE and shot noise."""
import math

import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, algorithms as al, devices, hamiltonians as hm
from quantum_circuit_simulator.experiments import solve
from quantum_circuit_simulator.statevector import StateVectorPlan, expectation_from_counts, sample


def random_circuit(n, noisy, params=10, seed=1):
    rng = np.random.default_rng(seed)
    c = Circuit(n, params, max_terms=500_000)
    k = 0
    for layer in range(3):
        for q in range(n):
            c.gate("RY", [q], rng.normal(), k % params, 1.3).gate("RX", [q], rng.normal(), (k + 1) % params, -0.7)
            k += 2
            if noisy:
                c.depolarize([q], 0.02).relax(q, 1e-7, 5e-5, 4e-5)
        for q in range(layer % 2, n - 1, 2):
            c.gate("CZ", [q, q + 1])
            if q % 3 == 0:
                c.gate("ECR", [q + 1, q])
            if noisy:
                c.depolarize([q, q + 1], 0.03)
    c.fsim([0, 1], 0.3, 0.2, 0)
    if noisy:
        c.readout(1, 0.02, 0.05)
    return c


OBS = [(1.0, {1: "Z"}), (0.5, {0: "X", 2: "Y"}), (0.3, {1: "Z", 3: "Z"})]


@pytest.mark.parametrize("noisy", [False, True])
def test_state_simulation_matches_pauli_propagation(noisy):
    c = random_circuit(5, noisy)
    shifts = np.random.default_rng(2).normal(size=10)
    pauli = c.prepare(OBS).evaluate(shifts)
    value, gradient = StateVectorPlan(c, OBS).evaluate(shifts, gradients=not noisy)
    assert value == pytest.approx(pauli.value, abs=1e-12)
    if not noisy:
        assert np.allclose(gradient, pauli.gradient, atol=1e-12)


def test_samples_follow_the_noisy_distribution():
    c = random_circuit(4, noisy=True)
    exact = c.prepare([(1.0, {1: "Z"})]).evaluate_value()
    for method in ("density_matrix", "trajectories"):
        counts = sample(c, 6000, seed=4, method=method)["counts"]
        mean, error = expectation_from_counts(counts, {1: "Z"})
        assert abs(mean - exact) < 4 * error, method


def test_clifford_sampling_beyond_the_state_vector_uses_stim():
    pytest.importorskip("stim")
    ghz = Circuit(120).gate("H", [0])
    for q in range(119):
        ghz.gate("H", [q + 1]).gate("CZ", [q, q + 1]).gate("H", [q + 1])
    result = sample(ghz, 200, seed=1)
    assert result["method"] == "stim" and set(result["counts"]) <= {"0" * 120, "1" * 120}
    ghz.gate("RY", [3], 0.3)
    with pytest.raises(ValueError, match="not Clifford"):
        sample(ghz, 10)


def test_vqe_approaches_the_ground_state_with_each_optimizer():
    h = hm.transverse_field_ising(hm.chain(4), j=1.0, h=1.0)
    ansatz = al.hardware_efficient(4, layers=2)
    import importlib.util
    methods = [("adam", 300, 0.05), ("spsa", 400, 0.4)]
    if importlib.util.find_spec("scipy"):
        methods.append(("lbfgs", 200, 0.05))
    for method, steps, slack in methods:
        result = al.vqe(h, ansatz, method=method, steps=steps, seed=1)
        assert result.reference < result.value < result.reference + slack, method
        assert result.trace and {"iteration", "value", "gradient_norm", "seconds"} <= set(result.trace[0])


def test_objective_gradient_beyond_sixteen_parameters_on_a_device():
    h = hm.heisenberg(hm.chain(6))
    ansatz = al.hardware_efficient(6, layers=2)            # 36 parameters
    chip = devices.get("square-grid", rows=2, columns=3)
    objective = al.Objective(ansatz, h, device=chip)
    x = np.linspace(-0.3, 0.4, objective.n_parameters)
    value, gradient, _ = objective(x)
    for j in (0, 17, 35):
        e = np.zeros_like(x)
        e[j] = 1e-6
        assert gradient[j] == pytest.approx((objective.value(x + e) - objective.value(x - e)) / 2e-6, abs=1e-6)
    assert objective.method == "graph" and objective.report["device"] == chip.name


def test_qaoa_finds_good_cuts():
    cost = hm.maxcut(hm.random_regular(8, 3, seed=3))
    result = al.qaoa(cost, depth=2, steps=150, seed=0)
    assert result.extra["approximation_ratio"] > 0.75
    assert result.extra["best_sampled_cost"] == pytest.approx(result.reference)


def test_zne_recovers_the_ideal_value_better_than_raw():
    from quantum_circuit_simulator.circuits import family
    qasm, observable = family("ghz", n=6)
    chip = devices.get("square-grid", rows=3, columns=3)
    import importlib.util
    fits = ("linear", "richardson") + (("exponential",) if importlib.util.find_spec("scipy") else ())
    for fit in fits:
        out = al.zne(qasm, observable, chip, fit=fit)
        assert abs(out["error"]) < abs(out["unmitigated_error"]), fit


def test_shot_noise_matches_sampling_statistics():
    program = al.hardware_efficient(3, 1).bind(np.linspace(0.2, 1.4, 12))
    h = hm.Hamiltonian(3, [(1.0, {0: "Z", 1: "Z"}), (0.5, {2: "Z"}), (0.7, {0: "X"})])
    estimate = al.shot_noise(program, h, 2000)
    assert estimate["circuit_variants"] == 2
    # The Z group's variance from the exact distribution.
    probabilities = np.abs(__import__("quantum_circuit_simulator.statevector", fromlist=["statevector"]).statevector(program)) ** 2
    index = np.arange(8)
    zz = 1 - 2 * ((index ^ (index >> 1)) & 1)
    z2 = 1 - 2 * ((index >> 2) & 1)
    samples = zz + 0.5 * z2
    variance = probabilities @ samples ** 2 - (probabilities @ samples) ** 2
    group = next(g for g in estimate["groups"] if g["terms"] == 2)
    assert group["variance"] == pytest.approx(variance, abs=1e-12)
    assert estimate["value"] == pytest.approx(solve(__import__("quantum_circuit_simulator.interop", fromlist=["convert"])
                                                    .convert(program).circuit, h.terms)["value"], abs=1e-12)
    assert 0 < estimate["standard_error"] < 0.1 and math.isfinite(estimate["standard_error"])
