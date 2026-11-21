# Pauli Cones: guide for agents

Pauli Cones computes expectation values of Pauli observables on noisy quantum circuits that start in |0...0>, and their exact gradients in up to 16 shared parameters. It propagates the observable backward through its light cone, so cost depends on the cone, not the register: a local observable on a 1,000-qubit shallow circuit is cheap, while a deep, noiseless, highly entangling circuit may be refused or need truncation.

## When to use it

- **Use it for:** expectation values and gradients of noisy or shallow circuits, VQE energies, local observables on many qubits, and sweeps over many parameter vectors.
- **Use something else for:**
  - sampling bitstrings, or quantities other than Pauli expectations;
  - mid-circuit measurement, resets or classical control;
  - initial states other than |0...0>;
  - deep noiseless circuits where `plan` shows wide cones and exact answers are required. For those, a tensor-network or statevector simulator may be faster.

## Five commands

Every command prints one JSON document. Exit code 0 is success, 2 an invalid request or failed evaluation, and 124 a timeout. On failure, read `error.hint`: it names the next thing to try.

```bash
pauli-cones capabilities                                        # gates, noise, limits, backends
pauli-cones plan circuit.qasm --observable "Z0*Z1"              # dry run: is exact feasible?
pauli-cones run circuit.qasm --observable "0.5*Z0*Z1 - X3"      # value and gradients
pauli-cones run circuit.qasm --observable "Z5" --min-coefficient 1e-4   # approximate, with a bound
pauli-cones schema                                              # JSON request format
```

Observables are written with wire indices, such as `"0.5*Z0*Z1 - X3"`, or as Qiskit labels, such as `"0.5*IXZ"`, where the rightmost character is wire 0. OpenQASM 2 is read directly; OpenQASM 3 needs Qiskit. Add noise to an OpenQASM circuit with `--noise-1q 0.001 --noise-2q 0.01 --readout 0.02,0.03`.

## Chips

`pauli-cones devices` lists the chip catalogue:
- superconducting heavy-hex (including the 127-qubit IBM Eagle-style layout), square-grid and line chips;
- all-to-all trapped ions;
- neutral-atom arrays;
- D-Wave-style Chimera, Pegasus and Zephyr annealer graphs.

`pauli-cones device square-grid --param rows=6 --param columns=6` describes one chip, and `--edges` prints its full JSON description, which is also the format for a custom chip.

Add `--device heavy-hex-127` (or `--device my_chip.json`) to `run` or `plan` to apply the chip's gate and readout errors and check its connectivity. The response's `device.uncoupled_two_qubit_operations` counts gates on qubit pairs the chip does not couple. Catalogue noise values are typical figures for each technology, not calibration data; scale them for noise studies in Python with `chip.scaled_noise(2.0)`.

To run a circuit on a chip, add `--compile`:

```bash
pauli-cones compile circuit.qasm --device heavy-hex-127                       # SWAPs, depth, duration, layouts
pauli-cones run circuit.qasm --observable "Z0*Z5" --device heavy-hex-127 --compile --noise-scale 2
```

Compiling does four things:
1. Places the circuit's qubits on the chip.
2. Inserts SWAPs so every two-qubit gate acts on coupled qubits. `--router sabre` uses Qiskit and usually needs fewer.
3. Schedules gates with the chip's durations.
4. Simulates gate errors (a SWAP counts as three two-qubit gates), T1/T2 decay while qubits idle, and readout error on the measured Paulis.

Observables keep the circuit's own qubit numbers; they are moved to the final layout for you. `--noise-scale` multiplies every error and decoherence rate, so 0 is noise-free and repeating a run at several scales is a noise sweep. In Python: `from pauli_cones.compiler import compile_circuit`, then `compile_circuit(circuit, chip).expectation("Z0*Z1")`.

## Recipe

1. **Plan.** `pauli-cones plan`: if `exact_guaranteed` is true, exact evaluation cannot exceed the term cap.
2. **Run exactly.** `pauli-cones run`. A `ConeOverflow` error means the exact expansion exceeded the cap. It never returns a wrong answer.
3. **Truncate if needed.** Rerun with `--min-coefficient 1e-4`, then `1e-5`. Accept the value once it stops changing between the two. Each result's `error_bound` is a rigorous worst case on |exact − returned|; it is usually far larger than the real error, so the convergence check is the practical test.
4. **Sweep parameters.** For many parameter vectors of one circuit, use `--backend graph`. It compiles once and is much faster per vector; put up to 128 vectors in one JSON request's `parameter_vectors`.
5. **Small cones.** When every cone has at most 6 wires, `--backend dense` is exact regardless of depth.

## Reading a result

- `results[i].value`: the expectation of the whole weighted observable at parameter vector i.
- `results[i].gradient`: d value / d parameter, one entry per shared parameter.
- `results[i].error_bound` and `gradient_error_bound`: present only for truncated runs.
- `backend`: the method actually used. `diagnostics.fallback_reason` explains any fallback.
- `diagnostics`: cone sizes and term counts. `numerics`: the cutoffs in force.

## Python

```python
from pauli_cones.interop import expectation
r = expectation(open("circuit.qasm").read(), "0.5*Z0*Z1 - X3")         # or a Qiskit QuantumCircuit
r = expectation(qiskit_circuit, "Z0", values={"theta": 0.3}, gradients=True)
print(r.value, r.gradient, r.error_bound)
```

As a drop-in Qiskit primitive: `from pauli_cones.estimator import Estimator`, then use it wherever a Qiskit `EstimatorV2` is expected. To simulate on a chip, use `from pauli_cones import devices`, then `chip = devices.get("all-to-all-ions", n=20)`, and pass `device=chip` to `expectation` or `Estimator`.

For repeated evaluation, convert once and reuse the plan:

```python
from pauli_cones.interop import convert, parse_observable
c = convert(qiskit_circuit)                       # Qiskit Parameters become shared parameters
plan = c.circuit.prepare(parse_observable("Z0*Z1"), backend="graph")
for values in sweep:
    print(plan.evaluate(c.shifts(values)).value)
```

## Verify before reporting

- **Cross-check small cases:** on a reduced instance (at most about 20 qubits), compare against an independent simulator, or against `--backend python` or `dense`.
- **Report truncation fully:** for truncated results, state the threshold and the convergence check, not only the bound.
- **Mind the noise convention:** noise probabilities are total non-identity Pauli error probabilities. Qiskit Aer's depolarizing parameter λ equals p·4^k/(4^k−1) for k wires.
