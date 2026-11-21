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

The compile report's `layout` is `exact` when every interacting pair sits on a coupler, so no SWAPs are needed; otherwise it is `greedy`. Observables keep the circuit's own qubit numbers; they are moved to the final layout for you. `--noise-scale` multiplies every error and decoherence rate, so 0 is noise-free and repeating a run at several scales is a noise sweep. In Python: `from pauli_cones.compiler import compile_circuit`, then `compile_circuit(circuit, chip).expectation("Z0*Z1")`.

## Sweeps

Compare chips, noise levels and sizes in one command; each row gives the value, the ideal noise-free value, the deviation, the method used, and the routing cost:

```bash
pauli-cones sweep --family qaoa-ring --sizes 8,12,16 --family-param p=2 \
    --device heavy-hex-127 --device "all-to-all-ions:n=16" --noise-scales 0,0.5,1,2 --format markdown
pauli-cones sweep circuit.qasm --observable "Z0*Z5" --device my_chip.json --noise-scales 1,2
```

Built-in families, each with its own observable:
- `ghz`;
- `hardware-efficient`;
- `qaoa-ring` (MaxCut);
- `ising-trotter`.

The solver is chosen per point: exact sparse propagation, then exact dense for cones of at most 6 wires, then truncation at 1e-4 and 1e-5. A truncated row has `exact: false`, an `error_bound`, and the `convergence_change` between the two tolerances. A chip too small for a size gives a row with `error` instead of stopping the sweep. In Python: `from pauli_cones.experiments import sweep, solve`.

## Designing chips

Build a new chip from a catalogue layout, then compare it with existing chips:

```bash
pauli-cones design --base "heavy-hex:rows=3,row_length=7" --modules 4 --links ring --link-error 0.02 \
    --remove-qubits 5,17 --t1 3e-4 --t2 2e-4 --time-1q 3e-8 --time-2q 2e-7 --control-error-2q 2e-3 \
    --name modular-hh --out modular.json
pauli-cones compare modular.json heavy-hex-127 "square-grid:rows=6,columns=6" --family qaoa-ring --sizes 8,12 \
    --format markdown
```

Design options:
- `--modules` tiles copies of the base and joins them in a line, ring or all-to-all. Links can have their own `--link-error`, usually worse than on-module couplers.
- `--remove-qubits` removes defective qubits and renumbers the rest.
- `--add-couplers` and `--remove-couplers` edit connectivity.
- Giving all of `--t1`, `--t2`, `--time-1q` and `--time-2q` derives gate errors from the exact coherence limit, plus `--control-error-1q/2q` for errors coherence does not capture.

More options:
- `--qubit-spread 0.2` gives each qubit its own one-qubit error, readout, T1 and T2, spread log-normally around the chip values as on a real chip.
- `--spectator-error` and `--zz-hz` add crosstalk. Neighbours of every two-qubit gate take a depolarizing error. Every coupled pair of active qubits picks up a ZZ conditional phase of 2π·ζ·t for as long as both are active, idle or not, except during their own two-qubit gate.
- Coherent ZZ multiplies the work of exact simulation. Sweeps and `solve` then switch to truncation and report its convergence; expect seconds rather than milliseconds per point.
- `--expand-to N` grows a chip to at least N qubits with the same architecture: heavy-hex stays heavy-hex, grids keep their aspect ratio, and modular chips add modules.

In a sweep, `--device fit:heavy-hex-127` grows the chip to fit each size.

The result is an ordinary device file: use it anywhere a device goes (`--device modular.json`). `compare` ranks chips by mean deviation from the ideal value on a benchmark family, with mean SWAPs, two-qubit gates and duration. In Python, `pauli_cones.design` has the same operations.

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
