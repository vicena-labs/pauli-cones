"""Optional adapters. Unsupported models fail explicitly rather than change noise."""
from __future__ import annotations

from itertools import product
import math
import numpy as np
from .models import DenseAdjoint, local_kraus, measurement_terms


class PauliCones:
    derivative_method = "native forward analytic gradient"

    def __init__(self, case, backend="native"):
        from pauli_cones import Circuit
        circuit = Circuit(case.n_qubits, case.n_parameters, max_terms=case.max_terms)
        for op in case.operations:
            kind, wires = op["kind"], op["wires"]
            if kind == "depolarize":
                circuit.depolarize(wires, op["probability"])
            elif kind == "pauli_error":
                circuit.pauli_error(wires[0], op["axis"], op["probability"])
            elif kind == "relax":
                circuit.relax(wires[0], op["duration"], op["t1"], op["t2"])
            else:
                circuit.gate(kind, wires, theta=op.get("theta", 0.), param=op.get("param", -1))
        for q, (p01, p10) in case.readout.items():
            circuit.readout(q, p01, p10)
        self.prepared = circuit.prepare(case.observables, backend=backend)
        self.derivative_method = ("reusable graph reverse analytic gradient" if self.prepared.backend == "graph"
                                  else "native forward analytic gradient")
        if self.prepared.backend == "dense":
            self.derivative_method = "Pauli Cones bounded dense analytic gradient"

    def evaluate(self, shifts, gradients=False):
        result = self.prepared.evaluate(shifts, gradients=gradients)
        return result.value, list(result.gradient)

    def diagnostics(self):
        value = getattr(self.prepared, "diagnostics", None)
        return value() if callable(value) else value


class QulacsDensity:
    derivative_method = "exact per-gate parameter shift; shared-parameter chain rule"

    def __init__(self, case):
        if case.n_qubits > 8:
            raise NotImplementedError("dense comparator capped at eight qubits")
        from qulacs import DensityMatrix, Observable, ParametricQuantumCircuit, gate
        self.case = case
        self.state = DensityMatrix(case.n_qubits)
        self.circuit = ParametricQuantumCircuit(case.n_qubits)
        self.rotations = []
        for op in case.operations:
            kind, wires = op["kind"], op["wires"]
            if kind in ("RX", "RY", "RZ"):
                if op.get("param", -1) >= 0:
                    # Qulacs' built-in R gates use exp(+i angle P/2).
                    getattr(self.circuit, f"add_parametric_{kind}_gate")(wires[0], -op["theta"])
                    self.rotations.append(op)
                else:
                    getattr(self.circuit, f"add_{kind}_gate")(wires[0], -op["theta"])
            elif kind in ("H", "X", "Y", "Z", "CZ"):
                self.circuit.add_gate(getattr(gate, kind)(*wires))
            elif kind == "depolarize":
                channel = (gate.DepolarizingNoise(wires[0], op["probability"]) if len(wires) == 1
                           else gate.TwoQubitDepolarizingNoise(*wires, op["probability"]))
                self.circuit.add_gate(channel)
            elif kind in ("relax", "pauli_error"):
                self.circuit.add_gate(gate.CPTP([gate.DenseMatrix(wires, k) for k in local_kraus(op)]))
            else:
                raise NotImplementedError(f"Qulacs adapter does not implement {kind}")
        self.observable = Observable(case.n_qubits)
        for weight, mapping in measurement_terms(case):
            text = " ".join(f"{axis} {q}" for q, axis in sorted(mapping.items())) or "I 0"
            self.observable.add_operator(weight, text)

    def _value(self):
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        return float(self.observable.get_expectation_value(self.state))

    def evaluate(self, shifts, gradients=False):
        angles = [op["theta"] + shifts[op["param"]] for op in self.rotations]
        for index, angle in enumerate(angles):
            self.circuit.set_parameter(index, -angle)
        value = self._value()
        gradient = [0.] * self.case.n_parameters if gradients else []
        if gradients:
            for index, (op, angle) in enumerate(zip(self.rotations, angles)):
                self.circuit.set_parameter(index, -(angle + math.pi / 2))
                plus = self._value()
                self.circuit.set_parameter(index, -(angle - math.pi / 2))
                minus = self._value()
                gradient[op["param"]] += (plus - minus) / 2
                self.circuit.set_parameter(index, -angle)
        return value, gradient


def qiskit_observable(case):
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    for weight, mapping in measurement_terms(case):
        label = "".join(mapping.get(q, "I") for q in reversed(range(case.n_qubits)))
        terms.append((label, weight))
    return SparsePauliOp.from_list(terms or [("I" * case.n_qubits, 0.)])


def qiskit_circuit(case, lindblad=False, per_gate=False):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_aer.noise import PauliLindbladError, depolarizing_error, pauli_error, thermal_relaxation_error
    circuit = QuantumCircuit(case.n_qubits)
    rotations = [op for op in case.operations if op.get("param", -1) >= 0]
    parameters = ParameterVector("a", len(rotations) if per_gate else case.n_parameters)
    index = 0
    for op in case.operations:
        kind, wires = op["kind"], op["wires"]
        if kind in ("RX", "RY", "RZ"):
            theta = op["theta"]
            if op.get("param", -1) >= 0:
                theta = parameters[index] if per_gate else theta + parameters[op["param"]]
                index += 1
            getattr(circuit, kind.lower())(theta, wires[0])
        elif kind in ("H", "X", "Y", "Z", "CZ", "SX"):
            getattr(circuit, kind.lower())(*wires)
        elif kind == "depolarize":
            n = len(wires)
            scale = 1 - op["probability"] * 4**n / (4**n - 1)
            if lindblad:
                if scale <= 0:
                    raise NotImplementedError("depolarizing channel has no finite real-rate Pauli-Lindblad mapping")
                labels = ["".join(label) for label in product("IXYZ", repeat=n) if any(a != "I" for a in label)]
                channel = PauliLindbladError(labels, [-math.log(scale) / 4**n] * len(labels))
            else:
                channel = depolarizing_error(op["probability"] * 4**n / (4**n - 1), n)
            circuit.append(channel, wires)
        elif kind == "pauli_error":
            p = op["probability"]
            if lindblad:
                if p >= .5:
                    raise NotImplementedError("Pauli error has no finite nonnegative Lindblad rate")
                channel = PauliLindbladError([op["axis"]], [-.5 * math.log1p(-2 * p)])
            else:
                channel = pauli_error([("I", 1 - p), (op["axis"], p)])
            circuit.append(channel, wires)
        elif kind == "relax":
            if lindblad:
                raise NotImplementedError("pauli-prop adapter supports Pauli-Lindblad channels; T1 relaxation is nonunital")
            circuit.append(thermal_relaxation_error(op["t1"], op["t2"], op["duration"], 0), wires)
        else:
            raise NotImplementedError(f"unsupported adapter operation {kind}")
    return circuit, parameters, rotations


class AerDensity:
    derivative_method = "exact per-gate parameter shift; shared-parameter chain rule"

    def __init__(self, case):
        if case.n_qubits > 8:
            raise NotImplementedError("dense comparator capped at eight qubits")
        from qiskit_aer import AerSimulator
        self.case = case
        self.backend = AerSimulator(method="density_matrix", max_parallel_threads=1,
                                    max_parallel_experiments=1, max_parallel_shots=1)
        self.circuit, self.parameters, self.rotations = qiskit_circuit(case, per_gate=True)
        self.circuit.save_expectation_value(qiskit_observable(case), list(range(case.n_qubits)), label="value")

    def _value(self, angles):
        bound = self.circuit.assign_parameters(dict(zip(self.parameters, angles)))
        result = self.backend.run(bound, shots=1).result()
        if not result.success:
            raise RuntimeError(str(result.status))
        return float(result.data(0)["value"])

    def evaluate(self, shifts, gradients=False):
        angles = [op["theta"] + shifts[op["param"]] for op in self.rotations]
        value = self._value(angles)
        gradient = [0.] * self.case.n_parameters if gradients else []
        if gradients:
            for index, (op, angle) in enumerate(zip(self.rotations, angles.copy())):
                angles[index] = angle + math.pi / 2
                plus = self._value(angles)
                angles[index] = angle - math.pi / 2
                minus = self._value(angles)
                gradient[op["param"]] += (plus - minus) / 2
                angles[index] = angle
        return value, gradient


class IBMPauliProp:
    derivative_method = "exact per-gate parameter shift of compiled Pauli rotations; shared chain rule"

    def __init__(self, case):
        from pauli_prop.propagation import circuit_to_rotation_gates, evolve_through_cliffords
        from qiskit.quantum_info import SparsePauliOp
        self.case = case
        circuit, parameters, _ = qiskit_circuit(case, lindblad=True)
        self.clifford, transformed = evolve_through_cliffords(circuit)
        symbolic = circuit_to_rotation_gates(transformed)
        zero = {p: 0. for p in parameters}
        # Parse structure once, including zero-angle paths. PauliEvolutionGate
        # retains its symbolic generator, so a zero base angle cannot erase it.
        self.bases, self.slopes = [], []
        for theta in symbolic.thetas:
            if hasattr(theta, "parameters") and theta.parameters:
                base = float(theta.bind({p: zero[p] for p in theta.parameters}))
                slopes = []
                for parameter in parameters:
                    if parameter not in theta.parameters:
                        slopes.append(0.)
                    else:
                        values = {p: float(p == parameter) for p in theta.parameters}
                        slopes.append(float(theta.bind(values)) - base)
            else:
                base, slopes = float(theta), [0.] * len(parameters)
            self.bases.append(base)
            self.slopes.append(slopes)
        self.bases = np.asarray(self.bases)
        self.slopes = np.asarray(self.slopes).reshape(len(self.bases), case.n_parameters)
        self.rotations = symbolic._replace(thetas=self.bases.tolist())
        self.observable = qiskit_observable(case)
        self.SparsePauliOp = SparsePauliOp
        self.max_discarded_l1 = 0.

    def _value(self, angles):
        from pauli_prop.propagation import propagate_through_rotation_gates
        self.rotations.thetas[:] = angles
        evolved, discarded_l1 = propagate_through_rotation_gates(
            self.observable, self.rotations, max_terms=self.case.max_terms, atol=0., frame="h")
        self.max_discarded_l1 = max(self.max_discarded_l1, float(discarded_l1))
        evolved = self.SparsePauliOp(evolved.paulis.evolve(self.clifford, frame="h"),
                                    evolved.coeffs, copy=False, ignore_pauli_phase=False)
        return float(np.sum(evolved.coeffs[~np.any(evolved.paulis.x, axis=1)]).real)

    def evaluate(self, shifts, gradients=False):
        active = np.flatnonzero(np.any(self.slopes != 0, axis=1))
        if gradients and len(active) > 48:
            raise NotImplementedError("benchmark limits per-gate gradient to 48 variable occurrences")
        if self.case.n_qubits > 8 and gradients:
            raise NotImplementedError("large truncated stress case is value-only in this bounded benchmark")
        angles = (self.bases + self.slopes @ np.asarray(shifts)).tolist()
        value = self._value(angles)
        gradient = np.zeros(self.case.n_parameters) if gradients else np.empty(0)
        if gradients:
            for index in active:
                angle = angles[index]
                angles[index] = angle + math.pi / 2
                plus = self._value(angles)
                angles[index] = angle - math.pi / 2
                minus = self._value(angles)
                gradient += self.slopes[index] * (plus - minus) / 2
                angles[index] = angle
        return value, gradient.tolist()

    def diagnostics(self):
        return {"max_discarded_coefficient_l1": self.max_discarded_l1,
                "clifford_structure_reused": True, "rotation_structure_reused": True}


def prepare(name, case):
    if name in ("native", "graph", "dense"):
        return PauliCones(case, backend=name)
    return {"dense_adjoint": DenseAdjoint, "qulacs": QulacsDensity,
            "aer": AerDensity, "pauli_prop": IBMPauliProp}[name](case)
