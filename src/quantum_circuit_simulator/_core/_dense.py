"""Bounded local density-matrix backend with analytic reverse-mode gradients.

This is an alternate representation, not a claim that dense simulation is
generally faster. Only cones with identical stage sets share forward states.
Channel adjoints are used in reverse mode; noisy channels are never inverted.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ._reference import ConeOverflow, P, P2, _idle_observable


class DenseLimitExceeded(ConeOverflow):
    """A local density cone exceeds its declared qubit or working-memory budget."""


@dataclass
class _Plan:
    indices: tuple[int, ...]
    wires: tuple[int, ...]
    operator: np.ndarray
    axes: dict[int, tuple[tuple[int, ...], tuple[int, ...]]]
    estimated_bytes: int


class DensePrepared:
    def __init__(self, circuit, observables, *, max_qubits=6, max_bytes=64*1024*1024):
        self.c = circuit
        self.bound = circuit
        self.max_qubits = max_qubits
        self.max_bytes = max_bytes
        groups = {}
        touched = {q for stage in circuit.stages for q in stage.wires}
        for term in observables:
            coefficient, ops = _idle_observable(term, touched, circuit.readout)
            if coefficient == 0.:
                continue
            active = set(ops)
            indices = []
            for index in range(len(circuit.stages)-1, -1, -1):
                stage = circuit.stages[index]
                if active.intersection(stage.wires):
                    active.update(stage.wires)
                    indices.append(index)
            wires = tuple(sorted(active))
            indices = tuple(reversed(indices))
            if len(wires) > max_qubits:
                raise DenseLimitExceeded(
                    f'dense cone needs {len(wires)} qubits; limit is {max_qubits}')
            groups.setdefault((indices, wires), []).append((coefficient, ops))

        self.plans = []
        total_operator_bytes = 0
        peak_working_bytes = 0
        for (indices, wires), terms in groups.items():
            n = len(wires)
            dimension = 1 << n
            matrix_bytes = dimension * dimension * 16
            # Reverse AD saves only states just before parameterized stages.
            parameter_stages = sum(circuit.stages[i].param >= 0 for i in indices)
            # Conservative allowance for states, transposes and NumPy temporaries.
            working_bytes = (parameter_stages + 12) * matrix_bytes + 32768
            total_operator_bytes += matrix_bytes
            peak_working_bytes = max(peak_working_bytes, working_bytes)
            estimate = total_operator_bytes + peak_working_bytes
            if estimate > max_bytes:
                raise DenseLimitExceeded(
                    f'dense arrays need an estimated {estimate} bytes; limit is {max_bytes}')
            position = {wire: i for i, wire in enumerate(wires)}
            operator = np.zeros((dimension, dimension), complex)
            for coefficient, ops in terms:
                # Readout acts on the measured Z, not on the evolving density.
                tensor = np.ones((1, 1), complex)
                for wire in reversed(wires):
                    axis = ops.get(wire, 'I')
                    local = P[{'I': 0, 'X': 1, 'Z': 2, 'Y': 3}[axis]]
                    if axis == 'Z' and wire in circuit.readout:
                        p01, p10 = circuit.readout[wire]
                        local = (p10-p01)*P[0] + (1-p01-p10)*P[2]
                    tensor = np.kron(tensor, local)
                operator += coefficient * tensor
            axes = {}
            for i in indices:
                local_wires = tuple(position[q] for q in circuit.stages[i].wires)
                selected = tuple(n-1-q for q in reversed(local_wires))
                selected += tuple(2*n-1-q for q in reversed(local_wires))
                order = selected + tuple(j for j in range(2*n) if j not in selected)
                inverse = tuple(np.argsort(order))
                axes[i] = (order, inverse)
            self.plans.append(_Plan(indices, wires, operator, axes, working_bytes))
        self.stats = {
            'representation': 'local_density_matrix',
            'density_groups': len(self.plans),
            'max_cone_qubits': max((len(p.wires) for p in self.plans), default=0),
            'estimated_array_bytes': total_operator_bytes + peak_working_bytes,
            'dense_max_bytes': max_bytes,
            'gradient_method': 'reverse_channel_adjoint',
        }

    def bind(self, shifts):
        self.bound = self.c.shifted(shifts)
        return self

    def evaluate(self, *, with_gradients=True, include_stats=False):
        total = 0.
        gradient = np.zeros(self.c.nparams if with_gradients else 0)
        for plan in self.plans:
            dimension = 1 << len(plan.wires)
            rho = np.zeros((dimension, dimension), complex)
            rho[0, 0] = 1.
            before = {}
            for index in plan.indices:
                stage = self.bound.stages[index]
                if with_gradients and stage.param >= 0:
                    before[index] = rho
                rho = _apply(rho, _superoperator(stage.matrix), plan.axes[index])
            total += float(np.vdot(plan.operator, rho).real)
            if with_gradients and self.c.nparams:
                adjoint = plan.operator
                for index in reversed(plan.indices):
                    stage = self.bound.stages[index]
                    if stage.param >= 0:
                        tangent = _apply(before.pop(index), _superoperator(stage.derivative),
                                         plan.axes[index])
                        gradient[stage.param] += float(np.vdot(adjoint, tangent).real)
                    adjoint = _apply(adjoint, _superoperator(stage.matrix).conj().T,
                                     plan.axes[index])
        if include_stats:
            return total, gradient, dict(self.stats)
        return total, gradient


def _superoperator(transfer):
    """Convert the local Heisenberg PTM to a Schrödinger density map."""
    basis = np.asarray(P if len(transfer) == 4 else P2)
    dimension = basis.shape[1]
    return (np.einsum('ab,aij,blk->ijkl', transfer, basis, basis,
                      optimize=True) / dimension).reshape(dimension**2, dimension**2)


def _apply(matrix, operator, axes):
    order, inverse = axes
    tensor = matrix.reshape((2,) * len(order))
    moved = tensor.transpose(order).reshape(operator.shape[1], -1)
    result = (operator @ moved).reshape((2,) * len(order)).transpose(inverse)
    return result.reshape(matrix.shape)
