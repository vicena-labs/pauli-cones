"""Large logical registers using exact backward causal-cone remapping.

The native Pauli key is 256 physical bits. A wide register may be much larger,
provided every *requested observable's backward causal cone* fits that bound.
No tensor product of disjoint states is constructed or approximated here.
"""
from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import numpy as np
from ._reference import ConeOverflow


class WidePrepared:
    """Plan over an arbitrarily numbered register, using compact local engines.

    Each observable receives its own exact backward light cone. Observables
    sharing a stage sequence and physical support use one inner prepared plan.
    Shared parameter indices and original chronological stage order are kept.
    """

    def __init__(self, circuit, observables, *, backend, graph_max_nodes,
                 graph_max_edges, dense_max_qubits, dense_max_bytes,
                 max_cone_qubits=256, max_cone_stages=100_000,
                 max_total_plan_stages=500_000):
        from ..api import Circuit, Prepared
        self.circuit = circuit
        self.nparams = circuit.nparams
        self.groups = []
        self.backend = backend
        self.constant = 0.
        if not 1 <= max_cone_qubits <= 256:
            raise ValueError('max_cone_qubits must be in 1..256')
        stages = circuit.stages
        grouped = defaultdict(list)
        largest_cone = 0
        total_plan_stages = 0
        # Propagate all observable supports in ONE reverse graph pass.
        # Each integer bit represents one query, independent of the physical
        # register width. This avoids O(number of observables * stages) scans.
        queries = tuple(observables)
        wire_queries = [0] * circuit.nqubits
        for i, (_, axes) in enumerate(queries):
            bit = 1 << i
            for q in axes:
                wire_queries[q] |= bit
        membership = [0] * len(stages)
        for i in range(len(stages) - 1, -1, -1):
            wires = stages[i].wires
            active = wire_queries[wires[0]]
            if len(wires) == 2:
                active |= wire_queries[wires[1]]
                wire_queries[wires[1]] = active
            wire_queries[wires[0]] = active
            membership[i] = active

        cone_wires = [set(axes) for _, axes in queries]
        per_query_indices = [[] for _ in queries]
        selection_count = 0
        for i, bitmask in enumerate(membership):
            if not bitmask:
                continue
            wires = stages[i].wires
            while bitmask:
                bit = bitmask & -bitmask
                query = bit.bit_length() - 1
                per_query_indices[query].append(i)
                cone_wires[query].update(wires)
                if len(cone_wires[query]) > max_cone_qubits:
                    raise ConeOverflow(f'backward cone needs >{max_cone_qubits} qubits; '
                                       'the wide-register engine cannot represent it exactly')
                if len(per_query_indices[query]) > max_cone_stages:
                    raise ConeOverflow('backward cone stage budget exceeded')
                selection_count += 1
                if selection_count > max_total_plan_stages:
                    raise ConeOverflow('wide preparation total selected stages exceeded')
                bitmask ^= bit

        idle_terms = 0
        for (coefficient, axes), indices, active in zip(queries, per_query_indices, cone_wires):
            if not indices:
                # Contraction for idle product inputs includes asymmetric
                # readout, and contributes zero gradient for every parameter.
                scalar = float(coefficient)
                for q, axis in axes.items():
                    if axis in ('X', 'Y'):
                        scalar = 0.
                        break
                    if axis == 'Z' and q in circuit.readout:
                        scalar *= 1. - 2.*circuit.readout[q][0]
                self.constant += scalar
                idle_terms += 1
                continue
            wires = tuple(sorted(active))
            largest_cone = max(largest_cone, len(wires))
            grouped[(tuple(indices), wires)].append((coefficient, axes))
        self.total_groups = len(grouped)
        for (indices, wires), items in grouped.items():
            total_plan_stages += len(indices)
            if total_plan_stages > max_total_plan_stages:
                raise ConeOverflow('wide preparation total stage budget exceeded')
            translation = {old: new for new, old in enumerate(wires)}
            local = Circuit(len(wires), self.nparams, circuit.max_terms)
            local._impl.stages = [replace(stages[i], wires=tuple(translation[q] for q in stages[i].wires))
                                  for i in indices]
            # Readout only acts on the requested observable support. Other
            # readout definitions act on identity and cannot influence result.
            local._impl.readout = {translation[q]: pair for q, pair in circuit.readout.items()
                                   if q in translation}
            remapped = [(coef, {translation[q]: axis for q, axis in axes.items()})
                        for coef, axes in items]
            plan = Prepared(local, remapped, backend=backend,
                            graph_max_nodes=graph_max_nodes, graph_max_edges=graph_max_edges,
                            dense_max_qubits=dense_max_qubits, dense_max_bytes=dense_max_bytes)
            self.groups.append(plan)
        self.stats = {'logical_qubits': circuit.nqubits,
                      'max_cone_qubits': largest_cone,
                      'unique_cone_groups': self.total_groups,
                      'total_prepared_cone_stages': total_plan_stages,
                      'idle_terms_contracted': idle_terms}

    def bind(self, shifts):
        self._shifts = tuple(shifts)
        return self

    def evaluate(self, *, include_stats=False, with_gradients=True):
        value = self.constant
        gradient = np.zeros(self.nparams if with_gradients else 0, dtype=float)
        peak = 0
        fallback = []
        actual = set()
        for plan in self.groups:
            result = plan.evaluate(self._shifts, gradients=with_gradients)
            actual.add(result.backend)
            value += result.value
            if with_gradients:
                gradient += np.asarray(result.gradient, dtype=float)
            d = plan.diagnostics
            peak = max(peak, d.get('max_single_cone_terms', d.get('max_terms', 0)))
            if d.get('fallback_reason'):
                fallback.append(d['fallback_reason'])
        actual_backend = 'wide-mixed' if len(actual) > 1 else 'wide-' + next(iter(actual), 'constant')
        statistics = dict(self.stats, max_single_cone_terms=peak,
                          fallback_reasons=fallback, backend=actual_backend,
                          underlying_backends=tuple(sorted(actual)))
        if include_stats:
            return value, gradient, statistics
        return value, gradient
