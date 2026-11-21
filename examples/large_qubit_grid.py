"""Runnable many-qubit circuit: all wires participate, small local Z cones.

Example: python examples/large_qubit_grid.py --rows 64 --columns 64 --observables 128
This is NOT an arbitrary full-state simulation on 4096 qubits.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pauli_cones import Circuit, Topology


def run(rows: int, columns: int, observable_count: int = 128, *, noise: float = .005,
        vertical: bool = False):
    grid = Topology.grid(rows, columns)
    n = grid.n_qubits
    if not 1 <= observable_count <= min(n, 1024):
        raise ValueError('observables must be in 1..min(qubits, 1024)')
    c = Circuit(n, n_parameters=1, max_terms=2000)
    # A parameterized layer on EVERY physical wire. This gate schedule has
    # shallow independent horizontal pairs, not an entangling whole-grid layer.
    for q in range(n):
        c.gate('RY', [q], theta=.2, param=0)
    pairs = 0
    for r in range(rows):
        for col in range(0, columns - 1, 2):
            q = r*columns + col
            c.gate('CZ', [q, q+1])
            if noise:
                c.depolarize([q,q+1], noise)
            pairs += 1
    vertical_pairs = 0
    if vertical:
        for r in range(0, rows-1, 2):
            for col in range(columns):
                q = r*columns + col
                c.gate('CZ', [q, q+columns])
                if noise:
                    c.depolarize([q, q+columns], noise)
                vertical_pairs += 1
    selected = sorted({int(i*(n-1)/max(1,observable_count-1))
                       for i in range(observable_count)})
    terms = [(1./len(selected), {q:'Z'}) for q in selected]
    t0 = time.perf_counter()
    prepared = c.prepare(terms, backend='native')
    prep = time.perf_counter()-t0
    t1 = time.perf_counter()
    output = prepared.evaluate([.1])
    execution = time.perf_counter()-t1
    # Shallow circuit only. The output is an actual noisy Pauli expectation,
    # not a substitute for a full 2**n state or a hardware calibration.
    return {
        'logical_qubits': n, 'horizontal_pairs': pairs, 'vertical_pairs': vertical_pairs,
        'observable_terms': len(selected),
        'mean_z': output.value, 'shared_gradient': list(output.gradient),
        'backend': output.backend,
        'preparation_seconds': round(prep,6),
        'evaluation_seconds': round(execution,6),
        'diagnostics': prepared.diagnostics,
        'not_full_state_simulation': True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--columns', type=int, default=64)
    parser.add_argument('--observables', type=int, default=128)
    parser.add_argument('--vertical', action='store_true', help='Add disjoint vertical CZ/noise matching')
    args=parser.parse_args()
    print(json.dumps(run(args.rows,args.columns,args.observables,vertical=args.vertical),indent=2))


if __name__ == '__main__':
    main()
