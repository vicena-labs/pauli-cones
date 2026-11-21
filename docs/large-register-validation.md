# Wide-register qualification — 0.3.0a1

Status: local development qualification on November 19–20, 2026; **not** an external specialist comparison or a Vicena deployment test.

The package now accepts **up to 16,384 logical qubits**. In the built-in engine, this does not enlarge the native Pauli key: its four 64-bit masks still support a maximum of 256 **active wires per backward causal cone**. A batched reverse dependency pass identifies the structurally relevant stages for each observable, remaps each cone (for `native`, each group of cones) to a compact local register, and contracts idle product-state wires. Resource limits still apply to cone width, selected stages, and Pauli terms.

## Executed checks

- 4,096-qubit circuit, 64×64 grid, `RY` on every wire, a horizontal and vertical disjoint `CZ` matching with nonzero two-wire depolarizing noise, 64 selected Z observables and complete one-parameter gradient: value `0.9451734072572915`, derivative `-0.2923763970317192`, largest cone 4 qubits. Local preparation `~0.156 s`; evaluation `~0.0033 s` in one illustrative execution. **These are not fresh-process or matched-baseline benchmarks.**
- 16,384-qubit circuit, 128×128 grid, `RY` on every wire, disjoint horizontal noisy `CZ` pairs, 32 selected Z observables: value `0.9502413611836033`, derivative `-0.29394409889247913`, largest cone 2 qubits. Local preparation `~0.298 s`, evaluation `~0.0015 s` in one illustrative execution.
- Independent correctness equivalence on large logical registers versus compact-register native evaluations for noisy two-wire gates and gradients; independent analytic checks for shallow noisy disjoint pairs; tests for X/Y/Z, asymmetric readout, shared parameters, refusal on oversized cones, 16,384-qubit upper limit, and third-party backend registration.

## What is not proven

- No arbitrary full density/statevector/sampling output on 4,096 or 16,384 qubits.
- No tractability claim for deep two-dimensional random circuits, long-range entanglement, or large (>256-wire) structural cones; those fail closed unless a suitable **external** backend is registered.
- No proof of a broad or universal speedup over Aer, Qulacs, Julia Pauli propagation, cuPauliProp or tensor/trajectory engines in this new width regime.
- No live Vicena agent/Science Computer/UI round trip, no sustained throughput measurements and no GPU benchmarking in this package.

## Reproduce

```bash
python -m pytest -q tests/test_wide_architecture.py
python examples/large_qubit_grid.py --rows 64 --columns 64 --observables 64 --vertical
python examples/large_qubit_grid.py --rows 128 --columns 128 --observables 32
```

The examples measure only in-process plan preparation and one local evaluate operation. Times are illustrative and depend on host, interpreter, CPU thread allocation and other load. Re-run on the target host and report accuracy and complete cold-process latency before a performance comparison.
