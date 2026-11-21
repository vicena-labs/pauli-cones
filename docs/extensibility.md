# Large logical registers and pluggable simulation engines

Pauli Cones 0.3.0a1 separates **register width** from **active causal-cone width**. The input can name 1–16,384 wires (including an arbitrary line, grid or manually provided coupling graph), but the bundled sparse native kernel still has a fixed 256-wire Pauli-key format. The new `wide-native` route uses a single bit-parallel reverse support pass over all requested observables to compute which native stages can influence each measurement. For `native`, observables whose combined cone fits one key width are grouped, each group is remapped into `[0, group_width)`, and all groups run in one native call over one shared bound copy of the circuit (see [architecture](architecture.md#preparation-and-ownership)). The per-cone planner used by the other backends remaps each cone separately, and equal cone signatures share preparation. Wires with no relevant gates are analytically contracted against the product zero state, including asymmetric Z readout.

This method is **exact in its structural cone selection** (only numerical floating-point filters remain), not an exponential-scaling breakthrough. The per-cone limit is 256 and the maximum total selected stage budget is 500,000, maximum 100,000 stages per query; each cone still has its independent Pauli-term cap (`max_terms`). If an active cone is too large, raise a `ConeOverflow` rather than silently truncate; truncation happens only when a `Truncation` is requested, and its error bound is then reported. `Circuit.prepare(..., max_cone_qubits=64)` reduces the per-cone limit for application memory policy. Deep entangling 2D circuits can exceed the cap, regardless of logical width. The optimizer does **not** implement arbitrary noisy density states, bitstring samples, or syndrome-measurement feedback.

## Example: 16,384 logical qubits

```python
from pauli_cones import Circuit
c = Circuit(16384, n_parameters=1)
c.gate('RY', [16383], theta=0.2, param=0)
p = c.prepare([(1.0, {16383: 'Z'})])
print(p.evaluate([0.1]).as_dict())
print(p.diagnostics)  # logical_qubits=16384, max_cone_qubits=1
```

`examples/large_qubit_grid.py` runs a fuller 4,096-qubit example, with gate operations on every wire and selected noisy Z observables. Both examples deliberately have narrow cones. Measure preparation and execution separately before quoting any speedup; input parsing, installation, and Vicena submission are not included in these examples.

## Register a new backend

The extension point is deliberately explicit. No plugin package is discovered, downloaded, or executed automatically. Registration executes trusted Python code inside the calling interpreter:

```python
from pauli_cones import register_backend, unregister_backend, available_backends

# A plan factory receives an immutable-to-the-caller private circuit snapshot
# (NoisyPauliCircuit), normalized Pauli observables, and explicit plan budgets.
def factory(circuit, observables, *, options):
    return MyPlan(circuit, observables, options)  # implement in your project

register_backend('ext-tensor-network', factory)
# Now Circuit(...).prepare(observables, backend='ext-tensor-network')
# passes the whole logical register, even for >256 wires, to your backend.
# MyPlan must implement bind(shifts) and evaluate(with_gradients=..., include_stats=...).
# The evaluate method returns: (value: float, gradient: vector, stats: dict).
# It must implement the declared Pauli/readout/noise and shared-parameter semantics.
unregister_backend('ext-tensor-network')
```

This contract accepts external implementations but **does not claim they exist** in the shipped distribution. For tensor-network/trajectory implementations, explicitly document physical noise channels, stochastic versus deterministic accuracy, gradient semantics, preparation costs, and resource limits. Backend injection is not available through the default JSON subprocess CLI: the CLI intentionally restricts engines to the preinstalled built-in list. Custom code should register the plugin and invoke the Python API in a trusted environment.

### Important boundaries

- `native` uses the grouped single-call solver when logical qubits >256, and so does `auto` with a `Truncation`. `graph` compiles the same grouped cones into one graph set, reporting `wide-graph`, and falls back to `wide-native` over its budgets. Exact `auto`, `dense` and `python` run on per-cone remapped plans, subject to their own limits. All report backend `wide-*`.
- `max_cone_qubits` can be less than 256, but not greater. It is enforced at every register width, including registers of 256 wires or fewer; it does not apply to `ext-*` backends. To solve truly wider interacting cones, implement a different backend rather than increasing an unsafe native mask constant.
- The default numerical engine is still limited to 16 shared rotation parameters. A new backend does not currently relax API-wide parameter constraints.
- This is research software, not a malicious-plugin execution boundary. Only register trusted factories.
- No universal performance or accuracy guarantee follows from addressable register width. Test against independent references where feasible.
