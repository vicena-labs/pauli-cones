"""Compile circuits onto a chip: layout, routing, scheduling and device noise.

    from pauli_cones import devices
    from pauli_cones.compiler import compile_circuit
    compiled = compile_circuit(qasm_or_qiskit_circuit, devices.get("heavy-hex-127"))
    compiled.report                     # SWAPs, two-qubit gates, depth, duration, layouts
    compiled.expectation("Z0*Z1")       # observables use the circuit's own (logical) qubits

Steps:
1. Layout places logical qubits on physical ones. "greedy" puts qubits
   that interact often on nearby qubits; "trivial" keeps the numbering;
   a list gives the physical qubit of each logical one.
2. Routing inserts SWAPs so every two-qubit gate acts on coupled qubits.
   "greedy" moves one qubit along a shortest path; "sabre" uses Qiskit's
   SABRE router, usually with fewer SWAPs, when Qiskit is installed.
3. Scheduling starts each gate as soon as its qubits are free, using the
   chip's gate durations (RZ is virtual and takes no time).
4. Noise follows each physical gate: the chip's one-qubit error after
   every one-qubit gate except RZ, its two-qubit error after every
   two-qubit gate (a SWAP is three), T1/T2 relaxation for every idle
   interval of a qubit that has started, and readout error on every
   measured Pauli. `noise_scale` multiplies every error and decoherence
   rate, so 0 is noise-free.

The result is exact for the routed circuit with the chosen noise model.
Routing is heuristic: the SWAP count is not minimal.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .api import Circuit
from .interop import (ConversionError, Converted, _DECOMPOSE, _emit, _lower, parse_observable, readout_expand,
                      source_instructions)

TWO_QUBIT = {"CZ", "ECR"}


@dataclass
class Compiled:
    """A circuit compiled onto a device, with its report and layouts."""
    circuit: Circuit
    device: object
    initial_layout: list
    final_layout: list
    parameters: list = field(default_factory=list)
    source_order: list | None = None
    report: dict = field(default_factory=dict)
    readout: tuple | None = None

    def observable(self, spec):
        """Observable terms on the final physical qubits, with the device's readout error applied."""
        n_logical = len(self.final_layout)
        moved = [(c, {self.final_layout[q]: a for q, a in p.items()}) for c, p in parse_observable(spec, n_logical)]
        return readout_expand(moved, self.readout)

    def shifts(self, values):
        return Converted(self.circuit, self.parameters, self.source_order).shifts(values)

    def expectation(self, observable, *, values=None, gradients=False, backend="native", truncation=None,
                    threads=1):
        plan = self.circuit.prepare(self.observable(observable), backend=backend, truncation=truncation,
                                    threads=threads)
        return plan.evaluate(self.shifts(values) if self.parameters else None, gradients=gradients)


def compile_circuit(circuit, device, *, layout="greedy", router="greedy", noise=True, idle_noise=True,
                    noise_scale=1.0, values=None, max_terms=1_000_000):
    """Compile OpenQASM text, a Qiskit circuit or a gates-only Pauli Cones Circuit onto `device`."""
    if device.family == "annealer":
        raise ConversionError(f"{device.name} is an annealer; circuits cannot be compiled onto it",
                              "choose a gate-model device from `pauli-cones devices`")
    if router == "sabre":
        n, physical_instructions, source_order, initial, final, swaps = _sabre(circuit, device, layout, values)
        ops, params = _lower(physical_instructions)
        routed_ops = ops
    elif router == "greedy":
        n, instructions, source_order = _instructions(circuit, values)
        if n > device.n_qubits:
            raise ConversionError(f"the circuit has {n} qubits; {device.name} has {device.n_qubits}",
                                  "choose a larger device")
        ops, params = _lower(instructions)
        initial = _layout(layout, n, ops, device)
        routed_ops, final, swaps = _route(ops, initial, device)
    else:
        raise ValueError('router must be "greedy" or "sabre"')
    gate_noise = device.noise_model(noise_scale) if noise else None
    scheduled, facts = _schedule(routed_ops, device, gate_noise, idle_noise and noise and noise_scale > 0,
                                 noise_scale if noise else 0.0)
    readout = gate_noise.readout if gate_noise else None
    converted = _emit(device.n_qubits, scheduled, params, None, max_terms)  # readout acts on observables
    report = {"device": device.name, "router": router, "layout": layout if isinstance(layout, str) else "given",
              "logical_qubits": n, "swaps": swaps, **facts, "noise": bool(noise), "idle_noise": bool(idle_noise and noise),
              "noise_scale": noise_scale, "initial_layout": initial, "final_layout": final}
    return Compiled(converted.circuit, device, initial, final, params, source_order, report, readout)


def _instructions(circuit, values):
    if isinstance(circuit, Circuit):
        instructions = []
        for stage in circuit._impl.stages:
            if stage.name not in {"H", "SX", "X", "Y", "Z", "RX", "RY", "RZ", "CZ", "ECR"}:
                raise ConversionError(f"cannot compile a {stage.name} stage", "compile a gates-only circuit; "
                                      "the device adds noise")
            if stage.param >= 0:
                raise ConversionError("compiling a Pauli Cones Circuit with shared parameters is not supported",
                                      "compile the OpenQASM or Qiskit form instead")
            angles = [stage.angle] if stage.name in {"RX", "RY", "RZ"} else []
            instructions.append((stage.name.lower(), list(stage.wires), angles))
        return circuit.n_qubits, instructions, None
    return source_instructions(circuit, values)


# ---------------------------------------------------------------- layout
def _layout(layout, n, ops, device):
    if isinstance(layout, (list, tuple)):
        if len(layout) != n or len(set(layout)) != n or not all(0 <= p < device.n_qubits for p in layout):
            raise ValueError(f"layout must give {n} distinct physical qubits of {device.name}")
        return list(layout)
    if layout == "trivial":
        return list(range(n))
    if layout != "greedy":
        raise ValueError('layout must be "greedy", "trivial" or a list of physical qubits')
    weights = Counter(tuple(sorted(w)) for prim, w, *_ in ops if prim in TWO_QUBIT)
    total = Counter()
    for (a, b), w in weights.items():
        total[a] += w
        total[b] += w
    adjacency = device._adjacency()
    order = sorted(range(n), key=lambda q: (-total[q], q))
    placed, used, distance_cache = {}, set(), {}

    def distances(p):
        if p not in distance_cache:
            distance_cache[p] = _bfs(adjacency, p)
        return distance_cache[p]

    hub = max(range(device.n_qubits), key=lambda p: (len(adjacency[p]), -p))
    for q in order:
        partners = [(placed[m], w) for (a, b), w in weights.items() for m in (a, b)
                    if q in (a, b) and m != q and m in placed]
        if not placed:
            choice = hub
        elif partners:
            free = [p for p in range(device.n_qubits) if p not in used]
            choice = min(free, key=lambda p: (sum(w * distances(pp)[p] for pp, w in partners), p))
        else:  # no placed partner: the free qubit nearest the placed ones
            near = _bfs_multi(adjacency, list(used))
            choice = min((p for p in range(device.n_qubits) if p not in used), key=lambda p: (near[p], p))
        placed[q] = choice
        used.add(choice)
    return [placed[q] for q in range(n)]


def _bfs(adjacency, source):
    return _bfs_multi(adjacency, [source])


def _bfs_multi(adjacency, sources):
    far = len(adjacency) + 1
    dist = [far] * len(adjacency)
    queue = deque(sources)
    for s in sources:
        dist[s] = 0
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if dist[b] == far:
                dist[b] = dist[a] + 1
                queue.append(b)
    return dist


# ---------------------------------------------------------------- routing
def _route(ops, layout, device):
    """Map ops to physical qubits, inserting SWAPs before two-qubit gates on uncoupled qubits."""
    adjacency = device._adjacency()
    coupled = set(device.edges)
    l2p = list(layout)
    p2l = {p: q for q, p in enumerate(l2p)}
    routed, swaps = [], 0
    for prim, wires, base, index, sign in ops:
        physical = [l2p[q] for q in wires]
        if prim in TWO_QUBIT and tuple(sorted(physical)) not in coupled:
            path = _path(adjacency, physical[0], physical[1])
            if path is None:
                raise ConversionError(f"qubits {physical} of {device.name} are not connected",
                                      "choose a connected device or another layout")
            for a, b in zip(path, path[1:-1]):  # walk the first qubit to beside the second
                for sw_prim, sw_wires, _ in _DECOMPOSE["swap"]([a, b], []):
                    routed.append((sw_prim, list(sw_wires), None, -1, 1))
                swaps += 1
                qa, qb = p2l.get(a), p2l.get(b)
                if qa is not None:
                    l2p[qa] = b
                if qb is not None:
                    l2p[qb] = a
                p2l[a], p2l[b] = qb, qa
            physical = [l2p[q] for q in wires]
        routed.append((prim, physical, base, index, sign))
    return routed, l2p, swaps


def _path(adjacency, source, target):
    parent = {source: None}
    queue = deque([source])
    while queue:
        a = queue.popleft()
        if a == target:
            path = []
            while a is not None:
                path.append(a)
                a = parent[a]
            return path[::-1]
        for b in adjacency[a]:
            if b not in parent:
                parent[b] = a
                queue.append(b)
    return None


def _sabre(circuit, device, layout, values):
    try:
        from qiskit import QuantumCircuit, qasm2, transpile
    except ImportError as exc:
        raise ConversionError("router='sabre' needs Qiskit", "pip install qiskit, or use router='greedy'") from exc
    if isinstance(circuit, Circuit):
        raise ConversionError("router='sabre' compiles OpenQASM or Qiskit circuits",
                              "pass the circuit as OpenQASM or Qiskit, or use router='greedy'")
    if isinstance(circuit, str):
        circuit = qasm2.loads(circuit, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS) \
            if "OPENQASM 3" not in circuit else __import__("qiskit").qasm3.loads(circuit)
    if values is not None:
        params = list(circuit.parameters)
        circuit = circuit.assign_parameters(dict(zip(params, values)) if not isinstance(values, dict)
                                            else {p: values[p.name] for p in params})
    if circuit.num_qubits > device.n_qubits:
        raise ConversionError(f"the circuit has {circuit.num_qubits} qubits; {device.name} has {device.n_qubits}",
                              "choose a larger device")
    from qiskit.transpiler import CouplingMap
    coupling = CouplingMap([list(e) for e in device.edges] + [[b, a] for a, b in device.edges])
    initial = None
    if isinstance(layout, (list, tuple)):
        initial = list(layout)
    elif layout == "trivial":
        initial = list(range(circuit.num_qubits))
    out = transpile(circuit, coupling_map=coupling, initial_layout=initial, routing_method="sabre",
                    layout_method=None if initial else "sabre", optimization_level=1, seed_transpiler=7,
                    basis_gates=["rz", "sx", "x", "rx", "ry", "h", "cz", "swap"])
    swaps = sum(1 for item in out.data if item.operation.name == "swap")
    final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(circuit.num_qubits))
    initial_layout = list(out.layout.initial_index_layout()[:circuit.num_qubits]) if out.layout is not None \
        else list(range(circuit.num_qubits))
    n, instructions, _ = source_instructions(out)
    order = list(circuit.parameters)
    return n, instructions, order, initial_layout, final[:circuit.num_qubits], swaps


# ---------------------------------------------------------------- schedule and noise
def _schedule(ops, device, noise, idle, scale=1.0):
    """Insert gate errors and idle relaxation; return ops and timing facts.

    `scale` multiplies decoherence rates as it does error rates: T1 and T2 are divided by it.
    """
    t1, t2 = device.t1, device.t2
    if idle and t1 is None and t2 is None:
        idle = False
    if idle:
        t1 = t1 if t1 is not None else 1e6 * t2   # ions: effectively no T1 decay
        t2 = min(t2 if t2 is not None else 2 * t1, 2 * t1)
        t1, t2 = t1 / scale, t2 / scale
    durations = {1: device.time_1q or 0.0, 2: device.time_2q or 0.0}
    busy, started, level = {}, set(), {}
    out, one, two = [], 0, 0
    for prim, wires, base, index, sign in ops:
        virtual = prim == "RZ"
        duration = 0.0 if virtual else durations[len(wires)]
        start = max(busy.get(w, 0.0) for w in wires)
        if idle:
            for w in wires:
                gap = start - busy.get(w, 0.0)
                if w in started and gap > 0:
                    out.append(("RELAX", [w], (gap, t1, t2), -1, 1))
        out.append((prim, wires, base, index, sign))
        if not virtual:
            depth = max(level.get(w, 0) for w in wires) + 1
            for w in wires:
                level[w] = depth
        if len(wires) == 2:
            two += 1
            # The coupler's own infidelity when the chip gives one, as a depolarizing probability.
            error = device.coupler_error(*wires) if noise is not None else 0.
            if error:
                out.append(("DEPOL", wires, min(0.9375, error * scale * 5 / 4), -1, 1))
        elif not virtual:
            one += 1
            if noise and noise.depolarizing_1q:
                out.append(("DEPOL", wires, noise.depolarizing_1q, -1, 1))
        for w in wires:
            busy[w] = start + duration
            started.add(w)
    end = max(busy.values(), default=0.0)
    if idle:
        for w in sorted(started):
            if end - busy[w] > 0:
                out.append(("RELAX", [w], (end - busy[w], t1, t2), -1, 1))
    return out, {"one_qubit_gates": one, "two_qubit_gates": two, "depth": max(level.values(), default=0),
                 "duration_s": end}
