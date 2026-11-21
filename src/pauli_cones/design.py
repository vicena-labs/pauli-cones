"""Design new chips and compare them on benchmark circuits.

    from pauli_cones import design, devices
    module = devices.get("heavy-hex", rows=3, row_length=7)
    chip = design.modular(module, modules=4, links="ring", link_error=2e-2)       # four linked modules
    chip = design.remove_qubits(chip, [5, 17])                                    # defective qubits
    chip = design.with_physics(chip, t1=300e-6, t2=200e-6, time_1q=30e-9, time_2q=200e-9,
                               control_error_2q=2e-3)
    design.compare([chip, devices.get("heavy-hex-127")], family="qaoa-ring", sizes=[8, 12])

Every function returns a new Device (the chip description used everywhere
else in the package), so a design can be saved with to_json(), loaded with
devices.get("chip.json"), compiled onto and swept like a catalogue chip.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .devices import Device
from .topology import Topology


def from_edges(name, n_qubits, edges, *, family="custom", native_gates=None, **noise):
    """A chip from a coupler list. noise: error_1q, error_2q, readout, t1, t2, time_1q, time_2q."""
    return Device(name=name, family=family, topology=Topology(n_qubits, tuple(tuple(e) for e in edges)),
                  native_gates=native_gates or {"one_qubit": ["rz", "sx", "x"], "two_qubit": ["cz"]},
                  description="Custom chip.", source="Designed with pauli_cones.design.",
                  parameters={"designed": True}, **noise)


def add_couplers(chip, couplers, *, error=None):
    """Add couplers; `error` sets their two-qubit infidelity (otherwise the chip's error_2q)."""
    new = {tuple(sorted(c)) for c in couplers} - set(chip.edges)
    errors = dict(chip.coupler_errors)
    if error is not None:
        errors.update({c: error for c in new})
    return _rebuild(chip, chip.n_qubits, list(chip.edges) + sorted(new), errors, f"{chip.name}+couplers")


def remove_couplers(chip, couplers):
    gone = {tuple(sorted(c)) for c in couplers}
    missing = gone - set(chip.edges)
    if missing:
        raise ValueError(f"not couplers of {chip.name}: {sorted(missing)}")
    errors = {k: v for k, v in chip.coupler_errors.items() if k not in gone}
    return _rebuild(chip, chip.n_qubits, [e for e in chip.edges if e not in gone], errors, f"{chip.name}-couplers")


def remove_qubits(chip, qubits):
    """Remove qubits (for example defective ones) and their couplers, renumbering the rest
    in order. The old-to-new numbering is kept in parameters["renumbering"]."""
    dead = set(qubits)
    if not dead <= set(range(chip.n_qubits)):
        raise ValueError(f"{chip.name} has qubits 0..{chip.n_qubits - 1}")
    keep = [q for q in range(chip.n_qubits) if q not in dead]
    if not keep:
        raise ValueError("cannot remove every qubit")
    renumber = {old: new for new, old in enumerate(keep)}
    edges = [(renumber[a], renumber[b]) for a, b in chip.edges if a not in dead and b not in dead]
    errors = {(renumber[a], renumber[b]): e for (a, b), e in chip.coupler_errors.items()
              if a not in dead and b not in dead}
    out = _rebuild(chip, len(keep), edges, errors, f"{chip.name}-{len(dead)}q")
    return replace(out, parameters={**out.parameters, "renumbering": {str(k): v for k, v in renumber.items()}})


def modular(module, modules, *, links="ring", links_per_pair=1, link_error=None, name=None):
    """Tile `modules` copies of a chip and join neighbouring copies with links.

    links: "line" joins copy i to i+1, "ring" also joins the last to the first,
    "all" joins every pair. Each join couples `links_per_pair` qubits of one
    copy to the same qubits of the other, chosen among the lowest-degree qubits
    (the module's edge). link_error is the links' two-qubit infidelity, usually
    worse than on-module couplers; None keeps the chip's error_2q.
    """
    if modules < 2:
        raise ValueError("a modular chip needs at least 2 modules")
    if links not in ("line", "ring", "all"):
        raise ValueError('links must be "line", "ring" or "all"')
    n = module.n_qubits
    degree = [len(a) for a in module._adjacency()]
    ports = sorted(range(n), key=lambda q: (degree[q], q))[:max(1, links_per_pair)]
    edges, errors = [], {}
    for m in range(modules):
        edges += [(a + m * n, b + m * n) for a, b in module.edges]
        errors.update({(a + m * n, b + m * n): e for (a, b), e in module.coupler_errors.items()})
    pairs = ([(m, m + 1) for m in range(modules - 1)] + ([(modules - 1, 0)] if links == "ring" and modules > 2 else [])
             if links != "all" else [(i, j) for i in range(modules) for j in range(i + 1, modules)])
    for i, j in pairs:
        for k, q in enumerate(ports):
            # Port k of module i links to port of the reversed order in module j, so links spread.
            a, b = q + i * n, ports[-1 - k] + j * n
            edges.append((min(a, b), max(a, b)))
            if link_error is not None:
                errors[(min(a, b), max(a, b))] = link_error
    family = module.family if module.family != "annealer" else "custom"
    out = replace(module, name=name or f"{module.name}x{modules}-{links}",
                  topology=Topology(n * modules, tuple(edges)), coupler_errors=errors, family=family,
                  description=f"{modules} copies of {module.name} joined in a {links} by {links_per_pair} "
                              "link(s) per pair of modules.",
                  parameters={"module": module.name, "modules": modules, "links": links,
                              "links_per_pair": links_per_pair, "link_error": link_error})
    return out


def coherence_limited_error(duration, t1, t2, qubits=1):
    """Average gate infidelity from T1/T2 decay alone during a gate of `duration`.

    One qubit: F = (3 + e^(-t/T1) + 2 e^(-t/T2)) / 6. Two qubits decaying
    independently: the process fidelity (1 + e^(-t/T1) + 2 e^(-t/T2)) / 4 is
    squared, and F = (4 F_process + 1) / 5. None for T1 means no amplitude decay.
    """
    a = math.exp(-duration / t1) if t1 else 1.0
    b = math.exp(-duration / t2) if t2 else 1.0
    process = (1 + a + 2 * b) / 4
    if qubits == 1:
        return 1 - (2 * process + 1) / 3
    if qubits == 2:
        return 1 - (4 * process ** 2 + 1) / 5
    raise ValueError("qubits must be 1 or 2")


def with_physics(chip, *, t1, t2, time_1q, time_2q, control_error_1q=0.0, control_error_2q=0.0, readout=None):
    """Set coherence times and gate durations, deriving gate errors from them.

    Each gate's infidelity is its coherence limit plus a control error
    (miscalibration, leakage, crosstalk) that coherence does not capture.
    Per-coupler errors are kept and raised by the same coherence term.
    """
    if t2 is not None and t1 is not None and t2 > 2 * t1:
        raise ValueError("T2 cannot exceed 2 T1")
    e1 = coherence_limited_error(time_1q, t1, t2, 1) + control_error_1q
    e2_coherence = coherence_limited_error(time_2q, t1, t2, 2)
    e2 = e2_coherence + control_error_2q
    couplers = {k: min(0.999, v + e2_coherence) for k, v in chip.coupler_errors.items()}
    return replace(chip, t1=t1, t2=t2, time_1q=time_1q, time_2q=time_2q, error_1q=min(0.999, e1),
                   error_2q=min(0.999, e2), coupler_errors=couplers,
                   readout=tuple(readout) if readout is not None else chip.readout,
                   source="Gate errors derived from coherence times and durations plus a control error "
                          "(pauli_cones.design.with_physics).")


def compare(chips, *, family="qaoa-ring", sizes=(8, 12), noise_scale=1.0, router="greedy", circuit=None,
            observable=None, family_parameters=None):
    """Benchmark chip designs: run a circuit family (or one circuit) on each chip and
    summarise per chip. Returns a SweepResult whose rows are one per chip:
    mean |deviation| from the ideal value, mean SWAPs, two-qubit gates and
    duration over the sizes it fits, and how many sizes failed."""
    from . import devices as catalogue
    from .experiments import SweepResult, _device, sweep
    resolved = [_device(chip, catalogue) for chip in chips]
    names = [chip.name for chip in resolved]
    if len(set(names)) != len(names):
        raise ValueError("chips to compare need distinct names")
    rows = sweep(circuit or family, observable, devices=resolved, noise_scales=[noise_scale],
                 sizes=None if circuit else list(sizes), router=router,
                 family_parameters=family_parameters).rows
    summary = []
    for name in names:
        mine = [r for r in rows if r["device"] == name]
        good = [r for r in mine if "error" not in r]
        mean = lambda key: sum(r[key] for r in good) / len(good) if good else None  # noqa: E731
        summary.append({"device": name, "sizes_run": len(good),
                        "sizes_failed": len(mine) - len(good),
                        "mean_abs_deviation": (sum(abs(r["deviation"]) for r in good) / len(good)) if good else None,
                        "mean_swaps": mean("swaps"), "mean_two_qubit_gates": mean("two_qubit_gates"),
                        "mean_duration_s": mean("duration_s")})
    summary.sort(key=lambda r: (r["mean_abs_deviation"] is None, r["mean_abs_deviation"] or 0))
    result = SweepResult(summary)
    result.COLUMNS = ("device", "sizes_run", "sizes_failed", "mean_abs_deviation", "mean_swaps",
                      "mean_two_qubit_gates", "mean_duration_s")
    return result


def _rebuild(chip, n, edges, errors, name):
    return replace(chip, name=name, topology=Topology(n, tuple(edges)), coupler_errors=errors,
                   parameters={**chip.parameters, "designed": True})
