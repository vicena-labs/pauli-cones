"""Fair-rerun tables (v5): every tool at 1 and 8 threads, same vectors, same machine.

    python summarize_v5.py <results-dir>  > v5_tables.md
"""
import json, math, sys
from collections import defaultdict

R = sys.argv[1]


def rows(name):
    try:
        return [json.loads(l) for l in open(f"{R}/{name}")]
    except FileNotFoundError:
        return []


def f(x):
    if x is None:
        return "—"
    return f"{x * 1e3:.3g} ms" if x < 1 else f"{x:.3g} s"


def label(tool):
    return {"pauli_cones": "PC native", "pauli_cones_graph": "PC graph", "pauli_prop": "pauli-prop",
            "lightning": "Lightning", "qulacs": "Qulacs", "aer": "Aer", "quimb": "quimb"}.get(tool, tool)


# exact[workload][(tool, threads)] = record
exact = defaultdict(dict)
for name, threads in (("v5_exact_1t.jsonl", 1), ("v5_exact_8t.jsonl", 8), ("v5_pp_exact_1t.jsonl", 1),
                      ("v5_pp_exact_8t.jsonl", 8), ("v5_pp_surrogate_1t.jsonl", 1), ("v5_pp_surrogate_8t.jsonl", 8)):
    for r in rows(name):
        exact[r["workload"]][(r["tool"], threads)] = r
order = [json.loads(l)["workload"] for l in open(f"{R}/v5_exact_1t.jsonl")]
order = list(dict.fromkeys(order))


def ok(r, key="warm_value_s"):
    return r is not None and r.get(key) is not None and not r.get("error")


def graph_ok(r):
    return ok(r) and (r.get("diagnostics") or {}).get("backend", "graph").endswith("graph")


def why(r):
    if r is None:
        return "not run"
    e = r.get("error", "")
    if "timeout" in e:
        return "timeout"
    if "ConeOverflow" in e or "exceeds cap" in e or "truncated" in e:
        return "over cap"
    if "only single-qubit" in e or "no noise" in e or "InexactError" in e:
        return "unsupported"
    return "failed"


print("# Fair rerun, Apple M1 Pro (6 performance + 2 efficiency cores), macOS\n")
print("Warm medians of 5 fresh parameter vectors per run, one fresh process per run, 120 s limit. "
      "Quantum Circuit Simulator ran at 1 thread; every competitor ran at 1 and 8 threads (pauli-prop is single-threaded). "
      "A competitor's figure is its faster configuration. PC graph rows count only runs where the graph did not fall back to native.\n")
print("**Caveats.** The laptop was not quiet: other applications used about 1.5-2 of the 8 cores throughout, so "
      "8-thread figures are unreliable and 1-thread figures carry run-to-run variation of up to several times. "
      "During the 8-thread PauliPropagation.jl pass, Julia start-up alone took 14-97 s, so its 8-thread timeouts are "
      "artifacts, not limits of the package. Publishable numbers need a dedicated quiet machine.\n")

print("## Exact values\n")
print("| Workload | PC native 1t | PC graph 1t | Graph prepare | Fastest competitor (config) | Its value | PC best vs it | Refused or unsupported |")
print("|---|---|---|---|---|---|---|---|")
for w in order:
    t = exact[w]
    native, graph = t.get(("pauli_cones", 1)), t.get(("pauli_cones_graph", 1))
    ours = [x for x in (native.get("warm_value_s") if ok(native) else None,
                        graph.get("warm_value_s") if graph_ok(graph) else None) if x]
    comp = [(r["warm_value_s"], f"{label(k[0])}, {k[1]}t") for k, r in t.items()
            if not k[0].startswith("pauli_cones") and ok(r)]
    best = min(comp) if comp else None
    # A tool counts as failed only when none of its configurations succeeded.
    succeeded = {k[0] for k, r in t.items() if ok(r)}
    failed = sorted({f"{label(k[0])} ({why(r)})" for k, r in t.items()
                     if not ok(r) and k[0] not in succeeded and not (k[1] == 8 and k[0] == "pauli_prop")})
    ratio = f"{best[0] / min(ours):.3g}×" if ours and best else ("PC loses" if best and not ours else "—")
    gnote = f(graph.get("warm_value_s")) if graph_ok(graph) else ("fell back" if ok(graph) else why(graph))
    print(f"| {w} | {f(native.get('warm_value_s')) if ok(native) else why(native)} | {gnote} | "
          f"{f(graph.get('prepare_s')) if graph_ok(graph) else '—'} | {best[1] if best else 'none finished'} | "
          f"{f(best[0]) if best else '—'} | {ratio} | {'; '.join(failed)} |")

print("\n## Gradients (all shared parameters)\n")
print("| Workload | PC 1t | Fastest competitor (config) | Its gradient | PC vs it |")
print("|---|---|---|---|---|")
for w in order:
    t = exact[w]
    native, graph = t.get(("pauli_cones", 1)), t.get(("pauli_cones_graph", 1))
    ours = [x for x in (native.get("warm_grad_s") if ok(native, "warm_grad_s") else None,
                        graph.get("warm_grad_s") if graph_ok(graph) and graph.get("warm_grad_s") else None) if x]
    comp = [(r["warm_grad_s"], f"{label(k[0])}, {k[1]}t") for k, r in t.items()
            if not k[0].startswith("pauli_cones") and ok(r, "warm_grad_s")]
    if not ours and not comp:
        continue
    best = min(comp) if comp else None
    print(f"| {w} | {f(min(ours)) if ours else '—'} | {best[1] if best else 'none'} | {f(best[0]) if best else '—'} | "
          f"{f'{best[0] / min(ours):.3g}×' if ours and best else '—'} |")

print("\n## Quantum Circuit Simulator threads (1 → 8)\n")
print("| Workload | Native 1t | Native 8t | Speed-up |")
print("|---|---|---|---|")
for w in order:
    a, b = exact[w].get(("pauli_cones", 1)), exact[w].get(("pauli_cones", 8))
    if ok(a) or ok(b):
        sp = f"{a['warm_value_s'] / b['warm_value_s']:.2g}×" if ok(a) and ok(b) else "—"
        print(f"| {w} | {f(a.get('warm_value_s')) if ok(a) else why(a)} | {f(b.get('warm_value_s')) if ok(b) else why(b)} | {sp} |")

# Exact references at vector 0, from every exact run that finished.
ref, spread = {}, {}
for w, t in exact.items():
    vals = [r["value"] for k, r in t.items() if r.get("value") is not None and not r.get("error")]
    if vals:
        ref[w] = vals[0]
        spread[w] = max(vals) - min(vals)
print("\n## Agreement between exact tools at the first vector\n")
print("Largest difference between any two exact results on the same input: "
      f"{max(spread.values()):.2g} (over {len(spread)} workloads).\n")

print("## Truncated, coefficient threshold only (max_weight = Inf)\n")
print("Thresholds are absolute for PauliPropagation.jl and pauli-prop and converted to Quantum Circuit Simulator's relative form by "
      "dividing by the observable l1 norm, so all tools drop the same coefficients. Errors are absolute, in units of "
      "the observable's l1 norm, against the exact references above where one exists. A value timing counts even "
      "when the same run later failed computing gradients.\n")
OBS_NORMS = {k: v["obs"] for k, v in json.load(open(f"{R}/workloads.json")).items()}
trunc = defaultdict(dict)
for name in ("v5_trunc_native.jsonl", "v5_trunc_graph.jsonl", "v5_trunc_pp.jsonl",
             "v5_trunc_pauliprop_0.0001.jsonl", "v5_trunc_pauliprop_1e-05.jsonl"):
    for r in rows(name):
        tol = r.get("abs_coeff", r.get("min_abs_coeff"))
        if tol is None:
            tol = 1e-4 if "0.0001" in name else 1e-5
        default = {"v5_trunc_native.jsonl": "pauli_cones_truncated", "v5_trunc_graph.jsonl": "pauli_cones_truncated_graph",
                   "v5_trunc_pp.jsonl": "PauliPropagation.jl"}.get(name, "pauli_prop")
        tool = r.get("tool", default)  # timed-out runs carry no tool field
        if tool == "pauli_cones_truncated_graph" and r.get("backend") and not r["backend"].endswith("graph"):
            tool = "pauli_cones_truncated_graph (fell back)"
        # Timed-out records carry no thread count; the script ran 1 thread before 8.
        slot = trunc[(r["workload"], float(tol))]
        threads = r.get("threads") or (8 if (tool, 1) in slot else 1)
        slot[(tool, int(threads))] = r
tools = [("pauli_cones_truncated", "PC native"), ("pauli_cones_truncated_graph", "PC graph"),
         ("PauliPropagation.jl", "PauliPropagation.jl"), ("pauli_prop", "pauli-prop")]
print("| Workload | Threshold | " + " | ".join(f"{n} 1t / 8t (error)" for _, n in tools) + " |")
print("|---|---|" + "---|" * len(tools))
for key in sorted(trunc, key=lambda k: (order.index(k[0]) if k[0] in order else 99, -k[1])):
    w, tol = key
    cells = []
    for tool, _ in tools:
        a, b = trunc[key].get((tool, 1)), trunc[key].get((tool, 8))
        if a is None and b is None and tool == "pauli_cones_truncated_graph" and trunc[key].get((tool + " (fell back)", 1)):
            cells.append("fell back"); continue
        valued = lambda r: r is not None and r.get("warm_value_s") is not None
        rec = a if valued(a) else b
        norm = sum(abs(c) for c, _ in OBS_NORMS.get(w, [(1.0, None)]))
        err = (f"{abs(rec['value'] - ref[w]) / norm:.1e}"
               if valued(rec) and w in ref and rec.get("value") is not None else "no ref" if valued(rec) else "")
        def state(r, threads):
            if valued(r):
                return f(r.get("warm_value_s"))
            if r is None and (tool + " (fell back)", threads) in trunc[key]:
                return "fell back"
            return "—" if (r is None and tool == "pauli_prop" and threads == 8) else why(r)
        times = f"{state(a, 1)} / {state(b, 8)}"
        cells.append(f"{times} ({err})" if err else times)
    print(f"| {w} | {tol:g} | " + " | ".join(cells) + " |")
