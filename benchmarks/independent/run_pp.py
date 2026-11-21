"""Drive pp_bench.jl: one fresh Julia process per (workload, threads, truncation).

    python run_pp.py <julia-env> <workloads.json> <out.jsonl> <threads> <min_abs_coeff> <max_weight> <workload ...>
    PP_MODE=surrogate times the reusable surrogate instead.
Cold wall time = process start (incl. Julia startup, package load, JIT) to exit.
"""
import json, os, subprocess, sys, time

env_dir, wl, out, threads, tol, maxw, *names = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
timeout = float(os.environ.get("BENCH_TIMEOUT", 180))
for name in names:
    t = time.perf_counter()
    try:
        p = subprocess.run(["julia", f"--project={env_dir}", "-t", threads, os.path.join(here, "pp_bench.jl"),
                            wl, name, tol, maxw, os.environ.get("PP_MODE", "propagate")], capture_output=True, text=True, timeout=timeout)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        rec = json.loads(lines[-1][7:]) if lines else dict(workload=name, error=p.stderr[-300:])
    except subprocess.TimeoutExpired:
        rec = dict(workload=name, error=f"timeout>{timeout:.0f}s")
    rec.setdefault("tool", "PauliPropagation.jl" + (" surrogate" if os.environ.get("PP_MODE") == "surrogate" else ""))
    rec.update(threads=int(threads), min_abs_coeff=float(tol), max_weight=maxw,
               cold_process_wall_s=time.perf_counter() - t)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print({k: rec.get(k) for k in ("workload", "threads", "min_abs_coeff", "value", "warm_value_s",
                                   "warm_grad_s", "final_terms", "error")}, flush=True)
