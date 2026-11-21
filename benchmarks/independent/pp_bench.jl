# PauliPropagation.jl adapter for bench.py workloads (exported with `bench.py export`).
# usage: julia --project=<env> -t <threads> pp_bench.jl <workloads.json> <workload> <min_abs_coeff> <max_weight>
# min_abs_coeff = 0 and max_weight = Inf is exact (no truncation).
# An optional fifth argument `surrogate` times PauliPropagation.jl's reusable
# surrogate instead: one propagation without angles (exact, or truncated by
# max_weight; min_abs_coeff does not apply), zerofilter!, then evaluate! per
# vector. It supports only Clifford gates and Pauli rotations, so noiseless
# workloads only, and it has no gradient.
using PauliPropagation, JSON3, Statistics

const t_start = time()
data = JSON3.read(read(ARGS[1], String))
w = data[Symbol(ARGS[2])]
tol = parse(Float64, ARGS[3])
maxw = ARGS[4] == "Inf" ? Inf : parse(Int, ARGS[4])
n = Int(w.n)
surrogate = length(ARGS) >= 5 && ARGS[5] == "surrogate"

function build(w)
    circ = Any[]
    rot = Tuple{Float64,Int}[]          # (base angle, shared parameter) per rotation, in order
    for op in w.ops
        k = String(op[1])
        if k in ("RX", "RY", "RZ")
            push!(circ, PauliRotation(Symbol(k[2:2]), Int(op[2]) + 1))
            push!(rot, (Float64(op[3]), Int(op[4])))
        elseif k == "CZ"
            push!(circ, CliffordGate(:CZ, [Int(op[2]) + 1, Int(op[3]) + 1]))
        elseif k == "depol"
            length(op[2]) == 1 || error("only single-qubit depolarizing is supported")
            # Quantum Circuit Simulator p (total non-identity probability) damps by 1-4p/3 = 1-lambda.
            push!(circ, DepolarizingNoise(Int(op[2][1]) + 1, 4 * Float64(op[3]) / 3))
        else
            error("unsupported op $k")
        end
    end
    psum = PauliSum(n)
    for (c, p) in w.obs
        qs = [parse(Int, String(q)) + 1 for q in keys(p)]
        ps = [Symbol(String(p[q])) for q in keys(p)]
        add!(psum, ps, qs, Float64(c))
    end
    return [g for g in circ], rot, psum
end

thetas(rot, s) = [b + (par >= 0 ? s[par + 1] : 0.0) for (b, par) in rot]

function value(circ, rot, psum, s)
    out = propagate(circ, psum, thetas(rot, s); min_abs_coeff=tol, max_weight=maxw)
    return overlapwithzero(out), length(out)
end

function gradient(circ, rot, psum, s, nparams)
    e, g = rewindgradient(circ, psum, thetas(rot, s), overlapwithzero; min_abs_coeff=tol, max_weight=maxw)
    shared = zeros(nparams)
    for (i, (_, par)) in enumerate(rot)
        par >= 0 && (shared[par + 1] += g[i])
    end
    return e, shared
end

rec = Dict{String,Any}("workload" => ARGS[2], "tool" => surrogate ? "PauliPropagation.jl surrogate" : "PauliPropagation.jl",
                       "threads" => Threads.nthreads(), "min_abs_coeff" => tol, "max_weight" => string(maxw))
try
    t = time(); circ, rot, psum = build(w); rec["build_s"] = time() - t
    vecs = [Float64.(v) for v in w.vecs]
    reps = (length(vecs) - 1) ÷ 2
    sur = nothing
    if surrogate
        w.noisy && error("the surrogate supports no noise channels")
        t = time()
        sur = propagate(circ, wrapcoefficients(psum, NodePathProperties); max_weight=maxw)
        sur = zerofilter!(sur)
        rec["prepare_s"] = time() - t
    end
    # A distinct name: assigning `value` inside this try block would make it
    # local to the whole block and hide the global function.
    timed_value = surrogate ?
        ((circ, rot, psum, s) -> (evaluate!(sur, thetas(rot, s)); (overlapwithzero(sur), length(sur)))) : value
    t = time(); v, nterms = timed_value(circ, rot, psum, vecs[1]); rec["first_value_s"] = time() - t
    rec["value"] = v; rec["final_terms"] = nterms
    rec["since_start_to_first_value_s"] = time() - t_start
    ts = Float64[]
    for s in vecs[2:1 + reps]
        t = time(); timed_value(circ, rot, psum, s); push!(ts, time() - t)
    end
    rec["warm_value_s"] = median(ts); rec["warm_value_samples"] = ts
    if surrogate
        rec["grad_unsupported"] = "the surrogate has no gradient"
    elseif !w.noisy
        t = time(); e, g = gradient(circ, rot, psum, vecs[1], Int(w.nparams)); rec["first_grad_s"] = time() - t
        rec["gradient"] = g; rec["grad_value"] = e
        ts = Float64[]
        for s in vecs[2 + reps:end]
            t = time(); gradient(circ, rot, psum, s, Int(w.nparams)); push!(ts, time() - t)
        end
        rec["warm_grad_s"] = median(ts); rec["warm_grad_samples"] = ts
    else
        rec["grad_unsupported"] = "rewindgradient does not support noise channels"
    end
catch e
    rec["error"] = sprint(showerror, e)[1:min(end, 200)]
end
rec["peak_rss_mb"] = Sys.maxrss() / 1e6
println("RESULT ", JSON3.write(rec))
