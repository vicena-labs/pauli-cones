# JSON interface, version 1

Start with `quantum-circuit-simulator example`. Submit the resulting document with `quantum-circuit-simulator run file.json`, or supply it on standard input with `quantum-circuit-simulator run -`.

Required fields are `schema_version` (exactly 1), `n_qubits`, `operations`, and `observables`. Optional fields are `n_parameters` (default 0), `max_terms` (default 50000), `readout` (default empty), `parameter_vectors` (default one zero shift vector), and `gradients` (a boolean, default true). Unknown fields and duplicate JSON keys are rejected.

Fixed gates are objects with `op` and `wires`: H, SX, X, Y, Z, CZ, ECR. RX/RY/RZ additionally require `theta` and optionally `param` (default -1 for a fixed angle). FSIM requires `theta` and `phi`; its optional `param` differentiates theta only. Named rotations implement exp(-i theta Pauli / 2).

Noise operations:

```json
{"op":"DEPOLARIZE","wires":[0,1],"probability":0.01}
{"op":"PAULI_ERROR","wire":0,"axis":"Z","probability":0.02}
{"op":"RELAX","wire":0,"duration":0.000001,"t1":0.000070,"t2":0.000090}
```

Readout entries contain `wire`, `p01`, and `p10`. Observable entries contain a finite `coefficient` and `pauli`, a mapping from decimal wire-index strings to I/X/Y/Z. Empty `pauli` denotes identity. The returned `value` is the weighted sum, not an array of individually measured terms. Its gradient has exactly `n_parameters` components unless gradients are disabled, in which case it is an empty array.

Every vector in `parameter_vectors` has exactly `n_parameters` additive shifts, relative to the original base angles. At most 128 vectors form one finite batch. Failure produces `ok:false` and a nonzero CLI exit; successful partial batches are never emitted. Timeout exits with code 124. Other request failures exit with code 2.

The response declares precision, filtering thresholds, selected backend, term cap and phase timings. Worker timing includes package imports; parent-worker wall time additionally includes launching and collecting the worker. Neither is a measurement of any hosted application's agent/network/UI latency.

Select `--backend native|python|graph|dense|auto`. The default is native. `--value-only` disables gradients even when the request omits `gradients` or sets it to true. `requested_backend`, actual result backends, and `diagnostics.fallback_reason` distinguish a selected method from a budget-triggered fallback. Dense uses the same supported physical channels; it does not ignore noise to satisfy a cap. Native/reference never silently approximate a cap failure. The API exposes advanced graph and dense budgets; the CLI uses their documented defaults.

`--min-coefficient F` and `--max-weight N` select the truncated mode on the native, graph and auto backends; each result then carries `error_bound` and, with gradients, `gradient_error_bound`. `--threads N` evaluates independent cone groups in parallel, with results that do not depend on N. These are command-line options rather than request fields, so version 1 requests are unchanged.

```bash
quantum-circuit-simulator run problem.json --value-only
quantum-circuit-simulator run problem.json --min-coefficient 1e-4 --max-weight 6 --threads 4
quantum-circuit-simulator run problem.json --backend graph
quantum-circuit-simulator run problem.json --backend auto
```

Return to [README](../README.md).
