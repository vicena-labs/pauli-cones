# Comparison methodology and coverage

The architecture cleanup should be assessed by common inputs and outputs, not by
the number of nominal wires or by comparing a gradient request with a value-only
request. The optional `benchmarks` package implements these rules and preserves
the complete case definitions in each JSON result.

## Common numerical problem

All cases begin in the zero product state. Wire zero is the least significant
bit. A parameter vector supplies **additive shifts** from each gate's stored base
angle. Several gates can share a parameter; every corresponding contribution
must appear in its derivative. All backends return the same real weighted sum of
Pauli expectations, with the same measured readout model.

The current suite exercises seeded single-qubit rotations, nearest-neighbor CZ,
one- and two-wire depolarizing channels, zero-temperature T1/T2 relaxation, and
independent asymmetric classical Z readout. A separate 32-wire constructed parity
case has an analytic answer and tests early input-boundary contraction. It is a
deliberate representation stress case, not evidence about arbitrary 32-wire or
127-wire circuits. The suite does not establish fSim/ECR external interoperability;
their core correctness tests are separate.

The depolarizing API uses total nonidentity Pauli-error probability `p` on `k`
wires. Aer uses `lambda = p * 4**k / (4**k - 1)`. Both yield nonidentity-Pauli
attenuation `s = 1 - p * 4**k / (4**k - 1)`. For positive `s`, the IBM adapter uses
all `4**k - 1` nonidentity generators with rate `-log(s) / 4**k`. A single Pauli
error of probability `p < 1/2` maps to rate `-log(1 - 2*p) / 2`. Unsupported rate
regimes are rejected; they are not approximated by a different channel.

Relaxation applies amplitude damping with survival `exp(-duration/T1)` followed by
dephasing with coherence factor `exp(-duration/T2 + duration/(2*T1))`. Readout
transforms each measured Z to `(p10 - p01) I + (1 - p01 - p10) Z`. Products are
expanded before constructing the competing observable, so asymmetric readout on
a Z product is not incorrectly replaced by a single global scaling factor.

The independent density oracle constructs Kraus maps and full matrices directly,
without using Pauli Cones PTMs, masks, or cone selection. Its reverse derivative
propagates an observable through **adjoint channels**, never inverse noisy
channels. Qulacs and Aer are checked against it, and the IBM adapter is checked
on their common Pauli-noise subset. Tests also rebind circuits prepared at a zero
angle and test repeated occurrences of the same shared parameter.

## Timing contract

Each backend/output pair runs in a separate process with a timeout. Worker
environment limits OpenMP, OpenBLAS, MKL, NumExpr and Accelerate to one thread;
Aer additionally receives explicit single-thread options. This is a shared CPU
host, not a dedicated or affinity-pinned performance laboratory. The result logs
CPU model, available affinity count, cgroup CPU quota when available, Python and
dependency versions, and source hashes. Source changes during the run are flagged.

The suite records four distinct costs:

1. **Cold process:** interpreter start, imports, one preparation, first binding
   and requested evaluation, serialization of the small result, and exit. Each
   cold sample is a fresh process. Installation and compilation are excluded.
2. **First setup:** package imports and first preparation inside the worker.
3. **Prepared setup:** repeated construction in an already imported process.
   Global gate caches may be warm; this cost is not labeled cold compilation.
4. **Warm evaluation:** binding plus requested value or value-and-gradient on a
   resident prepared object. Each round uses three deterministic **new** shift
   vectors, generated outside the timed block. This prevents repeated PTM-cache
   hits from masquerading as fresh optimization steps. Every timed vector and
   output is retained and oracle-checked. Per-vector samples and their median are
   preserved.

Value-only and value-and-gradient records are separate. Native and graph gradients
perform analytic differentiation. Qulacs, Aer, and the optional IBM gradient
adapter use the exact two-point shift rule **per variable gate occurrence**, then
sum via the shared-parameter chain rule. Shifting one shared parameter globally
by plus/minus pi/2 would generally be wrong. A separate NumPy analytic channel
adjoint addresses the different-algorithm concern; it is not represented as the
fastest possible dense implementation. IBM large stress-case gradient sweeps are
explicitly excluded by the harness budget.

The graph's preparation and persistent storage are charged and reported. A warm
speed advantage only pays off after approximately

`(graph_prepare - native_prepare) / (native_per_vector - graph_per_vector)`

vectors, when the denominator is positive. A negative denominator supplies no
amortization argument. This simple estimate uses prepared setup and warm medians;
it is not a measured end-to-end service latency. The default runtime remains
native because small cones can favor it.

The Aer adapter uses `assign_parameters` on a prepared symbolic circuit and a
separate simulator job for each requested or shifted value. It does **not** test
Aer runtime parameter binding or batching all shifted circuits into one job.
Consequently these timings describe this reproducible adapter, not Aer's fastest
possible optimizer/batch configuration. Qulacs uses direct updates on a resident
parametric density circuit. The IBM adapter reuses its Clifford transformation and
rotation structure. Further tuning must retain the same output/accuracy contract
and separately account for preparation and batching latency.

Peak process RSS includes imports, prepared objects, temporary arrays and native
allocations. It is measured per backend worker and is not interchangeable with a
graph's persistent numeric-array byte count, which excludes Python metadata.
Backend-specific diagnostics report fallback and storage separately.

## Accuracy and failure contract

Recorded comparison tolerance is absolute `1e-10` for values and `1e-9` for each
gradient component. The full returned vectors, independent reference values,
maximum errors and derivative methods are retained. The numerical contract of
each runtime still applies; these checks are not a global error certificate.

IBM uses `atol=0` and the same configured `max_terms` number. Its cap truncates
where Pauli Cones' native cap fails closed, so **any nonzero reported discarded
coefficient norm removes that result from eligible exact comparisons**, even if
this particular scalar answer happens to pass the oracle tolerance. Accuracy
failures, unsupported model errors, missing optional packages, subprocess errors,
and timeouts remain in the report. They are never silently dropped from a speed
ratio. No aggregate speedup is computed across different tasks or statuses.

## Coverage that remains unmeasured

PauliPropagation.jl surrogate/gradient configurations, CUDA cuPauliProp, noisy MPO
or MPDO, tensor-network contractions and stochastic trajectories are not ranked by
this CPU suite. Their absence is recorded, not interpreted as a win. Julia and a
CUDA GPU were unavailable for the recorded environment. Tensor/trajectory routes
need an explicitly validated adapter and a common error/variance target before
their timings can be compared. In particular, a unitary surrogate or an MPS
trajectory output cannot be silently substituted for deterministic T1/T2 noisy
expectations. Large calibrated 127-wire performance, production service behavior,
and all supported gate/noise combinations require additional qualification.

These gaps are separate from the implemented bounded dense fallback and reusable
graph: those give users concrete alternate representations now, with explicit
resource limits and diagnostics. No runtime selector claims to choose the fastest
possible representation.

## Primary upstream references

- [IBM Pauli propagation](https://quantum.cloud.ibm.com/docs/en/addons/pauli-prop)
  and [propagation API](https://quantum.cloud.ibm.com/docs/en/api/pauli-prop/propagation).
  The adapter also inspects the installed 0.2.1 API and reuses its documented
  Clifford and rotation preparation.
- [Aer PauliLindbladError](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.PauliLindbladError.html),
  [depolarizing_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.depolarizing_error.html),
  and [thermal_relaxation_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html).
  [AerSimulator options](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
  describe runtime parameter binding, an unmeasured tuning option in this suite.
- [Qulacs advanced guide](https://docs.qulacs.org/en/latest/guide/2.0_python_advanced.html).
- [PauliPropagation.jl](https://sparqlesim.github.io/PauliPropagation.jl/stable/)
  and [cuPauliProp overview](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/overview.html)
  for further specialist comparisons; neither is timed by this suite.
