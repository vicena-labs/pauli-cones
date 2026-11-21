# Recorded CPU comparison — 2026-11-19

The complete record is [cpu_comparison_2026-11-19.json](cpu_comparison_2026-11-19.json).
It contains all circuit definitions, base and fresh timed parameter vectors,
returned values/gradients, oracle values, samples, versions, hashes, diagnostics,
peak worker RSS, unsupported cases and timeouts. The runtime/configuration and
four imported benchmark modules were unchanged during the complete run:
`source_unchanged_during_run=true`.

Pauli Cones installed metadata and module both report **0.2.0a1**. Environment:
Python 3.12.14, NumPy 2.5.5, Qulacs 0.6.14, Qiskit 2.6.2, Aer 0.18.4,
IBM pauli-prop 0.2.1, Linux x86-64, Intel Xeon Platinum 8573C. Workers requested
one thread; CPU affinity exposed nine logical processors. This was not an
isolated, pinned performance laboratory.

## Coverage and accuracy

- **84 requested backend/output combinations:** 67 eligible successes,
  13 unsupported or adapter-budget exclusions, two explicit dense resource-limit
  rejections, one truncated IBM stress result, and one worker timeout.
- **No accuracy failures.** Among eligible results, maximum absolute value error
  was **9.55e-15** and maximum gradient-component error was **2.07e-14** against
  the independent dense Kraus adjoint or the 32q analytic formula. This covers
  the base vectors and every timed fresh vector, not only a preparation angle.
- **14 comparator-contract tests passed.** Tests cover shared-parameter chain
  rules, zero-angle rebinding, asymmetric-product readout and independent adjoint
  cross-checks. The upstream IBM real-cast warnings are retained in the JSON;
  these fixtures have real Hermitian coefficients.
- IBM T1 relaxation is not substituted with Pauli noise. Its incompatible cases
  are excluded explicitly. The 32q IBM value agrees for this scalar but reports
  discarded coefficient mass and therefore is **not** an eligible exact result.
- The timeout concerns Aer's entire 40-second worker budget for the denser 4q
  gradient measurement, including validation and many repeated calls. It does
  **not** say that one gradient evaluation took 40 seconds.

## Warm values, milliseconds per fresh vector

Medians of five rounds, each with three new parameter vectors. These are resident
API binding + evaluation times. The comparisons are all for the same output and
noise model within a row.

| Case | Native | Graph | Independent dense adjoint | Qulacs | IBM pauli-prop | Aer |
|---|---:|---:|---:|---:|---:|---:|
| 2q Pauli noise, depth 2 | 0.155 | 0.114 | 0.205 | 0.500 | 0.670 | 4.183 |
| 4q Pauli noise, depth 3 | 0.401 | 0.441 | 1.202 | 2.019 | 1.254 | 11.519 |
| 4q T1/T2 + readout, depth 3 | 0.333 | 0.472 | 1.505 | 2.902 | unsupported | 23.929 |
| 6q T1/T2 + readout, depth 2 | 0.295 | 0.436 | 20.636 | 13.027 | unsupported | 24.163 |
| 4q / 8 parameters, depth 5 | 1.983 | 0.453 | 2.906 | 3.902 | unsupported | 36.010 |

## Warm value + gradient, milliseconds per fresh vector

Native and graph use analytic differentiation. The independent dense baseline
uses a channel adjoint. Qulacs, IBM and Aer adapters use exact per-gate parameter
shift and sum shared-parameter contributions. These are different derivative
algorithms; this table is not a claim about every possible implementation in
those packages. Aer also does not exercise runtime parameter-binding batches.

| Case | Native | Graph | Independent dense adjoint | Qulacs | IBM pauli-prop | Aer |
|---|---:|---:|---:|---:|---:|---:|
| 2q Pauli noise, depth 2 | 0.119 | 0.266 | 0.458 | 6.777 | 11.824 | 72.304 |
| 4q Pauli noise, depth 3 | 0.375 | 1.147 | 2.933 | 93.131 | 66.808 | 577.224 |
| 4q T1/T2 + readout, depth 3 | 0.383 | 1.383 | 3.613 | 136.177 | unsupported | 1,083.240 |
| 6q T1/T2 + readout, depth 2 | 0.347 | 1.067 | 50.763 | 646.273 | unsupported | 1,285.090 |
| 4q / 8 parameters, depth 5 | 2.640 | 1.103 | 6.874 | 338.557 | unsupported | timeout |

The package's own bounded dense fallback is included in the full JSON. It is a
resource/correctness alternative, not the fastest backend on these small cases;
its corresponding value/gradient medians range from 2.918/6.183 ms on the 2q case
to 15.589/41.361 ms on the 4q T1/T2/readout case.

## Cold value requests, milliseconds

**One fresh-process sample per pair**, including interpreter start, imports,
preparation and the first value. Installation/compilation are excluded. Do not
interpret small differences here as statistically established speed advantages.
For example, Qulacs' 2q cold sample was **159.750 ms**, slightly below native's
**162.861 ms**. That observed competitor-faster result is retained, rather than
being hidden by the warm comparison.

| Case | Native | Graph | Independent dense adjoint | Qulacs | IBM pauli-prop | Aer |
|---|---:|---:|---:|---:|---:|---:|
| 2q Pauli noise, depth 2 | 162.861 | 167.815 | 205.549 | 159.750 | 602.320 | 541.055 |
| 4q Pauli noise, depth 3 | 173.802 | 214.985 | 176.226 | 211.635 | 622.413 | 650.686 |
| 4q T1/T2 + readout, depth 3 | 166.558 | 171.480 | 200.683 | 173.489 | unsupported | 866.599 |
| 6q T1/T2 + readout, depth 2 | 162.240 | 173.027 | 278.430 | 180.914 | unsupported | 927.198 |
| 4q / 8 parameters, depth 5 | 162.885 | 206.292 | 184.260 | 178.009 | unsupported | 847.162 |

## Where the graph helps, and where it does not

For the denser 4q / 8-parameter case, graph value + gradient took
**1.103 ms**, compared with native
**2.640 ms**, a **2.39x** warm advantage.
Prepared setup cost was **38.195 ms** versus
**3.218 ms**. Using these medians, the extra setup
cost is recovered after approximately **23 vectors**. This is a simple
amortization estimate, not a measured service latency. Its cold gradient request
was slower: **210.492 ms** versus
**169.221 ms**.

The graph is slower for most of the smaller gradient cones in this suite. Native
therefore remains the default. Reusable graph preparation is an explicit option,
with bounded memory and fallback diagnostics, not a universal optimization.
Some sub-millisecond value/gradient timings invert slightly between independent
workers; five short samples do not resolve small differences reliably.

## The constructed 32q stress case

Native evaluated the analytically solvable 32q parity value in **0.167 ms** and
value + gradient in **0.233 ms**; graph took **0.498 ms** and **0.881 ms**. Early
product-input contraction keeps this particular representation bounded. The
dense backend correctly refused its configured cone budget. Full-density external
adapters are capped at eight qubits and were not run. IBM's term-capped result is
marked truncated and excluded from exact performance comparisons. This case is
not representative of arbitrary 32q or 127q workloads.

Julia, CUDA cuPauliProp, noisy MPO/MPDO, tensor networks and trajectories are not
ranked. For the timing/noise contract and tuning limitations, see
[comparison methodology](../../docs/comparison-methodology.md). For exact raw
numbers and all unsuccessful/excluded cases, use the JSON record.

Record SHA-256: `30c75063086bb6bb3cf96a46878d46a53efe135e2dbd35ff421b7c67e862d649`.
