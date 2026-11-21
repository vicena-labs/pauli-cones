# Reproducible CPU comparisons

This optional suite compares identical noisy Pauli expectations and shared-angle
gradients. It is separate from the installable runtime. Results include failures,
unsupported noise, truncation, timeouts, versions, source hashes, and resource
settings; an inaccurate result is never counted as a speed win.

The [recorded results and tables](results/README.md) summarize the November 19
CPU run, including cases where an alternate method is faster. The separate
`compare_revision.py` script compares a supplied reviewed source tree against
this revision; see [performance evidence](../docs/performance.md).

The [independent harness](independent/README.md) compares version 0.4.0 with Aer, Qulacs, Lightning, pauli-prop, quimb and PauliPropagation.jl, including its surrogate mode and IBM's 127-qubit kicked-Ising circuits.

## Run

From the source checkout, with Python 3.12 and a C++17 compiler:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install -r benchmarks/requirements.txt
python -m pytest benchmarks/test_comparators.py
python -m benchmarks.compare --output benchmarks/results/local.json
```

The requirements file pins the optional packages used for the recorded run.
They are not required to install or use Pauli Cones. On Windows activate the
virtual environment using its platform-specific activation script.

A shorter comparison, or one without external dependencies:

```sh
python -m benchmarks.compare --backends native,graph,dense,dense_adjoint \
  --cases pauli_noise_2q_d2,repeated_gradient_4q_d5 \
  --repeats 5 --cold-repeats 1 --output benchmarks/results/small.json
```

Missing optional imports and unsupported models produce explicit records. The
default is five warm samples of three fresh parameter vectors each, one separate fresh-process
sample per successful backend/output pair, and a 40-second timeout per subprocess.
Every timed vector and result is retained and checked against the independent
oracle; timed rounds use new angles so they do not reuse native PTM cache entries.
Use more repetitions and paired machine-isolated runs for publication-grade
performance conclusions. There are no timing assertions in tests.

## What is compared

| Adapter | Representation | Gradient method | Noise in this suite |
|---|---|---|---|
| `native` | Pauli Cones sparse per-observable cones | One forward derivative propagation | Pauli errors, depolarizing, T1/T2, affine Z readout |
| `graph` | Pauli Cones reusable arithmetic graph | Reverse differentiation through graph | Same; bounded graph can report native fallback |
| `dense` | Pauli Cones bounded local density cones | Analytic channel adjoint | Same within configured cone limits |
| `dense_adjoint` | Independent full dense Kraus matrices in NumPy | Analytic reverse channel adjoint | Same, at most eight qubits |
| `qulacs` | Qulacs density matrix | Exact individual-gate parameter shift, summed into shared parameters | Same, at most eight qubits |
| `aer` | Aer density matrix | Exact individual-gate parameter shift, summed into shared parameters | Same, at most eight qubits |
| `pauli_prop` | IBM Rust Pauli propagation | Exact transformed-rotation parameter shift, summed into shared parameters | Pauli-Lindblad-compatible channels; T1 explicitly unsupported |

The independent NumPy adjoint is a transparent correctness and algorithm baseline,
not a claim to be an optimized third-party simulator. Qulacs and Aer gradients are
not finite differences. Their parameter-shift work is explicitly charged; the
timings do not establish the speed of a hypothetical native noisy adjoint in those
packages. The IBM adapter precomputes its Clifford transformation and rotation
structure and only rebinds the numeric angles during repeated evaluations.

See [comparison methodology](../docs/comparison-methodology.md) for timing and
accuracy rules, exact noise conversions, and the remaining coverage gaps.
