# Validation of source alphas 0.2.0a1 and 0.2.0a2

Recorded November 19, 2026, on Linux x86_64 with Python 3.12.14, NumPy 2.5.5, GCC 14.2 and pytest 9.2.0. This is a local qualification record, not a claim that hosted CI or every operating system passed.

## Numerical and integration checks

- The core/public-API/CLI suite passes **192 tests** from the working tree.
- Building a source distribution and local wheel succeeds. Installing that wheel into a separate environment and running the same suite passes **192 tests**. The tested runtime is the installed package, not the editable checkout.
- The optional external-adapter suite passes **14 tests** with the pinned packages in `benchmarks/requirements.txt`.
- Native C++ passes strict warning/syntax checks. The retained experimental fresh constructor and its reference to the maintained kernel compile through syntax checking; this does not qualify the experimental parser for untrusted service use.
- The final external comparison records 84 requested backend/output pairs, including 67 eligible successes and all unsupported, capped, truncated and timeout outcomes. There are no accuracy failures among returned comparisons. Source hashes match the delivered runtime and harness.

Tests cover independent Kraus density calculations, analytic noisy products, gradients through zero angles, repeated shared parameters, reversed and noncontiguous wires, internal plus inputs, masks above 64 wires, scalar ABI output boundaries, dense/graph budgets, preparation and evaluation fallbacks, idle readout, small graph coefficients and complete CLI error handling.

The final revision comparison builds both old and new native kernels with the same compiler, uses eleven distinct vectors and clears each worker's angle-transfer cache before timing. See [performance evidence](performance.md) and [complete external tables](../benchmarks/results/README.md) for scope and measurements.

## Source and release checks

The source export is selected from the standalone Git checkout. It contains source, tests, examples, documentation, license text and benchmark JSON. Compiled libraries, wheels, environment files, private calibration data and imported private repository history are excluded.

The built-in source hygiene checker passes the tracked tree and exported source. An additional detect-secrets 1.5.0 scan runs with live credential verification disabled. Entropy findings are reviewed against the benchmark's explicit SHA-256 fields; source/library digests are not credentials. No unresolved credential finding remains. These static checks cannot prove absence of unknown or intentionally obfuscated secrets.

The mechanical release check confirms Apache-2.0 metadata, LICENSE, NOTICE and security guidance. It does not determine ownership or trademark rights. Future contributed source and third-party material still need appropriate permission and notices.

No binary build is included in the source release. A clean local commit and downloadable source archive do not by themselves establish that a public repository exists; publication must be verified separately in GitHub.

## Remaining qualifications

The graph is experimental and can be slower or require more preparation memory. Dense fallback is deliberately bounded. Numeric filtering is declared, and no global numerical-error certificate is claimed. GPU/Julia/MPO/trajectory performance, the historical calibrated 127-wire family, hosted application latency and other operating systems remain outside this run's qualification. See [comparison methodology](comparison-methodology.md).

## Developer-toolkit extension: 0.2.0a2 (November 19, 2026)

Version 0.2.0a2 adds a validated undirected topology helper, an architecture
playground (line/grid/custom coupling graphs), a small MaxCut QAOA circuit, a
toy variational-energy loop and a short correctness-checked backend comparison.
The sparse numerical kernel, reference backend, graph and dense backends were
not modified. The original 0.2.0a1 performance numbers above do not acquire
new scope from these examples.

The complete Python test suite ran against an **installed copy** on Linux
x86_64 with CPython 3.13.5, NumPy 2.3.5 and pytest 9.0.2: **208 passed**.
The new tests check topology validation/edge matching, a custom graph,
architecture-probe parity, zero-parameter MaxCut expectations, analytic QAOA
gradients against independent finite differences and the bounded dense
backend, toy variational energy, and a five-backend smoke comparison. The
source build and example CLIs were also run successfully.

No hosted hardware, physical calibration, general chip routing, quantum error
correction, gate sampling or full-state simulations were added or validated.
The source release must pass its scanner again after any changes.

Return to [README](../README.md).

## Large-register extension: 0.3.0a1 (November 19–20, 2026)

The logical width increases to 16,384 without changing native 256-wire Pauli masks. A shared reverse query-bitset pass remaps every selected local cone into the existing native engine and adds opt-in trusted external backend registration. Wide-register tests include analytic/local-reference correctness, 4,096/16,384-wire examples, resource refusals, custom backends and topology limits. See [large-register qualification](large-register-validation.md) for exact coverage and observed local timings. This extension has no independent state-of-the-art head-to-head or live Vicena qualification.
