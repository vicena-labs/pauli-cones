# Changelog

## 0.4.0rc2 (release candidate, not yet published) — faster native solver and graph, truncated mode, interop and agent-first CLI

- Add an optional truncated mode, `prepare(..., truncation=Truncation(min_coefficient, max_weight))`, on the `native`, `graph` and `auto` backends. It discards terms whose coefficients fall below a fraction of the observable's l1 norm, or that act on more than `max_weight` open wires. Every evaluation returns a rigorous `error_bound` and `gradient_error_bound`; exact plans return `None`. Other backends refuse truncation.
- Rewrite the native term store as a flat open-addressing table that a stage updates in place. Terms with identity on the stage's wires are not touched, terms the stage maps to themselves are rescaled in place, and only moved or branching terms are rehashed. Keys use one to four 64-bit words, chosen from each group's wire count.
- Propagate observables jointly when their combined backward cone fits the key width the widest single cone needs. The circuit is linear in the observable, so the result is unchanged while shared Pauli terms merge. A group that exceeds the term cap is split back into single cones, so no request that succeeded before now fails.
- Evaluate all cone groups in one native call, `run_pauli_batch`, with an optional `threads` count. Results are summed in group order and do not depend on the thread count. Wide `native` registers now use this path with one shared bound copy of the circuit instead of one prepared plan per cone.
- Read circuits from OpenQASM 2 (built in), OpenQASM 3 and Qiskit (`pauli_cones.interop`), with exact decompositions into the native gates. Qiskit Parameters entering angles as `p + b` or `-p + b` become shared parameters; others are bound per evaluation. Observables can be written as `"0.5*Z0*Z1 - X3"`, Qiskit labels, `SparsePauliOp` or JSON terms. `pauli_cones.interop.expectation` evaluates one in one call.
- Add `pauli_cones.estimator.Estimator`, a Qiskit Estimator V2 primitive that reuses one plan per observable across a parameter sweep and reports truncation error bounds.
- Make the command line agent-first. `run` and the new `plan` dry run accept OpenQASM files with `--observable` and a uniform noise model. New `capabilities`, `schema` and `guide` commands describe the tool as JSON and Markdown. Every error carries a `hint`. The JSON request accepts an OpenQASM form.
- When several joint cone groups exceed the term cap, split them all at once and retry once, instead of retrying the whole batch once per group; the batch and graph compiler report every group's status.
- Compile and evaluate the `graph` backend in C++. Observables are grouped into cones as for the native solver; each group's propagation is stored as flat edge arrays, terms a stage leaves unchanged get no edge, and an evaluation is one pass over the edges with one reverse pass for gradients. Graph plans take `threads`, support registers wider than 256 wires (`wide-graph`) and fall back to `wide-native` over budget. `stats` adds `cone_groups`; `nodes` and `edges` count the smaller graph.
- Truncate graphs too. A truncated graph chooses its discarded terms at the angles bound during preparation, keeps each discarded term as a node no edge reads, and computes the error bound from those terms' coefficients at every evaluation's own angles, so the bound holds at any binding. Truncated gradients use forward tangents, which the gradient bound needs. `numerics['truncation']['discarded_terms_chosen']` says which policy applies. When the value bound exceeds `Truncation.reprepare_factor` (default 2) times its value where the graph was compiled, the graph compiles again at the current angles; a recompilation over budget keeps the current graph and stops retrying. Diagnostics report `graph_reprepared` and `reprepare_failed`.
- Rebind rotation angles with vectorised NumPy writes into the native stage array instead of a Python loop over stages.
- The original C symbols and structure layouts are unchanged.
- Make the sparse coefficient cutoff relative to the observable's l1 norm in the native kernel and Python reference. Observables with weights of 1e-16 or less previously returned exactly 0 on `native` and `python` while `graph` and `dense` returned the correct value. `numerics` reports `coefficient_cutoff_relative_to`.
- Enforce `max_cone_qubits` on registers of 256 wires or fewer; it was previously read only by the wide-register planner.

## 0.3.0a1 — scalable logical registers and backend extension

- Support 1–16,384 logical qubits through exact backward-causal-cone remapping to the existing <=256-wire native numerical engine; refuse wider active cones.
- Share the reverse graph pass for all requested observables, group identical local stage signatures and contract idle product-state wires analytically.
- Add explicit trusted external backend registration (`ext-*`) including full-width access, so non-Pauli engines can be developed independently.
- Add 4,096/16,384-qubit runnable demos, regression tests, and documented limits. Existing native numerical kernel and historical speed claims remain unchanged.


## 0.2.0a2 — developer toolkit

Add a validated, undirected coupling graph helper and runnable architecture playground for line/grid/custom graphs. Add a small educational MaxCut QAOA circuit with verified analytic gradients, a variational-energy gradient-descent example, an accuracy-checked within-package backend comparison, and a capability inventory that explicitly distinguishes integrated speedups from retained experimental runners and separate general-purpose quantum solvers. No numerical C++ or Python backend is changed.

## 0.2.0a1 — architecture cleanup and reproducible comparisons

Contract known product inputs as soon as backward propagation reaches each wire's first operation, including idle readout before expansion. Specialize occupied local PTM rows, preserve derivative-only transitions, and construct rotation transfers analytically. Add scalar native execution, public value-only methods and CLI controls; canonicalize duplicate observables.

Add bounded reusable arithmetic graphs with reverse differentiation, local density matrices with channel-adjoint gradients, and an explicit sparse-to-dense capacity fallback. Report the selected representation, fallback reason and numerical policy. Preserve existing native C ABI layouts and use one maintained kernel for experimental runners.

Add independent noisy numerical tests, matched external CPU comparisons, fresh-angle revision comparisons, source hashes and raw failure records. Apply Apache-2.0 licensing for source distribution. No compiled release or blanket simulator-speed claim is made.

## 0.1.0a1 — source-only alpha candidate

Extract a standalone package from the numerical research prototype. Retain the original sparse Pauli recurrence and float64 filtering. Add a validated Circuit/Prepared API, explicit native/reference backends, versioned JSON CLI with worker deadlines, self-contained examples and installed-wheel tests. Build the native extension at installation time rather than import time.

Calibrated device fixtures and private application content are excluded. The initial publication and licensing decisions remain pending. This is not a PyPI release or a production readiness claim.
