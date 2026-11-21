# Reusable execution graphs

The graph backend prepares the *structure* of backward Pauli propagation once
and reuses it for parameter sweeps. Each later evaluation is a single pass over
stored arithmetic. On deep or wide circuits, preparation costs about one native
evaluation and the graph is ahead from the first vector; on small circuits it
costs a few milliseconds, repaid after tens of vectors. The native backend
remains the choice for one-off evaluations, for truncation far from a fixed
point in parameter space, and when the structural expansion exceeds the graph
budgets. [Measured
performance](performance.md#choosing-graph-or-dense-execution) has the numbers.

## What the graph represents

Each graph node holds one scalar Pauli coefficient. An edge multiplies a node
by one entry of a local Pauli transfer matrix and adds it to another node:
`node[dst] += T[entry] * node[src]`. Paths reaching the same Pauli term after a
stage share a destination node. A term with identity on a stage's wires keeps
its node and gets no edge, so a stage costs edges only for the terms it changes.
Every node is written by one stage, or accumulated into before it is next read,
and read only by later stages; evaluation is therefore one pass over the edges
in order, and reverse differentiation one pass back.

Observables are grouped into backward cones exactly as for the
[native solver](architecture.md#preparation-and-ownership): observables whose
combined cone fits one key width share a graph, which merges the Pauli terms
they share, and a joint graph that exceeds `max_terms` is split back into one
graph per observable. Each group is renumbered to its own wires, so wide
registers work whenever each cone fits 256 wires, and all graphs read one
shared bound copy of the circuit's stage matrices. The C++ compiler builds the
graphs, optionally on several threads, and stores them as flat arrays: two
32-bit node numbers and one byte of matrix entry per edge.

As in the native kernel, a wire is contracted into its known product input as
soon as backward propagation passes its earliest relevant operation. For a zero
input, local X and Y terms contribute zero while I and Z contribute one. For a
plus input, Z and Y contribute zero while I and X contribute one. Contracted
wires leave the term key, so affine relaxation terms can merge immediately.
Public circuits currently expose the zero product input.

The graph contains arithmetic operations, not stored evaluation results.
`bind(shifts)` changes local matrix entries for additive parameter shifts without
rebuilding the graph or changing the original circuit.

## Why zero-angle preparation is safe

A path absent at a particular angle can be necessary after rebinding. It can
also have a nonzero derivative at that same angle. Consequently, graph support
must not be inferred from the nonzero entries of one bound transfer matrix.

For RX, RY and RZ, the graph obtains support from the exact constant, sine and
cosine coefficient matrices. For fSim with fixed phi and variable theta, it
writes the local unitary as

\[
U(\theta)=A+\cos(\theta)B+\sin(\theta)C.
\]

Expanding `U† P U` gives transfer coefficient matrices in the basis

\[
1,\quad\cos\theta,\quad\sin\theta,\quad
\cos(2\theta),\quad\sin(2\theta).
\]

The union of their nonzero coefficients defines structural support. This is an
analytic expansion rather than sampling several angles. Tiny nonzero Fourier
coefficients are retained conservatively. The graph supports exactly the
current public angle parameters; derivatives of phi, noise strengths, and
readout probabilities are not implemented.

## Gradients without inverting noisy channels

Evaluation first propagates scalar coefficient arrays through the graph. When
gradients are requested, it saves these arrays and runs reverse differentiation
through the arithmetic. For an edge from node `i` to node `j` with weight `t`,

\[
v_j \mathrel{+}= t v_i,\qquad
\bar v_i \mathrel{+}= t\bar v_j,\qquad
\frac{\partial f}{\partial \theta_p}
\mathrel{+}=\bar v_j v_i\frac{\partial t}{\partial\theta_p}.
\]

Repeated occurrences of a shared parameter add to the same gradient entry.
This differentiates a scalar computation graph through fixed noise operations;
it never attempts to invert amplitude damping or another noisy channel. Its
arithmetic cost is proportional to the graph's edges plus local matrix binding,
rather than multiplying every stored coefficient by the number of parameters.

Every node keeps its value, so the reverse pass needs no saved history.
Value-only evaluation skips the reverse pass. Binding still computes the small
local derivative matrices, so value-only does not promise zero
derivative-related preparation cost. With `threads`, groups evaluate in
parallel when the graph has enough edges to repay starting threads; results
are summed in group order and do not depend on the thread count.

## Truncated graphs

A plan prepared with `truncation=Truncation(...)` on the graph backend
propagates coefficients, with their tangents, at the angles bound when the
graph is compiled, and discards terms there by the same rules as the native solver:
largest coefficient below `min_coefficient` times the observable's l1 norm, or
more than `max_weight` open wires. Coefficients are compared at one binding, so
which terms are discarded is fixed until the graph is compiled again. Discarding the terms
during compilation is what lets a truncated graph cover circuits whose exact
structure exceeds `max_terms`.

Each discarded term stays in the graph as a node that no edge reads. Every
evaluation computes those nodes at its own angles and reports the native
solver's rigorous bound from them: the sum of their coefficients' magnitudes
for the value and, for each derivative, their tangents plus the coefficient
times the generator bounds of that parameter's earlier stages. The bound
therefore holds at any binding, and it grows when the parameters move to where
the discarded terms matter. Truncated gradients use forward tangents rather
than the reverse pass, because the bound needs each discarded term's
derivative.

Near the angles it was compiled at, a truncated graph is about as accurate as
truncated native evaluation, often more so, because it keeps a term while any
tangent is large. Far from them it degrades: on a noiseless 20-qubit circuit,
jumps of about 2 radians made a fixed graph's error about seven times native's,
and on a noisy 36-qubit circuit jumps of 0.5 radians made it about 2.6 times.

The graph therefore compiles itself again when its bound shows this. After
each evaluation, if the value bound exceeds `Truncation.reprepare_factor`
(default 2) times the bound at the angles the graph was compiled for, the
graph is compiled again at the current angles and the evaluation repeated.
Bounds smaller than one threshold's worth never trigger it. Weight-only
truncation discards the same terms at every binding, so it never recompiles.
If a recompilation exceeds a budget, the plan keeps its current graph, whose
results and bound remain valid, records the reason in `reprepare_failed`, and
stops trying. `graph_reprepared` counts recompilations, and
`reprepare_factor=None` turns them off.

Along an optimiser-like walk of 40 steps of 0.1 radians, recompilation kept
the worst error at 0.019 on the noiseless 20-qubit circuit (fixed graph 0.25,
truncated native 0.053) and at 2.8e-4 on the noisy 36-qubit circuit (fixed 9.5e-4,
native 1.2e-3), recompiling 5 and 9 times. Including recompilation, the walks
took 112 ms and 352 ms against native's 306 ms and 447 ms. For evaluations
scattered far apart, every evaluation can trigger a recompilation of tens of
milliseconds, and truncated native is then cheaper.

The same rule makes a truncated graph keep more live terms than a value-only
native evaluation, so it reaches `max_terms` or the node budget on the largest
settings, where native can still return values but not gradients. Preparation
then falls back to native after the time spent compiling, up to tens of
seconds in the qualification runs. Where the graph fits, its preparation takes
0.03–2 s and each evaluation is 3–42 times faster than truncated native.

## Resource limits and fallback

Internal construction accepts `max_nodes` and `max_edges`, defaulting to 200,000
and 1,000,000, as totals over all groups; `Circuit.prepare` passes its own
defaults of 2,000,000 nodes and 10,000,000 edges (about 120 MB at most), large
enough for ansatze with hundreds of parameters. It also enforces the circuit's
`max_terms` on the live structural terms of each group after every stage.
Counts are checked as nodes and edges are added.
Exceeding one of these limits raises `GraphLimitExceeded`, a `ConeOverflow`
subclass. It never returns a partial graph or truncated numeric answer.

Structural support may exceed a budget even when a particular binding would
have many zero coefficients. A caller can catch **only** `GraphLimitExceeded`
during preparation and select ordinary native propagation, reporting the actual
backend and reason. Other errors must remain visible. Preparation time spent
on an unsuccessful graph attempt belongs in the total cost of that request.

`stats` reports node and edge counts, the number of cone groups, the maximum
live structural terms in one group, and the bytes of the stored graph arrays
and their evaluation buffers. Array bytes exclude Python metadata, the circuit
snapshot and temporary construction containers. These limits are numerical work
budgets, not a process-memory sandbox or a peak-RSS guarantee.

## Numerical contract

Fixed operations keep the existing local transfer-entry roundoff filter of
`5e-15`. Parametric support is structural. Unlike ordinary sparse propagation,
the exact reusable graph does not prune small coefficients, including during
readout initialization: doing that at one binding would make graph topology
invalid at another. A truncated graph prunes deliberately and accounts for it
in its bound, as described above. Values and
gradients may therefore differ from the native path by floating-point
accumulation order and by its existing coefficient cutoff. Neither path claims
a rigorous end-to-end error certificate.

No GPU, optional external simulator or runtime code generation is required by
this backend; it uses the same compiled extension as the native solver. For
larger cones the graph can run out of preparation budget, and a refused
preparation still costs its time. Keep both outcomes in benchmark reports; do
not summarize only successful fast cases.

## Qualification

`tests/test_graph.py` checks:

- Analytic transfer values and tangents for all parameterized gate families,
  including special fSim phi values.
- Rebinding graphs prepared at zero angles to nonzero angles and back.
- Shared-parameter accumulation across repeated gates and gate families.
- Independent explicit Kraus density evolution with depolarizing noise,
  relaxation, asymmetric readout, and both supported internal product inputs.
- Value-only equivalence, high-index wires, no-stage inputs, and a 128-wire
  affine-relaxation example whose structural frontier stays at one term.
- Fail-closed node, edge, and term budgets and invalid parameter vectors.
- A 600-wire register against the native solver, with identical results on one
  and four threads; fallback to `wide-native` over budget; and a joint group
  over the term cap splitting into single cones.

`tests/test_truncation_and_batching.py` runs the truncation checks on both the
native and graph backends: error within the reported bound for value and
gradient, an analytic weight-limit case, inactive truncation being bitwise
exact, thread-count independence on 600 wires, and the bound holding at
shifts of up to three radians from graphs prepared at zero and at random
angles. It also checks that recompilation fires and lowers the error far from
the compiled angles, that it is off for `reprepare_factor=None` and weight-only
truncation, and that a recompilation over budget keeps the current graph.

Run the qualification with `python -m pytest -q tests/test_graph.py` after
installing the test dependencies. The public API tests separately qualify its
snapshot ownership, input validation, and reported fallback behavior.
