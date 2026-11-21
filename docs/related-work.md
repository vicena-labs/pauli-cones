# Related work and acknowledgements

Quantum Circuit Simulator's main solver, the Pauli-cone method, implements Heisenberg-picture Pauli propagation: an observable is expanded in the Pauli basis and propagated backward through the circuit, and its expectation is read off against the input state. The method is known under several names, including Pauli back-propagation, sparse Pauli dynamics, Pauli paths and LOWESA. They differ mainly in how they truncate. This page credits the papers and software Quantum Circuit Simulator builds on and states what it adds. Arxiv identifiers were checked on 2026-11-20.

## Theory of Pauli propagation under noise

- Aharonov, Gao, Landau, Liu and Vazirani showed that noise makes random-circuit sampling classically simulable in polynomial time through Pauli paths ([arXiv:2211.03999](https://arxiv.org/abs/2211.03999)).
- Schuster, Yin, Gao and Yao extended this to the expectation of any observable on any noisy circuit, with small error on average over input states ([arXiv:2407.12768](https://arxiv.org/abs/2407.12768)).
- Mele et al. showed that noise makes most circuits effectively shallow ([arXiv:2403.13927](https://arxiv.org/abs/2403.13927)). Angrisani et al. treated noiseless random circuits ([arXiv:2409.01706](https://arxiv.org/abs/2409.01706)) and arbitrary local noise ([arXiv:2501.13101](https://arxiv.org/abs/2501.13101)). Martinez et al. covered non-unital noise ([arXiv:2501.13050](https://arxiv.org/abs/2501.13050)). González-García, Cirac and Trivedi gave cases where the average-case guarantees fail ([arXiv:2407.16068](https://arxiv.org/abs/2407.16068)).
- Restricting a local expectation to its light cone predates all of these ([Bravyi, Gosset and Movassagh, arXiv:1909.11485](https://arxiv.org/abs/1909.11485)).

## Practical truncated propagation

- Clifford perturbation theory ([Begušić, Hejazi and Chan, arXiv:2306.04797](https://arxiv.org/abs/2306.04797)) and sparse Pauli dynamics ([Begušić and Chan, arXiv:2306.16372](https://arxiv.org/abs/2306.16372); [Begušić, Gray and Chan, arXiv:2308.05077](https://arxiv.org/abs/2308.05077); [arXiv:2409.03097](https://arxiv.org/abs/2409.03097)) simulated IBM's 127-qubit kicked-Ising experiment with coefficient truncation.
- LOWESA ([Fontana et al., arXiv:2306.05400](https://arxiv.org/abs/2306.05400); [Rudolph et al., arXiv:2308.09109](https://arxiv.org/abs/2308.09109)) truncates by trigonometric frequency and builds a surrogate valid across angles. Shao et al. analysed Pauli-path simulation of noisy variational circuits ([arXiv:2306.05804](https://arxiv.org/abs/2306.05804)).
- IBM's operator backpropagation uses the same expansion to shorten circuits run on hardware and budgets its truncation error ([Fuller et al., arXiv:2502.01897](https://arxiv.org/abs/2502.01897)).
- The benchmark circuit most of these papers share is the kicked-Ising experiment of Kim et al., Nature 618, 500 (2023), [doi:10.1038/s41586-023-06096-3](https://doi.org/10.1038/s41586-023-06096-3).

## Software

- **PauliPropagation.jl** and its framework paper by Rudolph, Jones, Teng, Angrisani and Holmes ([arXiv:2505.21606](https://arxiv.org/abs/2505.21606); [repository](https://github.com/MSRudolph/PauliPropagation.jl)) are the reference implementation and the standard review of the method. Its truncation options, its zero-state filtering, its reusable surrogate and its `rewindgradient`, based on Lin, Granet, Hémery and Dreyer's backpropagation through Pauli propagation ([arXiv:2607.15184](https://arxiv.org/abs/2607.15184)), are all prior art for features of Quantum Circuit Simulator.
- **IBM pauli-prop** ([repository](https://github.com/Qiskit/pauli-prop)) propagates Pauli sums through Qiskit circuits with coefficient and term-count truncation.
- **PauliStrings.jl** ([arXiv:2410.09654](https://arxiv.org/abs/2410.09654)), **Majorana propagation** ([arXiv:2503.18939](https://arxiv.org/abs/2503.18939)) and Begušić's **spd** ([repository](https://github.com/tbegusic/spd)) are further CPU implementations.
- **NVIDIA cuPauliProp** ([documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/overview.html)) and **PADO-Pauli** ([Kim and Lee, arXiv:2610.17755](https://arxiv.org/abs/2610.17755)) run Pauli propagation on GPUs with noise, truncation and gradients.
- Quantum Circuit Simulator's tests and benchmarks compare against Qiskit Aer, Qulacs ([arXiv:2011.13524](https://arxiv.org/abs/2011.13524)), PennyLane Lightning with adjoint differentiation ([arXiv:2403.02512](https://arxiv.org/abs/2403.02512); method from [Jones and Gacon, arXiv:2009.02823](https://arxiv.org/abs/2009.02823)), quimb and Stim ([arXiv:2103.02202](https://arxiv.org/abs/2103.02202)).

## Compiled graphs, gradients and error estimates

- Compiling the propagation once and replaying it for new parameters appears in LOWESA, in PauliPropagation.jl's surrogate, in symbolic Pauli propagation ([Monaco et al., arXiv:2512.16674](https://arxiv.org/abs/2512.16674)), in PADO-Pauli, which also propagates many observables in one shared basis, and in monoprop ([Miller, arXiv:2610.21000](https://arxiv.org/abs/2610.21000)).
- Reverse-mode gradients over Pauli propagation appear in PADO-Pauli and monoprop, which differentiate through the stored propagation, so noise channels never need inverting. Lin et al. instead save memory by exploiting the reversibility of unitary circuits, as adjoint methods such as Jones and Gacon's do.
- The value bound, the sum of the magnitudes of discarded coefficients, is the standard worst-case bound stated in the PauliPropagation.jl review and in IBM's operator backpropagation. PauliPropagation.jl's numerical certificates estimate the mean-square truncation error averaged over random angles by Monte Carlo sampling. LOWESA bounds the error averaged over parameters.

## What Quantum Circuit Simulator adds

Its Pauli-cone solver is an engineering contribution to this method. It does not claim the method, the light-cone restriction, the discarded-coefficient value bound, compiled replay or reverse-mode differentiation through noise. It contributes:

- **A bound at every binding for compiled truncated graphs.** A truncated graph keeps its discarded terms as nodes that no edge reads. Every evaluation computes the error bound from them at its own angles, so the bound holds at any binding, not only where the graph was compiled or on average. The graph compiles itself again when the bound grows. PADO-Pauli lists such a certificate as future work.
- **A gradient bound** for truncated evaluations: the discarded terms' derivatives plus their coefficients times the generator norms of earlier stages with the same parameter.
- **Two implementation choices:** each wire is contracted into an arbitrary product input state as soon as no remaining operation touches it, and observables are propagated jointly only when their light cones together fit one key width.
- **A tested CPU implementation** with a C++ kernel, a Python reference, local density matrices for small cones, and an independent qualification suite.

Return to [README](../README.md).
