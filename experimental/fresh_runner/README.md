# Experimental restricted native runner

These retained C++ constructors implement a restricted first-run and precompiled-plan interface. They now compile the maintained package kernel through a relative include, preserving its C ABI. They are not installed or invoked by `pauli-cones run` and are not covered by the public Circuit schema. No device calibration files are bundled.

The fresh constructor accepts only the prototype's compact 127-wire, 144-directed-coupler physical-input schema and two/four layers of its SX/RZ/ECR matching schedule. It returns an all-Z summed observable and complete layer-angle gradients. It is not an arbitrary-circuit frontend, density-matrix or sampling simulator.

Build (development machine with zlib headers):

```bash
g++ -O3 -std=c++17 -pthread fresh_circuit_runner.cpp native_pauli.cpp -lz -o fresh_circuit_runner
```

The included precompiled-runner implementation supplies helpers to the fresh constructor. Its narrow custom JSON parser, resource limits, binary-plan ABI and error-exit semantics need further hardening before service use. The source is covered by the project license. The packaged general API and fresh constructor have different performance scopes; do not transfer a speed claim between them.
