# Dependencies and third-party material

The standalone project includes the Pauli reference, native wrapper, indexed-cone wrapper, reusable graph, local density backend and C++ kernel. These sources use the project Apache-2.0 license. Preserve any contributor or third-party notices when extending them. A source inventory is not independent ownership or patent clearance.

NumPy is a runtime dependency, installed separately. Setuptools is a build dependency; pytest is a test dependency. They are not vendored in the source. The numerical extension uses the platform Python API and C++ standard library. Binary distribution review must account for any runtime libraries bundled into future platform wheels; the local wheel is not a portable manylinux release.

The optional experimental fresh-runner sources link to the system zlib library when compiled. They are not included in the installed Python wheel and are not compiled by the default package build.

No IBM fake-backend calibration snapshots, copied Qiskit/Aer/Stim/Qulacs source, old CI artifacts, private application source, credentials, infrastructure settings or private Git history are intentionally included. Examples and benchmarks use synthetic circuits and noise values. A future calibration import may require its own source license and notices.

The optional comparison tools install pauli-prop, Qulacs, Qiskit and Qiskit Aer separately. They are not runtime dependencies or vendored implementations. The recorded versions are pauli-prop 0.2.1 (Apache-2.0), Qulacs 0.6.14 (MIT), Qiskit 2.6.2 (Apache-2.0) and Aer 0.18.4 (Apache-2.0); consult each installed distribution's own notices. The adapters are original API integration code.

The complete Apache-2.0 text is in [LICENSE](LICENSE); the upstream source is https://www.apache.org/licenses/LICENSE-2.0.txt.
