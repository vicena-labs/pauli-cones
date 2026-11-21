"""Compile at installation/build time, never at import or evaluation time."""
import sys
from setuptools import Extension, setup
# The batch entry point runs independent cones on std::thread.
flags = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17", "-pthread"]
links = [] if sys.platform == "win32" else ["-pthread"]
setup(ext_modules=[Extension(
    "quantum_circuit_simulator._core._native_kernel",
    sources=["src/quantum_circuit_simulator/_core/native_pauli.cpp", "src/quantum_circuit_simulator/_core/module.cpp"],
    language="c++", extra_compile_args=flags, extra_link_args=links,
)])
