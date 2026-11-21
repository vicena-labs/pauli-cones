"""Compile at installation/build time, never at import or evaluation time."""
import sys
from setuptools import Extension, setup
# The batch entry point runs independent cones on std::thread.
flags = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17", "-pthread"]
links = [] if sys.platform == "win32" else ["-pthread"]
setup(ext_modules=[Extension(
    "pauli_cones._core._native_kernel",
    sources=["src/pauli_cones/_core/native_pauli.cpp", "src/pauli_cones/_core/module.cpp"],
    language="c++", extra_compile_args=flags, extra_link_args=links,
)])
