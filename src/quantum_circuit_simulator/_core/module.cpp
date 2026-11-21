// Package-loader shim only. Numerical C ABI remains in native_pauli.cpp.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_kernel", "Pauli transfer kernel", -1, nullptr};
PyMODINIT_FUNC PyInit__native_kernel(void) { return PyModule_Create(&module); }
