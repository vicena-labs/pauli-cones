"""Compare a reviewed source checkout with this checkout through the public API.

Requires Python with NumPy and a C++17 compiler. Run from this checkout:

    python benchmarks/compare_revision.py --baseline-root /path/to/old/pauli-cones \
        --compiler g++ --output benchmarks/results/revision_local.json

A short portability check:

    python benchmarks/compare_revision.py --baseline-root /path/to/old/pauli-cones \
        --compiler g++ --fixture six_qubit_16_parameter_depth4_z --repetitions 1 \
        --output /tmp/revision_smoke.json

The baseline must contain src/pauli_cones/api.py and the native C++ source.
This benchmark builds both C ABIs with the same compiler in a temporary directory;
it does not install packages, change either checkout, or retain compiled binaries.
Builds happen before timing. Both isolated Python workers use the invoking Python
and its NumPy. Circuit construction and prepare() are reported separately from
Prepared.evaluate(), which includes validation, cold-angle rebinding, propagation,
weighted accumulation and returned Evaluation construction. Process launch, imports
and installation are excluded. Native output is checked across every timed vector.

The 127q fixture is synthetic, not the historical calibrated heavy-hex circuit.
Recorded JSON files describe their particular source hashes and build artifacts;
rerunning on another checkout/compiler/machine is not expected to reproduce times.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

CURRENT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ('six_qubit_16_parameter_depth4_z',
            'synthetic_127q_grid_depth_4_mean_local_z')
NATIVE_SOURCE = Path('src/pauli_cones/_core/native_pauli.cpp')


def build_fixture(Circuit, fixture):
    if fixture == 'six_qubit_16_parameter_depth4_z':
        circuit = Circuit(6, 16, max_terms=50000)
        for layer in range(4):
            for q in range(6):
                circuit.gate('RY', (q,), .23 + q * .07, param=(q + layer * 6) % 16)
                circuit.gate('RZ', (q,), -.31 + layer * .05, param=(q + layer * 6 + 7) % 16)
            for q in range(5):
                circuit.gate('CZ', (q, q + 1)).depolarize((q, q + 1), .013)
            for q in range(6):
                circuit.relax(q, .07, 1., 1.2)
        return circuit, [(1., {2: 'Z'})]
    if fixture != 'synthetic_127q_grid_depth_4_mean_local_z':
        raise ValueError(fixture)
    circuit = Circuit(127, 4, max_terms=50000)
    for layer in range(4):
        for q in range(127):
            circuit.gate('RY', (q,), float(.21 + .06 * (layer % 3)), param=layer)
        edges = []
        if layer % 2 == 0:
            for r in range(11):
                for column in range((layer // 2) % 2, 11, 2):
                    edges.append((r * 12 + column, r * 12 + column + 1))
        else:
            for r in range((layer // 2) % 2, 10, 2):
                for column in range(12):
                    edges.append((r * 12 + column, (r + 1) * 12 + column))
        for a, b in edges:
            if a >= 127 or b >= 127:
                continue
            circuit.gate('CZ', (a, b)).depolarize((a, b), .005)
            circuit.relax(a, 2e-7, 70e-6, 90e-6).relax(b, 2e-7, 70e-6, 90e-6)
    for q in range(127):
        circuit.readout(q, .02, .014)
    return circuit, [(1 / 127, {q: 'Z'}) for q in range(127)]


def source_hashes(source):
    return {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / 'src/pauli_cones').rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.cpp')}


def load_library(source, library_path):
    sys.path.insert(0, str(source / 'src'))
    import pauli_cones
    from pauli_cones._core import _native, _indexed
    imported = Path(pauli_cones.__file__).resolve()
    if not imported.is_relative_to((source / 'src').resolve()):
        raise RuntimeError('Imported a different source checkout than requested')
    lib = ct.CDLL(str(library_path))
    signature = [ct.c_int, ct.c_int, ct.c_int, ct.POINTER(_native.CStage),
                 ct.c_int, ct.POINTER(_native.CTerm), ct.c_int, ct.c_uint64,
                 ct.POINTER(ct.c_double), ct.POINTER(_native.CStats)]
    lib.run_pauli.argtypes = signature
    lib.run_pauli.restype = ct.c_int
    if hasattr(lib, 'run_pauli_value'):
        lib.run_pauli_value.argtypes = signature
        lib.run_pauli_value.restype = ct.c_int
    _native.library = lambda: lib
    _indexed.library = lambda: lib
    return lib


def worker(args):
    source = args.baseline_root if args.worker == 'original' else CURRENT_ROOT
    before = source_hashes(source)
    lib = load_library(source, args.library)
    import numpy as np
    from pauli_cones import Circuit
    from pauli_cones._core import _reference
    plans = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request['operation']
            if operation == 'metadata':
                result = {'package_source': 'src/pauli_cones/__init__.py',
                          'source_hashes_before': before,
                          'source_hashes_after': source_hashes(source),
                          'native_library': args.library.name,
                          'native_library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
                          'python_version': sys.version,
                          'numpy_version': np.__version__}
            elif operation == 'prepare':
                _reference.cached_gate_ptm.cache_clear()
                start = time.perf_counter()
                circuit, observables = build_fixture(Circuit, request['fixture'])
                built = time.perf_counter()
                plan = circuit.prepare(observables, backend='native')
                ready = time.perf_counter()
                plans[request['fixture']] = plan
                result = {'construction_ms': (built - start) * 1000,
                          'preparation_ms': (ready - built) * 1000,
                          'nqubits': circuit.n_qubits, 'nparams': circuit.n_parameters,
                          'nstages': len(circuit._impl.stages), 'observables': len(observables)}
            elif operation == 'evaluate':
                plan = plans[request['fixture']]
                _reference.cached_gate_ptm.cache_clear()
                start = time.perf_counter()
                if args.worker == 'original':
                    evaluated = plan.evaluate(request['shifts'])
                else:
                    evaluated = plan.evaluate(request['shifts'], gradients=request['gradients'])
                end = time.perf_counter()
                expected_width = plan.n_parameters if request['gradients'] else 0
                if (len(evaluated.gradient) != expected_width or
                    not all(math.isfinite(x) for x in (evaluated.value, *evaluated.gradient))):
                    raise RuntimeError('Invalid output shape or nonfinite numerical result')
                result = {'milliseconds': (end - start) * 1000,
                          'value': evaluated.value, 'gradient': list(evaluated.gradient)}
                if hasattr(plan, 'diagnostics'):
                    result['diagnostics'] = plan.diagnostics
            else:
                raise ValueError(operation)
            print(json.dumps({'ok': True, 'result': result}), flush=True)
        except Exception as exc:
            print(json.dumps({'ok': False, 'exception': type(exc).__name__, 'message': str(exc)}), flush=True)


def build_libraries(baseline, directory, compiler):
    command = shlex.split(compiler)
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError('C++ compiler unavailable; pass --compiler with a C++17 compiler')
    version = subprocess.run([*command, '--version'], text=True, capture_output=True, check=True)
    extension = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
    flags = ['-O3', '-std=c++17', '-dynamiclib' if sys.platform == 'darwin' else '-shared']
    if sys.platform != 'win32':
        flags.append('-fPIC')
    libraries = {}
    for variant, root in (('original', baseline), ('current', CURRENT_ROOT)):
        target = directory / f'{variant}_pauli{extension}'
        subprocess.run([*command, *flags, str(root / NATIVE_SOURCE), '-o', str(target)], check=True)
        libraries[variant] = target
    return libraries, {'compiler': Path(command[0]).name,
                       'compiler_version': version.stdout.splitlines()[0], 'flags': flags,
                       'platform': platform.system(), 'machine': platform.machine(),
                       'build_timing': 'excluded; both libraries built before starting workers'}


def run(args, libraries):
    processes = {variant: subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), '--baseline-root', str(args.baseline_root),
         '--worker', variant, '--library', str(libraries[variant])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        for variant in ('original', 'current')}

    def request(variant, payload):
        process = processes[variant]
        process.stdin.write(json.dumps(payload) + '\n')
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f'{variant} worker exited: {process.poll()}')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response)
        return response['result']

    try:
        rows = []
        for fixture in args.fixture or FIXTURES:
            preparations = {variant: request(variant, {'operation': 'prepare', 'fixture': fixture})
                            for variant in ('original', 'current')}
            nparams = preparations['original']['nparams']
            series = {'original_gradient': [], 'current_gradient': [], 'current_value': []}
            for repetition in range(args.repetitions):
                shifts = [math.sin(repetition * .37 + parameter) * .01 for parameter in range(nparams)]
                order = list(series) if repetition % 2 == 0 else list(reversed(series))
                for engine in order:
                    variant = engine.split('_', 1)[0]
                    result = request(variant, {'operation': 'evaluate', 'fixture': fixture,
                                               'shifts': shifts, 'gradients': engine != 'current_value'})
                    series[engine].append({'repetition': repetition, 'shifts': shifts, **result})
            max_value_error = max(abs(a['value'] - b['value'])
                                  for a, b in zip(series['original_gradient'], series['current_gradient']))
            max_gradient_error = max(abs(x - y)
                                     for a, b in zip(series['original_gradient'], series['current_gradient'])
                                     for x, y in zip(a['gradient'], b['gradient']))
            max_scalar_error = max(abs(a['value'] - b['value'])
                                   for a, b in zip(series['original_gradient'], series['current_value']))
            if max(max_value_error, max_gradient_error, max_scalar_error) > args.atol:
                raise RuntimeError(f'Accuracy gate failed on {fixture}; no speed result accepted')
            row = {'fixture': fixture, 'setup': preparations,
                   'max_original_current_value_difference': max_value_error,
                   'max_original_current_gradient_difference': max_gradient_error,
                   'max_original_current_scalar_difference': max_scalar_error,
                   'timings': {engine: {'median_ms': statistics.median(r['milliseconds'] for r in records),
                                        'minimum_ms': min(r['milliseconds'] for r in records),
                                        'maximum_ms': max(r['milliseconds'] for r in records),
                                        'first_request_total_ms': preparations[engine.split('_', 1)[0]]['construction_ms']
                                        + preparations[engine.split('_', 1)[0]]['preparation_ms'] + records[0]['milliseconds'],
                                        'samples': records}
                               for engine, records in series.items()}}
            rows.append(row)
            print(json.dumps({'fixture': fixture,
                              'median_ms': {engine: item['median_ms'] for engine, item in row['timings'].items()},
                              'max_value_difference': max_value_error,
                              'max_gradient_difference': max_gradient_error}), flush=True)
        metadata = {variant: request(variant, {'operation': 'metadata'}) for variant in processes}
        for variant, details in metadata.items():
            if details['source_hashes_before'] != details['source_hashes_after']:
                raise RuntimeError(f'{variant} sources changed during timing; discard this run')
        return {'purpose': __doc__,
                'timing_contract': f'{args.repetitions} distinct parameter vectors; alternating engine order; angle cache cleared before every timed public evaluate call',
                'setup_contract': 'one construction/prepare observation per fixture and revision; no broad setup-speed inference',
                'accuracy_absolute_tolerance': args.atol, 'metadata': metadata, 'rows': rows}
    finally:
        for process in processes.values():
            process.stdin.close()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline-root', type=Path, required=True,
                        help='Source root of the reviewed/earlier pauli-cones package')
    parser.add_argument('--compiler', default=os.environ.get('CXX', 'c++'),
                        help='GCC/Clang-style C++17 compiler command (default CXX or c++)')
    parser.add_argument('--output', type=Path, default=Path('revision_comparison_local.json'))
    parser.add_argument('--repetitions', type=int, default=11)
    parser.add_argument('--fixture', choices=FIXTURES, action='append')
    parser.add_argument('--atol', type=float, default=2e-11)
    parser.add_argument('--worker', choices=('original', 'current'), help=argparse.SUPPRESS)
    parser.add_argument('--library', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.baseline_root = args.baseline_root.expanduser().resolve()
    if args.repetitions < 1 or not math.isfinite(args.atol) or args.atol <= 0:
        parser.error('repetitions and finite atol must be positive')
    for root in (args.baseline_root, CURRENT_ROOT):
        if not (root / NATIVE_SOURCE).is_file() or not (root / 'src/pauli_cones/api.py').is_file():
            parser.error('Both roots must contain the Pauli Cones public API and native C++ source')
    if args.worker and args.library is None:
        parser.error('worker requires its compiled library')
    return args


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix='pauli_revision_') as temporary:
        libraries, build = build_libraries(args.baseline_root, Path(temporary), args.compiler)
        result = run(args, libraries)
        result['build'] = build
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
