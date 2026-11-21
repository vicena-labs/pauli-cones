"""JSON-only command line for agents and scripts: every command prints one JSON
document (except `guide`, which prints Markdown), and every failure carries
error.type, error.message and error.hint with a concrete next step.

    pauli-cones capabilities | schema | guide | example | doctor
    pauli-cones devices | device NAME [--param k=v]   # chip catalogue and descriptions
    pauli-cones compile INPUT --device NAME           # route onto a chip: SWAPs, depth, duration, layouts
    pauli-cones sweep [INPUT | --family NAME --sizes 8,16] --device SPEC ... --noise-scales 0,1,2
    pauli-cones design --base NAME [--modules 4 --links ring] [--t1 ... --t2 ...] --out chip.json
    pauli-cones compare CHIP CHIP ... --family qaoa-ring --sizes 8,12   # rank chip designs
    pauli-cones plan  INPUT [--observable EXPR]     # dry run: feasibility and recommendations
    pauli-cones run   INPUT [--observable EXPR] [options]

INPUT is a JSON request, an OpenQASM file (.qasm), or - for standard input.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from . import __version__

MAX_INPUT = 4 * 1024 * 1024


def emit_error(kind, message, status=2, hint=None):
    error = {'type': kind, 'message': message}
    if hint:
        error['hint'] = hint
    print(json.dumps({'ok': False, 'error': error}))
    return status


def main(argv=None):
    p = argparse.ArgumentParser(prog='pauli-cones', description='Noisy Pauli expectations and shared-parameter '
                                'gradients. Every command prints JSON; start with `pauli-cones guide`.')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('guide', help='Print the usage guide for agents (Markdown)')
    sub.add_parser('capabilities', help='Supported gates, noise, limits, backends and exit codes as JSON')
    sub.add_parser('schema', help='JSON Schema of requests and the response fields')
    sub.add_parser('example', help='Print a self-contained noisy two-qubit request')
    sub.add_parser('doctor', help='Report installed backend availability without compiling')
    sub.add_parser('devices', help='The chip catalogue: architecture families and their parameters')
    dev = sub.add_parser('device', help='Describe one chip: connectivity summary, native gates, noise')
    dev.add_argument('name', help='Catalogue name (see `pauli-cones devices`) or a device .json file')
    dev.add_argument('--param', action='append', default=[], help='Catalogue parameter, e.g. --param rows=6')
    dev.add_argument('--edges', action='store_true', help='Print the full description with every coupler')
    for name, text in (('plan', 'Dry run: cone sizes, a rigorous term bound and recommended next commands'),
                       ('run', 'Evaluate one request: values, gradients and diagnostics'),
                       ('compile', 'Route an OpenQASM circuit onto a chip and report SWAPs, depth and duration')):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument('input', nargs='?', default='-', help='JSON request, OpenQASM file, or - for stdin')
        cmd.add_argument('--observable', help='For OpenQASM input: e.g. "0.5*Z0*Z1 - X3" or "0.5*IXZ"')
        cmd.add_argument('--values', help='For OpenQASM 3 input parameters: JSON list of values, or of vectors')
        cmd.add_argument('--noise-1q', type=float, default=0., help='OpenQASM input: depolarizing after 1-qubit gates')
        cmd.add_argument('--noise-2q', type=float, default=0., help='OpenQASM input: depolarizing after 2-qubit gates')
        cmd.add_argument('--readout', help='OpenQASM input: "p01,p10" readout error on every wire')
        cmd.add_argument('--max-terms', type=int, help='Term cap per cone (OpenQASM input; default 50000)')
        cmd.add_argument('--device', help='OpenQASM input: a chip from the catalogue or a .json file; applies '
                                          'its noise and checks its connectivity')
        cmd.add_argument('--device-param', action='append', default=[], help='Catalogue parameter, e.g. rows=6')
        cmd.add_argument('--compile', action='store_true', help='With --device: route the circuit onto the chip '
                         'and simulate it with gate errors, idle T1/T2 and readout')
        cmd.add_argument('--router', choices=('greedy', 'sabre'), default='greedy',
                         help='SWAP router; sabre needs Qiskit and usually inserts fewer SWAPs')
        cmd.add_argument('--layout', choices=('greedy', 'trivial'), default='greedy', help='Initial qubit placement')
        cmd.add_argument('--noise-scale', type=float, default=1.0, help='Multiply every device error (noise sweeps)')
        cmd.add_argument('--no-idle-noise', action='store_true', help='Skip T1/T2 relaxation of idle qubits')
        cmd.add_argument('--timeout', type=float, default=60.0, help='Worker wall deadline in seconds')
        if name == 'run':
            cmd.add_argument('--backend', choices=('native', 'python', 'graph', 'dense', 'auto'), default='native')
            cmd.add_argument('--value-only', action='store_true', help='Skip gradients')
            cmd.add_argument('--min-coefficient', type=float, default=None,
                             help='Truncate terms below this fraction of the observable l1 norm')
            cmd.add_argument('--max-weight', type=int, default=None,
                             help='Truncate terms acting on more open wires than this')
            cmd.add_argument('--threads', type=int, default=1, help='Threads for independent cone groups')
    sw = sub.add_parser('sweep', help='Compile and solve over chips, noise scales and sizes; one row per point')
    sw.add_argument('input', nargs='?', help='OpenQASM file; or use --family')
    sw.add_argument('--family', help='Built-in circuit family: ghz, hardware-efficient, qaoa-ring, ising-trotter')
    sw.add_argument('--sizes', help='Comma-separated qubit counts for --family, e.g. 8,16,32')
    sw.add_argument('--family-param', action='append', default=[], help='Family parameter, e.g. p=2')
    sw.add_argument('--observable', help='Observable; a family supplies its own when omitted')
    sw.add_argument('--device', action='append', default=[],
                    help='Chip, repeatable: catalogue name, "name:k=v,k=v", or a .json file')
    sw.add_argument('--noise-scales', default='1', help='Comma-separated noise multipliers, e.g. 0,0.5,1,2')
    sw.add_argument('--router', choices=('greedy', 'sabre'), default='greedy')
    sw.add_argument('--layout', choices=('greedy', 'trivial'), default='greedy')
    sw.add_argument('--no-idle-noise', action='store_true')
    sw.add_argument('--format', choices=('json', 'markdown', 'csv'), default='json')
    sw.add_argument('--timeout', type=float, default=600.0, help='Worker wall deadline in seconds')
    ds = sub.add_parser('design', help='Build a chip description: base layout, modules, defects, couplers, physics')
    ds.add_argument('--base', required=True, help='Catalogue name, "name:k=v,k=v", or a device .json file')
    ds.add_argument('--name', help='Name of the new chip')
    ds.add_argument('--modules', type=int, help='Tile this many copies of the base chip')
    ds.add_argument('--links', choices=('line', 'ring', 'all'), default='ring', help='How modules connect')
    ds.add_argument('--links-per-pair', type=int, default=1)
    ds.add_argument('--link-error', type=float, help='Two-qubit infidelity of inter-module links')
    ds.add_argument('--remove-qubits', help='Comma-separated defective qubits to remove (the rest are renumbered)')
    ds.add_argument('--add-couplers', help='Comma-separated a-b pairs, e.g. 0-5,3-9')
    ds.add_argument('--coupler-error', type=float, help='Infidelity of the added couplers')
    ds.add_argument('--remove-couplers', help='Comma-separated a-b pairs')
    for flag in ('--t1', '--t2', '--time-1q', '--time-2q'):
        ds.add_argument(flag, type=float, help='Seconds; with all four, gate errors follow from coherence')
    ds.add_argument('--control-error-1q', type=float, default=0.)
    ds.add_argument('--control-error-2q', type=float, default=0.)
    ds.add_argument('--readout', help='"p01,p10"')
    ds.add_argument('--out', help='Write the JSON description to this file as well as printing a summary')
    cp = sub.add_parser('compare', help='Rank chips on a benchmark family: deviation, SWAPs, gates, duration')
    cp.add_argument('chips', nargs='+', help='Catalogue names, "name:k=v,k=v" or device .json files')
    cp.add_argument('--family', default='qaoa-ring', help='ghz, hardware-efficient, qaoa-ring or ising-trotter')
    cp.add_argument('--sizes', default='8,12')
    cp.add_argument('--family-param', action='append', default=[])
    cp.add_argument('--noise-scale', type=float, default=1.0)
    cp.add_argument('--router', choices=('greedy', 'sabre'), default='greedy')
    cp.add_argument('--format', choices=('json', 'markdown', 'csv'), default='json')
    cp.add_argument('--timeout', type=float, default=600.0)
    args = p.parse_args(argv)
    if args.command == 'sweep':
        return _sweep(args)
    if args.command == 'design':
        return _design(args)
    if args.command == 'compare':
        return _compare(args)
    if args.command == 'guide':
        from .agent import guide
        # The guide is UTF-8 Markdown; write bytes so a console's legacy encoding cannot fail.
        sys.stdout.buffer.write(guide().encode('utf-8') + b'\n'); sys.stdout.flush(); return 0
    if args.command in ('capabilities', 'schema'):
        from . import agent
        print(json.dumps(getattr(agent, args.command)(), indent=2)); return 0
    if args.command == 'example':
        from ._example import EXAMPLE
        print(json.dumps(EXAMPLE, indent=2)); return 0
    if args.command in ('devices', 'device'):
        from . import devices
        try:
            if args.command == 'devices':
                print(json.dumps(devices.catalog(), indent=2)); return 0
            chip = devices.get(args.name, **_params(args.param))
        except (KeyError, ValueError, OSError) as exc:
            return emit_error(type(exc).__name__, str(exc).strip('"'),
                              hint='Run `pauli-cones devices` for catalogue names and parameters.')
        print(json.dumps(chip.to_dict() if args.edges else {**chip.summary(), 'description': chip.description,
                                                            'source': chip.source}, indent=2))
        return 0
    if args.command == 'doctor':
        print(json.dumps({'version': __version__, 'python': platform.python_version(),
                          'platform': platform.platform(),
                          'native_available': importlib.util.find_spec('pauli_cones._core._native_kernel') is not None,
                          'qiskit_available': importlib.util.find_spec('qiskit') is not None,
                          'license': 'Apache-2.0'})); return 0
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        return emit_error('InvalidTimeout', 'timeout must be finite and in (0, 3600]',
                          hint='Pass --timeout with a number of seconds up to 3600.')
    try:
        raw = _read(args.input)
        if len(raw) > MAX_INPUT:
            return emit_error('InputTooLarge', 'request exceeds 4 MiB', hint='Split the work into smaller requests.')
        if _is_qasm(args.input, raw):
            raw = _qasm_request(raw, args)
        mode = args.command if args.command in ('plan', 'compile') else ('value' if args.value_only else 'request')
        options = {'threads': getattr(args, 'threads', 1)}
        if getattr(args, 'min_coefficient', None) is not None or getattr(args, 'max_weight', None) is not None:
            options['truncation'] = {'min_coefficient': args.min_coefficient or 0., 'max_weight': args.max_weight}
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        started = time.perf_counter()
        child = subprocess.run([sys.executable, '-m', 'pauli_cones._worker', getattr(args, 'backend', 'native'),
                                mode, json.dumps(options)], input=raw, capture_output=True,
                               timeout=args.timeout, env=env)
        elapsed = time.perf_counter() - started
        # Parse a complete response before returning it. Never return a partial batch as success.
        output = json.loads(child.stdout)
        if not isinstance(output, dict) or not isinstance(output.get('ok'), bool):
            return emit_error('InvalidWorkerResponse', 'worker did not return a complete response')
        if child.returncode != 0 or not output['ok']:
            print(json.dumps(output, allow_nan=False)); return child.returncode or 2
        output['timings']['parent_worker_wall_seconds'] = elapsed
        print(json.dumps(output, allow_nan=False, separators=(',', ':'))); return 0
    except subprocess.TimeoutExpired:
        return emit_error('Timeout', 'worker exceeded wall deadline and was terminated', 124,
                          hint='Run `pauli-cones plan` first; raise --timeout, or approximate with '
                               '--min-coefficient 1e-4.')
    except _UsageError as exc:
        return emit_error('UsageError', str(exc), hint=exc.hint)
    except (OSError, ValueError, KeyError) as exc:
        return emit_error(type(exc).__name__, str(exc), hint='Check the input path and format; '
                          'see `pauli-cones schema`.')


class _UsageError(Exception):
    def __init__(self, message, hint):
        super().__init__(message)
        self.hint = hint


def _sweep(args):
    """Run a sweep in the worker and print its rows as JSON, Markdown or CSV."""
    if bool(args.input) == bool(args.family):
        return emit_error('UsageError', 'give an OpenQASM file or --family, not both or neither',
                          hint='e.g. `pauli-cones sweep --family ghz --sizes 8,16 --device heavy-hex-127`.')
    spec = {'noise_scales': [float(x) for x in args.noise_scales.split(',')], 'router': args.router,
            'layout': args.layout, 'idle_noise': not args.no_idle_noise, 'observable': args.observable}
    devices_spec = []
    for d in args.device:
        if d.endswith('.json'):
            devices_spec.append(json.loads(Path(d).read_text()))
        else:
            devices_spec.append(d)
    spec['devices'] = devices_spec
    if args.family:
        if not args.sizes:
            return emit_error('UsageError', '--family needs --sizes', hint='Add e.g. --sizes 8,16,32.')
        spec.update(family=args.family, sizes=[int(x) for x in args.sizes.split(',')],
                    family_parameters=_params(args.family_param))
    else:
        if not args.observable:
            return emit_error('UsageError', 'an OpenQASM sweep needs --observable', hint='Add e.g. --observable "Z0".')
        spec['circuit'] = _read(args.input).decode('utf-8')
    try:
        child = subprocess.run([sys.executable, '-m', 'pauli_cones._worker', 'native', 'sweep', '{}'],
                               input=json.dumps({'sweep': spec}).encode(), capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return emit_error('Timeout', 'sweep exceeded its deadline', 124,
                          hint='Raise --timeout, or sweep fewer sizes, chips or noise scales.')
    output = json.loads(child.stdout)
    if not output.get('ok'):
        print(json.dumps(output)); return child.returncode or 2
    if args.format == 'json':
        print(json.dumps(output, allow_nan=False, separators=(',', ':')))
    else:
        from .experiments import SweepResult
        table = SweepResult(output['rows'])
        print(table.markdown() if args.format == 'markdown' else table.csv(), end='\n')
    return 0


def _chip(spec):
    from . import devices
    from .experiments import _device
    if spec.endswith('.json'):
        return devices.get(spec)
    return _device(spec, devices)


def _pairs(text):
    return [tuple(int(x) for x in pair.split('-')) for pair in text.split(',') if pair]


def _design(args):
    """Build a chip from the command-line recipe and print its summary (and write its JSON)."""
    from . import design
    try:
        chip = _chip(args.base)
        if args.modules:
            chip = design.modular(chip, args.modules, links=args.links, links_per_pair=args.links_per_pair,
                                  link_error=args.link_error)
        if args.remove_couplers:
            chip = design.remove_couplers(chip, _pairs(args.remove_couplers))
        if args.add_couplers:
            chip = design.add_couplers(chip, _pairs(args.add_couplers), error=args.coupler_error)
        if args.remove_qubits:
            chip = design.remove_qubits(chip, [int(q) for q in args.remove_qubits.split(',')])
        physics = [args.t1, args.t2, args.time_1q, args.time_2q]
        if any(v is not None for v in physics):
            if any(v is None for v in physics):
                raise ValueError('physics needs all of --t1, --t2, --time-1q and --time-2q')
            chip = design.with_physics(chip, t1=args.t1, t2=args.t2, time_1q=args.time_1q, time_2q=args.time_2q,
                                       control_error_1q=args.control_error_1q, control_error_2q=args.control_error_2q,
                                       readout=[float(x) for x in args.readout.split(',')] if args.readout else None)
        elif args.readout:
            chip = chip.with_noise(readout=tuple(float(x) for x in args.readout.split(',')))
        if args.name:
            from dataclasses import replace
            chip = replace(chip, name=args.name)
    except (KeyError, ValueError, OSError) as exc:
        return emit_error(type(exc).__name__, str(exc).strip('"'),
                          hint='See `pauli-cones design --help`; base chips come from `pauli-cones devices`.')
    if args.out:
        chip.to_json(args.out)
    print(json.dumps({**chip.summary(), 'description': chip.description, 'source': chip.source,
                      'coupler_errors': len(chip.coupler_errors), 'written_to': args.out}, indent=2))
    return 0


def _compare(args):
    """Rank chips on a benchmark family in the worker and print the summary rows."""
    chips = []
    try:
        for spec in args.chips:
            chips.append(_chip(spec).to_dict())
    except (KeyError, ValueError, OSError) as exc:
        return emit_error(type(exc).__name__, str(exc).strip('"'), hint='Run `pauli-cones devices` for names.')
    request = {'compare': {'chips': chips, 'family': args.family, 'sizes': [int(x) for x in args.sizes.split(',')],
                           'noise_scale': args.noise_scale, 'router': args.router,
                           'family_parameters': _params(args.family_param)}}
    try:
        child = subprocess.run([sys.executable, '-m', 'pauli_cones._worker', 'native', 'compare', '{}'],
                               input=json.dumps(request).encode(), capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return emit_error('Timeout', 'comparison exceeded its deadline', 124,
                          hint='Raise --timeout, or compare fewer chips or sizes.')
    output = json.loads(child.stdout)
    if not output.get('ok'):
        print(json.dumps(output)); return child.returncode or 2
    if args.format == 'json':
        print(json.dumps(output, allow_nan=False, separators=(',', ':')))
    else:
        from .experiments import SweepResult
        table = SweepResult(output['rows'])
        table.COLUMNS = tuple(output['rows'][0]) if output['rows'] else ()
        print(table.markdown() if args.format == 'markdown' else table.csv(), end='\n')
    return 0


def _params(pairs):
    """Catalogue parameters from k=v strings, as int, float or None where they parse."""
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f'parameter {pair!r} must be written key=value')
        key, value = pair.split('=', 1)
        for kind in (int, float):
            try:
                out[key] = kind(value); break
            except ValueError:
                continue
        else:
            out[key] = None if value.lower() == 'none' else value
    return out


def _read(path):
    if path == '-':
        return sys.stdin.buffer.read(MAX_INPUT + 1)
    with Path(path).open('rb') as f:
        return f.read(MAX_INPUT + 1)


def _is_qasm(path, raw):
    head = re.sub(rb'^(\s|//[^\n]*\n)*', b'', raw[:4096])
    return path.endswith('.qasm') or head.startswith(b'OPENQASM')


def _qasm_request(raw, args):
    """Wrap OpenQASM text and command-line options into the OpenQASM request form."""
    compiling = args.command == 'compile' or args.compile
    if compiling and not args.device:
        raise _UsageError('compiling needs --device', 'Add e.g. --device heavy-hex-127; see `pauli-cones devices`.')
    if not args.observable and args.command != 'compile':
        raise _UsageError('OpenQASM input needs --observable',
                          'Add e.g. --observable "Z0" or --observable "0.5*Z0*Z1 - X3".')
    request = {'schema_version': 1, 'qasm': raw.decode('utf-8'), 'observables': args.observable or 'Z0'}
    if compiling:
        request['compile'] = {'router': args.router, 'layout': args.layout, 'noise_scale': args.noise_scale,
                              'idle_noise': not args.no_idle_noise}
    if args.max_terms is not None:
        request['max_terms'] = args.max_terms
    if args.device:
        from . import devices
        try:
            request['device'] = devices.get(args.device, **_params(args.device_param)).to_dict()
        except (KeyError, ValueError, OSError) as exc:
            raise _UsageError(str(exc).strip('"'), 'Run `pauli-cones devices` for catalogue names and parameters.')
    if args.values:
        values = json.loads(args.values)
        request['parameter_vectors'] = values if values and isinstance(values[0], list) else [values]
    noise = {}
    if args.noise_1q:
        noise['depolarizing_1q'] = args.noise_1q
    if args.noise_2q:
        noise['depolarizing_2q'] = args.noise_2q
    if args.readout:
        noise['readout'] = [float(x) for x in args.readout.split(',')]
    if noise:
        request['noise'] = noise
    if getattr(args, 'value_only', False):
        request['gradients'] = False
    if compiling and noise:
        raise _UsageError('compiling uses the device noise',
                          'Drop --noise-1q/--noise-2q/--readout and use --noise-scale to vary the device noise.')
    return json.dumps(request).encode('utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
