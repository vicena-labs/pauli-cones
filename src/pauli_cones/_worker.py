"""One finite request per process. The CLI parent enforces a wall deadline."""
from __future__ import annotations
import json
import sys
import time


def strict_json(text):
    def pairs(items):
        obj={}
        for k,v in items:
            if k in obj: raise ValueError(f"duplicate JSON key: {k}")
            obj[k]=v
        return obj
    def constant(value): raise ValueError(f"nonfinite JSON value {value}")
    return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)


def main():
    try:
        backend=sys.argv[1]
        start=time.perf_counter()
        data=strict_json(sys.stdin.buffer.read(4*1024*1024+1).decode('utf-8'))
        from .schema import decode_request
        c,obs,vecs=decode_request(data)
        device_check = None
        if isinstance(data, dict) and 'device' in data:
            from .devices import Device
            device = Device.from_dict(data['device'])
            pairs = device.uncoupled_pairs(c)
            device_check = {'name': device.name, 'n_qubits': device.n_qubits,
                            'noise_applied': 'noise' not in data,
                            'uncoupled_two_qubit_operations': len(pairs),
                            'first_uncoupled_pairs': [list(p) for p in pairs[:5]],
                            'note': 'two-qubit operations must act on coupled qubits to run on this chip; '
                                    'route the circuit onto the device first' if pairs else
                                    'every two-qubit operation acts on coupled qubits'}
        if len(sys.argv) > 2 and sys.argv[2] == 'plan':
            from .agent import plan
            output = {'ok': True, 'schema_version': 1, 'plan': plan(c, obs),
                      'timings': {'worker_seconds': time.perf_counter() - start}}
            if device_check:
                output['device'] = device_check
            print(json.dumps(output, allow_nan=False, separators=(',', ':')), flush=True)
            return 0
        gradients = data.get('gradients', True) and not (len(sys.argv)>2 and sys.argv[2]=='value')
        # Options from the CLI parent: threads and an optional truncation.
        options=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
        from .api import Truncation
        truncation=Truncation(**options['truncation']) if 'truncation' in options else None
        t=time.perf_counter()
        plan=c.prepare(obs,backend=backend,truncation=truncation,threads=options.get('threads',1))
        preparation=time.perf_counter()-t
        t=time.perf_counter(); results=plan.evaluate_many(vecs,gradients=gradients); computation=time.perf_counter()-t
        output={'ok':True,'schema_version':1,'backend':plan.backend,'requested_backend':backend,'n_qubits':c.n_qubits,
            'n_parameters':c.n_parameters,'results':[x.as_dict() for x in results],
            'gradients':gradients, 'diagnostics':plan.diagnostics,
            **({'device': device_check} if device_check else {}),
            'numerics':plan.numerics,
            'timings':{'preparation_seconds':preparation,'evaluate_batch_seconds':computation,
                       'worker_seconds':time.perf_counter()-start}}
        print(json.dumps(output,allow_nan=False,separators=(',',':')),flush=True)
        return 0
    except Exception as exc:
        from .agent import hint_for
        print(json.dumps({'ok':False,'error':{'type':type(exc).__name__,'message':str(exc),
                                              'hint':hint_for(exc)}}),flush=True)
        return 2

if __name__=='__main__':raise SystemExit(main())
