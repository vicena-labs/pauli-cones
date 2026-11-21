"""Versioned JSON schema validation; no code loading or arbitrary matrix input."""
from __future__ import annotations
import re
from .api import Circuit, _integer


def fields(obj, required, optional=()):
    if not isinstance(obj, dict): raise ValueError("expected a JSON object")
    missing=set(required)-obj.keys(); extra=obj.keys()-set(required)-set(optional)
    if missing or extra:
        raise ValueError(f"invalid fields: missing={sorted(missing)}, unknown={sorted(extra)}")


def decode_request(data):
    if isinstance(data, dict) and "qasm" in data:
        return decode_qasm_request(data)
    fields(data, ("schema_version", "n_qubits", "operations", "observables"),
           ("n_parameters", "max_terms", "readout", "parameter_vectors", "gradients"))
    _integer(data['schema_version'], 'schema_version', 1, 1)
    if not isinstance(data.get('gradients', True), bool):
        raise ValueError('gradients must be a boolean')
    c = Circuit(data['n_qubits'], data.get('n_parameters',0), data.get('max_terms',50_000))
    ops=data['operations']
    if not isinstance(ops,list) or len(ops)>10_000: raise ValueError("operations must be a list of at most 10000 entries")
    for entry in ops:
        if not isinstance(entry,dict) or not isinstance(entry.get('op'),str): raise ValueError("operation requires text op")
        op=entry['op'].upper()
        if op in {'H','SX','X','Y','Z','CZ','ECR'}:
            fields(entry,('op','wires')); c.gate(op,entry['wires'])
        elif op in {'RX','RY','RZ'}:
            fields(entry,('op','wires','theta'),('param',)); c.gate(op,entry['wires'],entry['theta'],entry.get('param',-1))
        elif op=='FSIM':
            fields(entry,('op','wires','theta','phi'),('param',)); c.fsim(entry['wires'],entry['theta'],entry['phi'],entry.get('param',-1))
        elif op=='DEPOLARIZE':
            fields(entry,('op','wires','probability')); c.depolarize(entry['wires'],entry['probability'])
        elif op=='PAULI_ERROR':
            fields(entry,('op','wire','axis','probability')); c.pauli_error(entry['wire'],entry['axis'],entry['probability'])
        elif op=='RELAX':
            fields(entry,('op','wire','duration','t1','t2')); c.relax(entry['wire'],entry['duration'],entry['t1'],entry['t2'])
        else: raise ValueError(f"unsupported operation {op!r}")
    ros=data.get('readout',[])
    if not isinstance(ros,list) or len(ros)>c.n_qubits: raise ValueError("invalid readout list")
    seen=set()
    for item in ros:
        fields(item,('wire','p01','p10'))
        q=_integer(item['wire'],'readout wire',0,c.n_qubits-1)
        if q in seen: raise ValueError("duplicate readout wire")
        seen.add(q); c.readout(q,item['p01'],item['p10'])
    rawobs=data['observables']
    if not isinstance(rawobs,list) or not 1<=len(rawobs)<=1024: raise ValueError("provide 1..1024 observable terms")
    obs=[]
    for item in rawobs:
        fields(item,('coefficient','pauli'))
        if not isinstance(item['pauli'],dict): raise ValueError("pauli must be a wire-to-axis JSON object")
        pauli={}
        for q,axis in item['pauli'].items():
            if not isinstance(q,str) or not re.fullmatch(r'0|[1-9][0-9]*',q): raise ValueError("Pauli wires must be decimal integer keys")
            pauli[int(q)]=axis
        obs.append((item['coefficient'],pauli))
    vecs=data.get('parameter_vectors',[[0.0]*c.n_parameters])
    if not isinstance(vecs,list) or not 1<=len(vecs)<=128 or any(not isinstance(v,list) for v in vecs):
        raise ValueError("parameter_vectors must contain 1..128 lists")
    return c,obs,vecs


def decode_qasm_request(data):
    """The OpenQASM request form: a circuit as OpenQASM text, observables as a
    string or JSON terms, an optional uniform noise model, and parameter
    vectors holding OpenQASM 3 input values in source order."""
    from .interop import from_qasm, parse_observable
    fields(data, ("schema_version", "qasm", "observables"),
           ("max_terms", "parameter_vectors", "gradients", "noise", "device"))
    _integer(data['schema_version'], 'schema_version', 1, 1)
    if not isinstance(data['qasm'], str): raise ValueError("qasm must be OpenQASM text")
    if not isinstance(data.get('gradients', True), bool):
        raise ValueError('gradients must be a boolean')
    noise = data.get('noise')
    if 'device' in data and noise is None:
        from .devices import Device
        noise = Device.from_dict(data['device']).noise_model()
    converted = from_qasm(data['qasm'], noise=noise, max_terms=data.get('max_terms', 50_000))
    c = converted.circuit
    raw = data['observables']
    if isinstance(raw, list):
        for item in raw:
            fields(item, ('coefficient', 'pauli'))
        raw = [(item['coefficient'], {int(q): a for q, a in item['pauli'].items()}) for item in raw]
    obs = parse_observable(raw, c.n_qubits)
    if not 1 <= len(obs) <= 1024: raise ValueError("provide 1..1024 observable terms")
    vecs = data.get('parameter_vectors', [[0.0] * len(converted.parameters)])
    if not isinstance(vecs, list) or not 1 <= len(vecs) <= 128 or any(not isinstance(v, list) for v in vecs):
        raise ValueError("parameter_vectors must contain 1..128 lists")
    return c, obs, [converted.shifts(v) if converted.parameters else v for v in vecs]
