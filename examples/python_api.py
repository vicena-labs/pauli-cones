"""Run after installing the wheel or `python -m pip install .`."""
from pauli_cones import Circuit

circuit = Circuit(2, n_parameters=1)
circuit.gate("RY", [0], theta=0.7, param=0)
circuit.gate("H", [1]).gate("CZ", [0, 1])
circuit.depolarize([0, 1], probability=0.01)
circuit.relax(0, duration=2e-7, t1=70e-6, t2=90e-6)
circuit.readout(0, p01=0.02, p10=0.01)
plan = circuit.prepare([(1.0, {0: "Z"})], backend="native")
for result in plan.evaluate_many([[0.0], [0.1], [-0.1]]):
    print(result.as_dict())
