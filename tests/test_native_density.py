"""Public native API against independent explicit Kraus density evolution."""
import math
import numpy as np
import pytest
from quantum_circuit_simulator import Circuit

I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1,-1]).astype(complex)
H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
CZ=np.diag([1,1,1,-1]).astype(complex)

def density_answer(theta):
    v=np.array([1,0,0,0],complex)
    ry=np.array([[math.cos(theta/2),-math.sin(theta/2)],
                 [math.sin(theta/2),math.cos(theta/2)]],complex)
    u=CZ @ np.kron(H,ry)
    v=u@v;rho=np.outer(v,v.conj())
    p=.031
    noisy=(1-p)*rho
    for j,a in enumerate((I,X,Y,Z)):
        for k,b in enumerate((I,X,Y,Z)):
            if j==k==0:continue
            q=np.kron(b,a);noisy+=(p/15)*(q@rho@q.conj().T)
    t,t1,t2=2e-6,65e-6,80e-6
    g=1-math.exp(-t/t1)
    a0=np.kron(I,np.diag([1,math.sqrt(1-g)]))
    a1=np.kron(I,np.array([[0,math.sqrt(g)],[0,0]],complex))
    damped=a0@noisy@a0.conj().T+a1@noisy@a1.conj().T
    q=np.kron(I,Z);dephase=math.exp(-t/t2)/math.sqrt(1-g)
    rho=(1+dephase)/2*damped+(1-dephase)/2*(q@damped@q)
    measured=.014-.023+(1-.014-.023)*np.trace(np.kron(I,Z)@rho).real
    return float(measured)

@pytest.mark.parametrize('angle',[-1.7,-.4,.0,.3,1.2])
def test_native_value_and_gradient_against_density(angle):
    c=Circuit(2,1).gate('RY',[0],angle,0).gate('H',[1]).gate('CZ',[0,1])
    c.depolarize([0,1],.031).relax(0,2e-6,65e-6,80e-6).readout(0,.023,.014)
    p=c.prepare([(1.,{0:'Z'})])
    for shift in (0.,.13):
        res=p.evaluate([shift]);x=angle+shift
        expected=density_answer(x)
        grad=(density_answer(x+2e-5)-density_answer(x-2e-5))/4e-5
        assert abs(res.value-expected)<2e-12
        assert abs(res.gradient[0]-grad)<2e-7
