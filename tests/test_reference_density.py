"""Independent small dense-matrix oracle and analytic/FD gradient qualifications."""
import math
import numpy as np
import pytest
from pauli_cones._core._reference import (NoisyPauliCircuit,ConeOverflow,GATE1,GATE2,rotation,fsim,
                        I,X,Y,Z,load_chip_grid)


def embed(operator,wires,n):
    """Reference embedding by explicit basis construction; not a Pauli PTM."""
    wires=list(wires);dim=1<<n
    result=np.zeros((dim,dim),complex)
    for col in range(dim):
        k=sum(((col>>w)&1)<<j for j,w in enumerate(wires))
        for r in range(1<<len(wires)):
            row=col
            for j,w in enumerate(wires):
                row=(row & ~(1<<w)) | (((r>>j)&1)<<w)
            result[row,col]=operator[r,k]
    return result


def unitary(rho,u,wires,n):
    b=embed(u,wires,n)
    return b@rho@b.conj().T


def depol2(rho,wires,p,n):
    basis=(I,X,Y,Z)
    out=(1-p)*rho
    for a in basis:
        for b in basis:
            if a is I and b is I:continue
            full=embed(np.kron(b,a),wires,n)
            out+=(p/15)*full@rho@full.conj().T
    return out


def relax(rho,wire,t,T1,T2,n):
    g=1-math.exp(-t/T1)
    A0=np.diag([1., math.sqrt(1-g)])
    A1=np.array([[0,math.sqrt(g)],[0,0]])
    full0,full1=embed(A0,(wire,),n),embed(A1,(wire,),n)
    rho=full0@rho@full0.conj().T + full1@rho@full1.conj().T
    pure=math.exp(-t/T2)/math.sqrt(1-g)
    fullz=embed(Z,(wire,),n)
    return ((1+pure)/2)*rho + ((1-pure)/2)*(fullz@rho@fullz)


def oracle(n,stages,measurement_wire,initial='zero',readout=None):
    v=np.array([1.+0j])
    for _ in range(n):v=np.kron(([1,1]/np.sqrt(2) if initial=='plus' else [1,0]),v)
    rho=np.outer(v,v.conj())
    for op,wires,value in stages:
        if op in GATE1:rho=unitary(rho,GATE1[op],wires,n)
        elif op in GATE2:rho=unitary(rho,GATE2[op],wires,n)
        elif op in ('RX','RY','RZ'):rho=unitary(rho,rotation(op[1],value)[0],wires,n)
        elif op=='FSIM':rho=unitary(rho,fsim(value[0],value[1])[0],wires,n)
        elif op=='D2':rho=depol2(rho,wires,value,n)
        elif op=='RELAX':rho=relax(rho,wires[0],*value,n)
        else:raise ValueError(op)
    z=np.trace(embed(Z,(measurement_wire,),n)@rho).real
    if readout:
        p01,p10=readout
        z=(p10-p01)+(1-p01-p10)*z
    return float(z),rho


def build_case(n,theta,theta2,seed=0):
    c=NoisyPauliCircuit(n,2,max_terms=4096)
    steps=[]
    def gate(name,wires,value=0.,param=-1):
        c.gate(name,wires,value,param)
        steps.append((name,tuple(wires),value))
    def two_noise(a,b,p):
        c.channel_depolarize((a,b),p)
        steps.append(('D2',(a,b),p))
    def relax_wire(q,t,T1,T2):
        c.channel_relax(q,t,T1,T2)
        steps.append(('RELAX',(q,),(t,T1,T2)))
    for q in range(n):gate('RY',(q,),theta+q*.11,param=0)
    for q in range(n-1):
        gate('CZ',(q,q+1))
        two_noise(q,q+1,.006)
        relax_wire(q,2.0e-7,65e-6,81e-6)
    if n>=2:
        gate('ECR',(0,1));two_noise(0,1,.012)
    if n>=2:
        c.fsim((0,1),theta=.31,phi=.13)
        steps.append(('FSIM',(0,1),(.31,.13)))
    for q in range(n):gate('RZ',(q,),theta2-q*.05,param=1)
    c.set_readout(0,.026,.017)
    return c,steps


def test_amplitude_damping_affine_and_readout():
    theta=.62;t=3.2e-6;T1=70e-6;T2=80e-6
    g=1-math.exp(-t/T1)
    c=NoisyPauliCircuit(127,1).gate('RY',(53,),theta,param=0).channel_relax(53,t,T1,T2).set_readout(53,.035,.022)
    val,grad,stats=c.evaluate([(1.,{53:'Z'})],include_stats=True)
    a=.022-.035;b=1-.035-.022
    assert val==pytest.approx(a+b*(g+(1-g)*math.cos(theta)),abs=1e-12)
    assert grad[0]==pytest.approx(b*(1-g)*(-math.sin(theta)),abs=1e-12)
    assert stats['max_terms']<=3


@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('initial',['zero','plus'])
@pytest.mark.parametrize('theta',[-.5,.12,.65])
def test_against_independent_density_and_fd(n,initial,theta):
    phi=.3
    c,steps=build_case(n,theta,phi)
    v,g=c.evaluate([(1.,{0:'Z'})],initial=initial)
    ref,_=oracle(n,steps,0,initial,readout=(.026,.017))
    assert abs(v-ref)<2e-11,(n,initial,theta,v,ref)
    eps=2e-5
    for j in range(2):
        def evaluate_shift(s):
            shifted=[(op,w,(value+s if j==0 else value) if op=='RY' else (value+s if j==1 else value) if op=='RZ' else value) for op,w,value in steps]
            return oracle(n,shifted,0,initial,readout=(.026,.017))[0]
        expected=(evaluate_shift(eps)-evaluate_shift(-eps))/(2*eps)
        assert abs(g[j]-expected)<2e-7,(n,initial,theta,j,g[j],expected)


def test_bounded_cone_catches_explosive_deep_circuit():
    c=NoisyPauliCircuit(6,0,max_terms=1)
    c.gate('H',(0,)).gate('CZ',(0,1)).gate('RY',(0,),.4)
    with pytest.raises(ConeOverflow):c.evaluate([(1.,{0:'Z'})])


def test_t2_physical_constraint():
    with pytest.raises(ValueError):NoisyPauliCircuit(1).channel_relax(0,.1,1.,3.)


def test_fsim_matrix_unitary():
    u,dt,df=fsim(.2,.34)
    assert np.max(np.abs(u@u.conj().T-np.eye(4)))<1e-15
    assert np.max(np.abs((fsim(.20001,.34)[0]-fsim(.19999,.34)[0])/(.00002)-dt))<1e-10


def test_2d_127_qubit_bounded_and_finite():
    c=load_chip_grid(11,12,depth=2,max_terms=4096)
    v,g,s=c.evaluate([(1.,{5*12+5:'Z'})],include_stats=True)
    assert math.isfinite(v) and np.all(np.isfinite(g))
    assert abs(v)<=1.000001
    assert 0<=s['max_terms']<=4096
