"""CPU scalar-block counterexample for M5, not a hardware acceptance result.

Each original 512x512 quadrant is constant, so scalar products times K model
these matrices. FP16 inputs and separate FP32 transform/multiply/add rounding
are explicit. The independent GPU model still decides its full42-case screen.
"""
import json
import math
import struct
from pathlib import Path

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def f16(x):return struct.unpack('e',struct.pack('e',x))[0]
def add(a,b):return f32(a+b)
def sub(a,b):return f32(a-b)
def mul(a,b):return f32(a*b)
def scale(peak):return 2.**math.ceil(math.log2(peak/127)) if peak else 1.
def q(x,bound):
    s=scale(bound);return max(-127,min(127,round(x/s))),s
def parts(v,bound,adaptive):
    h=f16(v);lo=sub(v,h)
    hb=abs(h) if adaptive else f16(bound)
    lb=abs(lo) if adaptive else min(bound,2.**max(math.ceil(math.log2(bound))-1-11,-25)) if bound else 0.
    qh,sh=q(h,hb);ql,sl=q(lo,lb)
    return h,lo,qh,ql,sh,sl

def run(adaptive):
    x=.75;delta=1/256;y=.75;z=-.74951171875
    a=list(map(f16,[x,-x+delta,x,-x+delta]));b=list(map(f16,[y,z,y,z]))
    s1=add(a[2],a[3]);s2=sub(s1,a[0]);t1=sub(b[1],b[0]);t2=sub(b[3],t1)
    s=[s1,s2,sub(a[0],a[2]),sub(a[1],s2)]
    t=[t1,t2,sub(b[3],b[1]),sub(t2,b[2])]
    ma=list(map(abs,a));mb=list(map(abs,b))
    sb1=ma[2]+ma[3];sb2=sb1+ma[0];tb1=mb[1]+mb[0];tb2=mb[3]+tb1
    sb=[sb1,sb2,ma[0]+ma[2],ma[1]+sb2]
    tb=[tb1,tb2,mb[3]+mb[1],tb2+mb[2]]
    sq=[parts(v,bound,adaptive) for v,bound in zip(s,sb)];tq=[parts(v,bound,adaptive) for v,bound in zip(t,tb)]
    ar=parts(a[3],abs(a[3]),True);br=parts(b[3],abs(b[3]),True)
    main=[(a[0],b[0]),(a[1],b[2]),(sq[3][0],b[3]),(a[3],tq[3][0]),(sq[0][0],tq[0][0]),(sq[1][0],tq[1][0]),(sq[2][0],tq[2][0])]
    pairs=[(sq[3],1,br,0),(ar,0,tq[3],1)]
    for j in range(3):pairs.extend([(sq[j],1,tq[j],0),(sq[j],0,tq[j],1)])
    signs=[(0,1,0,0),(0,0,-1,0),(0,1,0,1),(0,1,0,1),(0,1,1,1),(0,1,1,1),(0,0,1,1),(0,0,1,1)]
    dest=[(0,1,2,3),(0,),(1,),(2,),(1,3),(1,2,3),(2,3)]
    associated={5:(4,5),6:(6,7),4:(2,3),2:(0,),3:(1,)}
    out=[0.]*4
    for k in range(4):
        for prod in (0,1,5,6,4,2,3):
            d=mul(mul(*main[prod]),128.)
            for j in dest[prod]:out[j]=sub(out[j],d) if prod==3 else add(out[j],d)
            for term in associated.get(prod,()):
                xx,xi,yy,yi=pairs[term]
                d=mul(float(128*xx[2+xi]*yy[2+yi]),mul(xx[4+xi],yy[4+yi]))
                for j,sg in enumerate(signs[term]):
                    if sg:out[j]=add(out[j],d) if sg>0 else sub(out[j],d)
    ref=[512*(a[0]*b[0]+a[1]*b[2]),512*(a[0]*b[1]+a[1]*b[3]),512*(a[2]*b[0]+a[3]*b[2]),512*(a[2]*b[1]+a[3]*b[3])]
    ef=math.sqrt(sum((x-y)**2 for x,y in zip(out,ref))/sum(x*x for x in ref))
    return dict(quantization='adaptive' if adaptive else 'a_priori_bound',original_a_quadrants=a,original_b_quadrants=b,output_quadrants=out,reference_quadrants=ref,ef=ef,pass_fixed_floor=ef<=1e-4,S1=s[0],S1_high_integer=sq[0][2],S1_high_scale=sq[0][4],T1_residual=tq[0][1])

if __name__=='__main__':
    rows=[run(False),run(True)]
    assert not rows[0]['pass_fixed_floor'] and rows[1]['pass_fixed_floor']
    result=dict(scope=__doc__,shape=[1024,1024,1024],dtype='FP16',inside_exponents=[0,0],rows=rows)
    print(json.dumps(result,indent=2))
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
