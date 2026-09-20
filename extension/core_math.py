"""NumPy-only core helpers for the Chapter 4 notebooks and verification."""
from fractions import Fraction
import numpy as np

def relu(x): return np.maximum(np.asarray(x,dtype=float),0.0)
F={"phi":(1.,-2.,-3.,9.3),"theta":((0.,-1.),(0.,1.),(-.67,1.))}
G={"phi":(.5,-1.,-1.5,2.),"theta":((-.6,-1.),(.2,1.),(-.5,1.))}
def shallow(x,p):
    x=np.asarray(x,dtype=float); y=np.full_like(x,p['phi'][0],dtype=float)
    for c,(b,w) in zip(p['phi'][1:],p['theta']): y+=c*relu(b+w*x)
    return y
def compose(x,n=1,first=F,second=G):
    y=np.asarray(x,dtype=float)
    for _ in range(n): y=shallow(y,first)
    return shallow(y,second)
def q(v): return Fraction(str(v))
def merge(pieces):
    out=[]
    for p in pieces:
        if out and out[-1]['slope']==p['slope'] and out[-1]['intercept']==p['intercept']: out[-1]['right']=p['right']
        else: out.append(dict(p))
    return out
def shallow_pieces(p,lo=-1,hi=1):
    lo,hi=q(lo),q(hi); cuts={lo,hi}
    for b,w in p['theta']:
        b,w=q(b),q(w)
        if w and lo<-b/w<hi: cuts.add(-b/w)
    out=[]; edges=sorted(cuts)
    for l,r in zip(edges[:-1],edges[1:]):
        m=(l+r)/2; a,c=Fraction(0),q(p['phi'][0]); active=[]
        for coef,(b,w) in zip(p['phi'][1:],p['theta']):
            coef,b,w=q(coef),q(b),q(w); on=b+w*m>0; active.append(on)
            if on: a+=coef*w; c+=coef*b
        out.append({'left':l,'right':r,'slope':a,'intercept':c,'active':tuple(active),'source':'inner'})
    return merge(out)
def compose_pieces(inner,outer):
    out=[]
    for p in inner:
        cuts={p['left'],p['right']}; a,c=p['slope'],p['intercept']
        for b,w in outer['theta']:
            b,w=q(b),q(w)
            if w*a and p['left']<-(b+w*c)/(w*a)<p['right']: cuts.add(-(b+w*c)/(w*a))
        edges=sorted(cuts)
        for l,r in zip(edges[:-1],edges[1:]):
            m=(l+r)/2; slope,intercept=Fraction(0),q(outer['phi'][0]); active=[]
            for coef,(b,w) in zip(outer['phi'][1:],outer['theta']):
                coef,b,w=q(coef),q(b),q(w); on=b+w*(a*m+c)>0; active.append(on)
                if on: slope+=coef*w*a; intercept+=coef*(b+w*c)
            out.append({'left':l,'right':r,'slope':slope,'intercept':intercept,'active':tuple(active),'source':'outer_preimage'})
    return merge(out)
def iterate_pieces(n,first=F,second=G):
    pieces=shallow_pieces(first)
    for _ in range(1,n): pieces=compose_pieces(pieces,first)
    return pieces,compose_pieces(pieces,second)
def clipping_parameters():
    theta=np.zeros((4,2));psi=np.zeros((4,4));phi=np.zeros((4,1));theta[1:]=((.3,-1),(-1,2),(-.5,.65));psi[1:]=((.3,2,-1,7),(-.2,2,1.2,-8),(.3,-2.3,-.8,2));phi[:,0]=(0,.5,-1.5,2.2);return phi,psi,theta
def clipping(x,phi,psi,theta,activation=relu):
    x=np.asarray(x,float).reshape(1,-1);h=activation(theta[1:,0:1]+theta[1:,1:2]*x);z=psi[1:,0:1]+psi[1:,1:]@h;hp=activation(z);weighted=phi[1:,0:1]*hp;return phi[0,0]+weighted.sum(0),z,hp,weighted,h
def clipping_region_count(phi,psi,theta,lo=0.,hi=1.):
    """Small affine propagation for the two-layer clipping example (final, merged regions)."""
    cuts={lo,hi}
    for b,w in theta[1:]:
        if w and lo<-b/w<hi: cuts.add(float(-b/w))
    first=sorted(cuts); allcuts=set(first)
    for l,r in zip(first[:-1],first[1:]):
        m=(l+r)/2; active=(theta[1:,0]+theta[1:,1]*m)>0
        for row in psi[1:]:
            a=float(np.sum(row[1:]*theta[1:,1]*active));b=float(row[0]+np.sum(row[1:]*theta[1:,0]*active))
            if a and l<-b/a<r: allcuts.add(-b/a)
    edges=sorted(allcuts); pieces=[]
    for l,r in zip(edges[:-1],edges[1:]):
        m=(l+r)/2; hactive=(theta[1:,0]+theta[1:,1]*m)>0; za=psi[1:,1:]@(theta[1:,1]*hactive);zb=psi[1:,0]+psi[1:,1:]@(theta[1:,0]*hactive);on=za*m+zb>0;a=float(np.sum(phi[1:,0]*za*on));b=float(phi[0,0]+np.sum(phi[1:,0]*zb*on));pieces.append((a,b,l,r))
    merged=[]
    for p in pieces:
        if merged and np.allclose(merged[-1][:2],p[:2],atol=1e-12):merged[-1]=(merged[-1][0],merged[-1][1],merged[-1][2],p[3])
        else:merged.append(p)
    return merged
def parameter_count(widths): return sum((a+1)*b for a,b in zip(widths[:-1],widths[1:]))
