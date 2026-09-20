"""把教师提供的两个 MATLAB ReLU 演示按原参数移植为可复现 Python 证据。"""
from pathlib import Path
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "matlab_demo"


def relu(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def shallow(x, model):
    x = np.asarray(x, dtype=float)
    pre = model["theta0"][:, None] + model["theta1"][:, None] * x.reshape(1, -1)
    return model["phi0"] + model["phi"] @ relu(pre)


SHALLOW = [
    {"name": "a", "theta0": np.array([.55,-1.05,1.85]), "theta1": np.array([-1.,1.,-1.]), "phi": np.array([-.55,1.12,1.]), "phi0": -.9975},
    {"name": "b", "theta0": np.array([-.60,-1.10,-1.35]), "theta1": np.array([1.,1.,1.]), "phi": np.array([.65,-.85,-1.22]), "phi0": 0.},
    {"name": "c", "theta0": np.array([.25,.55,1.55]), "theta1": np.array([-1.,-1.,-1.]), "phi": np.array([-1.40,-1.65,1.25]), "phi0": -.43},
]

COMPOSED = [
    {"name":"network1", "theta0":np.array([1.2,0.,-.6]), "theta1":np.ones(3), "phi":np.array([2.,-16/3,25/3]), "phi0":-1.4},
    {"name":"network2", "theta0":np.array([1.2,.2,-1/3]), "theta1":np.ones(3), "phi":np.array([-1.75,3.775,-2.655]), "phi0":.95},
]


def knots(model, lo=-1., hi=1.):
    roots = -model["theta0"] / model["theta1"]
    return sorted(float(v) for v,c in zip(roots,model["phi"]) if c != 0 and lo < v < hi)


def preimages(target, model, lo=-1., hi=1.):
    edges = [lo, *knots(model,lo,hi), hi]
    vals = shallow(edges,model)
    roots=[]
    for a,b,ya,yb in zip(edges[:-1],edges[1:],vals[:-1],vals[1:]):
        if abs(yb-ya) < 1e-12: continue
        t=(target-ya)/(yb-ya)
        if -1e-12 <= t <= 1+1e-12: roots.append(a+min(1,max(0,t))*(b-a))
    return sorted(set(round(v,12) for v in roots))


def composed_regions():
    f,g=COMPOSED; cuts=set(knots(f))
    for target in knots(g): cuts.update(preimages(target,f))
    edges=[-1.,*sorted(cuts),1.]; vals=shallow(shallow(edges,f),g)
    rows=[]
    for a,b,ya,yb in zip(edges[:-1],edges[1:],vals[:-1],vals[1:]):
        slope=(yb-ya)/(b-a);rows.append((a,b,slope,ya-slope*a))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    query=np.array([0.,.5,1.,1.5,2.])
    with (OUT/"shallow_query.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["model","x","y"])
        for m in SHALLOW:
            for x,y in zip(query,shallow(query,m)):w.writerow([m["name"],x,y])
    rows=composed_regions();assert len(rows)==9
    with (OUT/"composed_regions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["region","x_left","x_right","slope","intercept"])
        for i,row in enumerate(rows,1):w.writerow([i,*row])
    target=-.5;xs=preimages(target,COMPOSED[0]);ys=shallow(xs,COMPOSED[0]);zs=shallow(ys,COMPOSED[1])
    with (OUT/"shared_intermediate_mapping.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["x","f_x","g_f_x"]);w.writerows(zip(xs,ys,zs))
    assert len(xs)==3 and np.allclose(ys,target) and np.allclose(zs,zs[0])
    print("MATLAB shallow query rows:",3*len(query))
    print("constructed MATLAB composition regions:",len(rows))
    print("shared intermediate mappings:",list(zip(xs,ys,zs)))


if __name__ == "__main__":
    main()
