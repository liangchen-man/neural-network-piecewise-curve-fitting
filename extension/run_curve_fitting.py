"""Evidence-first fitting under separate interpolation and contiguous-gap protocols."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; STEPS=750; CHECK_EVERY=50; PATIENCE=10; SEEDS=(0,1,2)
def feature(x,z):
 s=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(z)))];center=(np.median(x),np.median(z));return s,np.hypot(x-center[0],z-center[1]),center
def protocol(n,name,seed=20260919):
 rng=np.random.default_rng(seed)
 if name=='interpolation':q=rng.permutation(n);a=int(.7*n);b=int(.85*n);return q[:a],q[a:b],q[b:]
 gap=np.arange(int(.45*n),int(.55*n));visible=np.setdiff1d(np.arange(n),gap);q=rng.permutation(visible);a=int(.85*len(q));return q[:a],q[a:],gap
def scale(y,tr):
 med=float(np.median(y[tr]));sd=float(np.std(y[tr]));return med,sd if sd>1e-12 else 1.
def raw_metrics(y,p,grid):
 d=y-p;return dict(mae=float(np.mean(abs(d))),rmse=float(np.sqrt(np.mean(d*d))),roughness=float(np.mean(np.diff(grid,2)**2)),slope_tv=float(np.sum(abs(np.diff(grid)))))
class MLP(nn.Module):
 def __init__(self,k):
  super().__init__();w=[1,736,1] if k=='N1' else [1,32,32,32,1];a=nn.Tanh if k in ('N1','N3','N4','N5') else nn.ReLU;parts=[]
  for i,(u,v) in enumerate(zip(w[:-1],w[1:])):parts.append(nn.Linear(u,v));parts.extend([] if i==len(w)-2 else [a()])
  self.net=nn.Sequential(*parts)
 def forward(self,x):return self.net(x)
def fit_nn(t,y,tr,va,k,seed,curvature_lambda=1e-6):
 torch.manual_seed(seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=MLP(k).to(dev);assert sum(q.numel() for q in m.parameters())==2209
 tt=torch.tensor(t[:,None],dtype=torch.float32,device=dev);yy=torch.tensor(y[:,None],dtype=torch.float32,device=dev);it=torch.tensor(tr,device=dev);iv=torch.tensor(va,device=dev);opt=torch.optim.Adam(m.parameters(),lr=1e-3);best=(1e99,None,0);stale=0;hist=[]
 for step in range(1,STEPS+1):
  opt.zero_grad();p=m(tt[it]);loss=nn.functional.huber_loss(p,yy[it],delta=1.) if k=='N5' else nn.functional.mse_loss(p,yy[it]);reg=torch.tensor(0.,device=dev)
  if k in ('N4','N5'):
   col=torch.linspace(-1,1,128,device=dev).reshape(-1,1).requires_grad_(True);o=m(col);d=torch.autograd.grad(o,col,torch.ones_like(o),create_graph=True)[0];d2=torch.autograd.grad(d,col,torch.ones_like(d),create_graph=True)[0];reg=(d2*d2).mean();loss=loss+curvature_lambda*reg
  loss.backward();opt.step()
  if step%CHECK_EVERY==0:
   with torch.no_grad():v=torch.sqrt(torch.mean((m(tt[iv])-yy[iv])**2)).item()
   hist.append((step,float(loss.item()),float(reg.item()),float(v)))
   if v<best[0]:best=(v,{a:b.detach().cpu().clone() for a,b in m.state_dict().items()},step);stale=0
   else:stale+=1
   if stale>=PATIENCE:break
 m.load_state_dict(best[1])
 with torch.no_grad():pred=m(tt).cpu().numpy().ravel()
 return pred,hist,best[2],step,sum(q.numel() for q in m.parameters())
def baselines(t,y,tr,va):
 order=np.argsort(t[tr]);tx,ty=t[tr][order],y[tr][order];b0=np.interp(t,tx,ty);best=None
 for fac in (.01,.1,1.,10.):
  p=UnivariateSpline(tx,ty,s=len(tx)*np.var(ty)*fac)(t);score=float(np.sqrt(np.mean((p[va]-y[va])**2)));best=min(best or (1e99,None,None),(score,p,fac),key=lambda z:z[0])
 return b0,best[1],best[2]
def write_row(w,curve,source,kind,proto,seed,n,tr,va,ev,med,sd,t,r,p,seconds,updates,param,fac=''):
 grid=np.linspace(-1,1,1024);pg=np.interp(grid,np.sort(t),p[np.argsort(t)]);m=raw_metrics(r[ev],(med+sd*p)[ev],med+sd*pg)
 w.writerow(dict(curve_id=curve,source=source,protocol=proto,model_id=kind,seed=seed,n_points=n,n_train=len(tr),n_val=len(va),n_eval=len(ev),transductive_geometry=True,parameters=param,updates=updates,seconds=seconds,scaler_median=med,scaler_std=sd,status='MEASURED',spline_factor=fac,**m))
def run_one(curve,x,z,source,w,logs,preds,only_proto=None):
 s,r,_=feature(x,z);t=2*(s-s.min())/(s.max()-s.min())-1;n=len(t)
 for proto in ((only_proto,) if only_proto else ('interpolation','gap')):
  tr,va,ev=protocol(n,proto);med,sd=scale(r,tr);y=(r-med)/sd;b0,b1,fac=baselines(t,y,tr,va)
  for name,p in (('B0',b0),('B1',b1)):write_row(w,curve,source,name,proto,'',n,tr,va,ev,med,sd,t,r,p,0.,0,0,fac if name=='B1' else '');preds[(curve,proto,name,'')]=med+sd*p
  for name in ('N1','N2','N3','N4','N5'):
   for seed in SEEDS:
    st=time.perf_counter();p,h,best,steps,param=fit_nn(t,y,tr,va,name,seed);write_row(w,curve,source,name,proto,seed,n,tr,va,ev,med,sd,t,r,p,time.perf_counter()-st,steps,param);preds[(curve,proto,name,str(seed))]=med+sd*p;logs.extend(dict(curve_id=curve,protocol=proto,model_id=name,seed=seed,step=a,loss=b,reg=c,val_rmse_normalized=d) for a,b,c,d in h)
 return t,r
def synthetic():
 x=np.linspace(0,1,1024)
 for k in range(3):
  clean=.3*x+.2*x*x if k==0 else .3*x+.08*np.sin(8*np.pi*x)
  if k==2:clean=.3*x+.08*np.sin(8*np.pi*x)-.22*np.exp(-((x-.58)/.04)**2)
  for spike in (0,1):
   rng=np.random.default_rng(100+k+10*spike);obs=clean+rng.normal(0,.025,len(x));
   if spike:obs[rng.choice(len(x),len(x)//100,False)]+=rng.normal(0,.3,len(x)//100)
   yield f'synthetic_{k}_spike_{spike}',x,obs,clean
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--mode',choices=('train','retrain_verify'),default='train');a=ap.parse_args();OUT.mkdir(exist_ok=True);preds={};logs=[];old=list(csv.DictReader((OUT/'metrics_per_run.csv').open(encoding='utf8'))) if (OUT/'metrics_per_run.csv').exists() else [];sec=[float(x['seconds']) for x in old if x.get('model_id','').startswith('N') and x.get('seconds')]
 benchmark={'previous_runs':len(sec),'mean_seconds_per_250_updates':float(np.mean(sec)) if sec else None,'real_neural_runs':180,'frozen_max_updates':STEPS,'check_every':CHECK_EVERY,'patience_checks':PATIENCE,'projection_seconds':float(np.mean(sec))*180*STEPS/250 if sec else None,'reason':'750 updates is a bounded local rerun budget based on measured prior timing; it is below 5000 to complete the real 6×2×5×3 matrix.'};(OUT/'benchmark_projection.json').write_text(json.dumps(benchmark,indent=2),encoding='utf8')
 fields=['curve_id','source','protocol','model_id','seed','n_points','n_train','n_val','n_eval','transductive_geometry','parameters','updates','seconds','scaler_median','scaler_std','status','mae','rmse','roughness','slope_tv','spline_factor'];target=OUT/('retrain_verify_metrics.csv' if a.mode=='retrain_verify' else 'metrics_per_run.csv')
 with target.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();curves=np.load(ROOT/'data_small'/'real_curves.npz',allow_pickle=True)['curves']
  if a.mode=='retrain_verify':run_one('real_row_0_verify',curves[0]['x'],curves[0]['z'],'real CSV audited slice',w,logs,preds,'interpolation')
  else:
   for c in curves[:6]:run_one('real_row_'+str(c['row_index']),c['x'],c['z'],'real CSV audited slice',w,logs,preds)
   for name,x,obs,clean in synthetic():run_one(name,x,obs,'synthetic clean+noise',w,logs,preds,'interpolation');np.savez_compressed(OUT/(name+'_truth.npz'),x=x,clean=clean,observed=obs)
 with (OUT/('retrain_verify_training_log.csv' if a.mode=='retrain_verify' else 'training_log.csv')).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=['curve_id','protocol','model_id','seed','step','loss','reg','val_rmse_normalized']);w.writeheader();w.writerows(logs)
 np.savez_compressed(OUT/('retrain_verify_predictions.npz' if a.mode=='retrain_verify' else 'predictions.npz'),records=np.array([(k,v) for k,v in preds.items()],dtype=object))
 if a.mode=='train':
  rows=[]
  for name,x,obs,clean in synthetic():
   _,truth,_=feature(x,clean)
   for key,p in preds.items():
    if key[0]==name:
     amp=float(p.min()-truth.min());loc=float(x[np.argmin(p)]-x[np.argmin(truth)]);rows.append({'curve_id':name,'protocol':key[1],'model_id':key[2],'seed':key[3],'clean_rmse':float(np.sqrt(np.mean((p-truth)**2))),'depression_amplitude_error':amp,'depression_location_error':loc})
  with (OUT/'synthetic_truth_metrics.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print('completed',a.mode,'metric rows',sum(1 for _ in target.open())-1)
if __name__=='__main__':main()
