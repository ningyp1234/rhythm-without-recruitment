"""Independent sparse implementation of Pugliese et al. Eq. 1.
Actual signed connectome only; rates, NOT spikes. Parameters are sampled model
assumptions, not measurements. SciPy RNG differs from upstream JAX replicates.
"""
import numpy as np
from scipy import sparse
from scipy.stats import truncnorm
class RateNetwork:
 def __init__(self,weights_pre_post,sizes,seed=1,dt=.00025,scale=.03,size_reference=None):
  self.w=sparse.csr_matrix(weights_pre_post).T.tocsr()*scale
  self.count=self.w.shape[0];self.dt=dt;self.rng=np.random.default_rng(seed)
  def draw(m,s):return truncnorm.rvs(-m/s,np.inf,loc=m,scale=s,size=self.count,random_state=self.rng)
  self.tau=draw(.02,.002);a=draw(1,.1);th=draw(7.5,.6);self.cap=draw(200,10)
  sz=np.array(sizes,float);median=np.nanmedian(sz) if size_reference is None else float(size_reference);sz[~np.isfinite(sz)|(sz==0)]=median;self.size=sz/median;self.a=a/self.size;self.threshold=th*self.size
  self.rates=np.zeros(self.count);self.muted=np.zeros(self.count,bool);self.t=0.
 def rhs(self,r,inputs,cut=False):
  rr=np.where(self.muted,0,r);net=inputs+(0 if cut else self.w@rr)
  act=np.maximum(self.cap*np.tanh(self.a/self.cap*(net-self.threshold)),0);act[self.muted]=0
  return (act-r)/self.tau
 def advance(self,ids,inputs,milliseconds=10,cut=False):
  current=np.zeros(self.count);np.add.at(current,np.asarray(ids,int),inputs)
  n=round(milliseconds/1000/self.dt);h=milliseconds/1000/n
  for _ in range(n):
   r=self.rates;k1=self.rhs(r,current,cut);k2=self.rhs(r+h*k1/2,current,cut);k3=self.rhs(r+h*k2/2,current,cut);k4=self.rhs(r+h*k3,current,cut)
   self.rates=r+h*(k1+2*k2+2*k3+k4)/6
  self.t+=milliseconds/1000
  if not np.isfinite(self.rates).all() or np.min(self.rates)<-1e-6 or np.any(self.rates>self.cap+1e-6):raise RuntimeError('Rate integration outside physiological model bounds')
  return self.rates
