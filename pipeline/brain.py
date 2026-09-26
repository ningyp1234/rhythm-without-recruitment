"""Full MaleCNS spiking model. No target location, heading command or behavior rules.
LIF constants follow Shiu's public model; applying them to MaleCNS is uncalibrated.
The implementation is separately compared with Brian2 on deterministic spike input.
"""
from pathlib import Path
import json,os
import numpy as np,pandas as pd
from scipy import sparse
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
PARAMS=dict(dt_ms=.1,rest_mv=-52.,threshold_mv=-45.,tau_membrane_ms=20.,tau_synapse_ms=5.,refractory_ms=2.2,delay_ms=1.8,weight_mv=.275,input_jump_mv=68.75)
SIGN={'acetylcholine':1.,'gaba':-1.,'glutamate':-1.,'histamine':-1.}

@njit(cache=True)
def integrate(indptr,indices,weights,v,g,until,queue,start,external_ids,external_events,muted,cut,dt):
 n=len(v);counts=np.zeros(n,np.uint32);raster=np.zeros((len(external_events),n),np.uint8) if n<100 else np.empty((0,0),np.uint8)
 am=np.exp(-dt/20.);ag=np.exp(-dt/5.);coupling=(am-ag)/3.;delay=round(1.8/dt);refr=round(2.2/dt)
 fired=np.zeros(n,np.bool_);transmissions=0;synapses=0.;active=np.empty(n,np.bool_)
 for k in range(len(external_events)):
  step=start+k;slot=step%len(queue)
  for i in range(n):
   active[i]=step>=until[i]
   if active[i]:v[i]=v[i]*am+g[i]*coupling;g[i]*=ag
   fired[i]=active[i] and v[i]>7 and not muted[i]
   if fired[i]:counts[i]+=1
   if len(raster):raster[k,i]=fired[i]
  # Synaptic updates occur after threshold detection. Refractory postsynaptic
  # variables have Brian2's conditional-write semantics (do not accumulate).
  for i in range(n):
   if active[i]:g[i]+=queue[slot,i]
   queue[slot,i]=0
  for j in range(len(external_ids)):
   if external_events[k,j] and not muted[external_ids[j]]:v[external_ids[j]]+=68.75
  for i in range(n):
   if fired[i]:
    if not cut:
     dst=(step+delay)%len(queue)
     for e in range(indptr[i],indptr[i+1]):
      w=weights[e]
      if w!=0 and not muted[indices[e]]:
       queue[dst,indices[e]]+=w;transmissions+=1;synapses+=abs(w)/.275
    v[i]=0.;g[i]=0.;until[i]=step+refr
  # Directly stimulated cells have no refractory, as in the reference model.
  for j in range(len(external_ids)):until[external_ids[j]]=0
  for i in range(n):
   if muted[i]:v[i]=0.;g[i]=0.
 return counts,transmissions,synapses,raster

class Brain:
 def __init__(self,dt_ms=.1,model='current',backend=None):
  assert model in ('current','conductance');self.model=model
  self.backend=backend or os.environ.get('FLY_BRAIN_SOLVER','blocked')
  if self.backend not in ('reference','blocked'):raise ValueError('Unknown neural backend')
  from numba import set_num_threads,config
  self.threads=max(1,min(int(os.environ.get('FLY_BRAIN_THREADS','12')),config.NUMBA_NUM_THREADS))
  set_num_threads(self.threads)
  self.dt=dt_ms;self.n=pd.read_csv(ROOT/'data/neurons.csv.gz').fillna('');self.meta=json.loads((ROOT/'data/manifest.json').read_text())
  self.c=sparse.load_npz(ROOT/'data/synapse_counts.npz').tocsc();self.sign=self.n.consensus_nt.map(SIGN).fillna(0).to_numpy(float)
  self.weights=self.c.data*.275*np.repeat(self.sign,np.diff(self.c.indptr))
  self.count=len(self.n);self.edges=self.c.nnz;self.dynamic_edges=int(np.count_nonzero(self.weights));self.class_names=self.meta['classes'];self.classes=self.n.superclass.map({x:i for i,x in enumerate(self.class_names)}).to_numpy(np.int32)
  self.groups={}
  for typ in ['DNa02','DNp09','DNp01','DNp15','DNp20','LT11','R1-R6','R7p','R8p','T4a','T4b','T5a','T5b','ORN_DM1','Ti flexor MN','Ti extensor MN']:
   for side in ['L','R']:self.groups[typ+'_'+side]=np.flatnonzero(self.n.type.eq(typ).to_numpy()&self.n.side.eq(side).to_numpy())
  self.motor=np.flatnonzero(self.n.superclass.eq('vnc_motor'))
  self.descending=np.flatnonzero(self.n.superclass.eq('descending_neuron'))
  self.reset()
 def reset(self,seed=0):
  self.v=np.zeros(self.count);self.g=np.zeros(self.count);self.until=np.zeros(self.count,np.int64);self.queue=np.zeros((round(1.8/self.dt)+1,self.count));self.step_no=0;self.rng=np.random.default_rng(seed);self.rates=np.zeros(self.count);self.total_spikes=np.zeros(self.count,np.uint64);self.last_spikes=np.zeros(self.count,np.uint32);self.muted=np.zeros(self.count,np.bool_);self.total_transmissions=0;self.synaptic_events=0.;self.last_traffic=0;self.cut=False;self.window_flow=np.zeros((len(self.class_names),len(self.class_names)),np.int64);self.window_spikes=np.zeros(self.count,np.uint32)
  if self.model=='conductance':self.g=np.zeros((2,self.count));self.queue=np.zeros((2,round(1.8/self.dt)+1,self.count))
 def advance(self,ids,rates_hz,milliseconds=10,cut=False,silence_dn=False):
  ids=np.asarray(ids,np.int64);rates=np.broadcast_to(np.asarray(rates_hz,float),(len(ids),));steps=round(milliseconds/self.dt)
  assert steps>0 and np.all((rates>=0)&(rates<=1000))
  if bool(cut)!=self.cut:self.queue.fill(0);self.cut=bool(cut)
  self.muted.fill(False)
  if silence_dn:self.muted[self.descending]=True
  events=(self.rng.random((steps,len(ids)))<rates[None,:]*self.dt/1000)
  if self.model=='conductance':
   if self.backend=='blocked':
    from conductance_fast import integrate_blocked
    solver=integrate_blocked
   else:
    from conductance import integrate_conductance
    solver=integrate_conductance
  else:solver=integrate
  args=(self.c.indptr,self.c.indices,self.weights,self.v,self.g,self.until,self.queue,self.step_no,ids,events,self.muted,cut,self.dt)
  counts,traffic,synapses,_=solver(*args,True) if self.model=='conductance' and self.backend=='blocked' else solver(*args)
  self.step_no+=steps;self.last_spikes=counts;self.window_spikes+=counts;self.total_spikes+=counts;self.total_transmissions+=int(traffic);self.last_traffic=int(traffic);self.synaptic_events+=float(synapses)
  if not cut:self.window_flow+=flow_counts(self.c.indptr,self.c.indices,self.weights,counts,self.classes,self.muted,len(self.class_names))
  alpha=1-np.exp(-milliseconds/50);self.rates+=(counts*1000/milliseconds-self.rates)*alpha
  if not(np.isfinite(self.v).all() and np.isfinite(self.g).all()):raise RuntimeError('Nonfinite neural state')
  return counts
 def readout(self):
  return {key:float(self.rates[ix].mean()) if len(ix) else 0. for key,ix in self.groups.items()}
 def edge_view(self,limit=6000):
  from activity_view import edge_view
  return edge_view(self.c.indptr,self.c.indices,self.weights,self.window_spikes,self.muted,self.cut,limit)
 def class_traffic(self):return self.window_flow.tolist()

@njit(cache=True)
def flow_counts(indptr,indices,weights,counts,classes,muted,nclass):
 matrix=np.zeros((nclass,nclass),np.int64)
 for pre in range(len(counts)):
  if counts[pre] and not muted[pre]:
   for e in range(indptr[pre],indptr[pre+1]):
    post=indices[e]
    if weights[e]!=0 and not muted[post]:matrix[classes[pre],classes[post]]+=counts[pre]
 return matrix
