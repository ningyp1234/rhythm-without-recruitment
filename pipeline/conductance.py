"""Conductance LIF variant; generic, explicitly uncalibrated MaleCNS parameters.

Conductances are relative to leak conductance. For weak input at rest, a single
anatomical synapse matches the initial current of the legacy .275 mV model:
q_exc=.275/(0-(-52)); q_inh=.275/((-52)-(-80)). No weight normalization.
Exponential midpoint integration preserves the reversal-potential bounds without
voltage clipping. Positive conductances decay even during absolute refractory.
External Poisson events open an excitatory conductance (not a voltage jump).
"""
import numpy as np
from numba import njit
PARAMS=dict(name='conductance-lif-v1',dt_ms=.1,rest_mv=-52.,threshold_mv=-45.,excitatory_reversal_mv=0.,inhibitory_reversal_mv=-80.,tau_membrane_ms=20.,tau_synapse_ms=5.,refractory_ms=2.2,delay_ms=1.8,excitatory_event_g_over_leak=.275/52,inhibitory_event_g_over_leak=.275/28,input_event_g_over_leak=68.75/52,calibration='generic assumptions, not fitted to MaleCNS physiology')

@njit(cache=True)
def passive_step(v,ge,gi,dt):
 """Relative voltage v=Vm+52. Exact frozen-midpoint leak+conductance flow."""
 half=np.exp(-dt/10.);e=ge*half;i=gi*half;total=1+e+i
 equilibrium=(52*e-28*i)/total
 return equilibrium+(v-equilibrium)*np.exp(-dt*total/20.)

@njit(cache=True)
def integrate_conductance(indptr,indices,weights,v,g,until,queue,start,external_ids,external_events,muted,cut,dt):
 n=len(v);counts=np.zeros(n,np.uint32);steps=len(external_events)
 raster=np.zeros((steps,n),np.uint8) if n<100 else np.empty((0,0),np.uint8)
 decay=np.exp(-dt/5.);delay=round(1.8/dt);refr=round(2.2/dt);fired=np.zeros(n,np.bool_);transmissions=0;synapses=0.
 for k in range(steps):
  step=start+k;slot=step%queue.shape[1]
  for i in range(n):
   v[i]=passive_step(v[i],g[0,i],g[1,i],dt) if step>=until[i] else 0.
   g[0,i]*=decay;g[1,i]*=decay
   fired[i]=step>=until[i] and v[i]>7 and not muted[i]
   if fired[i]:counts[i]+=1
   if len(raster):raster[k,i]=fired[i]
   for c in range(2):g[c,i]+=queue[c,slot,i];queue[c,slot,i]=0.
  for j in range(len(external_ids)):
   if external_events[k,j] and not muted[external_ids[j]]:g[0,external_ids[j]]+=68.75/52
  for i in range(n):
   if fired[i]:
    if not cut:
     dst=(step+delay)%queue.shape[1]
     for e in range(indptr[i],indptr[i+1]):
      w=weights[e];post=indices[e]
      if w!=0 and not muted[post]:
       if w>0:queue[0,dst,post]+=w/52
       else:queue[1,dst,post]+=-w/28
       transmissions+=1;synapses+=abs(w)/.275
    v[i]=0.;until[i]=step+refr
   if muted[i]:v[i]=0.;g[0,i]=0.;g[1,i]=0.
 return counts,transmissions,synapses,raster
