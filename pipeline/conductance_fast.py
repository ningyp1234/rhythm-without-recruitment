"""Delay-blocked conductance solver; same dt, event ordering and float64 arithmetic.

No event emitted inside a block can arrive until that block has finished because
block length <= minimum transmission delay. Each neuron can therefore integrate
its block independently. Future queue slots are independent and scattered in parallel. Each slot retains
ascending presynaptic/edge accumulation order, as in the reference solver.
Global display traffic totals are reduced by timestep. Neural arithmetic is unchanged.
"""
import numpy as np
from numba import njit, prange
from conductance import passive_step


def _advance_block(v,g,until,queue,start,length,events,offset,head,next_input,muted,dt,fired,counts,raster):
 decay=np.exp(-dt/5.);refr=round(2.2/dt)
 for i in prange(len(v)):
  vv=v[i];ge=g[0,i];gi=g[1,i];u=until[i];count=0
  for k in range(length):
   step=start+k;slot=step%queue.shape[1]
   # The exact resting equilibrium remains zero without evaluating exp().
   if step<u:vv=0.
   elif vv!=0. or ge!=0. or gi!=0.:vv=passive_step(vv,ge,gi,dt)
   ge*=decay;gi*=decay
   spike=step>=u and vv>7 and not muted[i]
   fired[k,i]=spike
   if len(raster):raster[offset+k,i]=spike
   ge+=queue[0,slot,i];gi+=queue[1,slot,i]
   queue[0,slot,i]=0.;queue[1,slot,i]=0.
   j=head[i]
   while j>=0:
    if events[offset+k,j] and not muted[i]:ge+=68.75/52
    j=next_input[j]
   if spike:count+=1;vv=0.;u=step+refr
   if muted[i]:vv=0.;ge=0.;gi=0.
  v[i]=vv;g[0,i]=ge;g[1,i]=gi;until[i]=u;counts[i]+=count

_serial_block=njit(cache=True,nogil=True)(_advance_block)
_parallel_block=njit(cache=True,nogil=True,parallel=True)(_advance_block)

def _scatter_slots(indptr,indices,weights,queue,start,delay,length,fired,muted):
 # Within a delay block, each emission time maps to a distinct future queue slot.
 # Slots can be processed independently; each slot keeps presynaptic order.
 traffic=np.zeros(length,np.int64);weighted=np.zeros(length,np.float64)
 for k in prange(length):
  dst=(start+k+delay)%queue.shape[1];nt=0;ns=0.
  for i in range(len(indptr)-1):
   if fired[k,i]:
    for e in range(indptr[i],indptr[i+1]):
     w=weights[e];post=indices[e]
     if w!=0 and not muted[post]:
      if w>0:queue[0,dst,post]+=w/52
      else:queue[1,dst,post]+=-w/28
      nt+=1;ns+=abs(w)/.275
  traffic[k]=nt;weighted[k]=ns
 return traffic,weighted
_scatter_serial=njit(cache=True,nogil=True)(_scatter_slots)
_scatter_parallel=njit(cache=True,nogil=True,parallel=True)(_scatter_slots)

@njit(cache=True,nogil=True)
def integrate_blocked(indptr,indices,weights,v,g,until,queue,start,external_ids,external_events,muted,cut,dt,parallel=False):
 n=len(v);steps=len(external_events);delay=round(1.8/dt)
 if delay<1:raise ValueError('Delay-block solver requires dt <= transmission delay')
 counts=np.zeros(n,np.uint32)
 raster=np.zeros((steps,n),np.uint8) if n<100 else np.empty((0,0),np.uint8)
 fired=np.empty((min(delay,steps),n),np.bool_)
 head=np.full(n,-1,np.int64);next_input=np.full(len(external_ids),-1,np.int64)
 # Preserve source order, including repeated externally stimulated IDs.
 for j in range(len(external_ids)-1,-1,-1):
  i=external_ids[j];next_input[j]=head[i];head[i]=j
 transmissions=0;synapses=0.
 for offset in range(0,steps,delay):
  length=min(delay,steps-offset)
  if parallel:_parallel_block(v,g,until,queue,start+offset,length,external_events,offset,head,next_input,muted,dt,fired,counts,raster)
  else:_serial_block(v,g,until,queue,start+offset,length,external_events,offset,head,next_input,muted,dt,fired,counts,raster)
  if not cut:
   traffic,weighted=_scatter_parallel(indptr,indices,weights,queue,start+offset,delay,length,fired,muted) if parallel else _scatter_serial(indptr,indices,weights,queue,start+offset,delay,length,fired,muted)
   for k in range(length):transmissions+=traffic[k];synapses+=weighted[k]
 return counts,transmissions,synapses,raster
