"""Compiled sampling of real emitted edges; visualization only, never dynamics."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True)
def sample_edges(indptr,indices,weights,spikes,muted,cut,limit=6000):
 active=np.flatnonzero(spikes);budget=max(1,limit//max(1,len(active)));capacity=len(active)*budget
 pre=np.empty(capacity,np.int64);post=np.empty(capacity,np.int64);value=np.empty(capacity,np.float64);size=0
 if not cut:
  for i in active:
   if muted[i]:continue
   lo,hi=indptr[i:i+2];ix=np.flatnonzero((weights[lo:hi]!=0)&(~muted[indices[lo:hi]]))
   if len(ix)>budget:ix=ix[np.linspace(0,len(ix)-1,budget).astype(np.int64)]
   for e in lo+ix:
    pre[size]=i;post[size]=indices[e];value[size]=weights[e]*spikes[i];size+=1
 if size>limit:
  ix=np.linspace(0,size-1,limit).astype(np.int64);return pre[ix],post[ix],value[ix]
 return pre[:size],post[:size],value[:size]

def edge_view(indptr,indices,weights,spikes,muted,cut,limit=6000):
 pre,post,value=sample_edges(indptr,indices,weights,spikes,muted,cut,limit)
 return dict(pre=pre.tolist(),post=post.tolist(),value=value.tolist(),render_limit=limit,meaning='sample of actual emitted synaptic events; no invented edges')
