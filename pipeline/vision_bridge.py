"""Official pretrained FlyVis -> MaleCNS T4/T5 by type and hemisphere.

This is NOT cell-resolved cross-dataset registration. Receptive-field identity is
lost at the type mean. Rates (Hz per model activity unit) and eye projection are
explicit interface assumptions; no desired direction or motor command is used.
"""
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
TYPES=[family+direction for family in ['T4','T5'] for direction in 'abcd']

class VisionBridge:
 def __init__(self,neurons):
  d=ROOT/'data/vision';p=np.load(d/'parameters.npz');self.w=sparse.load_npz(d/'weights.npz');self.bias=p['bias'];self.tau=p['tau'];self.input_index=p['input_index'];self.az=p['azimuth'];self.el=p['elevation'];self.meta=json.loads((d/'manifest.json').read_text());n=pd.read_csv(d/'nodes.csv.gz');self.indices=[np.flatnonzero(n.type.eq(t)) for t in TYPES]
  self.targets=[np.flatnonzero(neurons.type.eq(t)&neurons.side.eq(side)) for side in ['L','R'] for t in TYPES]
  self.ids=np.concatenate(self.targets);self.mapping=[dict(side=s,type=t,flyvis_units=len(self.indices[k]),male_cns_body_ids=neurons.bodyId.iloc[self.targets[eye*8+k]].astype(int).tolist()) for eye,s in enumerate(['L','R']) for k,t in enumerate(TYPES)]
  self.dt=.002;self.reset()
 def reset(self):
  self.activity=np.tile(self.bias[:,None],(1,2));self.response=np.zeros((2,8));self.pixels=np.full((2,721),.05);self.rate_means=np.zeros((2,8))
  for _ in range(250):self.advance_pixels(self.pixels)
 def advance_pixels(self,pixels):
  drive=np.zeros_like(self.activity)
  for eye in range(2):drive[self.input_index,eye]=pixels[eye]
  self.activity+=self.dt/np.maximum(self.tau[:,None],self.dt)*(-self.activity+self.bias[:,None]+self.w@np.maximum(self.activity,0)+drive)
  self.response=np.array([np.maximum(self.activity[ix],0).mean(axis=0) for ix in self.indices]).T
  if not np.isfinite(self.activity).all():raise RuntimeError('Nonfinite FlyVis activity')
 def render(self,position,orientation,light,power):
  delta=orientation.T@(np.asarray(light)-np.asarray(position));distance=max(float(np.linalg.norm(delta)),1e-9);direction=delta/distance
  # Idealized compound eyes: mirrored horizontal coordinates, vertical elevation;
  # 5.8 degree hex pitch from earlier FlyVis experiment, not measured eye optics.
  pixels=[]
  for side in [1,-1]:
   az=side*(np.pi/3+self.az);el=self.el
   ray=np.column_stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
   separation=np.arccos(np.clip(ray@direction,-1,1));sigma=max(np.deg2rad(5.8),np.arctan2(.6,distance))
   intensity=power/(1+(distance/10)**2)*np.exp(-.5*(separation/sigma)**2)
   pixels.append(.05+.95*intensity)
  return np.asarray(pixels)
 def stimulus(self,position,orientation,light,power,rate_scale):
  self.pixels=self.render(position,orientation,light,power)
  for _ in range(5):self.advance_pixels(self.pixels)
  # Type averages deliberately preserve tonic activity. Never subtract left/right
  # outputs to create steering. Saturation is an explicit rate-interface bound.
  self.rate_means=np.minimum(300,rate_scale*self.response)
  rates=np.concatenate([np.full(len(ix),self.rate_means.ravel()[k]) for k,ix in enumerate(self.targets)])
  return self.ids,rates
 def telemetry(self):
  return dict(model='FlyVis flow/0000/000',units_per_eye=self.w.shape[0],eyes=2,types=TYPES,activity=self.response.tolist(),rates_hz=self.rate_means.tolist(),pixels=self.pixels.tolist(),mapping='type mean + hemisphere; not cell-resolved',mapped_CNS_cells=len(self.ids))
