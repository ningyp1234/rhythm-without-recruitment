"""Six-leg FeCO input using only exact existing MaleCNS annotations.
Missing right-front SNpp50 is left missing, never invented or mirrored.
"""
from pathlib import Path
import pandas as pd,numpy as np
from walking_body import LEGS
class WalkingFeedback:
 def __init__(self,neurons,annotations=None):
  a=pd.read_csv(Path(__file__).resolve().parents[1]/'data/proprioception/annotations.csv').fillna('') if annotations is None else annotations.fillna('');lookup={int(i):k for k,i in enumerate(neurons.bodyId)};self.groups=[];self.mapping=[]
  for leg,seg,side,nerve in LEGS:
   for typ in ['SNpp39','SNpp41','SNpp50','SNpp51']:
    ids=a.loc[a.type.eq(typ)&a.entryNerve.eq(nerve)&a.rootSide.eq(side),'bodyId'].to_numpy();ix=np.array([lookup[int(i)] for i in ids],dtype=np.int64);self.groups.append(ix);self.mapping.append(dict(leg=leg,type=typ,body_ids=ids.tolist(),entryNerve=nerve,rootSide=side))
  self.ids=np.concatenate(self.groups);self.rates=np.zeros(len(self.ids));self.type_rates=np.zeros((6,4))
 def encode(self,angles,velocities,enabled=True,gain=1.,polarity=1,
            velocity_gain=1.,position_gain=1.):
  velocities=np.asarray(velocities)*polarity
  flex=np.clip((angles-.4789)/(2.502-.4789),0,1);self.type_rates=np.array([np.minimum(150,30*np.maximum(-velocities,0))*velocity_gain,np.minimum(150,30*np.maximum(velocities,0))*velocity_gain,80*flex*position_gain,80*(1-flex)*position_gain]).T*gain if enabled else np.zeros((6,4));self.type_rates=np.clip(self.type_rates,0,300)
  self.rates=np.concatenate([np.full(len(g),r) for g,r in zip(self.groups,self.type_rates.ravel())]);return self.ids,self.rates.copy()

 def telemetry(self):
  return dict(cells=len(self.ids),types=['SNpp39','SNpp41','SNpp50','SNpp51'],type_rates_hz=self.type_rates.tolist(),scope='six-leg FeCO; assumed tuning')
