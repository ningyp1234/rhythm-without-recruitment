"""Predeclared walking gates: motion alone, sliding and falling do not pass.
Diagnostic DN stimulation is scored separately from autonomous visual behavior.
"""
from pathlib import Path
import json,time,argparse
import numpy as np,pandas as pd
from brain import Brain,ROOT
from walking_body import WalkingBody
from walking_feedback import WalkingFeedback

def score(frames,feet,contact):
 f=pd.DataFrame(frames);feet=np.asarray(feet);contact=np.asarray(contact,bool);cycles=[]
 dt=float(np.median(np.diff(f.t))) if 't' in f and len(f)>1 else .01
 if not np.isfinite(dt) or dt<=0:raise ValueError('Walking score requires a positive sample interval')
 sustained_samples=max(1,int(np.ceil(.03/dt-1e-12)))
 # Count sustained stance -> swing -> stance events, at least 30 ms each.
 for leg in range(6):
  runs=[];start=0
  for i in range(1,len(contact)+1):
   if i==len(contact) or contact[i,leg]!=contact[start,leg]:
    if i-start>=sustained_samples:runs.append(bool(contact[start,leg]))
    start=i
  compact=[v for i,v in enumerate(runs) if i==0 or v!=runs[i-1]];cycles.append(sum(compact[i-2:i+1]==[True,False,True] for i in range(2,len(compact))))
 grounded=contact[1:]&contact[:-1];slip=np.linalg.norm(np.diff(feet[:,:,:2],axis=0),axis=2)/dt
 median_slip=float(np.median(slip[grounded])) if np.any(grounded) else float('inf')
 gates=dict(forward_distance_at_least_5mm=float(f.x.iloc[-1]-f.x.iloc[0])>=5,body_stays_upright=float((f.upright>.8).mean())>=.99,body_clear_of_ground=float((f.z>.4).mean())>=.99,all_six_legs_complete_two_steps=min(cycles)>=2,stance_slip_below_2mm_s=median_slip<2)
 # WalkingBody.contact requires positive foot contact force. Zero total force
 # also gives a zero nonfoot fraction, so airborne frames must be excluded.
 support_fraction=float(((f.nonfoot_load_fraction<=.01)&contact.any(axis=1)).mean()) if 'nonfoot_load_fraction' in f else 0.
 gates['feet_provide_support_without_body_drag']=support_fraction>=.99
 return dict(scoring_version='walking-kinematics-v3-rate-aware-contact-support',sample_dt_seconds=dt,sustained_event_samples=sustained_samples,gates=gates,walking_pass=all(gates.values()),foot_only_support_fraction=support_fraction,forward_mm=float(f.x.iloc[-1]-f.x.iloc[0]),lateral_mm=float(f.y.iloc[-1]-f.y.iloc[0]),upright_fraction=float((f.upright>.8).mean()),min_height=float(f.z.min()),steps_per_leg=cycles,median_stance_slip_mm_s=median_slip)

def run(b,body,fb,mode='DNg100',seconds=2.,seed=17,feedback=True,motor=True,cut=False):
 b.reset(seed);body.reset();rows=[];feet=[];contacts=[];mn=[];inputs=[]
 ids=np.flatnonzero(b.n.type.isin(mode.split('+')));visual=None
 if mode=='retina':
  from vision_bridge import VisionBridge
  visual=VisionBridge(b.n)
 for step in range(round(seconds*100)):
  si,sr=fb.encode(*body.kinematics(),enabled=feedback)
  if visual is not None:ei,er=visual.stimulus(body.position(),body.orientation(),[10,8,5],1,150)
  else:ei,er=ids,np.full(len(ids),150.)
  ai=np.r_[ei,si].astype(np.int64);ar=np.r_[er,sr];b.advance(ai,ar,10,cut=cut);body.step(b.rates,10,enabled=motor);t=body.telemetry();p=t['position']
  rows.append(dict(t=(step+1)*.01,x=p[0],y=p[1],z=p[2],upright=t['upright'],nonfoot_load_fraction=t['nonfoot_load_fraction'],motor_spikes=int(b.last_spikes[body.motor_ids].sum()),sensory_spikes=int(b.last_spikes[fb.ids].sum()),torque_norm=float(np.linalg.norm(body.torques))))
  feet.append(t['feet']);contacts.append(t['contact']);mn.append(b.rates[body.motor_ids].copy());inputs.append(sr)
 return score(rows,feet,contacts),pd.DataFrame(rows),dict(feet=np.array(feet),contact=np.array(contacts),motor_rates=np.array(mn),feedback_rates=np.array(inputs))

def main():
 p=argparse.ArgumentParser();p.add_argument('--modes',default='DNg100,DNb08,DNg97,retina');p.add_argument('--seconds',type=float,default=2);p.add_argument('--coxa-axis',choices=['roll','pitch','yaw'],default='roll');p.add_argument('--gain',type=float,default=8);p.add_argument('--stiffness',type=float,default=32);p.add_argument('--output',type=Path,default=ROOT/'results/walking');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 b=Brain(model='conductance');body=WalkingBody(b.n,gain=a.gain,stiffness=a.stiffness,coxa_axis=a.coxa_axis);fb=WalkingFeedback(b.n);report={}
 (a.output/'motor-map.json').write_text(json.dumps(body.mapping,indent=2));(a.output/'sensory-map.json').write_text(json.dumps(fb.mapping,indent=2))
 for mode in a.modes.split(','):
  start=time.perf_counter();r,f,arrays=run(b,body,fb,mode,a.seconds);r.update(wall_seconds=time.perf_counter()-start,model_seconds=a.seconds,artificial_DN_stimulus=mode!='retina',mapped_MNs=len(body.motor_ids),sensory_cells=len(fb.ids),stiffness=a.stiffness,torque_gain=a.gain,coxa_axis=a.coxa_axis)
  f.to_csv(a.output/(mode+'.csv'),index=False);np.savez_compressed(a.output/(mode+'.npz'),**arrays);report[mode]=r;(a.output/'benchmark.json').write_text(json.dumps(report,indent=2));print(mode,json.dumps(r),flush=True)
if __name__=='__main__':main()
