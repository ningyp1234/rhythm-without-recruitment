"""Untethered six-leg research body with explicit anatomical-family torque bridges.
Official NeuroMechFly geometry/contact physics; NOT a measured whole-fly muscle
model. No CPG, desired joint angle, trajectory, global goal or body force.
Non-driven DoFs retain passive posture springs. All assumptions exported.
"""
from types import SimpleNamespace
import numpy as np,mujoco
from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly
from legacy_body import Body as SceneExporter
LEGS=[('lf','T1','L','ProLN'),('lm','T2','L','MesoLN'),('lh','T3','L','MetaLN'),('rf','T1','R','ProLN'),('rm','T2','R','MesoLN'),('rh','T3','R','MetaLN')]
FAMILIES=[('coxa',['Tergopleural/Pleural promotor MN'],['Pleural remotor/abductor MN','Sternal posterior rotator MN']),('trochanterfemur',['Tr flexor MN','Acc. tr flexor MN'],['Tr extensor MN','Sternotrochanter MN','Tergotr. MN']),('tibia',['Ti flexor MN','Acc. ti flexor MN'],['Ti extensor MN'])]
class WalkingBody:
 def __init__(self,neurons,stiffness=32.,gain=8.,coxa_axis='roll',include_anterior_rotators=True,experimental_adhesion=False):
  # Zero position gain: controllers are absent, passive springs are explicit.
  # Stiffness 32 is a declared support assumption: 8 let the hind coxae carry
  # ~57% of resting body weight. This is not a measured muscle calibration.
  self.fly=make_locomotion_fly(name='fly',actuator_gain=0.,joint_stiffness=stiffness,joint_damping=.12,add_adhesion=experimental_adhesion,colorize=True)
  self.experimental_adhesion=bool(experimental_adhesion)
  world=FlatGroundWorld();world.add_fly(self.fly,[0,0,.8],Rotation3D('quat',[1,0,0,0]),bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,add_ground_contact_sensors=False)
  self.sim=Simulation(world);m,d=self.sim.mj_model,self.sim.mj_data;self.root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fly/c_thorax');self.geom_ids=[i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
  self.dofs=self.fly.get_actuated_jointdofs_order('position');self.units=[];self.mapping=[];self.joints=[];self.qadr=[];self.signs=[];self.gain=gain;self.stiffness=stiffness
  self.tip_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fly/'+l+'_tarsus5') for l,*_ in LEGS]
  self.sim.reset();mujoco.mj_forward(m,d)
  for leg,seg,side,nerve in LEGS:
   mask=neurons.superclass.eq('vnc_motor')&neurons.somaNeuromere.eq(seg)&neurons.side.eq(side)
   tip=self.tip_ids[[l for l,*_ in LEGS].index(leg)];jac=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jac,None,tip)
   for child,positive,negative in FAMILIES:
    positive=list(positive)
    if child=='coxa' and include_anterior_rotators:positive.append('Sternal anterior rotator MN')
    idx=next(i for i,k in enumerate(self.dofs) if k.child.name==leg+'_'+child and k.axis.value==(coxa_axis if child=='coxa' else 'pitch'));jid=int(m.actuator_trnid[idx,0]);adr=int(m.jnt_dofadr[jid]);q=int(m.jnt_qposadr[jid])
    if child=='coxa':sign=float(np.sign(jac[0,adr])) # NMF local roll is the dominant anterior swing axis; verified by foot Jacobian
    elif child=='trochanterfemur':sign=float(np.sign(jac[2,adr])) # flexor elevates foot locally
    else:sign=float(np.sign(d.qpos[q])) # tibia flexion increases absolute bend
    if sign==0:raise ValueError('Undefined coordinate sign')
    a=np.flatnonzero(mask&neurons.type.str.strip().isin(positive));b=np.flatnonzero(mask&neurons.type.str.strip().isin(negative));
    if include_anterior_rotators and (not len(a) or not len(b)):raise ValueError(f'Missing motor population: {leg} {child}; empty is not silent')
    self.units.append((a,b));self.joints.append(adr);self.qadr.append(q);self.signs.append(sign)
    self.mapping.append(dict(leg=leg,joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid),positive_types=positive,negative_types=negative,positive_body_ids=neurons.bodyId.iloc[a].tolist(),negative_body_ids=neurons.bodyId.iloc[b].tolist(),sign=sign,positive_count=len(a),negative_count=len(b),assumption='Muscle-family torque approximation on one axis, including annotated sternal anterior rotator; actual multidimensional muscle moment arms uncalibrated'))
  self.joints=np.array(self.joints);self.qadr=np.array(self.qadr);self.signs=np.array(self.signs);self.motor_ids=np.unique(np.concatenate([x for pair in self.units for x in pair]));self.tibia_q=self.qadr[2::3]
  self.geom_leg=np.full(m.ngeom,-1)
  for i in range(m.ngeom):
   name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) or ''
   for j,(leg,*_) in enumerate(LEGS):
    if '/'+leg+'_tarsus' in name:self.geom_leg[i]=j
  self.reset()
 def reset(self):
  self.sim.reset();m,d=self.sim.mj_model,self.sim.mj_data;d.ctrl[:]=0
  # Same passive settling for every counterfactual; no neural/pose controller.
  for _ in range(round(.4/self.sim.timestep)):mujoco.mj_step(m,d)
  d.time=0.;self.t=0.;self.origin=self.position();self.initial_q=d.qpos.copy();self.torques=np.zeros(18);self.activation=np.zeros((18,2));self.rates=np.zeros((18,2));self.reference_orientation=self.orientation()
 def position(self):return self.sim.mj_data.xpos[self.root].copy()
 def orientation(self):return self.sim.mj_data.xmat[self.root].reshape(3,3).copy()
 def step(self,motor_rates,milliseconds=10,enabled=True):
  m,d=self.sim.mj_model,self.sim.mj_data
  self.rates=np.array([[(motor_rates[a].mean() if len(a) else 0),(motor_rates[b].mean() if len(b) else 0)] for a,b in self.units]);target=1-np.exp(-self.rates/100) if enabled else np.zeros_like(self.rates)
  for _ in range(round(milliseconds/1000/self.sim.timestep)):
   self.activation+=(target-self.activation)*(-np.expm1(-self.sim.timestep/.02));self.torques=self.gain*self.signs*(self.activation[:,0]-self.activation[:,1]);d.qfrc_applied[:]=0;d.qfrc_applied[self.joints]=self.torques;mujoco.mj_step(m,d)
  self.t+=milliseconds/1000
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning.number.sum():raise RuntimeError('Invalid walking-body dynamics')
 def kinematics(self):
  d=self.sim.mj_data;return abs(d.qpos[self.tibia_q]),np.sign(d.qpos[self.tibia_q])*d.qvel[self.joints[2::3]]
 def contacts(self):
  d=self.sim.mj_data;m=self.sim.mj_model;loads=np.zeros(6);counts=np.zeros(6,int)
  for c in range(d.ncon):
   a,b=map(int,d.contact[c].geom);la,lb=self.geom_leg[a],self.geom_leg[b]
   if (la>=0 and m.geom_bodyid[b]==0) or (lb>=0 and m.geom_bodyid[a]==0):
    k=max(la,lb);force=np.zeros(6);mujoco.mj_contactForce(m,d,c,force);loads[k]+=max(0,force[0]);counts[k]+=force[0]>1e-5
  return counts>0,loads
 def telemetry(self):
  d=self.sim.mj_data;contact,load=self.contacts();return dict(position=self.position().tolist(),displacement=(self.position()-self.origin).tolist(),upright=float(self.orientation()[2,2]),height=float(self.position()[2]),contact=contact.tolist(),load=load.tolist(),feet=d.xpos[self.tip_ids].tolist(),joint_degrees=np.rad2deg(d.qpos[self.qadr]).reshape(6,3).tolist(),motor_rates=self.rates.tolist(),torques=self.torques.tolist(),mapped_mn=len(self.motor_ids),motor_population_counts=[[len(a),len(b)] for a,b in self.units],passive_stiffness=self.stiffness,torque_gain=self.gain,physics_warnings=int(d.warning.number.sum()),**self.support())
 def support(self):
  """Distinguish normal foot support from dragging a hip, knee or body."""
  m,d=self.sim.mj_model,self.sim.mj_data;nonfoot=0.;total=0.;names=set()
  for k in range(d.ncon):
   a,b=map(int,d.contact[k].geom)
   if m.geom_bodyid[a]==0:body_geom=b
   elif m.geom_bodyid[b]==0:body_geom=a
   else:continue
   force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force);f=max(0.,float(force[0]));total+=f
   if self.geom_leg[body_geom]<0:
    nonfoot+=f
    if f>1e-5:names.add(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[body_geom])))
  return dict(nonfoot_load_fraction=nonfoot/max(total,1e-12),nonfoot_ground_contacts=sorted(names))
 poses=SceneExporter.poses
 export_scene=SceneExporter.export_scene
