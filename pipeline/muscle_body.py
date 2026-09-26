"""Official FlyMimic Hill muscles driven solely by annotated MaleCNS MN activity.
The source body is tethered, with only its LEFT FRONT LEG muscle driven.
No desired angle, gait, goal bearing or learned behavioral policy enters step().
"""
from pathlib import Path
from types import SimpleNamespace
import json,gzip
import numpy as np
import mujoco
from legacy_body import Body as SceneExporter
ROOT=Path(__file__).resolve().parents[1]
# Anatomical muscle-family bridge, not a cross-animal single-fiber registration.
MOTOR_TYPES=[
 ['Tergopleural/Pleural promotor MN'],['Tergopleural/Pleural promotor MN'],
 ['Pleural remotor/abductor MN'],['Tergopleural/Pleural promotor MN'],
 ['Sternal anterior rotator MN'],['Sternal posterior rotator MN'],['Sternal adductor MN'],
 ['Tr flexor MN'],['Sternotrochanter MN','Tergotr. MN'],['Sternotrochanter MN','Tergotr. MN'],
 ['Acc. tr flexor MN'],['Tr extensor MN'],['Tr flexor MN'],
 ['Ti flexor MN','Acc. ti flexor MN'],['Ti extensor MN']]

class MuscleBody:
 def __init__(self,neurons):
  m=mujoco.MjModel.from_xml_path(str(ROOT/'data/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'))
  d=mujoco.MjData(m);self.sim=SimpleNamespace(mj_model=m,mj_data=d,timestep=m.opt.timestep)
  self.root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')
  self.geom_ids=[i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
  self.joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,s) for s in ['joint_LFCoxa_yaw','joint_LFCoxa_pitch','joint_LFCoxa_roll','joint_LFTrochanter_yaw','joint_LFTrochanter_pitch','joint_LFTrochanter_roll','joint_LFTibia_pitch']]
  self.qadr=m.jnt_qposadr[self.joints];self.dadr=m.jnt_dofadr[self.joints];self.tibia_q=int(self.qadr[-1]);self.tibia_d=int(self.dadr[-1])
  self.units=[];self.mapping=[]
  motor=neurons.superclass.eq('vnc_motor')&neurons.somaNeuromere.eq('T1')&neurons.side.eq('L');types=neurons.type.str.strip()
  for i,typ in enumerate(MOTOR_TYPES):
   ix=np.flatnonzero((motor&types.isin(typ)).to_numpy());assert len(ix)>0,typ;self.units.append(ix)
   self.mapping.append(dict(actuator=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i),leg='lf',motor_types=typ,motor_body_ids=neurons.bodyId.iloc[ix].tolist(),mapping_level='muscle-family, pooled equally; branch identity unresolved'))
  self.motor_ids=np.unique(np.concatenate(self.units));self.muscle_names=[x['actuator'] for x in self.mapping]
  # Equilibrate the source's passive mechanics before starting neural time.
  # This removes startup settling as a false positive for neural movement.
  mujoco.mj_resetDataKeyframe(m,d,0);d.ctrl[:]=0
  for _ in range(round(3/m.opt.timestep)):mujoco.mj_step(m,d)
  self.initial_qpos=d.qpos.copy();self.initial_qvel=d.qvel.copy();self.reset()
 def reset(self):
  m,d=self.sim.mj_model,self.sim.mj_data;mujoco.mj_resetData(m,d);d.qpos[:]=self.initial_qpos;d.qvel[:]=self.initial_qvel;mujoco.mj_forward(m,d)
  self.t=0.;self.torques=np.zeros(6);self.mean_rates=np.zeros(15);self.excitations=np.zeros(15);self.reference_q=d.qpos[self.qadr].copy()
 def position(self):return self.sim.mj_data.xpos[self.root].copy()
 def orientation(self):return self.sim.mj_data.xmat[self.root].reshape(3,3).copy()
 def step(self,motor_rates,milliseconds=10,enabled=True):
  m,d=self.sim.mj_model,self.sim.mj_data
  self.mean_rates=np.array([motor_rates[ix].mean() for ix in self.units])
  # Explicit uncalibrated neuromuscular recruitment: 100 Hz -> 0.632 excitation.
  # Bounded monotonic response; actual activation delay/force/tendon mechanics
  # are those of the official source, not prescribed joint angles.
  self.excitations=1-np.exp(-np.maximum(self.mean_rates,0)/100) if enabled else np.zeros(15)
  d.ctrl[:]=self.excitations;d.qfrc_applied[:]=0
  for _ in range(round(milliseconds/1000/m.opt.timestep)):mujoco.mj_step(m,d)
  self.t+=milliseconds/1000
  self.torques[:]=0;self.torques[0]=d.qfrc_actuator[self.tibia_d] if enabled else 0.
  if not(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or d.warning.number.sum():raise RuntimeError('Musculoskeletal integration failed')
 def kinematics(self):
  d=self.sim.mj_data;return float(d.qpos[self.tibia_q]),float(d.qvel[self.tibia_d])
 def telemetry(self):
  m,d=self.sim.mj_model,self.sim.mj_data
  return dict(kind='flymimic_left_front',scope='tethered left front leg',muscle_names=self.muscle_names,excitation=self.excitations.tolist(),activation=d.act.tolist(),force=d.actuator_force.tolist(),mn_rates_hz=self.mean_rates.tolist(),joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in self.joints],joint_degrees=np.rad2deg(d.qpos[self.qadr]).tolist(),joint_delta_degrees=np.rad2deg(d.qpos[self.qadr]-self.reference_q).tolist(),joint_velocity_rad_s=d.qvel[self.dadr].tolist(),mapped_motor_units=len(self.motor_ids),warning_count=int(d.warning.number.sum()),leg_points=[d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)].tolist() for name in ['LFCoxa','LFFemur','LFTibia','LFTarsus1','LFTarsus5']])
 poses=SceneExporter.poses
 export_scene=SceneExporter.export_scene
