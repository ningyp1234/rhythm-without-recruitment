"""Neural-to-muscle test interface; no gait generator, pursuit rule or flight force.
Only annotated tibia flexor/extensor motor units are connected to six joint torques.
Static pose springs support the physical test body. This is not a validated muscle model.
"""
import numpy as np,mujoco,json,gzip
from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly,PreprogrammedSteps,LocomotionAction,apply_locomotion_action

class Body:
 def __init__(self,neurons):
  self.fly=make_locomotion_fly(name='fly',actuator_gain=45.,add_adhesion=False,colorize=True)
  world=FlatGroundWorld();world.add_fly(self.fly,[0,0,.8],Rotation3D('quat',[1,0,0,0]),bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,add_ground_contact_sensors=False)
  self.sim=Simulation(world);m=self.sim.mj_model;self.root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fly/c_thorax');self.geom_ids=[i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
  self.dofs=self.fly.get_actuated_jointdofs_order('position');self.rest=PreprogrammedSteps().default_pose_by_dof_order(self.dofs);self.units=[];self.joints=[];self.mapping=[]
  for leg,segment,side in [('lf','T1','L'),('lm','T2','L'),('lh','T3','L'),('rf','T1','R'),('rm','T2','R'),('rh','T3','R')]:
   idx=next(i for i,d in enumerate(self.dofs) if d.child.name==leg+'_tibia' and d.axis.value=='pitch')
   aid=idx;jid=int(m.actuator_trnid[aid,0]);self.joints.append(int(m.jnt_dofadr[jid]))
   motor=neurons.superclass.eq('vnc_motor')&neurons.somaNeuromere.eq(segment)&neurons.side.eq(side)
   flex=np.flatnonzero((motor&neurons.type.isin(['Ti flexor MN','Acc. ti flexor MN'])).to_numpy());ext=np.flatnonzero((motor&neurons.type.eq('Ti extensor MN')).to_numpy());self.units.append((flex,ext))
   self.mapping.append(dict(leg=leg,joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid),flexor_body_ids=neurons.bodyId.iloc[flex].tolist(),extensor_body_ids=neurons.bodyId.iloc[ext].tolist()))
  self.reset()
 def reset(self):
  self.sim.reset();apply_locomotion_action(self.sim,'fly',LocomotionAction(joint_angles=self.rest,adhesion_onoff=None));self.sim.warmup();self.t=0.;self.torques=np.zeros(6);self.sim.mj_data.qfrc_applied[:]=0
 def position(self):return self.sim.mj_data.xpos[self.root].copy()
 def orientation(self):return self.sim.mj_data.xmat[self.root].reshape(3,3).copy()
 def step(self,motor_rates,milliseconds=10,enabled=True):
  # Hz -> antagonist contraction difference -> torque. Scaling and signs are
  # uncalibrated interface assumptions, NOT inferred biological motor parameters.
  torques=np.array([.005*((motor_rates[f].mean() if len(f) else 0)-(motor_rates[e].mean() if len(e) else 0)) for f,e in self.units])
  self.torques=np.clip(torques,-2,2) if enabled else np.zeros(6)
  d=self.sim.mj_data;d.qfrc_applied[:]=0;d.qfrc_applied[self.joints]=self.torques
  for _ in range(round(milliseconds/1000/self.sim.timestep)):self.sim.step()
  self.t+=milliseconds/1000
  assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
 def poses(self):
  d=self.sim.mj_data;out=[]
  for g in self.geom_ids:
   q=np.empty(4);mujoco.mju_mat2Quat(q,d.geom_xmat[g]);out.append([*d.geom_xpos[g].tolist(),*q.tolist()])
  return out
 def export_scene(self,path):
  m=self.sim.mj_model;meshes={};geoms=[]
  for g in self.geom_ids:
   k=int(m.geom_dataid[g]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)
   if k not in meshes:
    v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]];meshes[k]={'vertices':v.ravel().tolist(),'faces':f.ravel().tolist()}
   geoms.append(dict(mesh=k,name=name))
  with gzip.open(path,'wt') as f:json.dump(dict(meshes=meshes,geoms=geoms,poses=self.poses()),f,separators=(',',':'))
