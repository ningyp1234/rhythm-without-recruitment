"""EXPERIMENTAL six-leg geometrical transfer of official FlyMimic LF tendons.

No gait, target angle, phase, heading or root force enters this candidate.
Each copied Hill muscle is driven by its actual MaleCNS motor-family rate.
Only the source LEFT FRONT anatomy is measured/published. Every transfer to
NeuroMechFly (including LF) is an uncalibrated geometric hypothesis, not an
official six-leg muscle reconstruction. Production WalkingBody is unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from walking_body import WalkingBody, LEGS
from muscle_body import MOTOR_TYPES

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'


def _rotation_between(a,b):
    """Proper rotation carrying source longitudinal bone axis onto target axis."""
    a=np.asarray(a,float)/np.linalg.norm(a);b=np.asarray(b,float)/np.linalg.norm(b)
    v=np.cross(a,b);c=float(np.dot(a,b))
    if c < -1+1e-10:
        raise ValueError('Antipodal bone axes need an explicit anatomical frame')
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+k@k/(1+c)


class HomologousHillBody(WalkingBody):
    """Native Hill-muscle candidate, compatible with Experiment's 18-group summary.

    stiffness remains the existing support assumption (32). Source force maxima
    are copied without amplitude tuning. source force-length intervals are
    multiplied by the target/source tendon length ratio at each model's neutral
    keyframe, preserving normalized resting operating length as a hypothesis.
    """
    def __init__(self,neurons,stiffness=32.,source_force_scale=1.,experimental_adhesion=False):
        if source_force_scale != 1.:
            raise ValueError('This candidate does not allow outcome-tuned muscle-force scaling')
        super().__init__(neurons,stiffness=stiffness,gain=8.,experimental_adhesion=experimental_adhesion)
        old_m=self.sim.mj_model
        spec=self.sim.world.mjcf_root.copy()
        source_xml=ET.parse(SOURCE).getroot()
        source_bodies={node.attrib['name']:node for node in source_xml.iter('body')}
        source_sites={}
        for name,node in source_bodies.items():
            for site in node.findall('site'):
                source_sites[site.attrib['name']]=(name,np.fromstring(site.attrib.get('pos','0 0 0'),sep=' '))
        position=lambda name:np.fromstring(source_bodies[name].attrib.get('pos','0 0 0'),sep=' ')
        src_coxa_origin=position('LFCoxa')
        src_vectors={
            'coxa':position('LFTrochanter'),
            'femur':position('LFFemur')+position('LFTibia'),
            'tibia':position('LFTarsus1'),
        }
        source_m=mujoco.MjModel.from_xml_path(str(SOURCE));source_d=mujoco.MjData(source_m)
        mujoco.mj_resetDataKeyframe(source_m,source_d,0);mujoco.mj_forward(source_m,source_d)
        # Site transfer is defined in structural (zero-joint) body-local frames,
        # not world/keyframe frames. These particular source segment rotations
        # are all identity, so adding the split femur offset is valid. Refuse a
        # future differently rigged source instead of silently mixing frames.
        self.frame_audit={'source_body_quaternions':{},'target_body_quaternions':{}}
        for name in ['LFCoxa','LFTrochanter','LFFemur','LFTibia']:
            bi=mujoco.mj_name2id(source_m,mujoco.mjtObj.mjOBJ_BODY,name)
            quat=source_m.body_quat[bi]
            self.frame_audit['source_body_quaternions'][name]=quat.tolist()
            if not np.allclose(quat,[1,0,0,0],atol=1e-12):
                raise ValueError('Unvalidated source structural frame rotation: '+name)
        actuator_xml=list(source_xml.find('actuator'))
        tendons=list(source_xml.find('tendon'))
        self.hill_units=[];self.hill_mapping=[];self.transfer_sites=[];self.unmapped_source_templates=[]
        source_muscle_indices=[]
        tendon_names=[];muscle_names=[]
        for li,(leg,neuromere,side,_) in enumerate(LEGS):
            mirror=np.diag([1.,-1. if side=='R' else 1.,1.])
            lookup={name:mujoco.mj_name2id(old_m,mujoco.mjtObj.mjOBJ_BODY,'fly/'+leg+'_'+name)
                    for name in ['coxa','trochanterfemur','tibia','tarsus1']}
            for name,bi in lookup.items():
                quat=old_m.body_quat[bi]
                self.frame_audit['target_body_quaternions'][leg+'_'+name]=quat.tolist()
                if not np.allclose(quat,[1,0,0,0],atol=1e-12):
                    raise ValueError('Unvalidated target structural frame rotation: '+leg+'_'+name)
            target_vectors={
                'coxa':old_m.body_pos[lookup['trochanterfemur']],
                'femur':old_m.body_pos[lookup['tibia']],
                'tibia':old_m.body_pos[lookup['tarsus1']],
            }
            transforms={bone:_rotation_between(mirror@v,target_vectors[bone])@mirror
                         * (np.linalg.norm(target_vectors[bone])/np.linalg.norm(v))
                         for bone,v in src_vectors.items()}
            target_coxa_origin=old_m.body_pos[lookup['coxa']]
            site_map={}
            all_site_names={s.attrib['site'] for tendon in tendons for s in tendon.findall('site')}
            for site_name in sorted(all_site_names):
                src_body,point=source_sites[site_name]
                if src_body=='Thorax':
                    target_body='fly/c_thorax'
                    mapped=target_coxa_origin+transforms['coxa']@(point-src_coxa_origin)
                elif src_body=='LFCoxa':
                    target_body='fly/'+leg+'_coxa';mapped=transforms['coxa']@point
                elif src_body in ['LFTrochanter','LFFemur']:
                    target_body='fly/'+leg+'_trochanterfemur'
                    combined=point+(position('LFFemur') if src_body=='LFFemur' else 0)
                    mapped=transforms['femur']@combined
                elif src_body=='LFTibia':
                    target_body='fly/'+leg+'_tibia';mapped=transforms['tibia']@point
                else:
                    raise ValueError('No declared anatomical transfer for '+src_body)
                new_name='homologous/'+leg+'/'+site_name
                spec.body(target_body).add_site(name=new_name,pos=mapped,size=[.002,.002,.002],group=4)
                site_map[site_name]=new_name
                self.transfer_sites.append(dict(leg=leg,source_site=site_name,source_body=src_body,
                                               target_body=target_body,target_local_pos_mm=mapped.tolist()))
            mask=neurons.superclass.eq('vnc_motor')&neurons.somaNeuromere.eq(neuromere)&neurons.side.eq(side)
            for mi,(tendon,node) in enumerate(zip(tendons,actuator_xml)):
                units=np.flatnonzero(mask&neurons.type.str.strip().isin(MOTOR_TYPES[mi]))
                if not len(units):
                    self.unmapped_source_templates.append(dict(leg=leg,source_actuator=node.attrib['name'],
                        motor_types=MOTOR_TYPES[mi],reason='No matching annotated motor neuron: no target muscle actuator created; absence is not silence or proof of absent biological muscle'))
                    continue
                tn='homologous/'+leg+'/'+tendon.attrib['name']
                new_tendon=spec.add_tendon(name=tn)
                for site in tendon.findall('site'):new_tendon.wrap_site(site_map[site.attrib['site']])
                tendon_names.append(tn)
                an='homologous/'+leg+'/'+node.attrib['name'];muscle_names.append(an)
                source_muscle_indices.append(mi)
                self.hill_units.append(units)
                self.hill_mapping.append(dict(leg=leg,source_actuator=node.attrib['name'],
                    actuator=an,tendon=tn,motor_types=MOTOR_TYPES[mi],motor_body_ids=neurons.bodyId.iloc[units].tolist(),
                    source_geometry='FlyMimic left front',transfer='Uncalibrated homologous bone-axis/length mapping; right side mirrored'))
        # First compile only sites/tendons to read actual neutral target lengths.
        temporary=spec.copy().compile();td=mujoco.MjData(temporary)
        key=mujoco.mj_name2id(temporary,mujoco.mjtObj.mjOBJ_KEY,'neutral')
        mujoco.mj_resetDataKeyframe(temporary,td,key);mujoco.mj_forward(temporary,td)
        for i,(tn,an) in enumerate(zip(tendon_names,muscle_names)):
            mi=source_muscle_indices[i];node=actuator_xml[mi]
            tid=mujoco.mj_name2id(temporary,mujoco.mjtObj.mjOBJ_TENDON,tn)
            src_tid=mujoco.mj_name2id(source_m,mujoco.mjtObj.mjOBJ_TENDON,tendons[mi].attrib['name'])
            length_scale=float(td.ten_length[tid]/source_d.ten_length[src_tid])
            if not np.isfinite(length_scale) or length_scale<=0:raise ValueError('Invalid transferred tendon length')
            a=spec.add_actuator(name=an,target=tn,trntype=mujoco.mjtTrn.mjTRN_TENDON,
                dyntype=mujoco.mjtDyn.mjDYN_MUSCLE,gaintype=mujoco.mjtGain.mjGAIN_MUSCLE,
                biastype=mujoco.mjtBias.mjBIAS_MUSCLE,ctrllimited=True,ctrlrange=[0,1],
                dynprm=np.fromstring(node.attrib['dynprm'],sep=' '),
                gainprm=np.fromstring(node.attrib['gainprm'],sep=' '),
                biasprm=np.fromstring(node.attrib['biasprm'],sep=' '),
                lengthrange=np.fromstring(node.attrib['lengthrange'],sep=' ')*length_scale)
            self.hill_mapping[i].update(rest_length_scale=length_scale,
                source_rest_length_mm=float(source_d.ten_length[src_tid]),target_rest_length_mm=float(td.ten_length[tid]),
                source_force_scale=1.)
        for key in spec.keys:
            key.ctrl=np.concatenate([key.ctrl,np.zeros(len(muscle_names))])
            key.act=np.zeros(len(muscle_names))
        model=spec.compile();data=mujoco.MjData(model)
        neutral=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'neutral')
        def sim_reset():mujoco.mj_resetDataKeyframe(model,data,neutral)
        self.sim=SimpleNamespace(mj_model=model,mj_data=data,timestep=model.opt.timestep,reset=sim_reset)
        assert model.nq==old_m.nq and model.nv==old_m.nv
        for adr in self.joints:
            assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(model.dof_jntid[adr]))==mujoco.mj_id2name(old_m,mujoco.mjtObj.mjOBJ_JOINT,int(old_m.dof_jntid[adr]))
        self.hill_actuators=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,an) for an in muscle_names])
        self.motor_ids=np.unique(np.concatenate(self.hill_units))
        # Adhesion actuators target bodies, not joints; never interpret their
        # target IDs as joint IDs when this explicitly experimental path is on.
        self.native_joint_dofs=np.array([int(model.jnt_dofadr[model.actuator_trnid[i,0]]) for i in range(len(self.dofs))])
        self.adhesion_actuators=np.array([
            mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,
                              'fly/'+leg+'_tarsus5-adhesion') for leg,*_ in LEGS
        ]) if self.experimental_adhesion else np.array([],dtype=int)
        assert not self.experimental_adhesion or np.all(self.adhesion_actuators>=0)
        self.native_joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(model.dof_jntid[adr])) for adr in self.native_joint_dofs]
        self.gain=1. # Source force multiplier; not the old torque bridge gain.
        self.reset()

    def reset(self):
        super().reset()
        if hasattr(self,'hill_actuators'):
            self.hill_rates=np.zeros(len(self.hill_actuators));self.hill_excitation=np.zeros(len(self.hill_actuators))
            self.full_joint_torques=self.sim.mj_data.qfrc_actuator.copy()
            self.torques=self.full_joint_torques[self.joints].copy()

    def step(self,motor_rates,milliseconds=10,enabled=True):
        m,d=self.sim.mj_model,self.sim.mj_data
        self.rates=np.array([[motor_rates[a].mean(),motor_rates[b].mean()] for a,b in self.units])
        self.hill_rates=np.array([motor_rates[u].mean() for u in self.hill_units])
        self.hill_excitation=1-np.exp(-np.maximum(self.hill_rates,0)/100) if enabled else np.zeros(len(self.hill_actuators))
        d.ctrl[:]=0;d.ctrl[self.hill_actuators]=self.hill_excitation
        d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
        for _ in range(round(milliseconds/1000/self.sim.timestep)):mujoco.mj_step(m,d)
        self.t+=milliseconds/1000
        self.full_joint_torques=d.qfrc_actuator.copy()
        self.torques=self.full_joint_torques[self.joints].copy() # Compatibility summary only.
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning.number.sum():
            raise RuntimeError('Invalid homologous Hill candidate dynamics')

    def telemetry(self):
        row=super().telemetry()
        if hasattr(self,'hill_actuators'):
            row.update(kind='experimental_homologous_hill',
                muscle_scope='15 official LF muscle templates transferred to 6 legs; all target geometry is an uncalibrated hypothesis',
                summary_torques_scope='18 historical axes only; actual muscle torques span all native leg axes',
                native_joint_names=self.native_joint_names,
                native_joint_torques=self.full_joint_torques[self.native_joint_dofs].tolist(),
                muscle_rates_hz=self.hill_rates.tolist(),muscle_excitation=self.hill_excitation.tolist(),
                muscle_forces=self.sim.mj_data.actuator_force[self.hill_actuators].tolist(),
                actual_hill_muscles=len(self.hill_actuators),missing_source_templates=len(self.unmapped_source_templates),source_force_scale=1.)
        return row

    def manifest(self):
        manifest=dict(candidate='HomologousHillBody',production=False,
            source_url='https://github.com/gizemozd/FlyMimic/tree/9ea1131626cd76f7203b74076ef8f0e9cab30bef',
            source_license='Apache-2.0',source_xml_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            assumptions=[
                'Every target muscle is a geometric transfer from the official LF source, including target LF; none is measured target geometry.',
                'Source bone longitudinal axes align with target bone axes by minimal rotation; attachment coordinates scale uniformly by bone length.',
                'Right-side attachment sites mirror source local y; source split trochanter/femur sites merge into target trochanterfemur.',
                'Thoracic attachments translate to each coxa origin and scale with that coxa; this is especially uncertain for middle/hind legs.',
                'Source force maxima, Hill parameters and activation constants copy unchanged; no behavioral score tuning.',
                'Force-length interval scales by actual neutral target/source tendon length; physiological operating lengths remain uncalibrated.',
                f'Uniform passive support stiffness is {self.stiffness:g}; antagonist coactivation and multi-axis torques emerge from native Hill/tendon geometry.',
                'Attachment transfer uses structural zero-joint body-local frames; source/target segment rest quaternions are checked identity. Thorax world pose is not part of a body-local attachment vector.',
                'MN-family mean-rate recruitment 1-exp(-rate/100) is uncalibrated; no target angles, gait phase, heading, body/root force or fallback movement.'],
            mapped_motor_neurons=int(len(self.motor_ids)),native_active_axes=len(self.native_joint_dofs),muscle_count=len(self.hill_actuators),
            passive_stiffness=self.stiffness,frame_audit=self.frame_audit,
            muscle_mapping=self.hill_mapping,unmapped_source_templates=self.unmapped_source_templates,site_transfer=self.transfer_sites)
        return manifest

    def write_manifest(self,path):
        Path(path).write_text(json.dumps(self.manifest(),indent=2))
