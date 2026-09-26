"""Physical six-leg load/FeCO feedback into source MANC under fixed DN co-stim.

This is a paired *experimental* neural-body loop: real MANC cell identities
and connectome, Pugliese equation/parameters, physical foot forces/joints,
uncalibrated 80 Hz campaniform and 30 Hz/(rad/s) FeCO transduction, 78 Hill
muscles. Artificial DNg100+DNg75 input remains; no gait phase, target angle,
external root force, cloned cells, or scripted movement is supplied.
"""
from __future__ import annotations

import json
import time

import mujoco
import numpy as np
import pandas as pd
from scipy import sparse

from cpg_context_diagnostic import rhythm
from homologous_hill_body import HomologousHillBody
from probe_author_motor_pool_aggregation import original_rates
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS,WEIGHTS,load_parameters
from source_latest_vnc_hill_interface import compatible_neurons
from test_coscender_manc_body import diagnostic
from walking_benchmark import score
from walking_body import LEGS

OUT=ROOT/"results/published-dn-costimulation-body-20260924"
CASES=(("no_sensory",False,False),("FeCO_only",True,False),
       ("load_only",False,True),("FeCO_plus_load",True,True),
       ("FeCO_plus_load_outgoing_cut",True,True))


class MancPhysicalSensors:
    def __init__(self,neurons,body):
        self.body=body
        self.feco=[];self.load=[];self.mapping=[]
        for leg,_,side,nerve in LEGS:
            key={}
            for ty in ("SNpp39","SNpp41","SNpp53"):
                expected=f"{ty}_{nerve}_{side}"
                ids=np.flatnonzero((neurons.type==ty)&(neurons.instance==expected)&
                                   (neurons["class"]=="sensory neuron"))
                key[ty]=ids.astype(int)
                self.mapping.append(dict(leg=leg,type=ty,expected_instance=expected,
                                         source_body_ids=neurons.bodyId.iloc[ids].astype(int).tolist()))
            assert len(key["SNpp39"])>0 and len(key["SNpp41"])>0
            self.feco.append((key["SNpp39"],key["SNpp41"]))
            self.load.append(key["SNpp53"])
        self.all=np.unique(np.concatenate([ids for pair in self.feco for ids in pair]+self.load))
        self.unresolved_snpp53=neurons.loc[(neurons.type=="SNpp53")&
            ~neurons.index.isin(np.concatenate(self.load)),["bodyId","instance"]].to_dict("records")
        m,d=body.sim.mj_model,body.sim.mj_data
        self.jids=np.array([np.flatnonzero(m.jnt_dofadr==body.joints[3*i+1])[0] for i in range(6)])
        geom_leg=np.full(m.ngeom,-1)
        for i in range(m.ngeom):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) or ''
            for j,(leg,*_) in enumerate(LEGS):
                if name.startswith('fly/'+leg+'_') and 'coxa' not in name:geom_leg[i]=j
        self.geom_leg=geom_leg
        weight=m.body_mass.sum()*np.linalg.norm(m.opt.gravity)
        lever=np.linalg.norm(d.xpos[body.tip_ids]-d.xanchor[self.jids],axis=1)
        self.reference_moment=weight*np.maximum(lever,.1)/3

    def _ground_moments(self):
        m,d=self.body.sim.mj_model,self.body.sim.mj_data
        moments=np.zeros(6)
        for k in range(d.ncon):
            c=d.contact[k];a,b=map(int,c.geom)
            la,lb=self.geom_leg[a],self.geom_leg[b]
            if la>=0 and m.geom_bodyid[b]==0:leg=la;sign=-1.
            elif lb>=0 and m.geom_bodyid[a]==0:leg=lb;sign=1.
            else:continue
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,k,wrench)
            rotation=c.frame.reshape(3,3).T
            force=sign*rotation@wrench[:3]
            torque=sign*rotation@wrench[3:]
            jid=self.jids[leg]
            moments[leg]+=np.dot(np.cross(c.pos-d.xanchor[jid],force)+torque,d.xaxis[jid])
        return moments

    def encode(self,fe_enabled,load_enabled):
        angles,velocities=self.body.kinematics()
        feco_rates=np.array([np.minimum(150,30*np.maximum(-velocities,0)),
                             np.minimum(150,30*np.maximum(velocities,0))]).T if fe_enabled else np.zeros((6,2))
        moments=self._ground_moments()
        load_rates=80*np.tanh(np.abs(moments)/self.reference_moment) if load_enabled else np.zeros(6)
        indices=[];values=[]
        for leg in range(6):
            for kind in range(2):
                ids=self.feco[leg][kind]
                indices.extend(ids);values.extend([feco_rates[leg,kind]]*len(ids))
            ids=self.load[leg]
            indices.extend(ids);values.extend([load_rates[leg]]*len(ids))
        return np.asarray(indices,dtype=int),np.asarray(values,dtype=float),dict(
            feco_rates=feco_rates,load_rates=load_rates,moments=moments,angles=angles,velocities=velocities)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(NEURONS)
    neurons=compatible_neurons(raw)
    body=HomologousHillBody(neurons,stiffness=32,experimental_adhesion=False)
    sensors=MancPhysicalSensors(raw,body)
    assert len(sensors.all)>0 and len(sensors.unresolved_snpp53)==1
    params=load_parameters()
    tau=params["tau"][:,21].astype(float);gain=params["gain"][:,21].astype(float)
    cap=params["cap"][:,21].astype(float);threshold=params["threshold"][:,21].astype(float)
    current=params["input"][:,21].astype(float).copy()
    dn=np.flatnonzero(raw.type.eq("DNg75"))
    assert len(dn)==2 and np.all(current[dn]==0)
    current[dn]=150.
    weights=sparse.load_npz(WEIGHTS).T.tocsr()*.03
    weights_cut=weights.copy()
    cut_edges=np.isin(weights_cut.indices,sensors.all)
    assert np.count_nonzero(cut_edges)>0
    weights_cut.data[cut_edges]=0.
    weights_cut.eliminate_zeros()
    e1=np.flatnonzero(raw.type.eq("IN17A001"))
    author_selected,_=original_rates(117)
    selected=np.unique(np.r_[author_selected,e1])
    records=[];trace_by_case={}
    for name,fe_enabled,load_enabled in CASES:
        sensory_edges_cut=name.endswith("_outgoing_cut")
        active_weights=weights_cut if sensory_edges_cut else weights
        body.reset()
        rates=np.zeros(len(raw),dtype=float)
        neuron_trace=np.zeros((len(selected),200),np.float32)
        e1_trace=np.zeros((6,2000),np.float32)
        frames=[];feet=[];contacts=[];sensory=[]
        start=time.perf_counter()
        for tick in range(200):
            ids,encoded,sense=sensors.encode(fe_enabled,load_enabled)
            assert len(ids)==len(sensors.all) and len(np.unique(ids))==len(ids)
            for micro in range(10):
                t=(tick*10+micro)*.001
                total=(current if .02<=t<=1.999 else 0.)+active_weights@rates
                activation=np.maximum(cap*np.tanh((gain/cap)*(total-threshold)),0.)
                rates+=(activation-rates)*(.001/tau)
                rates[ids]=encoded
                e1_trace[:,tick*10+micro]=rates[e1]
            neuron_trace[:,tick]=rates[selected]
            body.step(rates,10)
            state=body.telemetry();p=state["position"]
            frames.append(dict(t=(tick+1)*.01,x=p[0],y=p[1],z=p[2],
                               upright=state["upright"],nonfoot_load_fraction=state["nonfoot_load_fraction"]))
            feet.append(state["feet"]);contacts.append(state["contact"])
            sensory.append(np.r_[sense["feco_rates"].ravel(),sense["load_rates"],sense["moments"]])
        result=score(frames,feet,contacts)
        lookup={int(index):i for i,index in enumerate(selected)}
        neural=diagnostic(raw,selected,neuron_trace)
        e1_stats=[rhythm(x,dt=.001) for x in e1_trace]
        neural["six_E1_sustained_count"]=sum(x["diagnostic_sustained_rhythm"] for x in e1_stats)
        neural["E1_frequency_hz"]=[x["frequency_hz"] for x in e1_stats]
        result.update(case=name,fe_enabled=fe_enabled,load_enabled=load_enabled,
                      sensory_outgoing_edges_cut=sensory_edges_cut,
                      neural=neural,physics_warnings=int(body.sim.mj_data.warning.number.sum()),
                      zero_native_joint_position_gain=bool(np.all(body.sim.mj_model.actuator_gainprm[:42]==0)),
                      no_external_body_force=bool(np.all(body.sim.mj_data.xfrc_applied==0) and
                                                  np.all(body.sim.mj_data.qfrc_applied==0)),
                      wall_seconds=time.perf_counter()-start)
        np.savez_compressed(OUT/f"physical_feedback_{name}.npz",selected_indices=selected,
                            selected_body_ids=raw.bodyId.iloc[selected].to_numpy(),
                            selected_rates_10ms=neuron_trace,E1_rates_1ms=e1_trace,
                            sensory_10ms=np.asarray(sensory),position=np.array([[f["x"],f["y"],f["z"]] for f in frames]),
                            feet=np.asarray(feet),contact=np.asarray(contacts))
        trace_by_case[name]=neuron_trace
        records.append(result)
        print(name,"forward",round(result["forward_mm"],4),"E1",neural["six_E1_sustained_count"],
              "tibia",neural["six_tibia_both_pools"],"walking",result["walking_pass"],flush=True)
    baseline=trace_by_case["no_sensory"]
    for record in records:
        record["selected_MN_E1_mean_abs_change_from_no_sensory_hz"]=float(np.abs(trace_by_case[record["case"]]-baseline).mean())
    baseline_record=next(r for r in records if r["case"]=="no_sensory")
    cut_record=next(r for r in records if r["case"]=="FeCO_plus_load_outgoing_cut")
    checks=dict(five_paired_cases=len(records)==5,
                exact_MANC_23532_cells=len(raw)==23532,
                mapped_sensory_outgoing_edges_removed=int(np.count_nonzero(cut_edges))>0,
                sensory_cut_recovers_baseline_neural_trace=bool(np.array_equal(trace_by_case[cut_record["case"]],baseline)),
                sensory_cut_recovers_baseline_body=abs(cut_record["forward_mm"]-baseline_record["forward_mm"])<1e-10,
                six_leg_FeCO_annotation_groups_complete=all(len(pair[0])>0 and len(pair[1])>0 for pair in sensors.feco),
                load_groups_only_named_source_cells=sum(map(len,sensors.load))==8,
                unknown_front_source_SNpp53_not_fabricated=len(sensors.unresolved_snpp53)==1,
                all_physics_finite=all(r["physics_warnings"]==0 for r in records),
                all_native_position_gains_zero=all(r["zero_native_joint_position_gain"] for r in records),
                no_external_body_force=all(r["no_external_body_force"] for r in records))
    report=dict(date="2026-09-24",hypothesis_source="https://www.biorxiv.org/content/10.64898/2026.04.29.721658v2",
                method="1ms source MANC Euler step with 10ms physical feedback, original graph and row117 params, bilateral artificial DNg100 380 + DNg75 150",
                sensory_boundary="actual MANC SNpp39/41 and 8 certain SNpp53 source IDs clamped to uncalibrated physical encoding; unidentified ninth SNpp53 excluded",
                uncertainty="No real rate-to-force or contact-moment-to-spike calibration; Euler differs from adaptive source solver; artificial DNs remain, this is not autonomous brain behaviour.",
                mapping=sensors.mapping,unresolved_snpp53=sensors.unresolved_snpp53,
                mapped_sensory_outgoing_edges_removed=int(np.count_nonzero(cut_edges)),
                results=records,checks=checks,passed=all(checks.values()),goal_complete=False)
    (OUT/"physical_feedback_report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"passed":report["passed"],"walking_passes":sum(r["walking_pass"] for r in records),
                      "feedback_rate_changes_hz":{r["case"]:r["selected_MN_E1_mean_abs_change_from_no_sensory_hz"] for r in records}},indent=2))


if __name__=="__main__":main()
