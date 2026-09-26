"""A2 diagnostic: latest author MANC VNC dynamics drive the 78-muscle body.

No trajectory, gait clock, target angle, joint servo, root force or fallback is
present.  Constant author-specified DNg100/DNb08 current is still an artificial
diagnostic command, so this experiment cannot establish A3 autonomy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cpg_rate import RateNetwork
from homologous_hill_body import HomologousHillBody
from source_calibration import ROOT, sha
from walking_benchmark import score
from walking_feedback import WalkingFeedback


OUT = ROOT / "results/source-latest-vnc-hill-interface-20260922"
DATA_N = ROOT / "data/cpg/manc-neurons-20260522.csv.gz"
DATA_W = ROOT / "data/cpg/manc-pre-post-all-synapses-20260522.npz"
DATA_MANIFEST = ROOT / "data/cpg/manifest-20260522.json"
CORE_TYPES = ["DNg100", "DNb08", "IN17A001", "INXXX466", "IN16B036"]
CONDITIONS = {
    # Exact single-cell identities and currents from the author experiment configs.
    "DNg100_all_proprio": dict(stim_body_ids=[10093], current=[250.], feedback=True,
                               velocity_gain=1., position_gain=1., cut=False),
    "DNg100_position_only": dict(stim_body_ids=[10093], current=[250.], feedback=True,
                                 velocity_gain=0., position_gain=1., cut=False),
    "DNg100_no_proprio": dict(stim_body_ids=[10093], current=[250.], feedback=False,
                              velocity_gain=0., position_gain=0., cut=False),
    "DNg100_synapse_cut": dict(stim_body_ids=[10093], current=[250.], feedback=True,
                               velocity_gain=0., position_gain=1., cut=True),
    "DNb08_position_only": dict(stim_body_ids=[14061], current=[65.], feedback=True,
                                velocity_gain=0., position_gain=1., cut=False),
}


def compatible_neurons(raw):
    n = raw.copy().fillna("")
    n["superclass"] = n["class"].replace({"motor neuron": "vnc_motor"})
    n["side"] = n.somaSide.str[:1]
    return n


def annotations(n):
    a = n.copy()
    a["entryNerve"] = a.instance.str.extract(r"_([^_]+)_[LR]")[0].fillna("")
    a["rootSide"] = a.rootSide.str[:1]
    return a


def run_condition(name, spec, n, w, seed=1, seconds=2.0):
    lookup = {int(v): i for i, v in enumerate(n.bodyId)}
    stimulus = np.array([lookup[x] for x in spec["stim_body_ids"]], dtype=np.int64)
    body = HomologousHillBody(n, stiffness=32.0)
    feedback = WalkingFeedback(n, annotations(n))
    network = RateNetwork(w, n["size"], seed=seed, dt=.001)
    motor = body.motor_ids
    core = np.flatnonzero(n.type.isin(CORE_TYPES))
    recorded = np.unique(np.r_[motor, core])
    rows=[]; feet=[]; contacts=[]; motor_rates=[]; recorded_rates=[]
    feedback_rates=[]; excitation=[]; activation=[]; muscle_force=[]; positions=[]
    started=time.perf_counter()
    for tick in range(round(seconds * 100)):
        sid, srates = feedback.encode(
            *body.kinematics(), enabled=spec["feedback"],
            velocity_gain=spec["velocity_gain"],
            position_gain=spec["position_gain"],
        )
        ids = np.r_[stimulus, sid]
        values = np.r_[np.asarray(spec["current"], float), srates]
        network.advance(ids, values, 10, cut=spec["cut"])
        body.step(network.rates, 10, enabled=True)
        tel=body.telemetry(); pos=np.asarray(tel["position"])
        rows.append(dict(t=(tick+1)/100, x=pos[0], y=pos[1], z=pos[2],
                         upright=tel["upright"],
                         nonfoot_load_fraction=tel["nonfoot_load_fraction"],
                         mapped_mn_mean_hz=float(network.rates[motor].mean()),
                         mapped_mn_max_hz=float(network.rates[motor].max()),
                         core_mean_hz=float(network.rates[core].mean())))
        feet.append(tel["feet"]);contacts.append(tel["contact"])
        motor_rates.append(network.rates[motor].copy())
        recorded_rates.append(network.rates[recorded].copy())
        feedback_rates.append(srates.copy());excitation.append(body.hill_excitation.copy())
        activation.append(body.sim.mj_data.act.copy())
        muscle_force.append(body.sim.mj_data.actuator_force[body.hill_actuators].copy())
        positions.append(pos)
    result=score(rows,feet,contacts)
    result.update(run_id=name, mode="A2", seed=seed, model_seconds=seconds,
                  wall_seconds=time.perf_counter()-started, neurons=len(n),
                  directed_pairs=int(w.nnz), signed_synapses=int(np.abs(w.data).sum()),
                  mapped_motor_neurons=len(motor), hill_muscles=len(body.hill_actuators),
                  missing_source_templates=len(body.unmapped_source_templates),
                  sensory_cells=len(feedback.ids), artificial_dn_stimulation=True,
                  stimulus_body_ids=spec["stim_body_ids"], stimulus_model_current=spec["current"],
                  synapse_cut=spec["cut"], feedback_enabled=spec["feedback"],
                  velocity_feedback_gain=spec["velocity_gain"],
                  position_feedback_gain=spec["position_gain"],
                  native_position_authority_zero=bool(np.all(body.sim.mj_model.actuator_gainprm[:len(body.native_joint_dofs)]==0)
                                                     and np.all(body.sim.mj_model.actuator_biasprm[:len(body.native_joint_dofs)]==0)),
                  max_abs_applied_generalized_force=float(np.max(np.abs(body.sim.mj_data.qfrc_applied))),
                  max_abs_external_body_force=float(np.max(np.abs(body.sim.mj_data.xfrc_applied))),
                  warnings=int(body.sim.mj_data.warning.number.sum()))
    arrays=dict(feet=np.asarray(feet),contact=np.asarray(contacts),
                motor_rates=np.asarray(motor_rates),motor_indices=motor,
                recorded_rates=np.asarray(recorded_rates),recorded_indices=recorded,
                feedback_rates=np.asarray(feedback_rates),feedback_indices=feedback.ids,
                muscle_excitation=np.asarray(excitation),muscle_activation=np.asarray(activation),
                muscle_force=np.asarray(muscle_force),position=np.asarray(positions))
    return result, pd.DataFrame(rows), arrays, body, feedback


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n=compatible_neurons(pd.read_csv(DATA_N))
    w=sparse.load_npz(DATA_W).tocsr()
    if w.shape != (len(n),len(n)):
        raise ValueError("Latest MANC neuron/matrix dimensions differ")
    results={}; all_rows=[]; body_manifest=None; feedback_mapping=None
    for name,spec in CONDITIONS.items():
        result,rows,arrays,body,feedback=run_condition(name,spec,n,w)
        results[name]=result;all_rows.append(rows.assign(run_id=name))
        np.savez_compressed(OUT/f"{name}.npz",**arrays)
        (OUT/f"{name}.json").write_text(json.dumps(result,indent=2)+"\n")
        if body_manifest is None:body_manifest=body.manifest();feedback_mapping=feedback.mapping
        print(name,json.dumps({k:result[k] for k in (
            "forward_mm","foot_only_support_fraction","steps_per_leg",
            "median_stance_slip_mm_s","walking_pass","wall_seconds")}),flush=True)
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/"traces.csv",index=False)
    (OUT/"motor-muscle-manifest.json").write_text(json.dumps(body_manifest,indent=2)+"\n")
    (OUT/"sensory-mapping.json").write_text(json.dumps(feedback_mapping,indent=2)+"\n")
    causal = results["DNg100_position_only"]["forward_mm"] - results["DNg100_synapse_cut"]["forward_mm"]
    report={
        "scope": "A2 latest MANC VNC rate dynamics to free six-leg 78-Hill-muscle body",
        "sources": {p.name:{"path":str(p),"sha256":sha(p)} for p in (DATA_N,DATA_W,DATA_MANIFEST)},
        "author_model": {"repository":"https://github.com/smpuglie/Pugliese_2026",
                         "commit":"10e7661bf414ba7b4c2edf795cd36d0f878c17c0",
                         "dt_seconds":.001,"parameter_seed":1},
        "controller_authority": {"artificial_DN_current":True,"gait_phase_clock":False,
            "future_reference":False,"joint_position_servo":False,"root_force":0,
            "external_body_force":0,"adhesion":False,
            "physical_output":"actual MANC motor rates -> 78 transferred Hill muscles"},
        "proprioception_bracket": {
            "all": "SNpp39/41 velocity and SNpp50/51 position channels",
            "position_only": "velocity channels set to zero; position channels retained",
            "interpretation": "causal bracket motivated by reported movement-channel suppression; zero is not claimed as the measured gain",
        },
        "conditions":results,"causal_forward_delta_over_synapse_cut_mm":causal,
        "classification": {"A2_full_six_leg_neural_motor_interface":
            "pass" if any(r["walking_pass"] for k,r in results.items() if not r["synapse_cut"]) else "incomplete",
            "A3_neural_closed_loop":"not run; DN command is artificial"},
        "limitations":[
            "MANC and the transferred FlyMimic body are different specimens.",
            "Muscles remain uncalibrated left-front homologous transfers with 12 unresolved templates.",
            "Model currents and proprioceptive gains are model units, not measured firing-rate calibrations.",
        ],"goal_complete":False}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"classification":report["classification"],"causal_forward_delta_mm":causal,
                      "goal_complete":False},ensure_ascii=False))


if __name__=="__main__":main()
