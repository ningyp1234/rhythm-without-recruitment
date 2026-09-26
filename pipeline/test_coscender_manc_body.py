"""Test published co-stimulation hypothesis in the source MANC and free body.

Neha Sapkal et al. 2026 report faster real walking under DNg100+DNg97 or
DNg100+DNg75 optogenetic co-activation. This *computational* counterfactual
uses the exact Pugliese row117 VNC equation and adds a matching source DNg100
current to the two annotated DN cells. The current match is a declared model
choice, not an inferred optical-current calibration. There is no brain
sensory decision or proprioceptive return in this experiment.
"""
from __future__ import annotations

import json
import time
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

from homologous_hill_body import HomologousHillBody
from probe_author_motor_pool_aggregation import pool_trace
from probe_source_manc_foot_adhesion import ltm_commands, run_condition
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import WEIGHTS, NEURONS, exact_selected, load_parameters
from source_latest_vnc_hill_interface import compatible_neurons

OUT=ROOT/"results/published-dn-costimulation-body-20260924"
BASE=ROOT/"results/brk-source-crosswalk-20260924/row117_brk_adaptive_cut_selected_rates.npz"
PARAMETER_ROW=21
GLOBAL_ROW=117
PAIR_TYPES=("DNg97","DNg75")
LTM_TYPES=("ltm MN","ltm1-tibia MN","ltm2-femur MN")


def diagnostic(neurons, selected, rates):
    lookup={int(index):i for i,index in enumerate(selected)}
    motor=np.flatnonzero(neurons["class"].eq("motor neuron").to_numpy())
    active_motor=sum(rates[lookup[i]].max()>1 for i in motor if i in lookup)
    active_tibia={}
    for segment in ("T1","T2","T3"):
        for side in ("L","R"):
            mask=neurons.somaNeuromere.eq(segment)&neurons.somaSide.str[:1].eq(side)
            entry={}
            for typ in ("Ti flexor MN","Ti extensor MN"):
                ids=np.flatnonzero(mask&neurons.type.eq(typ))
                entry[typ]=int(sum(rates[lookup[i]].max()>1 for i in ids if i in lookup))
            active_tibia[segment+side]=entry
    e1=np.flatnonzero(neurons.type.eq("IN17A001"))
    start=500 if rates.shape[1]>=1000 else 50
    e1_means={str(int(neurons.bodyId.iloc[i])):float(rates[lookup[i],start:].mean()) for i in e1 if i in lookup}
    return dict(motor_cells_active_peak_over_1hz=int(active_motor),tibia_active_counts=active_tibia,
                six_tibia_both_pools=all(v["Ti flexor MN"]>0 and v["Ti extensor MN"]>0 for v in active_tibia.values()),
                E1_mean_hz=e1_means)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-solved",action="store_true",help="Resume after report failure using already saved, identity-checked source trajectories")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(NEURONS)
    neurons=compatible_neurons(raw)
    index={int(body):i for i,body in enumerate(raw.bodyId)}
    with np.load(BASE) as saved:
        base_indices=saved["source_indices"].astype(int)
        base_rates=saved["baseline_rates_1ms"].astype(np.float64)
    assert np.array_equal(np.sort(base_indices),base_indices)
    assert base_rates.shape==(len(base_indices),2001)
    params=load_parameters()
    dng100=np.flatnonzero(raw.type.eq("DNg100"))
    source_current=float(params["input"][dng100[0],PARAMETER_ROW])
    assert len(dng100)==2 and source_current==380 and np.all(params["input"][dng100,PARAMETER_ROW]==source_current)
    weights=sparse.load_npz(WEIGHTS).T.tocsr()*.03
    body=HomologousHillBody(neurons,stiffness=32,experimental_adhesion=True)
    results=[]
    cases=[("DNg100_source_baseline",base_indices,base_rates,0.)]
    for ty in PAIR_TYPES:
        cells=np.flatnonzero(raw.type.eq(ty))
        assert len(cells)==2 and np.all(params["input"][cells,PARAMETER_ROW]==0)
        path=OUT/f"DNg100_plus_{ty}_selected_rates.npz"
        if args.reuse_solved:
            with np.load(path) as saved:
                selected=saved["source_indices"].astype(int)
                rates=saved["rates_1ms"].copy()
                assert saved["added_dn_type"].item()==ty
                assert np.array_equal(saved["added_dn_body_ids"],raw.bodyId.iloc[cells].to_numpy())
                assert np.array_equal(saved["added_current"],np.full(2,source_current))
                assert np.array_equal(saved["source_body_ids"],raw.bodyId.to_numpy()[selected])
            solver_s=None
        else:
            altered=dict(params)
            altered["input"]=params["input"].copy()
            altered["input"][cells,PARAMETER_ROW]=source_current
            full,solver_s=exact_selected(weights,altered,PARAMETER_ROW)
            selected=np.unique(np.r_[base_indices,cells])
            rates=full[selected].astype(np.float32)
            np.savez_compressed(path,source_indices=selected,source_body_ids=raw.bodyId.to_numpy()[selected],
                                rates_1ms=rates,added_dn_type=ty,
                                added_dn_body_ids=raw.bodyId.iloc[cells].to_numpy(),
                                added_current=np.array([source_current,source_current]))
            del full
        assert np.isfinite(rates).all()
        cases.append((f"DNg100_plus_{ty}",selected,rates,solver_s))
        print("source",ty,"seconds",round(solver_s,2) if solver_s is not None else "reused verified saved rates",flush=True)
    for case,selected,rates,solver_s in cases:
        pooled=pool_trace(body,selected,rates,"mean")
        neural=diagnostic(raw,selected,rates)
        for pooling in ("mean","sum"):
            grip,counts=ltm_commands(neurons,selected,rates,types=LTM_TYPES,pooling=pooling)
            for actuator in ("released","ltm_rate_gated"):
                result,trace=run_condition(body,pooled,grip,actuator)
                result.update(case=case,ltm_pooling=pooling,foot_adhesion=actuator,
                              source_neural=neural,source_solver_seconds=solver_s,
                              ltm_cells_by_leg=counts,ltm_control_peak=float(grip.max()),
                              stimulus_current_each_added_dn=source_current if case!="DNg100_source_baseline" else 0.)
                results.append(result)
                if case!="DNg100_source_baseline" and pooling=="sum" and actuator=="ltm_rate_gated":
                    np.savez_compressed(OUT/f"{case}_body_trace.npz",**trace,
                                        pooled_rates=pooled,source_grip=grip)
                print(case,pooling,actuator,round(result["forward_mm"],4),result["walking_pass"],flush=True)
    checks=dict(same_source_parameter_row_all_cases=all(r["case"] in [x[0] for x in cases] for r in results),
                all_original_MANC_cells_solved=len(raw)==23532,
                both_published_dn_pair_types_present=all(sum(r["case"]==f"DNg100_plus_{ty}" for r in results) == 4 for ty in PAIR_TYPES),
                all_six_foot_cells_modelled=len(body.adhesion_actuators)==6,
                zero_native_joint_position_gain=all(r["zero_native_joint_position_gain"] for r in results),
                all_physics_finite=all(r["physics_warnings"]==0 for r in results),
                no_external_body_force=all(r["no_applied_force"] for r in results),
                source_baseline_rates_identical_to_saved_pair=bool(np.array_equal(cases[0][2],base_rates)))
    report=dict(date="2026-09-24",source_paper="https://www.biorxiv.org/content/10.64898/2026.04.29.721658v2",
                vnc_source="https://github.com/smpuglie/Pugliese_2026",
                model_scope="Pugliese source 23,532-cell isolated MANC row117 adaptive VNC model; paired artificial current co-stimulation and 78-Hill six-leg open-loop physical replay",
                intervention="Add 380 source-current units to both DNg97 or both DNg75 MANC cells while original bilateral DNg100=380 each; all other row117 parameters unchanged",
                physiological_current_calibrated=False,reused_completed_neural_solutions=bool(args.reuse_solved),goal_complete=False,
                cases=[x[0] for x in cases],results=results,checks=checks,passed=all(checks.values()))
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"passed":report["passed"],"walking_passes":sum(r["walking_pass"] for r in results),
                      "neural":{case:diagnostic(raw,s,r) for case,s,r,_ in cases}},indent=2),flush=True)


if __name__=="__main__":main()
