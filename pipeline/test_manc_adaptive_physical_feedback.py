"""Adaptive source-MANC/physical-body loop under a fixed artificial DN input.

Each 10 ms body step holds measured sensory rates constant while the published
23,532-cell rate equation is integrated by RK45 at <=1 ms internal steps.
The unclamped condition is compared with the saved one-shot adaptive source
solution. This is a numerical/causal validation, not autonomous brain walking.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from cpg_context_diagnostic import rhythm
from homologous_hill_body import HomologousHillBody
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS,WEIGHTS,load_parameters
from source_latest_vnc_hill_interface import compatible_neurons
from test_coscender_manc_body import diagnostic
from test_manc_physical_feedback_costim import MancPhysicalSensors
from walking_benchmark import score

OUT=ROOT/"results/manc-adaptive-physical-feedback-20260924"
REFERENCE=ROOT/"results/published-dn-costimulation-body-20260924/DNg100_plus_DNg75_150_adaptive_selected_rates.npz"
CASES=("unclamped_source","no_sensory","FeCO_plus_load","FeCO_plus_load_outgoing_cut")


def comparisons(arrays,reference_rates):
    extra={}
    if "unclamped_source" in arrays:
        observed=arrays["unclamped_source"]["selected_rates_10ms"]
        extra["unclamped_selected_MAE_vs_saved_one_shot_RK45_hz"]=float(np.abs(observed-reference_rates).mean())
        extra["unclamped_selected_max_error_vs_saved_one_shot_RK45_hz"]=float(np.abs(observed-reference_rates).max())
    if "no_sensory" in arrays and "FeCO_plus_load_outgoing_cut" in arrays:
        baseline=arrays["no_sensory"]
        cut=arrays["FeCO_plus_load_outgoing_cut"]
        extra["sensory_cut_selected_rates_mean_abs_difference_hz"]=float(np.abs(baseline["selected_rates_10ms"]-cut["selected_rates_10ms"]).mean())
        extra["sensory_cut_selected_rates_max_abs_difference_hz"]=float(np.abs(baseline["selected_rates_10ms"]-cut["selected_rates_10ms"]).max())
        extra["sensory_cut_body_position_max_abs_difference_mm"]=float(np.abs(baseline["position"]-cut["position"]).max())
        extra["sensory_cut_contacts_exact_baseline"]=bool(np.array_equal(baseline["contact"],cut["contact"]))
        if "FeCO_plus_load" in arrays:
            feedback=arrays["FeCO_plus_load"]
            effect=float(np.abs(baseline["selected_rates_10ms"]-feedback["selected_rates_10ms"]).mean())
            extra["feedback_selected_rates_mean_abs_effect_hz"]=effect
            extra["feedback_effect_to_cut_numerical_difference_ratio"]=effect/max(extra["sensory_cut_selected_rates_mean_abs_difference_hz"],1e-30)
    return extra


def integrate_case(name,raw,body,sensors,params,weights,weights_cut,selected,e1):
    tau=params["tau"][:,21].astype(float)
    gain=params["gain"][:,21].astype(float)
    cap=params["cap"][:,21].astype(float)
    threshold=params["threshold"][:,21].astype(float)
    current=params["input"][:,21].astype(float).copy()
    dn=np.flatnonzero(raw.type.eq("DNg75"))
    assert len(dn)==2 and np.all(current[dn]==0)
    current[dn]=150.
    active_weights=weights_cut if name.endswith("_outgoing_cut") else weights
    clamped=name!="unclamped_source"
    fe=clamped and name!="no_sensory"
    load=fe
    rates=np.zeros(len(raw),float)
    body.reset()
    selected_trace=np.empty((len(selected),201),np.float32)
    selected_trace[:,0]=0.
    e1_trace=np.empty((len(e1),2000),np.float32)
    sensory_trace=[];frames=[];feet=[];contacts=[]
    start=time.perf_counter()
    for tick in range(200):
        ids,encoded,sense=sensors.encode(fe,load)
        if clamped:rates[ids]=encoded
        def rhs(t,state):
            stimulus=current if .02<=t<=1.999 else 0.
            total=stimulus+active_weights@state
            activation=np.maximum(cap*np.tanh((gain/cap)*(total-threshold)),0.)
            derivative=(activation-state)/tau
            if clamped:derivative[ids]=0.
            return derivative
        a=tick/100;b=(tick+1)/100
        solution=solve_ivp(rhs,(a,b),rates,t_eval=np.linspace(a,b,11)[1:],
                           method="RK45",rtol=2e-6,atol=5e-9,max_step=.001)
        if not solution.success:raise RuntimeError(f"{name} tick {tick}: {solution.message}")
        rates=solution.y[:,-1].copy()
        if clamped:rates[ids]=encoded
        e1_trace[:,tick*10:(tick+1)*10]=solution.y[e1]
        selected_trace[:,tick+1]=rates[selected]
        body.step(rates,10)
        state=body.telemetry();position=state["position"]
        frames.append(dict(t=b,x=position[0],y=position[1],z=position[2],
                           upright=state["upright"],nonfoot_load_fraction=state["nonfoot_load_fraction"]))
        feet.append(state["feet"]);contacts.append(state["contact"])
        sensory_trace.append(np.r_[sense["feco_rates"].ravel(),sense["load_rates"],sense["moments"]])
        if tick in (49,99,149,199):print(name,"tick",tick+1,"seconds",round(time.perf_counter()-start,1),flush=True)
    summary=score(frames,feet,contacts)
    neural=diagnostic(raw,selected,selected_trace[:,1:])
    e1_stats=[rhythm(trace,dt=.001) for trace in e1_trace]
    neural["six_E1_sustained_count"]=sum(s["diagnostic_sustained_rhythm"] for s in e1_stats)
    neural["E1_frequency_hz"]=[s["frequency_hz"] for s in e1_stats]
    summary.update(case=name,neural=neural,wall_seconds=time.perf_counter()-start,
                   physics_warnings=int(body.sim.mj_data.warning.number.sum()),
                   zero_native_joint_position_gain=bool(np.all(body.sim.mj_model.actuator_gainprm[:42]==0)),
                   no_external_body_force=bool(np.all(body.sim.mj_data.xfrc_applied==0) and np.all(body.sim.mj_data.qfrc_applied==0)),
                   sensory_clamped=clamped,sensory_outgoing_edges_cut=name.endswith("_outgoing_cut"))
    trace=dict(selected_indices=selected,selected_body_ids=raw.bodyId.iloc[selected].to_numpy(),
               selected_rates_10ms=selected_trace,E1_rates_1ms=e1_trace,
               sensory_10ms=np.asarray(sensory_trace),
               position=np.array([[f["x"],f["y"],f["z"]] for f in frames]),
               feet=np.asarray(feet),contact=np.asarray(contacts))
    np.savez_compressed(OUT/f"{name}.npz",**trace)
    print(name,"forward_mm",round(summary["forward_mm"],6),"E1",neural["six_E1_sustained_count"],
          "tibia",neural["six_tibia_both_pools"],"walking",summary["walking_pass"],flush=True)
    return summary,trace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cases",default=",".join(CASES),help="Comma-separated subset; rerun with all cases for complete report")
    parser.add_argument("--reuse-saved",action="store_true",help="Recompute comparisons after report-only correction; never reuse to represent fresh integration")
    args=parser.parse_args()
    requested=args.cases.split(",")
    assert requested and len(set(requested))==len(requested) and set(requested)<=set(CASES)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.reuse_saved:
        report=json.loads((OUT/"report.json").read_text())
        assert report["cases"]==requested and all((OUT/f"{name}.npz").is_file() for name in requested)
        arrays={}
        for name in requested:
            with np.load(OUT/f"{name}.npz") as saved:
                arrays[name]={key:saved[key].copy() for key in saved.files}
        with np.load(REFERENCE) as reference:
            reference_rates=reference["rates_1ms"][:,::10].astype(np.float64)
        report["numerical_and_causal_comparisons"]=comparisons(arrays,reference_rates)
        report["reused_saved_traces_for_report_correction"]=True
        (OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps(report["numerical_and_causal_comparisons"],indent=2));return
    raw=pd.read_csv(NEURONS)
    neurons=compatible_neurons(raw)
    body=HomologousHillBody(neurons,stiffness=32,experimental_adhesion=False)
    sensors=MancPhysicalSensors(raw,body)
    with np.load(REFERENCE) as reference:
        selected=reference["source_indices"].astype(int)
        reference_rates=reference["rates_1ms"][:,::10].astype(np.float64)
        assert np.array_equal(reference["source_body_ids"],raw.bodyId.iloc[selected].to_numpy())
    e1=np.flatnonzero(raw.type.eq("IN17A001"))
    assert len(e1)==6 and np.all(np.isin(e1,selected))
    params=load_parameters()
    weights=sparse.load_npz(WEIGHTS).T.tocsr()*.03
    weights_cut=weights.copy()
    cut_mask=np.isin(weights_cut.indices,sensors.all)
    assert np.any(cut_mask)
    weights_cut.data[cut_mask]=0.;weights_cut.eliminate_zeros()
    records=[];arrays={}
    for name in requested:
        row,trace=integrate_case(name,raw,body,sensors,params,weights,weights_cut,selected,e1)
        records.append(row);arrays[name]=trace
    extra=comparisons(arrays,reference_rates)
    checks=dict(source_neurons_23532=len(raw)==23532,
                same_DNg100_source_current=bool(np.array_equal(params["input"][np.flatnonzero(raw.type.eq("DNg100")),21],[380,380])),
                all_source_body_ids_saved=all(np.array_equal(arrays[name]["selected_body_ids"],raw.bodyId.iloc[selected].to_numpy()) for name in requested),
                all_physics_finite=all(row["physics_warnings"]==0 for row in records),
                all_native_position_gains_zero=all(row["zero_native_joint_position_gain"] for row in records),
                no_external_body_force=all(row["no_external_body_force"] for row in records))
    report=dict(date="2026-09-24",source="https://zenodo.org/records/22260924",
                method="Published full MANC equation, RK45 per 10ms physical feedback interval with max 1ms internal step, 78 Hill muscle six-leg body",
                input="Artificial bilateral DNg100 current 380 plus DNg75 current 150, source parameter row21/global row117",
                calibrated_physiological_sensory_and_motor_interface=False,
                cases=requested,mapped_sensory_outgoing_connections_cut=int(np.count_nonzero(cut_mask)),
                results=records,numerical_and_causal_comparisons=extra,
                checks=checks,passed=all(checks.values()),goal_complete=False)
    (OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"comparisons":extra,"walking_passes":sum(r["walking_pass"] for r in records)},indent=2),flush=True)
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
