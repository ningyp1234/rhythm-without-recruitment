"""Separate the old 1-ms sensory sampling lag from Euler/RK45 differences."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from cpg_context_diagnostic import rhythm
from homologous_hill_body import HomologousHillBody
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS,WEIGHTS,load_parameters
from source_latest_vnc_hill_interface import compatible_neurons
from test_coscender_manc_body import diagnostic
from test_manc_physical_feedback_costim import MancPhysicalSensors
from walking_benchmark import score

OUT=ROOT/"results/manc-adaptive-physical-feedback-20260924"
OLD=ROOT/"results/published-dn-costimulation-body-20260924/physical_feedback_FeCO_plus_load.npz"
NAMES=("legacy_after_first_microstep","aligned_before_first_microstep")


def main():
    raw=pd.read_csv(NEURONS)
    body=HomologousHillBody(compatible_neurons(raw),stiffness=32,experimental_adhesion=False)
    sensors=MancPhysicalSensors(raw,body)
    params=load_parameters()
    tau=params["tau"][:,21].astype(float);gain=params["gain"][:,21].astype(float)
    cap=params["cap"][:,21].astype(float);threshold=params["threshold"][:,21].astype(float)
    current=params["input"][:,21].astype(float).copy()
    dn=np.flatnonzero(raw.type.eq("DNg75"))
    assert len(dn)==2 and np.all(current[dn]==0)
    current[dn]=150.
    weights=sparse.load_npz(WEIGHTS).T.tocsr()*.03
    with np.load(OLD) as old:
        selected=old["selected_indices"].astype(int)
        old_selected=old["selected_rates_10ms"].copy()
        old_position=old["position"].copy()
    with np.load(OUT/"FeCO_plus_load.npz") as adaptive:
        assert np.array_equal(selected,adaptive["selected_indices"])
        adaptive_selected=adaptive["selected_rates_10ms"][:,1:].copy()
    e1=np.flatnonzero(raw.type.eq("IN17A001"))
    records=[]
    for name in NAMES:
        aligned=name==NAMES[1]
        body.reset();rates=np.zeros(len(raw),float)
        selected_trace=np.zeros((len(selected),200),np.float32)
        e1_trace=np.zeros((len(e1),2000),np.float32)
        frames=[];feet=[];contacts=[]
        for tick in range(200):
            ids,encoded,_=sensors.encode(True,True)
            if aligned:rates[ids]=encoded
            for micro in range(10):
                t=(tick*10+micro)*.001
                stimulus=current if .02<=t<=1.999 else 0.
                total=stimulus+weights@rates
                activation=np.maximum(cap*np.tanh((gain/cap)*(total-threshold)),0.)
                rates+=(activation-rates)*(.001/tau)
                rates[ids]=encoded
                e1_trace[:,tick*10+micro]=rates[e1]
            selected_trace[:,tick]=rates[selected]
            body.step(rates,10)
            state=body.telemetry();p=state["position"]
            frames.append(dict(t=(tick+1)*.01,x=p[0],y=p[1],z=p[2],
                               upright=state["upright"],nonfoot_load_fraction=state["nonfoot_load_fraction"]))
            feet.append(state["feet"]);contacts.append(state["contact"])
        result=score(frames,feet,contacts)
        e1_stats=[rhythm(x,dt=.001) for x in e1_trace]
        motor=diagnostic(raw,selected,selected_trace)
        result.update(case=name,six_E1_sustained_count=int(sum(s["diagnostic_sustained_rhythm"] for s in e1_stats)),
                      six_tibia_both_pools=bool(motor["six_tibia_both_pools"]),
                      physics_warnings=int(body.sim.mj_data.warning.number.sum()),
                      no_applied_force=bool(np.all(body.sim.mj_data.xfrc_applied==0) and np.all(body.sim.mj_data.qfrc_applied==0)),
                      selected_mae_vs_adaptive_hz=float(np.abs(selected_trace-adaptive_selected).mean()),
                      selected_mae_vs_legacy_hz=float(np.abs(selected_trace-old_selected).mean()),
                      position_max_abs_diff_vs_legacy_mm=float(np.abs(np.asarray([[f["x"],f["y"],f["z"]] for f in frames])-old_position).max()))
        records.append(result)
        np.savez_compressed(OUT/f"{name}.npz",selected_indices=selected,selected_rates_10ms=selected_trace,
                            E1_rates_1ms=e1_trace,position=np.asarray([[f["x"],f["y"],f["z"]] for f in frames]),
                            feet=np.asarray(feet),contact=np.asarray(contacts))
        print(name,"E1",result["six_E1_sustained_count"],"forward",result["forward_mm"],
              "MAE vs adaptive",result["selected_mae_vs_adaptive_hz"],flush=True)
    checks=dict(legacy_selected_bit_exact=np.array_equal(old_selected,np.load(OUT/f"{NAMES[0]}.npz")["selected_rates_10ms"]),
                legacy_body_bit_exact=np.array_equal(old_position,np.load(OUT/f"{NAMES[0]}.npz")["position"]),
                same_source_cells=len(raw)==23532,
                both_same_body_no_external_force=all(r["physics_warnings"]==0 and r["no_applied_force"] for r in records))
    report=dict(date="2026-09-24",purpose="Disentangle 1ms sensory-application timing from integration method",
                source="https://zenodo.org/records/22260924",records=records,checks=checks,
                passed=all(checks.values()),goal_complete=False)
    (OUT/"sensory_sample_order_diagnostic.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":checks},indent=2))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
