"""Replay saved physical afferents to locate adaptive-source tibia MN margins.

No parameter is tuned: only the two already-run sensory conditions are replayed.
The full MANC equation is rerun at the same 10 ms sensory hold / RK45 settings.
Its selected-neuron rates must reproduce the saved embodied loop before the
subthreshold input margins are interpreted.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from homologous_hill_body import HomologousHillBody
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS,WEIGHTS,load_parameters
from source_latest_vnc_hill_interface import compatible_neurons
from test_manc_physical_feedback_costim import MancPhysicalSensors

OUT=ROOT/"results/manc-adaptive-physical-feedback-20260924"
CASES=("no_sensory","FeCO_plus_load")
MODULES=("tibia flex","tibia extend")


def sensory_values(sensors,encoded_row):
    feco=encoded_row[:12].reshape(6,2)
    load=encoded_row[12:18]
    ids=[];values=[]
    for leg in range(6):
        for kind in range(2):
            members=sensors.feco[leg][kind]
            ids.extend(members);values.extend([feco[leg,kind]]*len(members))
        members=sensors.load[leg]
        ids.extend(members);values.extend([load[leg]]*len(members))
    return np.asarray(ids,int),np.asarray(values,float)


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
    motor=np.flatnonzero(raw["motor module"].isin(MODULES))
    assert len(motor)==86 and weights.shape==(len(raw),len(raw))
    motor_weights=weights[motor]
    records=[];saved_margins={}
    for name in CASES:
        with np.load(OUT/f"{name}.npz") as source:
            sensory=source["sensory_10ms"].copy()
            selected=source["selected_indices"].astype(int)
            expected=source["selected_rates_10ms"].copy()
        rates=np.zeros(len(raw),float)
        selected_trace=np.empty_like(expected);selected_trace[:,0]=0.
        margin=np.empty((len(motor),2000),np.float32)
        motor_rate=np.empty_like(margin)
        start=time.perf_counter()
        for tick in range(200):
            ids,encoded=sensory_values(sensors,sensory[tick])
            assert np.array_equal(np.unique(ids),sensors.all)
            rates[ids]=encoded
            def rhs(t,state):
                stimulus=current if .02<=t<=1.999 else 0.
                total=stimulus+weights@state
                activation=np.maximum(cap*np.tanh((gain/cap)*(total-threshold)),0.)
                derivative=(activation-state)/tau
                derivative[ids]=0.
                return derivative
            a=tick/100;b=(tick+1)/100
            times=np.linspace(a,b,11)[1:]
            result=solve_ivp(rhs,(a,b),rates,t_eval=times,method="RK45",
                             rtol=2e-6,atol=5e-9,max_step=.001)
            if not result.success:raise RuntimeError(result.message)
            rates=result.y[:,-1].copy();rates[ids]=encoded
            selected_trace[:,tick+1]=rates[selected]
            presynaptic=motor_weights@result.y
            local_stimulus=np.where((times>=.02)&(times<=1.999),current[motor,None],0.)
            margin[:,tick*10:(tick+1)*10]=(local_stimulus+presynaptic-threshold[motor,None]).astype(np.float32)
            motor_rate[:,tick*10:(tick+1)*10]=result.y[motor].astype(np.float32)
            if tick in (99,199):print(name,"tick",tick+1,"seconds",round(time.perf_counter()-start,1),flush=True)
        residual=float(np.abs(selected_trace-expected).mean())
        assert residual<1e-6,(name,residual)
        saved_margins[name]=margin
        rows=[]
        for leg in ("T1L","T1R","T2L","T2R","T3L","T3R"):
            segment,side=leg[:2],leg[2]
            for module in MODULES:
                mask=(raw.somaNeuromere.iloc[motor].eq(segment)&
                      raw.somaSide.iloc[motor].str[:1].eq(side)&
                      raw["motor module"].iloc[motor].eq(module)).to_numpy()
                local_margin=margin[mask,500:1999]
                local_rate=motor_rate[mask,500:1999]
                assert mask.any()
                rows.append(dict(leg=leg,motor_module=module,motor_body_ids=raw.bodyId.iloc[motor[mask]].astype(int).tolist(),
                                 source_types=raw.type.iloc[motor[mask]].tolist(),
                                 cell_count=int(mask.sum()),peak_margin=float(local_margin.max()),
                                 mean_margin=float(local_margin.mean()),positive_margin_fraction=float((local_margin>0).mean()),
                                 peak_rate_hz=float(local_rate.max()),active_cells_over_1hz=int((local_rate.max(axis=1)>1).sum())))
        records.append(dict(case=name,selected_replay_mae_hz=residual,
                            source_neurons=len(raw),tibia_motor_cells=len(motor),
                            rows=rows,wall_seconds=time.perf_counter()-start))
        np.savez_compressed(OUT/f"{name}_tibia_margins.npz",motor_source_indices=motor,
                            motor_body_ids=raw.bodyId.iloc[motor].to_numpy(),margin_1ms=margin,
                            motor_rates_1ms=motor_rate,selected_replayed_10ms=selected_trace)
    baseline,feedback=saved_margins.values()
    checks=dict(two_replayed_embodied_conditions=len(records)==2,
                both_selected_traces_reproduce=all(r["selected_replay_mae_hz"]<1e-6 for r in records),
                all_86_annotated_tibia_MNs=len(motor)==86,
                all_twelve_leg_pool_rows_each=all(len(r["rows"])==12 for r in records),
                all_finite=bool(np.isfinite(baseline).all() and np.isfinite(feedback).all()))
    report=dict(date="2026-09-24",source="https://zenodo.org/records/22260924",
                scope="Exact source-model pre-threshold net input for all 86 motor-module tibia flex/extend MNs, including 48 accessory tibia flexor cells, replaying recorded physical afferents; artificial DNg100+DNg75 remains",
                definition="margin = source current + signed 0.03-scaled presynaptic input - source cell threshold; positive margin is not automatically a sustained spike or a physical step",
                records=records,mean_abs_margin_change_from_feedback=float(np.abs(feedback-baseline).mean()),
                checks=checks,passed=all(checks.values()),goal_complete=False)
    (OUT/"adaptive_tibia_threshold_margins.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"mean_abs_margin_change":report["mean_abs_margin_change_from_feedback"],
                      "feedback_active_pools":[(r["leg"],r["motor_module"],r["active_cells_over_1hz"]) for r in records[1]["rows"] if r["active_cells_over_1hz"]]},indent=2),flush=True)
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
