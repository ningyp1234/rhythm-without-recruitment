"""Independently recompute the saved adaptive source/physical-loop controls."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS
from test_manc_adaptive_physical_feedback import OUT,REFERENCE,CASES


def main():
    report=json.loads((OUT/"report.json").read_text())
    timing=json.loads((OUT/"sensory_sample_order_diagnostic.json").read_text())
    n=pd.read_csv(NEURONS)
    saved={}
    for name in CASES:
        with np.load(OUT/f"{name}.npz") as archive:
            saved[name]={key:archive[key].copy() for key in archive.files}
    with np.load(REFERENCE) as archive:
        source_indices=archive["source_indices"]
        reference_rates=archive["rates_1ms"][:,::10]
    source=saved["unclamped_source"]
    mae=float(np.abs(source["selected_rates_10ms"]-reference_rates).mean())
    model_checks=[];forward_checks=[]
    for record in report["results"]:
        trace=saved[record["case"]]
        p=trace["position"]
        model_checks.append(np.array_equal(trace["selected_indices"],source_indices)
                            and np.array_equal(trace["selected_body_ids"],n.bodyId.iloc[source_indices].to_numpy())
                            and trace["selected_rates_10ms"].shape==(len(source_indices),201)
                            and np.isfinite(trace["selected_rates_10ms"]).all()
                            and trace["sensory_10ms"].shape==(200,24)
                            and np.isfinite(trace["position"]).all())
        forward_checks.append(abs(float(p[-1,0]-p[0,0])-record["forward_mm"])<1e-12)
    baseline=saved["no_sensory"]
    cut=saved["FeCO_plus_load_outgoing_cut"]
    feedback=saved["FeCO_plus_load"]
    with np.load(OUT/"aligned_before_first_microstep.npz") as aligned:
        aligned_rates=aligned["selected_rates_10ms"].copy()
    with np.load(ROOT/"results/published-dn-costimulation-body-20260924/physical_feedback_FeCO_plus_load.npz") as legacy:
        legacy_rates=legacy["selected_rates_10ms"].copy()
    cut_mae=float(np.abs(cut["selected_rates_10ms"]-baseline["selected_rates_10ms"]).mean())
    feedback_mae=float(np.abs(feedback["selected_rates_10ms"]-baseline["selected_rates_10ms"]).mean())
    cut_position_max=float(np.abs(cut["position"]-baseline["position"]).max())
    comparisons=report["numerical_and_causal_comparisons"]
    checks=dict(all_four_cases=report["cases"]==list(CASES),
                official_source_neuron_identity=len(n)==23532 and all(model_checks),
                chunked_adaptive_matches_one_shot_rk45=mae<.01 and abs(mae-report["numerical_and_causal_comparisons"]["unclamped_selected_MAE_vs_saved_one_shot_RK45_hz"])<1e-9,
                body_forward_recomputed_from_saved_positions=all(forward_checks),
                sensory_cut_nearly_recovers_no_sensory_with_solver_roundoff=cut_mae<.001 and cut_position_max<.0001
                and np.array_equal(cut["contact"],baseline["contact"])
                and abs(cut_mae-comparisons["sensory_cut_selected_rates_mean_abs_difference_hz"])<1e-9,
                sensory_signal_really_present=np.max(feedback["sensory_10ms"][:,:18])>0 and np.max(cut["sensory_10ms"][:,:18])>0,
                physical_feedback_changes_source_network_above_cut_roundoff=feedback_mae>100*cut_mae
                and abs(feedback_mae-comparisons["feedback_selected_rates_mean_abs_effect_hz"])<1e-8,
                sample_order_control_preserves_euler_five_E1=timing["passed"]
                and np.array_equal(np.load(OUT/"legacy_after_first_microstep.npz")["selected_rates_10ms"],legacy_rates)
                and [r["six_E1_sustained_count"] for r in timing["records"]]==[5,5]
                and abs(float(np.abs(aligned_rates-feedback["selected_rates_10ms"][:,1:]).mean())
                        -timing["records"][1]["selected_mae_vs_adaptive_hz"])<1e-8,
                walking_not_claimed=all(not row["walking_pass"] for row in report["results"]) and report["goal_complete"] is False)
    validation=dict(date="2026-09-24",checks={k:bool(v) for k,v in checks.items()},
                    passed=all(checks.values()),chunked_vs_one_shot_source_mae_hz=mae,
                    cut_vs_baseline_selected_mae_hz=cut_mae,feedback_vs_baseline_selected_mae_hz=feedback_mae,
                    sensory_cut_body_position_max_difference_mm=cut_position_max,
                    tolerance_note="Cut-control tolerances are exploratory post-result numerical diagnostics, not a preregistered biological effect threshold.",goal_complete=False)
    (OUT/"independent_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(json.dumps(validation,indent=2))
    if not validation["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
