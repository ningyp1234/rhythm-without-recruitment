"""High-resolution Euler *screen only* for a possible premotor dose window.

Any interesting condition must be independently rerun with adaptive RK45;
fixed-step candidates are not valid neural or walking success evidence.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from probe_fullmanc_premotor_source_solver import DATA, NEURONS, OUT, parameters, measure, source_graph
from source_latest_vnc_hill_interface import compatible_neurons


DOSES = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160], np.float32)
DT = .0005


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = compatible_neurons(pd.read_csv(NEURONS))
    byid = {int(v): i for i, v in enumerate(n.bodyId)}
    assert n.type.iloc[byid[12650]] == "IN21A004"
    selected = np.unique(np.r_[np.flatnonzero(n.type.eq("IN17A001").to_numpy()),
                               np.flatnonzero(n["class"].eq("motor neuron").to_numpy()),
                               [byid[k] for k in (12650, 10827, 17678)]])
    W = source_graph()
    p = parameters()
    current = np.repeat(p["current"].astype(np.float32)[:, None], len(DOSES), axis=1)
    current[byid[12650]] += DOSES
    tau = p["tau"].astype(np.float32)[:, None]
    gain_cap = (p["gain"] / p["cap"]).astype(np.float32)[:, None]
    cap = p["cap"].astype(np.float32)[:, None]
    threshold = p["threshold"].astype(np.float32)[:, None]
    rates = np.zeros_like(current)
    trace = np.zeros((len(DOSES), len(selected), 201), np.float32)
    started = time.perf_counter()
    for step in range(4000):
        t = step * DT
        stimulus = current if .02 <= t <= 1.999 else 0.0
        drive = stimulus + W @ rates
        activation = np.maximum(cap * np.tanh(gain_cap * (drive - threshold)), 0)
        rates += np.float32(DT) * (activation - rates) / tau
        if (step + 1) % 20 == 0:
            trace[:, :, (step + 1) // 20] = rates[selected].T
    rows = []
    for col, dose in enumerate(DOSES):
        m = measure(n, selected, trace[col], byid)
        rows.append({"dose": float(dose), "E1": m["six_E1_sustained_count"],
                     "left_front_flex_active": m["legs"]["lf"]["tibia flex"]["active_over_1_hz"],
                     "left_front_ext_active": m["legs"]["lf"]["tibia extend"]["active_over_1_hz"],
                     "left_front_both": m["left_front_tibia_both"],
                     "six_tibia_both_count": m["six_leg_tibia_both_count"],
                     "IN21A004_mean_hz": m["named_cell_mean_hz_after_0p5s"]["12650"],
                     "LF_flexor_12134_mean_hz": m["named_cell_mean_hz_after_0p5s"]["12134"],
                     "gate_candidate_only": m["six_E1_sustained_count"] == 6 and m["left_front_tibia_both"]})
    np.savez_compressed(OUT / "fixed_step_dose_screen.npz", doses=DOSES,
                        selected_indices=selected, rates_10ms=trace)
    source_baseline_E1 = 6
    baseline_matches_author = bool(rows[0]["E1"] == source_baseline_E1)
    result = {"method": "0.5ms forward Euler multi-dose selection only; exact source W, mask, same-run cell parameters; not a neural success gate",
              "data": str(DATA), "doses_model_current_units": DOSES.tolist(),
              "elapsed_seconds": time.perf_counter() - started,
              "rows": rows, "author_original_baseline_E1_count": source_baseline_E1,
              "fixed_step_baseline_matches_author": baseline_matches_author,
              "dose_ranking_valid": baseline_matches_author,
              "interpretation": "INVALID for ranking: fixed-step no-intervention baseline has 3/6 E1 vs 6/6 in author trace and source-aligned adaptive rerun" if not baseline_matches_author else "Adaptive confirmation required for any selected candidate",
              "adaptive_confirmation_required": True,
              "autonomous_walking_achieved": False}
    (OUT / "fixed_step_dose_screen.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
