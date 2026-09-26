"""Screen every author's saved full-MANC DNg100 neural trajectory.

Use the published sparse rates directly. The same predeclared E1/tibia gate
and cell ordering from screen_author_fullmanc_32 are applied to all four runs.
No local ODE solution, synthetic phase signal or body output is used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_context_diagnostic import rhythm
from screen_author_fullmanc_32 import LEGS, pool
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from source_calibration import ROOT


RUNS = (30662613, 33195857, 33195876, 33195882)
SOURCE = ROOT / "data/pugliese-zenodo-22260924/full-manc-all-runs"
OLD = ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882"
OUT = ROOT / "results/author-fullmanc-128-walking-20260924"
PUBLISHED_MOTOR_COUNTS = ROOT / "external/Pugliese_2026/figures/DNg100_Stim_fullManc-hyak-30662613-33195857-33195876-33195882/nMnsActive_allLegs.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    neurons = compatible_neurons(pd.read_csv(NEURONS))
    e1 = np.flatnonzero(neurons.type.eq("IN17A001").to_numpy())
    motor = np.flatnonzero(neurons["class"].eq("motor neuron").to_numpy())
    selected = np.unique(np.r_[e1, motor])
    assert (len(neurons), len(e1), len(motor), len(selected)) == (23532, 6, 732, 738)
    lookup = np.full(len(neurons), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected))
    pools = {}
    for leg, neuromere, side in LEGS:
        for module in ("tibia flex", "tibia extend", "coxa swing", "coxa stance"):
            ids = np.flatnonzero((neurons.somaNeuromere.eq(neuromere) & neurons.side.eq(side)
                                  & neurons["motor module"].eq(module)).to_numpy())
            assert len(ids)
            pools[(leg, module)] = lookup[ids]

    old_archive = np.load(ROOT / "results/row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz")
    assert np.array_equal(old_archive["selected_indices"], selected)
    assert np.array_equal(old_archive["selected_body_ids"], neurons.bodyId.iloc[selected].astype(int).to_numpy())
    published_counts = pd.read_csv(PUBLISHED_MOTOR_COUNTS)
    assert len(published_counts) == 128
    annotated_selected = neurons["motor module"].iloc[selected].astype(str).str.strip().ne("").to_numpy()
    all_rows, source_runs = [], []
    for block, run_id in enumerate(RUNS):
        folder = OLD if run_id == 33195882 else SOURCE / str(run_id)
        source_path = folder / "DNg100_Stim_fullManc_Rs.npz"
        if run_id == 33195882:
            trace = old_archive["rates_1ms"]
            shape = (1, 32, 23532, 2001)
        else:
            with np.load(source_path, allow_pickle=False) as sparse:
                shape = tuple(int(x) for x in sparse["shape"])
                assert shape == (1, 32, 23532, 2001) and float(sparse["fill_value"]) == 0.
                coords, values = sparse["coords"], sparse["data"]
            take = (coords[0] == 0) & (lookup[coords[2]] >= 0)
            trace = np.zeros((32, len(selected), 2001), np.float32)
            trace[coords[1, take], lookup[coords[2, take]], coords[3, take]] = values[take]
            del coords, values, take
            np.savez_compressed(OUT / f"run{run_id}_selected_738_neurons.npz",
                                rates_1ms=trace, selected_indices=selected,
                                selected_body_ids=neurons.bodyId.iloc[selected].astype(int).to_numpy())
        with h5py.File(folder / "neuron_params.h5") as params:
            currents = params["input_currents"][0]
            assert currents.shape == (32, len(neurons))
            active_input = np.flatnonzero(np.any(currents, axis=0))
            input_summary = [{"body_id": int(neurons.bodyId.iloc[i]),
                              "type": str(neurons.type.iloc[i]),
                              "range_current": [float(currents[:, i].min()), float(currents[:, i].max())]}
                             for i in active_input]
            n_input = np.count_nonzero(currents, axis=1).astype(int).tolist()
        source_runs.append({"run_id": run_id, "global_rows": [block * 32, block * 32 + 31],
                            "source_tensor_shape": shape,
                            "input_cells": input_summary, "input_cells_per_row": n_input,
                            "selected_trace_finite": bool(np.isfinite(trace).all()),
                            "published_figure4_active_motor_counts_match": bool(np.array_equal(
                                (trace[:, annotated_selected].max(axis=2) > .01).sum(axis=1),
                                published_counts["all legs"].iloc[block*32:block*32+32].to_numpy()))})
        for row in range(32):
            values = trace[row]
            e1_stats = [rhythm(values[lookup[i], 500::10], dt=.01) for i in e1]
            legs = {}
            for leg, _, _ in LEGS:
                flex = pool(values, pools[(leg, "tibia flex")])
                extend = pool(values, pools[(leg, "tibia extend")])
                swing = pool(values, pools[(leg, "coxa swing")])
                stance = pool(values, pools[(leg, "coxa stance")])
                both = bool(flex["ever_over_1hz"] and extend["ever_over_1hz"] and
                            flex["range_hz"] >= .5 and extend["range_hz"] >= .5)
                legs[leg] = {"tibia_flex": flex, "tibia_extend": extend,
                             "coxa_swing": swing, "coxa_stance": stance,
                             "both_tibia_pools_variable": both}
            sustained = sum(x["diagnostic_sustained_rhythm"] for x in e1_stats)
            all_rows.append({"global_row": block * 32 + row, "run_id": run_id,
                             "parameter_row": row, "sustained_E1_count": sustained,
                             "six_leg_tibia_both_count": sum(x["both_tibia_pools_variable"] for x in legs.values()),
                             "left_front_tibia_both": legs["lf"]["both_tibia_pools_variable"],
                             "all_six_E1_and_left_tibia": bool(sustained == 6 and legs["lf"]["both_tibia_pools_variable"]),
                             "legs": legs})
        print(f"run={run_id} E1_six={sum(x['sustained_E1_count']==6 for x in all_rows[-32:])} "
              f"joint={sum(x['all_six_E1_and_left_tibia'] for x in all_rows[-32:])}", flush=True)
        del trace
    prior = json.loads((ROOT / "results/row21-body-diagnostic-20260924/author_fullmanc_32_motor_screen.json").read_text())
    old_equivalent = all(
        all_rows[96 + i][field] == prior["rows"][i][field]
        for i in range(32)
        for field in ("sustained_E1_count", "six_leg_tibia_both_count", "left_front_tibia_both", "legs"))
    checks = {"all_128_author_rows": len(all_rows) == 128,
              "four_32_row_tensors": all(x["source_tensor_shape"] == (1, 32, 23532, 2001) for x in source_runs),
              "source_inputs_are_present": all(min(x["input_cells_per_row"]) > 0 for x in source_runs),
              "published_figure4_all128_motor_counts_reproduced": all(x["published_figure4_active_motor_counts_match"] for x in source_runs),
              "run33195882_old_32row_screen_reproduced": old_equivalent,
              "all_selected_rates_finite": all(x["selected_trace_finite"] for x in source_runs)}
    report = {"source": "https://zenodo.org/records/22260924",
              "source_run_order": list(RUNS),
              "scope": "All 128 author-saved full-MANC DNg100 parameter rows, source trajectories only, no behavioral claim",
              "predeclared_gate": "Same six E1 and tibia pool diagnostic as screen_author_fullmanc_32.py",
              "runs": source_runs, "rows": all_rows,
              "six_E1_rows": [r["global_row"] for r in all_rows if r["sustained_E1_count"] == 6],
              "six_tibia_rows": [r["global_row"] for r in all_rows if r["six_leg_tibia_both_count"] == 6],
              "joint_gate_rows": [r["global_row"] for r in all_rows if r["all_six_E1_and_left_tibia"]],
              "checks": checks, "passed": all(checks.values()),
              "autonomous_walking_achieved": False}
    (OUT / "author_fullmanc_128_neural_screen.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"six_E1_rows": report["six_E1_rows"],
                      "six_tibia_rows": report["six_tibia_rows"],
                      "joint_gate_rows": report["joint_gate_rows"],
                      "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
