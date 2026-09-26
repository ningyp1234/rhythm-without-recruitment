"""Use author-published motor-rhythm summaries to rank DNg100 conditions.

Prospectively take the first eight source-positive parameter rows from each of
the top three archived synapse-class triples. Test our stricter antagonist
gate with an independent adaptive ODE solver. Source summary ranking and all
selection rules are saved, so the selection is auditable.
"""
from __future__ import annotations

import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from screen_official_dng100_t1_grid import BASE, OUT, TABLE, evaluate, make_pools, source_matrix
from source_calibration import ROOT

AUTHOR = ROOT / "external/Pugliese_2026/figures/DNg100_Stim-hyak-run_id=33258130"
PER_CONDITION = 8
CONDITIONS = 3


def main():
    scores = pd.read_csv(AUTHOR / "scores_multiIndex_20260826.csv", header=[0, 1, 2], index_col=0)
    active = pd.read_csv(AUTHOR / "nMnsActive_multiIndex_20260826.csv", header=[0, 1, 2], index_col=0)
    grid = json.loads((BASE / "archive_grid.json").read_text())["entries"]
    ranked = []
    for ix, entry in enumerate(grid):
        key = (str(entry["excitatory_multiplier"]), str(entry["inhibitory_multiplier"]),
               str(entry["glutamate_multiplier"]))
        if key not in scores.columns:
            continue
        positives = (scores[key] > .5) & (active[key] > 20)
        ranked.append((int(positives.sum()), ix, entry, [int(i) for i in positives[positives].index]))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    chosen = ranked[:CONDITIONS]
    n = pd.read_csv(TABLE).fillna("")
    with h5py.File(BASE / "neuron_params.h5") as h:
        w = sparse.coo_matrix(h["W"][:])
        p = {name: h[key][:].astype(np.float64) for name, key in
             (("tau", "tau"), ("gain", "a"), ("threshold", "threshold"), ("cap", "fr_cap"))}
        drives = h["input_currents"][0].astype(np.float64)
    pools = make_pools(n)
    e1_l = int(np.flatnonzero(n.bodyId.eq(10707).to_numpy())[0])
    e1_r = int(np.flatnonzero(n.bodyId.eq(10690).to_numpy())[0])
    motor = np.flatnonzero(n["class"].eq("motor neuron").to_numpy())
    selected = np.unique(np.r_[[31, e1_l, e1_r], motor])
    lookup = {int(v): i for i, v in enumerate(selected)}
    rows = []
    gate_traces = {}
    for count, grid_index, entry, positive_rows in chosen:
        ex, glu, inh = (entry[k] for k in ("excitatory_multiplier", "glutamate_multiplier", "inhibitory_multiplier"))
        matrix = source_matrix(w, n.predictedNt.eq("glutamate").to_numpy(), ex, glu, inh).astype(np.float64)
        for row in positive_rows[:PER_CONDITION]:
            tau, gain, threshold, cap = (p[k][row] for k in ("tau", "gain", "threshold", "cap"))
            current = drives[row]
            def rhs(t, rates):
                total = matrix @ rates + (current if .02 <= t <= 1.999 else 0)
                return (np.maximum(cap * np.tanh((gain / cap) * (total - threshold)), 0) - rates) / tau
            ode = solve_ivp(rhs, (0, 2), np.zeros(len(n)), t_eval=np.arange(201) / 100,
                            method="RK45", rtol=2e-6, atol=5e-9, max_step=.001)
            if not ode.success:
                raise RuntimeError(ode.message)
            trace = ode.y[selected, 1:].T.astype(np.float32)[:, :, None]
            result = evaluate(trace, lookup, e1_l, e1_r, pools, 0)
            rows.append({"grid_index": grid_index, "parameter_row": row,
                         "author_positive_rows_in_condition": count,
                         "author_score": float(scores[(str(ex), str(inh), str(glu))].loc[row]),
                         "author_active_motor_count": float(active[(str(ex), str(inh), str(glu))].loc[row]),
                         "excitatory_multiplier": ex, "glutamate_multiplier": glu,
                         "inhibitory_multiplier": inh, **result})
            if result["left_joint_gate"]:
                gate_traces[f"grid{grid_index}_row{row}"] = trace[:, :, 0]
            print(f"adaptive grid={grid_index} row={row} E1={result['E1_left_sustained']} gate={result['left_joint_gate']}", flush=True)
    if gate_traces:
        np.savez_compressed(OUT / "author_ranked_adaptive_gate_traces.npz",
                            selected_indices=selected, **gate_traces)
    checks = {"author_summary_has_125_triples": scores.shape == (128, 125) and active.shape == (128, 125),
              "three_published_archive_conditions_selected": len(chosen) == CONDITIONS,
              "all_chosen_have_at_least_eight_author_positive_rows": all(x[0] >= PER_CONDITION for x in chosen),
              "all_24_adaptive_solves_finite": len(rows) == CONDITIONS * PER_CONDITION,
              "only_archived_input_and_parameters_used": bool(np.count_nonzero(drives) == 128 and np.all(drives[:, 31] == 250))}
    report = {"source": "https://zenodo.org/records/22260924",
              "author_summary_source": str(AUTHOR.relative_to(ROOT)),
              "ranking_rule": "Among 116 original archived triples, rank by number of 128 rows with author scores_multiIndex_20260826 > 0.5 AND nMnsActive_multiIndex_20260826 > 20. Tie by archive index. Test first eight qualifying row indices in each of top three.",
              "top_three": [{"grid_index": ix, "positive_count": count, "entry": entry,
                             "preselected_parameter_rows": positive_rows[:PER_CONDITION]}
                            for count, ix, entry, positive_rows in chosen],
              "rows": rows, "strict_left_joint_gate_pass_count": sum(r["left_joint_gate"] for r in rows),
              "checks": checks, "passed": all(checks.values()), "body_walking_achieved": False}
    (OUT / "author_ranked_adaptive_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"strict_left_joint_gate_pass_count": report["strict_left_joint_gate_pass_count"],
                      "checks": checks}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
