"""Adaptive check of source grid conditions rich in tibia antagonists under Euler.

This is an exploratory numerical audit, not a held-out biological validation.
The four highest-count grid triples (excluding already checked grid 90) and
first four flagged rows each are selected before any adaptive results.
"""
from __future__ import annotations

from collections import defaultdict
import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from screen_official_dng100_t1_grid import BASE, GRID, OUT, TABLE, evaluate, make_pools, source_matrix


def main():
    discovery = json.loads((OUT / "discovery_report.json").read_text())
    grouped = defaultdict(list)
    for item in discovery["rows"]:
        if item["sides"]["LHS"]["tibia_both_variable"] and item["grid_index"] != 90:
            grouped[item["grid_index"]].append(item["parameter_row"])
    chosen = sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0]))[:4]
    grid = json.loads(GRID.read_text())["entries"]
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
    for grid_index, source_rows in chosen:
        item = grid[grid_index]
        ex, glu, inh = (item[k] for k in ("excitatory_multiplier", "glutamate_multiplier", "inhibitory_multiplier"))
        matrix = source_matrix(w, n.predictedNt.eq("glutamate").to_numpy(), ex, glu, inh).astype(np.float64)
        for row in source_rows[:4]:
            tau, gain, threshold, cap = (p[k][row] for k in ("tau", "gain", "threshold", "cap"))
            current = drives[row]
            def rhs(t, rate):
                total = matrix @ rate + (current if .02 <= t <= 1.999 else 0)
                return (np.maximum(cap * np.tanh((gain / cap) * (total - threshold)), 0) - rate) / tau
            ode = solve_ivp(rhs, (0, 2), np.zeros(len(n)), t_eval=np.arange(201)/100,
                            method="RK45", rtol=2e-6, atol=5e-9, max_step=.001)
            if not ode.success:
                raise RuntimeError(ode.message)
            trace = ode.y[selected, 1:].T.astype(np.float32)[:, :, None]
            result = evaluate(trace, lookup, e1_l, e1_r, pools, 0)
            rows.append({"grid_index": grid_index, "parameter_row": row, **result})
            print(f"grid={grid_index} row={row} E1={result['E1_left_sustained']} gate={result['left_joint_gate']}", flush=True)
    report = {"source": "https://zenodo.org/records/22260924",
              "selection_rule": "Four archived triples with most left tibia flex/extend variable rows in prospectively fixed 16-row Euler discovery split, excluding grid90; first four flagged row IDs in each. Adaptive outcome unseen at selection.",
              "chosen": [{"grid_index": k, "discovery_both_variable_count": len(v),
                          "preselected_rows": v[:4], "multiplier_entry": grid[k]} for k,v in chosen],
              "rows": rows, "strict_gate_count": sum(r["left_joint_gate"] for r in rows),
              "checks": {"sixteen_adaptive_cases_tested": len(rows) == 16,
                         "four_source_grid_triples_tested": len(chosen) == 4,
                         "source_input_single_DNg100_250": bool(np.count_nonzero(drives) == 128 and np.all(drives[:,31] == 250))},
              "body_walking_achieved": False}
    report["passed"] = all(report["checks"].values())
    (OUT / "high_motor_grid_adaptive_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"strict_gate_count": report["strict_gate_count"], "checks": report["checks"]}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
