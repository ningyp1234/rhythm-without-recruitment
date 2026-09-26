"""Test whether the T1 DNg100 motor gate survives integration refinement."""
from __future__ import annotations

import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from screen_official_dng100_t1_grid import OUT, TABLE, evaluate, make_pools
from validate_official_dng100_t1_candidate import DATA, EXC, INH

ROWS = (10, 78)
STEPS = (.001, .0005, .00025, .000125)


def main():
    n = pd.read_csv(TABLE).fillna("")
    with h5py.File(DATA) as h:
        w = sparse.coo_matrix(h["W"][:].T)
        p = {k: h[v][list(ROWS)].T.astype(np.float64) for k, v in
             (("tau", "tau"), ("gain", "a"), ("threshold", "threshold"), ("cap", "fr_cap"))}
        drive = h["input_currents"][0, list(ROWS)].T.astype(np.float64)
    matrix = sparse.csr_matrix((w.data * np.where(w.data >= 0, EXC, INH),
                                (w.row, w.col)), shape=w.shape)
    pools = make_pools(n)
    e1_l = int(np.flatnonzero(n.bodyId.eq(10707).to_numpy())[0])
    e1_r = int(np.flatnonzero(n.bodyId.eq(10690).to_numpy())[0])
    selected = np.unique(np.r_[[31, e1_l, e1_r], *pools.values()])
    lookup = {int(v): i for i, v in enumerate(selected)}
    records = []
    arrays = {}
    for dt in STEPS:
        steps_per_frame = round(.01 / dt)
        rates = np.zeros_like(p["tau"])
        trace = np.empty((200, len(selected), 2), np.float32)
        for step in range(round(2 / dt)):
            t = step * dt
            total = (drive if .02 <= t <= 1.999 else 0) + matrix @ rates
            active = np.maximum(p["cap"] * np.tanh((p["gain"] / p["cap"]) *
                                                  (total - p["threshold"])), 0)
            rates += dt * (active - rates) / p["tau"]
            if (step + 1) % steps_per_frame == 0:
                trace[(step + 1) // steps_per_frame - 1] = rates[selected]
        arrays[f"dt_{dt:g}_rates_10ms"] = trace
        for col, row in enumerate(ROWS):
            item = evaluate(trace, lookup, e1_l, e1_r, pools, col)
            records.append({"dt_s": dt, "parameter_row": row, **item})
        print(f"Euler dt={dt:g}s complete", flush=True)
    adaptive = json.loads((OUT / "candidate_adaptive_report.json").read_text())
    for x in adaptive["rows"]:
        records.append({"dt_s": "adaptive_RK45", **x})
    np.savez_compressed(OUT / "candidate_numerical_convergence.npz",
                        selected_indices=selected, selected_body_ids=n.bodyId.iloc[selected].astype(int).to_numpy(),
                        **arrays)
    report = {"source": "https://zenodo.org/records/22260924", "rows": records,
              "scope": "Exact author checkpoint; four Euler step sizes 1ms to 0.125ms plus independent adaptive RK45; two predeclared rows 10 and 78",
              "checks": {"all_euler_traces_finite": all(np.isfinite(v).all() for v in arrays.values()),
                         "all_euler_step_sizes_tested": len(arrays) == 4},
              "body_walking_achieved": False}
    report["passed"] = all(report["checks"].values())
    (OUT / "candidate_numerical_convergence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{"dt_s": r["dt_s"], "parameter_row": r["parameter_row"],
                       "left_joint_gate": r["left_joint_gate"],
                       "E1_left": r["E1_left_sustained"],
                       "tibia_correlation": r["sides"]["LHS"]["tibia_correlation"]}
                      for r in records], indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
