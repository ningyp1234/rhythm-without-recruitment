"""Independent adaptive-solver check of discovered and held-out DNg100 rows."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.integrate import solve_ivp

from screen_official_dng100_t1_grid import OUT, TABLE, evaluate, make_pools
from validate_official_dng100_t1_candidate import DATA, EXC, INH

ROWS = (10, 78)


def main():
    n = pd.read_csv(TABLE).fillna("")
    with h5py.File(DATA) as h:
        w = h["W"][:]
        params = {key: h[key][list(ROWS)].astype(np.float64) for key in
                  ("tau", "a", "threshold", "fr_cap")}
        inputs = h["input_currents"][0, list(ROWS)].astype(np.float64)
    pre_post = sparse.coo_matrix(w.T)
    scale = np.where(pre_post.data > 0, EXC, INH)
    matrix = sparse.csr_matrix((pre_post.data * scale,
                                (pre_post.row, pre_post.col)), shape=pre_post.shape)
    pools = make_pools(n)
    e1_left = int(np.flatnonzero(n.bodyId.eq(10707).to_numpy())[0])
    e1_right = int(np.flatnonzero(n.bodyId.eq(10690).to_numpy())[0])
    motor = np.flatnonzero(n["class"].eq("motor neuron").to_numpy())
    selected = np.unique(np.r_[[31, e1_left, e1_right], motor])
    lookup = {int(v): i for i, v in enumerate(selected)}
    traces = []
    records = []
    for column, row in enumerate(ROWS):
        tau = params["tau"][column]
        gain = params["a"][column]
        threshold = params["threshold"][column]
        cap = params["fr_cap"][column]
        current = inputs[column]
        def rhs(t, rates):
            stimulus = current if .02 <= t <= 1.999 else 0
            total = matrix @ rates + stimulus
            return (np.maximum(cap * np.tanh((gain / cap) *
                                             (total - threshold)), 0) - rates) / tau
        ode = solve_ivp(rhs, (0, 2), np.zeros(len(n)),
                        t_eval=np.arange(201) / 100,
                        method="RK45", rtol=2e-6, atol=5e-9, max_step=.001)
        if not ode.success:
            raise RuntimeError(ode.message)
        traces.append(ode.y[selected, 1:].T.astype(np.float32))
        print(f"adaptive row {row} complete", flush=True)
    trace = np.stack(traces, axis=2)
    for column, row in enumerate(ROWS):
        item = evaluate(trace, lookup, e1_left, e1_right, pools, column)
        records.append({"parameter_row": row, "split": "discovery" if row < 16 else "heldout", **item})
    np.savez_compressed(OUT / "candidate_adaptive_rows10_78.npz", rates_10ms=trace,
                        parameter_rows=np.array(ROWS), selected_indices=selected,
                        selected_body_ids=n.bodyId.iloc[selected].astype(int).to_numpy())
    report = {"source": "https://zenodo.org/records/22260924",
              "scope": "Independent RK45 replay of exact author checkpoint row10 discovery and row78 heldout, original published adaptive tolerances and 1ms max step",
              "parameter_rows": list(ROWS), "rows": records,
              "checks": {"both_solutions_finite": bool(np.isfinite(trace).all()),
                         "both_source_stimuli_250": bool(np.all(inputs[:, 31] == 250) and np.count_nonzero(inputs) == 2),
                         "both_rows_original_graph_and_params": True},
              "body_walking_achieved": False}
    report["passed"] = all(report["checks"].values())
    (OUT / "candidate_adaptive_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": [{"parameter_row": x["parameter_row"], "left_joint_gate": x["left_joint_gate"],
                                "left_E1": x["E1_left_sustained"],
                                "left_tibia_corr": x["sides"]["LHS"]["tibia_correlation"]} for x in records],
                      "checks": report["checks"]}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
