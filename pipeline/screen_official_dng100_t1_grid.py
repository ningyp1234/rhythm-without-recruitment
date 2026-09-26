"""Prospective DNg100 source synaptic-parameter grid, T1 discovery split.

The 116 multiplier triples are listed as checkpoints in the author's Zenodo
archive. We use its exact seed-145 neuron parameters, MANC T1 graph, single
DNg100 input, and a fixed 1 ms numerical approximation. The first 16 of 128
parameter rows are a declared discovery split; no body or gait score is used.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import h5py
from scipy import sparse

from cpg_context_diagnostic import rhythm
from source_calibration import ROOT

BASE = ROOT / "data/pugliese-zenodo-22260924/dng100-t1-seed145-baseline"
DATA = BASE / "neuron_params.h5"
GRID = BASE / "archive_grid.json"
TABLE = ROOT / "external/Pugliese_2026/data/manc t1 connectome data/wTable_20250813_DNtoMN_unsorted_withModules.csv"
OUT = ROOT / "results/official-dng100-t1-grid-20260924"
DISCOVERY = 16


def read_source():
    n = pd.read_csv(TABLE).fillna("")
    with h5py.File(DATA) as h:
        w = sparse.coo_matrix(h["W"][:])
        p = {name: h[key][:DISCOVERY].T.astype(np.float32) for name, key in
             (("tau", "tau"), ("gain", "a"), ("threshold", "threshold"), ("cap", "fr_cap"))}
        current = h["input_currents"][0, :DISCOVERY].T.astype(np.float32)
        mask_true = all(bool(np.all(h["W_mask"][j])) for j in (0, DISCOVERY - 1))
    return n, w, p, current, mask_true


def source_matrix(w, glutamate_pre, ex, glu, inh):
    # Source implementation first multiplies entire glutamate presynaptic
    # rows by glu/inh, then reweights positive and negative W by ex and inh.
    factor = np.where(glutamate_pre[w.row], glu / inh, 1.).astype(np.float32)
    factor *= np.where(w.data >= 0, ex, inh).astype(np.float32)
    return sparse.csr_matrix((w.data.astype(np.float32) * factor,
                              (w.col, w.row)), shape=w.shape)


def make_pools(n):
    pools = {}
    for side in ("LHS", "RHS"):
        for module in ("coxa stance", "coxa swing", "femur/tr extend", "femur/tr flex", "tibia extend", "tibia flex"):
            pools[f"{side}:{module}"] = np.flatnonzero((n.somaSide.eq(side) & n["motor module"].eq(module)).to_numpy())
    return pools


def evaluate(trace, lookup, e1_left, e1_right, pools, column):
    e1_left_stats = rhythm(trace[50:, lookup[e1_left], column], dt=.01)
    e1_right_stats = rhythm(trace[50:, lookup[e1_right], column], dt=.01)
    side_results = {}
    for side in ("LHS", "RHS"):
        measures = {}
        times = {}
        for module in ("coxa stance", "coxa swing", "tibia flex", "tibia extend"):
            ids = pools[f"{side}:{module}"]
            values = trace[50:, [lookup[int(v)] for v in ids], column]
            times[module] = values.mean(axis=1)
            measures[module] = {"ever_over_1_hz": int((values.max(axis=0) > 1).sum()),
                                "mean_hz": float(values.mean()),
                                "pool_range_hz": float(np.ptp(times[module]))}
        a = times["tibia flex"].astype(np.float64)
        b = times["tibia extend"].astype(np.float64)
        da, db = a - a.mean(), b - b.mean()
        den = float(np.sqrt(np.dot(da, da) * np.dot(db, db)))
        corr = float(np.dot(da, db) / den) if den > 1e-12 else None
        both = bool(measures["tibia flex"]["ever_over_1_hz"] and
                    measures["tibia extend"]["ever_over_1_hz"] and
                    measures["tibia flex"]["pool_range_hz"] >= .5 and
                    measures["tibia extend"]["pool_range_hz"] >= .5)
        side_results[side] = {"pools": measures, "tibia_both_variable": both,
                              "tibia_correlation": corr,
                              "tibia_antiphase": bool(both and corr is not None and corr < -.25)}
    return {"E1_left_sustained": bool(e1_left_stats["diagnostic_sustained_rhythm"]),
            "E1_right_sustained": bool(e1_right_stats["diagnostic_sustained_rhythm"]),
            "E1_left_frequency_hz": (float(e1_left_stats["frequency_hz"])
                                     if e1_left_stats["frequency_hz"] is not None
                                     and np.isfinite(e1_left_stats["frequency_hz"]) else None),
            "sides": side_results,
            "left_joint_gate": bool(e1_left_stats["diagnostic_sustained_rhythm"] and side_results["LHS"]["tibia_antiphase"])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n, w, p, drive, mask_true = read_source()
    grid = json.loads(GRID.read_text())["entries"]
    assert len(grid) == 116 and len(n) == 4604 and w.nnz == 196535
    assert n.bodyId.iloc[31] == 10093 and n.type.iloc[31] == "DNg100"
    assert np.count_nonzero(drive) == DISCOVERY and np.all(drive[31] == 250)
    assert mask_true
    glupre = n.predictedNt.eq("glutamate").to_numpy()
    pools = make_pools(n)
    e1_left = int(np.flatnonzero(n.bodyId.eq(10707).to_numpy())[0])
    e1_right = int(np.flatnonzero(n.bodyId.eq(10690).to_numpy())[0])
    selected = np.unique(np.r_[[31, e1_left, e1_right],
                               *[v for x in pools.values() for v in [x]]])
    lookup = {int(v): i for i, v in enumerate(selected)}
    rows = []
    best = None
    best_trace = None
    for number, entry in enumerate(grid):
        ex, glu, inh = (entry[k] for k in
                        ("excitatory_multiplier", "glutamate_multiplier", "inhibitory_multiplier"))
        matrix = source_matrix(w, glupre, ex, glu, inh)
        rates = np.zeros_like(p["tau"])
        trace = np.empty((200, len(selected), DISCOVERY), np.float32)
        for step in range(2000):
            total = (drive if step >= 20 else 0) + matrix @ rates
            active = np.maximum(p["cap"] * np.tanh((p["gain"] / p["cap"]) *
                                                  (total - p["threshold"])), 0)
            rates += np.float32(.001) * (active - rates) / p["tau"]
            if (step + 1) % 10 == 0:
                trace[(step + 1) // 10 - 1] = rates[selected]
        for cell_row in range(DISCOVERY):
            item = {"grid_index": number, "parameter_row": cell_row,
                    "exc": ex, "glu": glu, "inh": inh,
                    **evaluate(trace, lookup, e1_left, e1_right, pools, cell_row)}
            rows.append(item)
            left = item["sides"]["LHS"]
            score = (int(item["left_joint_gate"]),
                     int(item["E1_left_sustained"] and left["tibia_both_variable"]),
                     int(item["E1_left_sustained"]),
                     left["pools"]["tibia flex"]["pool_range_hz"] *
                     left["pools"]["tibia extend"]["pool_range_hz"])
            if best is None or score > best[0]:
                best = (score, number, cell_row)
                best_trace = trace[:, :, cell_row].copy()
        if number % 20 == 0 or number == len(grid) - 1:
            print(f"grid {number+1}/{len(grid)} | best {best[:3]}", flush=True)
    checks = {"source_grid_116_unique_archived_conditions": len(set((x["exc"], x["glu"], x["inh"]) for x in rows)) == 116,
              "source_exact_16_neuron_parameter_rows": len(rows) == 116 * DISCOVERY,
              "single_source_DNg100_input_250": bool(np.count_nonzero(drive) == DISCOVERY and np.all(drive[31] == 250)),
              "all_analyses_finite": bool(np.isfinite(best_trace).all()),
              "source_graph_4604_196535": len(n) == 4604 and w.nnz == 196535}
    report = {"source": "https://zenodo.org/records/22260924", "archive": "DNg100_Stim__edf2_a.zip",
              "scope": "Published synapse-class multiplier grid on source MANC T1 DNg100 model; discovery parameter rows 0-15 only; no body",
              "parameter_rows_available": 128, "parameter_rows_discovery": DISCOVERY,
              "published_multiplier_conditions": len(grid), "neurons": len(n), "source_directed_pairs": int(w.nnz),
              "rows": rows, "best_grid_index": best[1], "best_parameter_row": best[2],
              "source_result_is_body_walking": False, "checks": checks, "passed": all(checks.values())}
    (OUT / "discovery_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / "best_discovery_trace.npz", rates_10ms=best_trace,
                        selected_indices=selected, selected_body_ids=n.bodyId.iloc[selected].astype(int).to_numpy(),
                        grid_index=best[1], parameter_row=best[2])
    summary = {"checks": checks,
               "left_joint_gate_cases": sum(r["left_joint_gate"] for r in rows),
               "left_E1_plus_both_variable_cases": sum(r["E1_left_sustained"] and r["sides"]["LHS"]["tibia_both_variable"] for r in rows),
               "left_tibia_flexor_recruited_cases": sum(r["sides"]["LHS"]["pools"]["tibia flex"]["ever_over_1_hz"] > 0 for r in rows),
               "best_grid_index": best[1], "best_parameter_row": best[2], "best_score": best[0]}
    print(json.dumps(summary, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
