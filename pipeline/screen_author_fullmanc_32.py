"""Screen all 32 author-saved full-MANC trajectories, no numerical rerun.

The original published sparse rate tensor is the source of every value. Only
the six E1 cells and 732 motor neurons are extracted; no synthetic rates,
phase templates or body scores enter candidate selection.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from cpg_context_diagnostic import rhythm
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from source_calibration import ROOT


DATA = ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882/DNg100_Stim_fullManc_Rs.npz"
OUT = ROOT / "results/row21-body-diagnostic-20260924"
LEGS = (("lf", "T1", "L"), ("lm", "T2", "L"), ("lh", "T3", "L"),
        ("rf", "T1", "R"), ("rm", "T2", "R"), ("rh", "T3", "R"))


def pool(trace: np.ndarray, indices: np.ndarray) -> dict:
    if len(indices) == 0:
        return {"size": 0, "ever_over_1hz": 0, "mean_hz": 0., "range_hz": 0.}
    values = trace[indices, 500::10]
    mean = values.mean(axis=0)
    return {"size": len(indices), "ever_over_1hz": int((values.max(axis=1) > 1).sum()),
            "mean_hz": float(values.mean()), "range_hz": float(np.ptp(mean))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = compatible_neurons(pd.read_csv(NEURONS))
    e1 = np.flatnonzero(n.type.eq("IN17A001").to_numpy())
    motor = np.flatnonzero(n["class"].eq("motor neuron").to_numpy())
    selected = np.unique(np.r_[e1, motor])
    assert len(e1) == 6 and len(motor) == 732 and len(selected) == 738
    lookup = np.full(len(n), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected))
    with np.load(DATA, allow_pickle=False) as source:
        shape = tuple(source["shape"])
        assert shape == (1, 32, 23532, 2001) and float(source["fill_value"]) == 0.
        coords, values = source["coords"], source["data"]
    take = (coords[0] == 0) & (lookup[coords[2]] >= 0)
    trace = np.zeros((32, len(selected), 2001), np.float32)
    trace[coords[1, take], lookup[coords[2, take]], coords[3, take]] = values[take]
    del coords, values, take
    pools = {}
    for leg, neu, side in LEGS:
        for module in ("tibia flex", "tibia extend", "coxa swing", "coxa stance"):
            ids = np.flatnonzero((n.somaNeuromere.eq(neu) & n.side.eq(side)
                                  & n["motor module"].eq(module)).to_numpy())
            pools[(leg, module)] = lookup[ids]
            assert len(ids) > 0
    rows = []
    for row in range(32):
        values = trace[row]
        e1_stats = [rhythm(values[lookup[i], 500::10], dt=.01) for i in e1]
        leg_results = {}
        for leg, _, _ in LEGS:
            f, x = pool(values, pools[(leg, "tibia flex")]), pool(values, pools[(leg, "tibia extend")])
            swing, stance = pool(values, pools[(leg, "coxa swing")]), pool(values, pools[(leg, "coxa stance")])
            both = bool(f["ever_over_1hz"] and x["ever_over_1hz"] and
                        f["range_hz"] >= .5 and x["range_hz"] >= .5)
            leg_results[leg] = {"tibia_flex": f, "tibia_extend": x,
                                "coxa_swing": swing, "coxa_stance": stance,
                                "both_tibia_pools_variable": both}
        count_e1 = sum(x["diagnostic_sustained_rhythm"] for x in e1_stats)
        rows.append({"parameter_row": row, "sustained_E1_count": count_e1,
                     "left_front_tibia_both": leg_results["lf"]["both_tibia_pools_variable"],
                     "six_leg_tibia_both_count": sum(v["both_tibia_pools_variable"] for v in leg_results.values()),
                     "all_six_E1_and_left_tibia": count_e1 == 6 and leg_results["lf"]["both_tibia_pools_variable"],
                     "legs": leg_results})
    checks = {"all_32_source_rows_extracted": len(rows) == 32,
              "published_shape_verified": shape == (1, 32, 23532, 2001),
              "selected_738_source_cell_ids": len(selected) == 738,
              "selected_trace_finite": bool(np.isfinite(trace).all())}
    report = {"source": "https://zenodo.org/records/22260924",
              "member": "run_id=33195882/ckpt/DNg100_Stim_fullManc_Rs.npz",
              "scope": "All 32 author-saved full-MANC rate trajectories; six E1 and anatomically annotated motor pools; no local solver or body",
              "predeclared_motor_gate": "After .5 s, each tibia flexor and extensor pool needs >=1 neuron above 1 Hz and mean-pool range >=.5 Hz; six E1 sustained additionally required for joint gate",
              "rows": rows,
              "rows_with_six_E1": [r["parameter_row"] for r in rows if r["sustained_E1_count"] == 6],
              "rows_with_left_tibia_both": [r["parameter_row"] for r in rows if r["left_front_tibia_both"]],
              "rows_with_joint_gate": [r["parameter_row"] for r in rows if r["all_six_E1_and_left_tibia"]],
              "checks": checks, "passed": all(checks.values()), "brain_autonomous_walking_achieved": False}
    np.savez_compressed(OUT / "author_fullmanc_32_selected_neurons.npz",
                        rates_1ms=trace, selected_indices=selected,
                        selected_body_ids=n.bodyId.iloc[selected].astype(int).to_numpy())
    (OUT / "author_fullmanc_32_motor_screen.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"six_E1_rows": report["rows_with_six_E1"],
                      "left_tibia_both_rows": report["rows_with_left_tibia_both"],
                      "joint_gate_rows": report["rows_with_joint_gate"],
                      "max_six_tibia_both": max(r["six_leg_tibia_both_count"] for r in rows),
                      "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
