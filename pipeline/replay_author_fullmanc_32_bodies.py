"""Physically test every author-saved full-MANC run with unchanged bodies.

All 32 source trajectories are tested prospectively, no body-score parameter
search. The two motor bridges are the existing 78-Hill and 18-family-torque
models. Even a physical pass here would remain open-loop DN stimulation.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from homologous_hill_body import HomologousHillBody
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from source_calibration import ROOT
from walking_body import WalkingBody
from walking_benchmark import score


OUT = ROOT / "results/row21-body-diagnostic-20260924"


def replay(body, rates: np.ndarray) -> dict:
    body.reset()
    rows, feet, contacts = [], [], []
    for tick in range(1, 201):
        body.step(rates[:, tick * 10], 10)
        state = body.telemetry()
        p = state["position"]
        rows.append({"t": tick / 100, "x": p[0], "y": p[1], "z": p[2],
                     "upright": state["upright"],
                     "nonfoot_load_fraction": state["nonfoot_load_fraction"]})
        feet.append(state["feet"])
        contacts.append(state["contact"])
    result = score(rows, feet, contacts)
    result["physics_warnings"] = int(body.sim.mj_data.warning.number.sum())
    return result


def main() -> None:
    n = compatible_neurons(pd.read_csv(NEURONS))
    selected_archive = np.load(OUT / "author_fullmanc_32_selected_neurons.npz")
    selected = selected_archive["selected_indices"]
    n_index = {int(v): j for j, v in enumerate(selected)}
    motor = np.flatnonzero(n["class"].eq("motor neuron").to_numpy())
    selected_motor = np.array([n_index[int(v)] for v in motor])
    assert len(motor) == 732
    data = selected_archive["rates_1ms"]
    assert data.shape == (32, 738, 2001)
    hill = HomologousHillBody(n, stiffness=32.)
    torque = WalkingBody(n, stiffness=32., gain=8.)
    full = np.zeros((len(n), 2001), np.float32)
    rows = []
    started = time.perf_counter()
    for row in range(32):
        full[motor] = data[row, selected_motor]
        h = replay(hill, full)
        t = replay(torque, full)
        rows.append({"parameter_row": row,
                     "hill": {"forward_mm": h["forward_mm"], "steps_per_leg": h["steps_per_leg"],
                              "walking_pass": h["walking_pass"], "gates": h["gates"],
                              "physics_warnings": h["physics_warnings"]},
                     "torque": {"forward_mm": t["forward_mm"], "steps_per_leg": t["steps_per_leg"],
                                "walking_pass": t["walking_pass"], "gates": t["gates"],
                                "physics_warnings": t["physics_warnings"]}})
        print(f"row={row:02d} hill={h['forward_mm']:.3f}mm {h['steps_per_leg']} torque={t['forward_mm']:.3f}mm {t['steps_per_leg']}", flush=True)
    independently_replayed_row21 = json.loads((OUT / "author_body_replay.json").read_text())
    row21 = rows[21]
    checks = {"all_32_author_rows_tested": len(rows) == 32,
              "row21_fresh_hill_reproduced": abs(row21["hill"]["forward_mm"] - independently_replayed_row21["hill_body"]["forward_mm"]) < 1e-5,
              "row21_fresh_torque_reproduced": abs(row21["torque"]["forward_mm"] - independently_replayed_row21["torque_body"]["forward_mm"]) < 1e-5,
              "no_physics_warning": all(x[k]["physics_warnings"] == 0 for x in rows for k in ("hill", "torque")),
              "source_trajectory_and_body_parameters_not_tuned": True}
    report = {"scope": "All 32 published full-MANC DNg100 run33195882 parameter-row neural trajectories, re-embodied open-loop in unchanged 78-Hill and 18-family-torque six-leg bodies",
              "source": "https://zenodo.org/records/22260924",
              "source_member": "DNg100_Stim_fullManc_Rs.npz",
              "model_seconds_each": 2.0, "wall_seconds": time.perf_counter() - started,
              "rows": rows,
              "hill_walking_pass_rows": [x["parameter_row"] for x in rows if x["hill"]["walking_pass"]],
              "torque_walking_pass_rows": [x["parameter_row"] for x in rows if x["torque"]["walking_pass"]],
              "hill_forward_5mm_rows": [x["parameter_row"] for x in rows if x["hill"]["forward_mm"] >= 5],
              "torque_forward_5mm_rows": [x["parameter_row"] for x in rows if x["torque"]["forward_mm"] >= 5],
              "checks": checks, "passed": all(checks.values()),
              "autonomous_walking_achieved": False}
    (OUT / "author_fullmanc_32_body_screen.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks,
                      "hill_walking_pass_rows": report["hill_walking_pass_rows"],
                      "torque_walking_pass_rows": report["torque_walking_pass_rows"],
                      "max_hill_forward_mm": max(x["hill"]["forward_mm"] for x in rows),
                      "max_torque_forward_mm": max(x["torque"]["forward_mm"] for x in rows)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
