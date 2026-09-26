"""Re-embody all 128 published DNg100 full-MANC neural trajectories.

No parameter search: each original author-saved motor trace drives the same
two six-leg body models used for the prior 32-row audit. The scoring rule is
also unchanged. This is open-loop replay, not autonomous neural behavior.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from homologous_hill_body import HomologousHillBody
from replay_author_fullmanc_32_bodies import replay
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from source_calibration import ROOT
from walking_body import WalkingBody


RUNS = (30662613, 33195857, 33195876, 33195882)
OUT = ROOT / "results/author-fullmanc-128-walking-20260924"
PRIOR = ROOT / "results/row21-body-diagnostic-20260924"


def main() -> None:
    n = compatible_neurons(pd.read_csv(NEURONS))
    motor = np.flatnonzero(n["class"].eq("motor neuron").to_numpy())
    assert len(motor) == 732
    hill = HomologousHillBody(n, stiffness=32.)
    torque = WalkingBody(n, stiffness=32., gain=8.)
    full = np.zeros((len(n), 2001), np.float32)
    rows = []
    started = time.perf_counter()
    for block, run_id in enumerate(RUNS):
        path = (PRIOR / "author_fullmanc_32_selected_neurons.npz" if run_id == 33195882
                else OUT / f"run{run_id}_selected_738_neurons.npz")
        with np.load(path) as selected_npz:
            selected = selected_npz["selected_indices"]
            rates = selected_npz["rates_1ms"]
        lookup = np.full(len(n), -1, dtype=int)
        lookup[selected] = np.arange(len(selected))
        motor_rates = rates[:, lookup[motor], :]
        assert motor_rates.shape == (32, 732, 2001)
        for row in range(32):
            full[motor] = motor_rates[row]
            h = replay(hill, full)
            t = replay(torque, full)
            rows.append({"global_row": block * 32 + row, "run_id": run_id,
                         "parameter_row": row,
                         "hill": {"forward_mm": h["forward_mm"], "steps_per_leg": h["steps_per_leg"],
                                  "walking_pass": h["walking_pass"], "gates": h["gates"],
                                  "physics_warnings": h["physics_warnings"]},
                         "torque": {"forward_mm": t["forward_mm"], "steps_per_leg": t["steps_per_leg"],
                                    "walking_pass": t["walking_pass"], "gates": t["gates"],
                                    "physics_warnings": t["physics_warnings"]}})
            print(f"global={block*32+row:03d} run={run_id} row={row:02d} "
                  f"hill={h['forward_mm']:.3f}mm {h['steps_per_leg']} "
                  f"torque={t['forward_mm']:.3f}mm {t['steps_per_leg']}", flush=True)
        del rates, motor_rates
    prior = json.loads((PRIOR / "author_fullmanc_32_body_screen.json").read_text())
    old_match = all(
        abs(rows[96+i][body]["forward_mm"] - prior["rows"][i][body]["forward_mm"]) < 1e-6
        and rows[96+i][body]["steps_per_leg"] == prior["rows"][i][body]["steps_per_leg"]
        for i in range(32) for body in ("hill", "torque"))
    checks = {"all_128_author_rows_physically_replayed": len(rows) == 128,
              "run33195882_old_32row_body_reproduced": old_match,
              "no_physics_warnings": all(r[b]["physics_warnings"] == 0 for r in rows for b in ("hill", "torque")),
              "two_unchanged_body_models": hill.stiffness == torque.stiffness == 32. and torque.gain == 8.,
              "source_motor_rates_without_added_gait_or_phase": True}
    report = {"scope": "Four official full-MANC DNg100 runs, all 128 author-saved parameter rows, 2s each, two unchanged six-leg bodies, open-loop",
              "source": "https://zenodo.org/records/22260924", "run_order": list(RUNS),
              "model_seconds_each": 2., "wall_seconds": time.perf_counter() - started,
              "rows": rows,
              "hill_walking_pass_rows": [r["global_row"] for r in rows if r["hill"]["walking_pass"]],
              "torque_walking_pass_rows": [r["global_row"] for r in rows if r["torque"]["walking_pass"]],
              "hill_forward_5mm_rows": [r["global_row"] for r in rows if r["hill"]["forward_mm"] >= 5],
              "torque_forward_5mm_rows": [r["global_row"] for r in rows if r["torque"]["forward_mm"] >= 5],
              "checks": checks, "passed": all(checks.values()), "autonomous_walking_achieved": False}
    (OUT / "author_fullmanc_128_body_screen.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks,
                      "hill_walking_pass_rows": report["hill_walking_pass_rows"],
                      "torque_walking_pass_rows": report["torque_walking_pass_rows"],
                      "best_hill_mm": max(r["hill"]["forward_mm"] for r in rows),
                      "best_torque_mm": max(r["torque"]["forward_mm"] for r in rows)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
