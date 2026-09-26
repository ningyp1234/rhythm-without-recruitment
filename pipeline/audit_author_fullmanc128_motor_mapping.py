"""Separate exact author MN coverage from earlier local-RK mapping counts."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from homologous_hill_body import HomologousHillBody
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from source_calibration import ROOT


RUNS = (30662613, 33195857, 33195876, 33195882)
OUT = ROOT / "results/author-fullmanc-128-walking-20260924"
PRIOR = ROOT / "results/row21-body-diagnostic-20260924"


def main() -> None:
    neurons = compatible_neurons(pd.read_csv(NEURONS))
    body = HomologousHillBody(neurons, stiffness=32.)
    annotated_leg = np.flatnonzero(neurons["motor module"].astype(str).str.strip().ne("").to_numpy())
    motor = np.flatnonzero(neurons["class"].eq("motor neuron").to_numpy())
    assert len(motor) == 732 and len(body.hill_actuators) == 78
    rows = []
    for block, run_id in enumerate(RUNS):
        path = (PRIOR / "author_fullmanc_32_selected_neurons.npz" if run_id == 33195882
                else OUT / f"run{run_id}_selected_738_neurons.npz")
        with np.load(path) as archive:
            selected = archive["selected_indices"]
            trace = archive["rates_1ms"]
        for row in range(32):
            active_selected = np.flatnonzero(trace[row, :, 500:].max(axis=1) > 1.)
            active = selected[active_selected]
            active_motor = np.intersect1d(active, motor)
            active_leg = np.intersect1d(active_motor, annotated_leg)
            mapped = np.intersect1d(active_leg, body.motor_ids)
            rows.append({"global_row": block * 32 + row, "run_id": run_id,
                         "parameter_row": row,
                         "active_motor_neurons": len(active_motor),
                         "active_annotated_leg_motor_neurons": len(active_leg),
                         "mapped_active_annotated_leg_motor_neurons": len(mapped),
                         "unmapped_active_annotated_leg_motor_neurons": len(active_leg)-len(mapped),
                         "unmapped_body_ids": neurons.bodyId.iloc[np.setdiff1d(active_leg, mapped)].astype(int).tolist()})
    old = json.loads((PRIOR / "signal_chain.json").read_text())
    exact = rows[117]
    checks = {"all_128_source_rows": len(rows) == 128,
              "global_117_is_run33195882_row21": exact["run_id"] == 33195882 and exact["parameter_row"] == 21,
              "exact_author_row117_mapped_plus_unmapped": exact["mapped_active_annotated_leg_motor_neurons"]
                  + exact["unmapped_active_annotated_leg_motor_neurons"] == exact["active_annotated_leg_motor_neurons"],
              "local_RK_counts_are_separate_from_author":
                  old["active_annotated_leg_motor_neurons"] != exact["active_annotated_leg_motor_neurons"],
              "all_mapped_counts_bounded": all(x["mapped_active_annotated_leg_motor_neurons"] <= x["active_annotated_leg_motor_neurons"] for x in rows)}
    report = {"scope": "Exact author-saved full-MANC rates, all four runs, after .5s >1Hz annotated leg motor mapping into the unchanged 78-Hill candidate",
              "source": "https://zenodo.org/records/22260924", "rows": rows,
              "author_global_row117_exact": exact,
              "earlier_local_RK_row21_counts": {
                  "active_annotated_leg_motor_neurons": old["active_annotated_leg_motor_neurons"],
                  "mapped_active_annotated_leg_motor_neurons": old["active_annotated_leg_motor_neurons_reaching_muscle"],
                  "unmapped_active_annotated_leg_motor_neurons": old["active_annotated_leg_motor_neurons_without_muscle"]},
              "checks": checks, "passed": all(checks.values())}
    (OUT / "author_fullmanc_128_motor_mapping.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"exact_row117": {k:exact[k] for k in (
        "active_annotated_leg_motor_neurons", "mapped_active_annotated_leg_motor_neurons",
        "unmapped_active_annotated_leg_motor_neurons")},
                      "local_RK_prior": report["earlier_local_RK_row21_counts"],
                      "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
