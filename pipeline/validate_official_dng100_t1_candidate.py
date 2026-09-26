"""Validate the one prospectively discovered T1 DNg100 condition on 112 held-out rows.

Loads the *exact* author grid-90 checkpoint, not a reconstructed matrix. The
condition was selected solely on neuron parameter rows 0-15; rows 16-127 are
held out. A rate-network gate is not evidence of physical walking.
"""
from __future__ import annotations

import hashlib
import json
import zlib
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from screen_official_dng100_t1_grid import BASE, OUT, TABLE, make_pools, evaluate, source_matrix

DATA = BASE / "neuron_params_candidate_grid90.h5"
EXC, GLU, INH = .045, .045, .01333
CHECKPOINT_CRC32 = 0x1d770f31
ROWS = 128


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = DATA.read_bytes()
    n = pd.read_csv(TABLE).fillna("")
    with h5py.File(DATA) as h:
        original = h["W"][:]
        p = {name: h[key][:].T.astype(np.float32) for name, key in
             (("tau", "tau"), ("gain", "a"), ("threshold", "threshold"), ("cap", "fr_cap"))}
        drive = h["input_currents"][0].T.astype(np.float32)
        masks_true = all(bool(np.all(h["W_mask"][j:j+4])) for j in range(0, ROWS, 4))
    with h5py.File(BASE / "neuron_params.h5") as base:
        w0 = sparse.coo_matrix(base["W"][:])
        neuron_params_match = all(np.array_equal(base[key][:], p[name].T) for name, key in
                                  (("tau", "tau"), ("gain", "a"), ("threshold", "threshold"), ("cap", "fr_cap")))
    modeled = source_matrix(w0, n.predictedNt.eq("glutamate").to_numpy(), EXC, GLU, INH)
    # source_matrix includes global sign multipliers; compare with the exact
    # archived W after applying the same source sign operation.
    archived_reweighted = sparse.csr_matrix(
        np.maximum(original.T, 0) * EXC + np.minimum(original.T, 0) * INH)
    weight_diff = modeled - archived_reweighted
    exact_W_match = weight_diff.nnz == 0 or np.max(np.abs(weight_diff.data)) <= 5e-5
    pools = make_pools(n)
    e1_left = int(np.flatnonzero(n.bodyId.eq(10707).to_numpy())[0])
    e1_right = int(np.flatnonzero(n.bodyId.eq(10690).to_numpy())[0])
    selected = np.unique(np.r_[[31, e1_left, e1_right], *pools.values()])
    lookup = {int(v): i for i, v in enumerate(selected)}
    rates = np.zeros_like(p["tau"])
    trace = np.empty((200, len(selected), ROWS), np.float32)
    for step in range(2000):
        total = (drive if step >= 20 else 0) + archived_reweighted @ rates
        active = np.maximum(p["cap"] * np.tanh((p["gain"] / p["cap"]) *
                                              (total - p["threshold"])), 0)
        rates += np.float32(.001) * (active - rates) / p["tau"]
        if (step + 1) % 10 == 0:
            trace[(step + 1) // 10 - 1] = rates[selected]
    rows = []
    for j in range(ROWS):
        item = {"parameter_row": j, "split": "discovery" if j < 16 else "heldout",
                **evaluate(trace, lookup, e1_left, e1_right, pools, j)}
        rows.append(item)
    saved = np.load(OUT / "best_discovery_trace.npz")
    replay_difference = saved["rates_10ms"] - trace[:, :, 10]
    replay_rmse = float(np.sqrt(np.mean(replay_difference.astype(np.float64) ** 2)))
    replay_max = float(np.max(np.abs(replay_difference)))
    discovery_reproduced = bool(replay_rmse < .01 and replay_max < .1)
    checks = {"exact_author_checkpoint_crc32": zlib.crc32(raw) == CHECKPOINT_CRC32,
              "all_128_source_parameter_rows": p["tau"].shape == (4604, 128),
              "all_128_author_masks_true": masks_true,
              "single_DNg100_input_250": bool(np.count_nonzero(drive) == 128 and np.all(drive[31] == 250)),
              "neuron_parameters_same_as_seed145_baseline": neuron_params_match,
              "reconstructed_source_W_matches_exact_archived_condition": bool(exact_W_match),
              "predeclared_discovery_trace_reproduced": discovery_reproduced,
              "all_rates_finite": bool(np.isfinite(trace).all())}
    heldout = rows[16:]
    report = {"source": "https://zenodo.org/records/22260924",
              "archive": "DNg100_Stim__edf2_a.zip",
              "checkpoint_member": json.loads((BASE / "archive_grid.json").read_text())["entries"][90]["archive_member"],
              "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
              "selected_grid_index": 90, "selection_only_used_parameter_rows": list(range(16)),
              "heldout_parameter_rows": list(range(16, 128)),
              "source_matrix_float32_replay_rmse_hz": replay_rmse,
              "source_matrix_float32_replay_max_abs_hz": replay_max,
              "multipliers": {"excitatory": EXC, "glutamate": GLU, "inhibitory": INH},
              "neurons": len(n), "signed_directed_pairs": int(np.count_nonzero(original)),
              "discovery_joint_gate_count": sum(r["left_joint_gate"] for r in rows[:16]),
              "heldout_joint_gate_count": sum(r["left_joint_gate"] for r in heldout),
              "heldout_E1_left_count": sum(r["E1_left_sustained"] for r in heldout),
              "heldout_both_tibia_variable_count": sum(r["sides"]["LHS"]["tibia_both_variable"] for r in heldout),
              "heldout_tibia_antiphase_count": sum(r["sides"]["LHS"]["tibia_antiphase"] for r in heldout),
              "rows": rows, "checks": checks, "passed": all(checks.values()),
              "body_walking_achieved": False}
    (OUT / "candidate_heldout_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / "candidate_all128_neural_traces.npz", rates_10ms=trace,
                        selected_indices=selected, selected_body_ids=n.bodyId.iloc[selected].astype(int).to_numpy())
    print(json.dumps({k: report[k] for k in ("discovery_joint_gate_count", "heldout_joint_gate_count",
                                             "heldout_E1_left_count", "heldout_both_tibia_variable_count",
                                             "heldout_tibia_antiphase_count", "checks")}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
