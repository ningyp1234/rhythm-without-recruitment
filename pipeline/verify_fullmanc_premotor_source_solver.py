"""Independently check source-baseline fidelity, adaptive outcomes and path IDs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from probe_fullmanc_premotor_source_solver import NEURONS, OUT, ROW, RUN, WEIGHTS, measure
from source_calibration import ROOT
from source_latest_vnc_hill_interface import compatible_neurons


def main() -> None:
    n = compatible_neurons(pd.read_csv(NEURONS))
    byid = {int(v): i for i, v in enumerate(n.bodyId)}
    report = json.loads((OUT / "report.json").read_text())
    paths = json.loads((OUT / "premotor_proprioceptive_paths.json").read_text())
    coarse = json.loads((OUT / "fixed_step_dose_screen.json").read_text())
    with np.load(OUT / "selected_neural_traces.npz") as a:
        traces, indices, body_ids = a["rates_10ms"], a["selected_indices"], a["selected_body_ids"]
    with np.load(ROOT / f"results/author-fullmanc-128-walking-20260924/run{RUN}_selected_738_neurons.npz") as a:
        published_indices = a["selected_indices"]
        published = a["rates_1ms"][ROW, :, ::10]
    assert traces.shape == (4, 741, 201)
    assert np.array_equal(body_ids, n.bodyId.iloc[indices].astype(np.int64).to_numpy())
    assert np.array_equal(indices[np.searchsorted(indices, published_indices)], published_indices)
    control = traces[0, np.searchsorted(indices, published_indices)]
    rmse = float(np.sqrt(np.mean((control - published) ** 2)))
    max_abs = float(np.max(np.abs(control - published)))
    assert rmse < .001 and max_abs < .02
    assert abs(rmse - report["conditions"][0]["author_baseline_comparison"]["author_selected_738_rate_rmse_hz"]) < 1e-9
    measured = [measure(n, indices, trace, byid) for trace in traces]
    e1 = [m["six_E1_sustained_count"] for m in measured]
    flex = [m["legs"]["lf"]["tibia flex"]["active_over_1_hz"] for m in measured]
    ext = [m["legs"]["lf"]["tibia extend"]["active_over_1_hz"] for m in measured]
    assert e1 == [6, 4, 3, 4] and flex == [0, 2, 4, 4] and ext == [0, 0, 0, 0]
    assert all(not m["joint_six_E1_and_six_tibia_gate"] for m in measured)
    assert report["passed"] and not report["autonomous_walking_achieved"]
    assert not coarse["fixed_step_baseline_matches_author"] and not coarse["dose_ranking_valid"]
    W = sparse.load_npz(WEIGHTS).tocsr().astype(np.int64)
    target = byid[12650]
    path_checks = {}
    for sensory_id in (100531, 163745):
        source = byid[sensory_id]
        product = int((W[source, :] @ W[:, target])[0, 0])
        entry = next(item for item in paths["targets"][0][
            "top_positive_two_edge_signed" if product > 0 else "top_negative_two_edge_signed"]
            if item["body_id"] == sensory_id)
        assert product == entry["two_edge_signed_count_product_sum"]
        path_checks[str(sensory_id)] = product
    assert path_checks["100531"] < 0 < path_checks["163745"]
    checks = {"author_baseline_738_neurons_within_0p001hz_rmse": True,
              "four_adaptive_neural_gates_recomputed": True,
              "fixed_step_bad_baseline_rejected": True,
              "opposite_SNpp50_two_edge_scores_recomputed": True,
              "walking_claim_false": True}
    summary = {"checks": checks, "passed": all(checks.values()),
               "baseline_rmse_hz": rmse, "baseline_max_abs_hz": max_abs,
               "adaptive_E1_counts": e1, "adaptive_lf_flexor_active": flex,
               "adaptive_lf_extensor_active": ext,
               "SNpp50_signed_two_edge_counts": path_checks}
    (OUT / "independent_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
