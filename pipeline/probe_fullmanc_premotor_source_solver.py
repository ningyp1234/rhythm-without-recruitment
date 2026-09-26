"""Prospective full-MANC premotor perturbation with source-aligned RK45.

Artificial currents are a causal diagnostic, never sensory feedback or a
walking controller. The run refuses to use the local sparse graph unless it
exactly matches the author's same-run W and unpruned W_mask.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from cpg_context_diagnostic import rhythm
from screen_author_fullmanc_32 import LEGS
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS, WEIGHTS
from source_latest_vnc_hill_interface import compatible_neurons


RUN = 33195857
ROW = 13
GLOBAL_ROW = 45
DATA = ROOT / f"data/pugliese-zenodo-22260924/full-manc-all-runs/{RUN}"
OUT = ROOT / "results/fullmanc-premotor-source-solver-20260924"
CASES = (
    ("author_DNg100_only", {}),
    ("IN21A004_12650_plus80", {12650: 80.0}),
    ("IN21A004_12650_plus160", {12650: 160.0}),
    ("IN21A004_12650_plus160_IN04B031_17678_plus60", {12650: 160.0, 17678: 60.0}),
)


def source_graph() -> sparse.csr_matrix:
    local = sparse.load_npz(WEIGHTS).tocsr()
    with h5py.File(DATA / "neuron_params.h5") as h:
        original = h["W"][:]
        mask = h["W_mask"][ROW]
    assert original.shape == local.shape == (23532, 23532)
    assert bool(mask.all()), "This row prunes edges; sparse graph cannot replace source W"
    source = sparse.csr_matrix(original)
    assert (source - local).nnz == 0, "Local weights differ from author run"
    return (source.T.tocsr() * np.float32(.03)).astype(np.float32)


def parameters() -> dict[str, np.ndarray]:
    with h5py.File(DATA / "neuron_params.h5") as h:
        return {
            "tau": h["tau"][ROW].astype(np.float64),
            "gain": h["a"][ROW].astype(np.float64),
            "threshold": h["threshold"][ROW].astype(np.float64),
            "cap": h["fr_cap"][ROW].astype(np.float64),
            "current": h["input_currents"][0, ROW].astype(np.float64),
        }


def run(weights: sparse.csr_matrix, p: dict[str, np.ndarray],
        additions: dict[int, float], byid: dict[int, int], selected: np.ndarray) -> tuple[np.ndarray, dict]:
    current = p["current"].copy()
    for body_id, dose in additions.items():
        current[byid[body_id]] += dose
    def derivative(t: float, rates: np.ndarray) -> np.ndarray:
        stimulus = current if .02 <= t <= 1.999 else 0.0
        total = stimulus + weights @ rates
        activation = np.maximum(p["cap"] * np.tanh((p["gain"] / p["cap"]) *
                                                  (total - p["threshold"])), 0)
        return (activation - rates) / p["tau"]
    started = time.perf_counter()
    sol = solve_ivp(derivative, (0., 2.), np.zeros(len(current)),
                    t_eval=np.arange(201) / 100, method="RK45",
                    rtol=2e-6, atol=5e-9, max_step=.001)
    if not sol.success:
        raise RuntimeError(sol.message)
    values = sol.y[selected].astype(np.float32)
    assert np.isfinite(values).all()
    return values, {"wall_seconds": time.perf_counter() - started,
                    "solver": "scipy RK45 / Dormand-Prince 5(4)",
                    "rtol": 2e-6, "atol": 5e-9, "max_step_seconds": .001,
                    "source_solver_family": "author Diffrax Dopri5; independent implementation, compared with author's archived baseline"}


def measure(n: pd.DataFrame, selected: np.ndarray, trace: np.ndarray,
            byid: dict[int, int]) -> dict:
    lookup = np.full(len(n), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected))
    e1 = np.flatnonzero(n.type.eq("IN17A001").to_numpy())
    e1_count = sum(rhythm(trace[lookup[t], 50:200], dt=.01)["diagnostic_sustained_rhythm"]
                   for t in e1)
    legs = {}
    for leg, neu, side in LEGS:
        modules = {}
        for name in ("tibia flex", "tibia extend"):
            ids = np.flatnonzero((n.somaNeuromere.eq(neu) & n.side.eq(side)
                                  & n["motor module"].eq(name)).to_numpy())
            values = trace[lookup[ids], 50:200]
            mean = values.mean(axis=0)
            modules[name] = {"neurons": len(ids),
                             "active_over_1_hz": int((values.max(axis=1) > 1).sum()),
                             "mean_pool_range_hz": float(np.ptp(mean)),
                             "mean_pool_rate_hz": float(values.mean())}
        f, x = modules["tibia flex"], modules["tibia extend"]
        modules["both_variable"] = bool(f["active_over_1_hz"] and x["active_over_1_hz"]
                                        and f["mean_pool_range_hz"] >= .5
                                        and x["mean_pool_range_hz"] >= .5)
        legs[leg] = modules
    return {
        "six_E1_sustained_count": int(e1_count),
        "left_front_tibia_both": legs["lf"]["both_variable"],
        "six_leg_tibia_both_count": sum(leg["both_variable"] for leg in legs.values()),
        "joint_six_E1_and_six_tibia_gate": bool(e1_count == 6 and all(x["both_variable"] for x in legs.values())),
        "named_cell_mean_hz_after_0p5s": {str(k): float(trace[lookup[byid[k]], 50:200].mean())
                                            for k in (12650, 10827, 17678, 12134, 12704)},
        "legs": legs,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = compatible_neurons(pd.read_csv(NEURONS))
    byid = {int(v): i for i, v in enumerate(n.bodyId)}
    assert len(n) == 23532 and n.type.iloc[byid[12650]] == "IN21A004"
    assert n.type.iloc[byid[17678]] == "IN04B031"
    selected = np.unique(np.r_[np.flatnonzero(n.type.eq("IN17A001").to_numpy()),
                               np.flatnonzero(n["class"].eq("motor neuron").to_numpy()),
                               [byid[k] for k in (12650, 10827, 17678)]])
    assert len(selected) == 741
    weights = source_graph()
    p = parameters()
    with np.load(ROOT / f"results/author-fullmanc-128-walking-20260924/run{RUN}_selected_738_neurons.npz") as a:
        original_ids = a["selected_indices"]
        original = a["rates_1ms"][ROW, :, ::10]
    reports = []
    traces = []
    for name, additions in CASES:
        print("running", name, flush=True)
        rates, numerical = run(weights, p, additions, byid, selected)
        metrics = measure(n, selected, rates, byid)
        comparison = None
        if not additions:
            take = np.searchsorted(selected, original_ids)
            assert np.array_equal(selected[take], original_ids)
            diff = rates[take] - original
            comparison = {"author_selected_738_rate_rmse_hz": float(np.sqrt(np.mean(diff ** 2))),
                          "author_selected_738_rate_max_abs_hz": float(np.max(np.abs(diff))),
                          "author_original_six_E1_count": 6,
                          "author_original_left_front_tibia_both": False,
                          "qualitative_E1_and_left_tibia_match_author": bool(
                              metrics["six_E1_sustained_count"] == 6 and not metrics["left_front_tibia_both"])}
        reports.append({"case": name, "added_artificial_input_current_by_body_id": additions,
                        "numerics": numerical, "author_baseline_comparison": comparison,
                        "neural": metrics})
        traces.append(rates)
        print(name, "E1", metrics["six_E1_sustained_count"], "left tibia both",
              metrics["left_front_tibia_both"], "six tibia both",
              metrics["six_leg_tibia_both_count"], flush=True)
    np.savez_compressed(OUT / "selected_neural_traces.npz", rates_10ms=np.stack(traces),
                        selected_indices=selected,
                        selected_body_ids=n.bodyId.iloc[selected].astype(np.int64).to_numpy(),
                        case_names=np.array([x[0] for x in CASES]))
    checks = {
        "source_graph_bit_exact_and_unpruned": True,
        "original_738_source_cells_matched": reports[0]["author_baseline_comparison"]["qualitative_E1_and_left_tibia_match_author"],
        "four_23k_neuron_2s_adaptive_cases": len(reports) == 4,
        "all_traces_finite": bool(np.isfinite(traces).all()),
    }
    report = {"source": "https://zenodo.org/records/22260924", "run_id": RUN,
              "source_parameter_row": ROW, "figure4_global_row": GLOBAL_ROW,
              "scope": "Synthetic upstream-current causal probe in the full author MANC graph, not natural sensory feedback or autonomous walking",
              "conditions": reports, "checks": checks, "passed": all(checks.values()),
              "autonomous_walking_achieved": False, "goal_complete": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "results": [{"case": x["case"],
        "author_comparison": x["author_baseline_comparison"],
        "E1": x["neural"]["six_E1_sustained_count"],
        "left_tibia_both": x["neural"]["left_front_tibia_both"],
        "six_tibia_both": x["neural"]["six_leg_tibia_both_count"]} for x in reports]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
