"""Explain silent left-front tibia MNs using exact published full-MANC traces.

For each selected original parameter row, reconstruct 10-ms whole-VNC rates
from the author's sparse tensor. Apply that run's W, mask, multipliers and
neuron thresholds to the published rate equation. This is a read-only audit:
the calculated input does not create a new simulated trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons


SELECTED = ((30662613, 2, 2), (33195857, 4, 36), (33195857, 6, 38),
            (33195857, 7, 39), (33195857, 13, 45),
            (33195876, 8, 72), (33195882, 21, 117))
SOURCE = ROOT / "data/pugliese-zenodo-22260924"
OUT = ROOT / "results/author-motor-input-margins-20260924"


def folder(run_id: int) -> Path:
    tail = "full-manc-run33195882" if run_id == 33195882 else f"full-manc-all-runs/{run_id}"
    return SOURCE / tail


def reconstruction(source: Path, parameter_rows: list[int]) -> dict[int, np.ndarray]:
    """Extract author rates at every 10th original 1-ms sample."""
    with np.load(source, allow_pickle=False) as archive:
        shape = tuple(int(x) for x in archive["shape"])
        assert shape == (1, 32, 23532, 2001), shape
        assert float(archive["fill_value"]) == 0.0
        coords = archive["coords"]
        values = archive["data"]
    assert coords.shape[0] == 4 and np.isfinite(values).all()
    result = {}
    for row in parameter_rows:
        take = (coords[0] == 0) & (coords[1] == row) & (coords[3] % 10 == 0)
        rates = np.zeros((shape[2], 201), dtype=np.float32)
        rates[coords[2, take], coords[3, take] // 10] = values[take]
        assert np.isfinite(rates).all()
        assert np.count_nonzero(rates) == int(np.count_nonzero(take))
        result[row] = rates
    return result


def summarize(n: pd.DataFrame, params: h5py.File, run_id: int, row: int,
              global_row: int, rates: np.ndarray, targets: np.ndarray) -> dict:
    thresholds = params["threshold"][row, targets]
    gain = params["a"][row, targets]
    cap = params["fr_cap"][row, targets]
    tau = params["tau"][row, targets]
    input_current = params["input_currents"][0, row, targets]
    # Source reweight_connectivity transposes the pre-by-post W, then scales
    # the positive and negative entries separately.
    W = np.column_stack([params["W"][:, int(t)] for t in targets])
    mask = np.column_stack([params["W_mask"][row, :, int(t)] for t in targets])
    W *= mask
    assert np.isfinite(W).all()
    config = json.loads((ROOT / "results/author-fullmanc-128-walking-20260924/author_fullmanc_128_neural_screen.json").read_text())
    assert global_row == next(x["global_row"] for x in config["rows"] if x["run_id"] == run_id and x["parameter_row"] == row)
    # All four publisher configurations use 0.03 for both signs. Require the
    # saved YAML to agree before using this value; no guessed weight scaling.
    cfg_paths = list(folder(run_id).rglob("*.yaml"))
    yaml_text = "\n".join(p.read_text() for p in cfg_paths)
    assert "excitatoryMultiplier: 0.03" in yaml_text
    assert "inhibitoryMultiplier: 0.03" in yaml_text
    positive = (0.03 * np.maximum(W, 0)).T @ rates
    negative = (0.03 * np.minimum(W, 0)).T @ rates
    synaptic = positive + negative
    stimulus = np.zeros_like(synaptic)
    stimulus[:, 2:200] = input_current[:, None]
    margin = synaptic + stimulus - thresholds[:, None]
    activation = np.maximum(cap[:, None] * np.tanh((gain / cap)[:, None] * margin), 0)
    # Expected derivative from exactly the source's published ODE; finite
    # differences at 10 ms are checked only away from discontinuities.
    expected_dr = (activation - rates[targets]) / tau[:, None]
    observed_dr = np.gradient(rates[targets], .01, axis=1)
    active = (rates[targets, 50:200] > 1).any(axis=1)
    output = []
    dynamic_checks = []
    for j, t in enumerate(targets):
        segment = slice(50, 200)
        r = rates[t, segment]
        m = margin[j, segment]
        dynamic = rates[t, 55:195] > 1
        if int(dynamic.sum()) >= 10 and float(np.std(observed_dr[j, 55:195][dynamic])) > 1:
            dynamic_checks.append({
                "body_id": int(n.bodyId.iloc[t]),
                "samples": int(dynamic.sum()),
                "source_ode_vs_10ms_derivative_correlation": float(np.corrcoef(
                    expected_dr[j, 55:195][dynamic], observed_dr[j, 55:195][dynamic])[0, 1]),
                "median_abs_derivative_difference_hz_per_s": float(np.median(np.abs(
                    expected_dr[j, 55:195][dynamic] - observed_dr[j, 55:195][dynamic]))),
            })
        output.append({
            "index": int(t), "body_id": int(n.bodyId.iloc[t]),
            "module": str(n["motor module"].iloc[t]),
            "rate_max_hz": float(r.max()), "rate_mean_hz": float(r.mean()),
            "threshold": float(thresholds[j]),
            "mean_positive_drive": float(positive[j, segment].mean()),
            "mean_negative_drive": float(negative[j, segment].mean()),
            "mean_drive_minus_threshold": float(m.mean()),
            "max_drive_minus_threshold": float(m.max()),
            "above_threshold_fraction": float((m > 0).mean()),
            "source_masked_nonzero_edges": int(np.count_nonzero(W[:, j])),
            "source_mask_removed_nonzero_edges": int(np.count_nonzero(params["W"][:, int(t)] * (~mask[:, j]))),
            "median_ode_derivative_residual_hz_per_s": float(np.median(np.abs(expected_dr[j, 55:195] - observed_dr[j, 55:195]))),
        })
    flex = [x for x in output if x["module"] == "tibia flex"]
    extend = [x for x in output if x["module"] == "tibia extend"]
    return {
        "global_row": global_row, "run_id": run_id, "parameter_row": row,
        "author_source_rate_shape": [23532, 201],
        "left_front_tibia_target_count": len(targets),
        "left_front_tibia_active_over_1hz": int(active.sum()),
        "flexor_active_over_1hz": sum(x["rate_max_hz"] > 1 for x in flex),
        "extensor_active_over_1hz": sum(x["rate_max_hz"] > 1 for x in extend),
        "all_silent_cells_subthreshold_entire_0p5_to_2s": all(
            x["max_drive_minus_threshold"] <= 1e-5 for x in output if x["rate_max_hz"] < 1e-4),
        "median_abs_ode_derivative_residual_hz_per_s": float(np.median(np.abs(expected_dr[:, 55:195] - observed_dr[:, 55:195]))),
        "active_rate_derivative_checks": dynamic_checks,
        "cells": output,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = compatible_neurons(pd.read_csv(NEURONS))
    targets = np.flatnonzero(((n.somaNeuromere == "T1") & (n.side == "L") &
                              n["motor module"].isin(("tibia flex", "tibia extend"))).to_numpy())
    assert len(n) == 23532 and len(targets) == 17
    results = []
    contrast_traces: dict[int, np.ndarray] = {}
    for run_id in dict.fromkeys(x[0] for x in SELECTED):
        cases = [x for x in SELECTED if x[0] == run_id]
        print(f"reading author sparse tensor {run_id}: {[x[1] for x in cases]}", flush=True)
        traces = reconstruction(folder(run_id) / "DNg100_Stim_fullManc_Rs.npz", [x[1] for x in cases])
        if run_id == 33195857:
            contrast_traces[45] = traces[13]
        if run_id == 33195882:
            contrast_traces[117] = traces[21]
        with h5py.File(folder(run_id) / "neuron_params.h5") as params:
            for _, row, global_row in cases:
                results.append(summarize(n, params, run_id, row, global_row, traces[row], targets))
                print(f"row {global_row}: tibia active {results[-1]['left_front_tibia_active_over_1hz']}/17, "
                      f"silent cells source-equation subthreshold={results[-1]['all_silent_cells_subthreshold_entire_0p5_to_2s']}", flush=True)
    # Two contrasting source conditions use the same incoming graph for the
    # relevant MNs. Record exactly what differs without implying causality
    # from a one-step retrospective comparison.
    with h5py.File(folder(33195857) / "neuron_params.h5") as a, h5py.File(folder(33195882) / "neuron_params.h5") as b:
        wa = np.column_stack([a["W"][:, int(t)] for t in targets])
        wb = np.column_stack([b["W"][:, int(t)] for t in targets])
        ma = np.column_stack([a["W_mask"][13, :, int(t)] for t in targets])
        mb = np.column_stack([b["W_mask"][21, :, int(t)] for t in targets])
        comparison = {
            "contrasted_global_rows": [45, 117],
            "incoming_W_columns_bit_identical": bool(np.array_equal(wa, wb)),
            "incoming_mask_columns_bit_identical": bool(np.array_equal(ma, mb)),
            "nonzero_edges_masked_out_row45": int(np.count_nonzero((~ma) & (wa != 0))),
            "nonzero_edges_masked_out_row117": int(np.count_nonzero((~mb) & (wb != 0))),
            "mean_threshold_row45": float(a["threshold"][13, targets].mean()),
            "mean_threshold_row117": float(b["threshold"][21, targets].mean()),
            "max_absolute_threshold_difference": float(np.max(np.abs(
                a["threshold"][13, targets] - b["threshold"][21, targets]))),
        }
        mean_rate_difference = (contrast_traces[117][:, 50:200].mean(axis=1)
                                - contrast_traces[45][:, 50:200].mean(axis=1))
        incoming = []
        for body_id in (12134, 12704):
            target = int(n.index[n.bodyId == body_id][0])
            delta = .03 * a["W"][:, target] * mean_rate_difference
            def entries(indices):
                return [{"body_id": int(n.bodyId.iloc[i]), "type": str(n.type.iloc[i]),
                         "class": str(n["class"].iloc[i]),
                         "change_in_mean_input": float(delta[i]),
                         "mean_rate_row45_hz": float(contrast_traces[45][i, 50:200].mean()),
                         "mean_rate_row117_hz": float(contrast_traces[117][i, 50:200].mean())}
                        for i in indices]
            incoming.append({
                "motor_body_id": body_id,
                "net_change_in_mean_synaptic_input": float(delta.sum()),
                "largest_increased_inputs": entries(np.argsort(-delta)[:10]),
                "largest_decreased_inputs": entries(np.argsort(delta)[:10]),
            })
        comparison["retrospective_incoming_contributors_not_causal_targets"] = incoming
    report = {
        "source": "https://zenodo.org/records/22260924",
        "method": "Author archived sparse R at 10 ms, author same-run W/W_mask/a/fr_cap/tau/threshold/input, source ODE reweighting and activation; retrospective input audit, not a new simulation",
        "cases": results,
        "row45_vs_row117_incoming_graph": comparison,
        "checks": {
            "all_seven_original_rows": len(results) == 7,
            "all_17_left_front_tibia_mns": all(x["left_front_tibia_target_count"] == 17 for x in results),
            "silent_cells_below_threshold_by_source_equation": all(x["all_silent_cells_subthreshold_entire_0p5_to_2s"] for x in results),
            "active_author_trace_matches_source_rate_equation_at_10ms": all(
                x["source_ode_vs_10ms_derivative_correlation"] > .90
                for r in results for x in r["active_rate_derivative_checks"]),
            "contrasting_conditions_same_incoming_graph": comparison["incoming_W_columns_bit_identical"]
                and comparison["incoming_mask_columns_bit_identical"]
                and comparison["nonzero_edges_masked_out_row45"] == 0
                and comparison["nonzero_edges_masked_out_row117"] == 0,
        },
        "autonomous_walking_achieved": False,
    }
    report["passed"] = all(report["checks"].values())
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": report["checks"], "case_summary": [
        {k: x[k] for k in ("global_row", "left_front_tibia_active_over_1hz", "median_abs_ode_derivative_residual_hz_per_s")}
        for x in results]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
