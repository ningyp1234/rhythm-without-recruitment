"""Probe how an uncalibrated equal-mean MN pool readout affects body replay.

All neural trajectories are original author-saved MANC outputs. Max and sum
pooling are sensitivity bounds, not physiological motor-unit calibrations.
They must not be interpreted as valid autonomous neural walking even if they
pass a body gate: DNg100 input is artificial and all trials are open-loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from homologous_hill_body import HomologousHillBody
from replay_author_fullmanc_32_bodies import replay
from screen_author_fullmanc_all128 import RUNS
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from walking_benchmark import score


OUT = ROOT / "results/author-motor-pool-aggregation-20260924"
PREVIOUS = ROOT / "results/author-fullmanc-128-walking-20260924"
SELECTED_ROWS = (2, 36, 38, 39, 45, 72, 117)
MODES = ("mean", "max", "sum")


def selected_path(global_row: int) -> Path:
    rid = RUNS[global_row // 32]
    if rid == 33195882:
        return ROOT / "results/row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz"
    return PREVIOUS / f"run{rid}_selected_738_neurons.npz"


def original_rates(global_row: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(selected_path(global_row), allow_pickle=False) as archive:
        selected = archive["selected_indices"].copy()
        rates = archive["rates_1ms"][global_row % 32].copy()
    assert rates.shape == (738, 2001) and len(selected) == 738
    return selected, rates


def pool_trace(body: HomologousHillBody, selected: np.ndarray,
               source_rates: np.ndarray, mode: str) -> np.ndarray:
    pools = []
    for group in body.hill_units:
        indices = np.searchsorted(selected, group)
        assert np.array_equal(selected[indices], group)
        subset = source_rates[indices, 10:2001:10]
        if mode == "mean":
            pooled = subset.mean(axis=0)
        elif mode == "max":
            pooled = subset.max(axis=0)
        elif mode == "sum":
            pooled = subset.sum(axis=0)
        else:
            raise ValueError(mode)
        pools.append(pooled)
    return np.asarray(pools, np.float64)


def body_replay(body: HomologousHillBody, pooled_rates: np.ndarray) -> tuple[dict, dict]:
    body.reset()
    model, data = body.sim.mj_model, body.sim.mj_data
    physics_per_tick = round(0.01 / body.sim.timestep)
    assert abs(physics_per_tick * body.sim.timestep - 0.01) < 1e-12
    rows, feet, contacts = [], [], []
    commands, activations, forces, positions = [], [], [], []
    upright, nonfoot = [], []
    for tick in range(200):
        body.hill_rates = pooled_rates[:, tick].copy()
        body.hill_excitation = 1 - np.exp(-np.maximum(body.hill_rates, 0) / 100)
        data.ctrl[:] = 0
        data.ctrl[body.hill_actuators] = body.hill_excitation
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        for _ in range(physics_per_tick):
            mujoco.mj_step(model, data)
        state = body.telemetry()
        p = np.asarray(state["position"])
        rows.append({"t": (tick + 1) / 100, "x": p[0], "y": p[1], "z": p[2],
                     "upright": state["upright"],
                     "nonfoot_load_fraction": state["nonfoot_load_fraction"]})
        feet.append(state["feet"])
        contacts.append(state["contact"])
        commands.append(body.hill_excitation.copy())
        # Native joint actuators have no activation state; data.act is Hill-only.
        activations.append(data.act.copy())
        forces.append(data.actuator_force[body.hill_actuators].copy())
        positions.append(p.copy())
        upright.append(float(state["upright"]))
        nonfoot.append(float(state["nonfoot_load_fraction"]))
    result = score(rows, feet, contacts)
    result["physics_warnings"] = int(data.warning.number.sum())
    result["zero_native_actuator_gain"] = bool(np.all(model.actuator_gainprm[:len(body.native_joint_dofs)] == 0))
    result["no_applied_force"] = bool(np.all(data.qfrc_applied == 0) and np.all(data.xfrc_applied == 0))
    arrays = {"pooled_rates": pooled_rates, "excitation": np.asarray(commands),
              "activation": np.asarray(activations), "muscle_force": np.asarray(forces),
              "position": np.asarray(positions), "feet": np.asarray(feet),
              "contact": np.asarray(contacts), "upright": np.asarray(upright),
              "nonfoot_load_fraction": np.asarray(nonfoot)}
    return result, arrays


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    neurons = compatible_neurons(pd.read_csv(NEURONS))
    body = HomologousHillBody(neurons, stiffness=32)
    prior = json.loads((PREVIOUS / "author_fullmanc_128_body_screen.json").read_text())
    results = []
    for global_row in SELECTED_ROWS:
        selected, rate = original_rates(global_row)
        for mode in MODES:
            pooled = pool_trace(body, selected, rate, mode)
            result, arrays = body_replay(body, pooled)
            result.update(global_row=global_row, run_id=RUNS[global_row // 32],
                          local_parameter_row=global_row % 32, pooling=mode,
                          peak_pooled_rate_hz=float(pooled.max()),
                          pools_ever_over_1hz=int(np.count_nonzero(pooled.max(axis=1) > 1)))
            if global_row in (45, 117):
                np.savez_compressed(OUT / f"row{global_row}_{mode}.npz", **arrays)
            results.append(result)
            print(global_row, mode, result["forward_mm"], result["steps_per_leg"],
                  result["walking_pass"], flush=True)
    mean_matches_old = all(
        abs(next(x for x in results if x["global_row"] == i and x["pooling"] == "mean")["forward_mm"]
            - prior["rows"][i]["hill"]["forward_mm"]) < 1e-8
        and next(x for x in results if x["global_row"] == i and x["pooling"] == "mean")["steps_per_leg"]
            == prior["rows"][i]["hill"]["steps_per_leg"]
        for i in SELECTED_ROWS)
    checks = {"seven_predeclared_source_rows": len(results) == 7 * 3,
              "mean_replays_exact_prior_body": mean_matches_old,
              "pool_hierarchy": all(next(x for x in results if x["global_row"] == i and x["pooling"] == "mean")["peak_pooled_rate_hz"]
                                    <= next(x for x in results if x["global_row"] == i and x["pooling"] == "max")["peak_pooled_rate_hz"]
                                    <= next(x for x in results if x["global_row"] == i and x["pooling"] == "sum")["peak_pooled_rate_hz"]
                                    for i in SELECTED_ROWS),
              "no_physics_warnings": all(x["physics_warnings"] == 0 for x in results),
              "muscle_only_no_external_force": all(x["zero_native_actuator_gain"] and x["no_applied_force"] for x in results)}
    report = {"scope": "A2 readout sensitivity: seven author-saved open-loop DNg100 full-MANC rows, same 78-muscle free body and 100Hz transduction; max and sum are uncalibrated bounds, not measured motor units",
              "source": "https://zenodo.org/records/22260924",
              "source_rows": list(SELECTED_ROWS), "pooling_modes": list(MODES),
              "results": results, "checks": checks, "passed": all(checks.values()),
              "autonomous_walking_achieved": False, "goal_complete": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks,
                      "physical_passes": [(x["global_row"], x["pooling"]) for x in results if x["walking_pass"]],
                      "max_forward_mm_by_mode": {mode: max(x["forward_mm"] for x in results if x["pooling"] == mode) for mode in MODES}}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
