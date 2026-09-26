"""Screen the author's exact full-MANC parameter ensemble, then drive the body.

The 32 parameter rows and inputs are official Zenodo artifacts.  A fixed-step
screen ranks neural rhythm only.  The selected row is rerun independently with
an adaptive solver before its actual motor-neuron rates drive the free body.
This remains A2 because DNg100 is stimulated artificially.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from cpg_context_diagnostic import rhythm
from homologous_hill_body import HomologousHillBody
from source_calibration import ROOT
from source_latest_vnc_hill_interface import compatible_neurons
from walking_benchmark import score


DATA = ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882"
NEURONS = ROOT / "data/cpg/manc-neurons-20251006.csv.gz"
WEIGHTS = ROOT / "data/cpg/manc-pre-post-20251006.npz"
GRAPH_MANIFEST = ROOT / "data/cpg/manifest-20251006.json"
OUT = ROOT / "results/source-fullmanc-exact-ensemble-20260923"
CORE_TYPES = ["DNg100", "IN17A001", "INXXX466", "IN16B036"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_rates(body, rates):
    # rates is neuron x parameter-row, returning muscle x parameter-row.
    return np.stack([rates[indices].mean(axis=0) for indices in body.hill_units])


def load_parameters():
    with h5py.File(DATA / "neuron_params.h5", "r") as source:
        return {
            "tau": source["tau"][:].T,
            "gain": source["a"][:].T,
            "threshold": source["threshold"][:].T,
            "cap": source["fr_cap"][:].T,
            "input": source["input_currents"][0].T,
            "motor_indices": source["mn_idxs"][:],
            "seeds": source["seeds"][:],
        }


def screen_ensemble(weights, params, body, core_indices):
    count, parameter_rows = params["tau"].shape
    rates = np.zeros((count, parameter_rows), dtype=np.float32)
    core_trace = []
    muscle_trace = []
    dt = 0.001
    started = time.perf_counter()
    for step in range(2000):
        stimulus = params["input"] if step >= 20 else 0.0
        total = stimulus + weights @ rates
        activation = np.maximum(
            params["cap"]
            * np.tanh(
                (params["gain"] / params["cap"])
                * (total - params["threshold"])
            ),
            0.0,
        )
        rates += dt * (activation - rates) / params["tau"]
        if (step + 1) % 10 == 0:
            core_trace.append(rates[core_indices].copy())
            muscle_trace.append(group_rates(body, rates))
    return (
        np.asarray(core_trace),
        np.asarray(muscle_trace),
        time.perf_counter() - started,
    )


def rank_screen(n, core_indices, core_trace, muscle_trace):
    rows = []
    keep = slice(50, None)
    e1_rows = np.flatnonzero(n.type.iloc[core_indices].eq("IN17A001").to_numpy())
    for parameter_row in range(core_trace.shape[2]):
        core_stats = [
            rhythm(core_trace[keep, index, parameter_row], dt=0.01)
            for index in range(len(core_indices))
        ]
        e1_stats = [core_stats[index] for index in e1_rows]
        muscle_stats = [
            rhythm(muscle_trace[keep, index, parameter_row], dt=0.01)
            for index in range(muscle_trace.shape[1])
        ]
        excitation = 1 - np.exp(-np.maximum(muscle_trace[keep, :, parameter_row], 0) / 100)
        rows.append(
            {
                "parameter_row": parameter_row,
                "rhythmic_E1_homologues": sum(x["diagnostic_sustained_rhythm"] for x in e1_stats),
                "rhythmic_core_cells": sum(x["diagnostic_sustained_rhythm"] for x in core_stats),
                "rhythmic_muscle_pools": sum(x["diagnostic_sustained_rhythm"] for x in muscle_stats),
                "active_variable_muscle_pools": int((np.ptp(excitation, axis=0) >= 0.05).sum()),
                "mean_excitation_std": float(excitation.std(axis=0).mean()),
                "mean_muscle_pool_rate_hz": float(muscle_trace[keep, :, parameter_row].mean()),
                "max_muscle_pool_rate_hz": float(muscle_trace[keep, :, parameter_row].max()),
                "E1_frequencies_hz": [x["frequency_hz"] for x in e1_stats],
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "rhythmic_E1_homologues",
            "rhythmic_muscle_pools",
            "active_variable_muscle_pools",
            "mean_excitation_std",
        ],
        ascending=False,
    ).reset_index(drop=True)


def exact_selected(weights, params, selected):
    tau = params["tau"][:, selected]
    gain = params["gain"][:, selected]
    threshold = params["threshold"][:, selected]
    cap = params["cap"][:, selected]
    current = params["input"][:, selected]

    def rhs(t, rates):
        stimulus = current if 0.02 <= t <= 1.999 else 0.0
        total = stimulus + weights @ rates
        activation = np.maximum(
            cap * np.tanh((gain / cap) * (total - threshold)), 0.0
        )
        return (activation - rates) / tau

    started = time.perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, 2.0),
        np.zeros(len(tau)),
        t_eval=np.arange(2001) / 1000,
        method="RK45",
        rtol=2e-6,
        atol=5e-9,
        max_step=0.001,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y, time.perf_counter() - started


def replay_body(n, rates):
    body = HomologousHillBody(n, stiffness=32.0)
    rows, feet, contacts = [], [], []
    excitation, activation, force, positions = [], [], [], []
    for tick in range(1, 201):
        body.step(rates[:, tick * 10], 10)
        telemetry = body.telemetry()
        position = np.asarray(telemetry["position"])
        rows.append(
            {
                "t": tick / 100,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "upright": telemetry["upright"],
                "nonfoot_load_fraction": telemetry["nonfoot_load_fraction"],
            }
        )
        feet.append(telemetry["feet"])
        contacts.append(telemetry["contact"])
        excitation.append(body.hill_excitation.copy())
        activation.append(body.sim.mj_data.act.copy())
        force.append(body.sim.mj_data.actuator_force[body.hill_actuators].copy())
        positions.append(position)
    result = score(rows, feet, contacts)
    result.update(
        native_position_authority_zero=bool(
            np.all(body.sim.mj_model.actuator_gainprm[: len(body.native_joint_dofs)] == 0)
            and np.all(body.sim.mj_model.actuator_biasprm[: len(body.native_joint_dofs)] == 0)
        ),
        max_abs_applied_generalized_force=float(np.max(np.abs(body.sim.mj_data.qfrc_applied))),
        max_abs_external_body_force=float(np.max(np.abs(body.sim.mj_data.xfrc_applied))),
        warnings=int(body.sim.mj_data.warning.number.sum()),
    )
    arrays = {
        "feet": np.asarray(feet),
        "contact": np.asarray(contacts),
        "muscle_excitation": np.asarray(excitation),
        "muscle_activation": np.asarray(activation),
        "muscle_force": np.asarray(force),
        "position": np.asarray(positions),
    }
    return result, arrays, body


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = compatible_neurons(pd.read_csv(NEURONS))
    pre_post = sparse.load_npz(WEIGHTS).tocsr()
    incoming = pre_post.T.tocsr() * 0.03
    params = load_parameters()
    if incoming.shape != params["tau"].shape[:1] * 2:
        raise ValueError("Graph and official parameter dimensions differ")
    body_map = HomologousHillBody(n, stiffness=32.0)
    core_indices = np.flatnonzero(n.type.isin(CORE_TYPES))
    core_trace, muscle_trace, screen_wall = screen_ensemble(
        incoming, params, body_map, core_indices
    )
    ranked = rank_screen(n, core_indices, core_trace, muscle_trace)
    ranked.to_csv(OUT / "neural-screen.csv", index=False)
    selected = int(ranked.iloc[0].parameter_row)

    exact, exact_wall = exact_selected(incoming, params, selected)
    result, arrays, body = replay_body(n, exact)
    result.update(
        run_id=f"official_fullmanc_row_{selected}",
        mode="A2",
        parameter_row=selected,
        artificial_bilateral_DNg100_stimulation=True,
        neurons=len(n),
        directed_pairs=int(pre_post.nnz),
        hill_muscles=len(body.hill_actuators),
        mapped_motor_neurons=len(body.motor_ids),
    )
    np.savez_compressed(
        OUT / "selected-exact-body.npz",
        t=np.arange(2001) / 1000,
        core_indices=core_indices,
        core_rates=exact[core_indices],
        motor_indices=body.motor_ids,
        motor_rates=exact[body.motor_ids],
        **arrays,
    )
    (OUT / "selected-exact-body.json").write_text(json.dumps(result, indent=2) + "\n")
    cut, cut_arrays, _ = replay_body(n, np.zeros_like(exact))
    np.savez_compressed(OUT / "zero-neural-cut.npz", **cut_arrays)
    (OUT / "zero-neural-cut.json").write_text(json.dumps(cut, indent=2) + "\n")

    report = {
        "scope": "Official exact full-MANC 32-row ensemble to free 78-Hill-muscle body",
        "sources": {
            "official_parameter_manifest": {"path": str(DATA / "manifest.json"), "sha256": sha(DATA / "manifest.json")},
            "neurons": {"path": str(NEURONS), "sha256": sha(NEURONS)},
            "weights": {"path": str(WEIGHTS), "sha256": sha(WEIGHTS)},
            "graph_manifest": {"path": str(GRAPH_MANIFEST), "sha256": sha(GRAPH_MANIFEST)},
        },
        "official_condition": {
            "run_id": 33195882,
            "experiment_seed": 424,
            "parameter_rows": 32,
            "stimulated_indices": [59, 282],
            "stimulus_model_current": 380,
        },
        "screen": {
            "method": "1 ms forward Euler, neural ranking only",
            "wall_seconds": screen_wall,
            "selection_rule": "E1 homologues, muscle-pool rhythms, variable pools, excitation variation",
            "selected_parameter_row": selected,
            "selected_metrics": ranked.iloc[0].to_dict(),
        },
        "frozen_exact_evaluation": {
            "method": "SciPy RK45, rtol 2e-6, atol 5e-9, maximum step 1 ms",
            "wall_seconds": exact_wall,
            "physical": result,
            "zero_neural_cut": cut,
        },
        "controller_authority": {
            "artificial_DNg100_current": True,
            "gait_clock": False,
            "future_reference": False,
            "joint_servo": False,
            "root_force": 0,
            "external_body_force": 0,
            "adhesion": False,
        },
        "classification": {
            "official_full_MANC_neural_ensemble": "pass",
            "A2_full_six_leg_neural_motor_interface": "pass" if result["walking_pass"] else "incomplete",
            "A3_neural_closed_loop": "not run; DNg100 command remains artificial",
        },
        "goal_complete": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "neural": report["screen"]["selected_metrics"],
                "forward_mm": result["forward_mm"],
                "steps_per_leg": result["steps_per_leg"],
                "walking_pass": result["walking_pass"],
                "classification": report["classification"],
                "goal_complete": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
