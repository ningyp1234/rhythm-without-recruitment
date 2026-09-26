"""Independently verify source neural integration and saved body trajectories."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from evaluate_official_running_dynamic_replay import CONTACT_PATH, INVERSE_PATH, MODEL_TO_MM, ROOT
from calibrate_official_running_contact_compliance import make_model
from train_official_running_proprioceptive_policy import build_sequences, cycles
from train_official_running_endurance_policy import best_cycle, tile_cycle
import train_official_running_cpg_policy as cp
from source_fullmanc_exact_ensemble import NEURONS, WEIGHTS, DATA, load_parameters

OUT = ROOT / "results/full-manc-phase-driven-walking-20260924"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((OUT / "report.json").read_text())
    saved = np.load(OUT / "body_and_neural_rollouts.npz")
    assert sha256(NEURONS) == report["sources"]["neurons_sha256"]
    assert sha256(WEIGHTS) == report["sources"]["weights_sha256"]
    assert sha256(DATA / "neuron_params.h5") == report["sources"]["official_neuron_params_sha256"]
    model, ground, claws, _ = make_model(.005)
    sequences = build_sequences(model, ground, claws, np.load(INVERSE_PATH), np.load(CONTACT_PATH))
    ck = torch.load(ROOT / "results/official-running-cpg-recovery-20260924/checkpoint.pt", map_location="cpu", weights_only=False)
    source = next(item for item in sequences if item["key"] == ck["source_bout"])
    _, start, period, _, _ = best_cycle(source)
    canonical = tile_cycle(source, start, period)
    canonical["cycle_period"] = period
    standard, _, _ = cp.cycle_from_phase(canonical, 0)
    references = {item["key"]: item["reference"] for item in sequences if item["split"] == "test"}
    references["canonical"] = standard
    params = load_parameters()
    raw = sparse.load_npz(WEIGHTS).tocsr()
    matrix = raw.T.tocsr() * .03
    row_number = report["neural_source"]["parameter_row"]
    tau = params["tau"][:, row_number]
    gain = params["gain"][:, row_number]
    threshold = params["threshold"][:, row_number]
    cap = params["cap"][:, row_number]
    current = params["input"][:, row_number]
    max_neural_error = 0.0
    for entry in report["rows"]:
        prefix = entry["body_state"] + "_" + entry["condition"] + "_"
        qpos = saved[prefix + "qpos"]
        body = saved[prefix + "body_contact"]
        foot = saved[prefix + "foot_ground_contact"]
        foot_z = saved[prefix + "foot_z_mm"]
        phase = saved[prefix + "phase"]
        e1 = saved[prefix + "e1_rates_hz"]
        motor = saved[prefix + "motor_rates_hz"]
        initial = saved[prefix + "initial_full_neural_rates"]
        terminal = saved[prefix + "terminal_full_neural_rates"]
        assert len(qpos) == len(body) == len(foot) == len(phase) == len(e1) == len(motor) == 1601
        assert math.isclose((len(qpos) - 1) / 800, 2.0, abs_tol=1e-12)
        assert initial.shape == terminal.shape == (23532,)
        assert np.isfinite(qpos).all() and np.isfinite(initial).all() and np.isfinite(terminal).all()
        assert np.isfinite(e1).all() and np.isfinite(motor).all()
        reference = references[entry["body_state"]]
        direction = reference[-1, :2] - reference[0, :2]
        direction /= max(np.linalg.norm(direction), 1e-12)
        forward_mm = float(np.dot(qpos[-1, :2] - qpos[0, :2], direction) * MODEL_TO_MM)
        assert math.isclose(forward_mm, entry["distance_2s_mm"], abs_tol=1e-5)
        assert math.isclose(float(body.mean()), entry["body_contact_fraction"], abs_tol=1e-12)
        assert cycles(foot_z) == entry["cycles_per_leg"]
        phase_cycles = float((np.unwrap(2 * np.pi * phase / period)[-1] - np.unwrap(2 * np.pi * phase / period)[0]) / (2 * np.pi))
        assert math.isclose(phase_cycles, entry["neural_phase_cycles"], abs_tol=1e-12)
        assert math.isclose(float(motor.mean()), entry["mean_motor_neuron_rate_hz"], abs_tol=1e-6)
        assert int(np.count_nonzero(motor.max(axis=0) > 1.0)) == entry["source_motor_neurons_ever_above_1_hz"]
        assert motor.shape[1] == entry["source_motor_neuron_count"]
        if entry["condition"] == "neural_state_frozen":
            assert np.array_equal(initial, terminal)
        else:
            rates = initial.astype(np.float64).copy()
            for _ in range(2000):
                drive = 0.0 if entry["condition"] == "DNg100_drive_removed" else current
                total = drive + matrix @ rates
                activation = np.maximum(cap * np.tanh((gain / cap) * (total - threshold)), 0.0)
                rates += .001 * (activation - rates) / tau
            error = float(np.max(np.abs(rates - terminal)))
            max_neural_error = max(max_neural_error, error)
            assert error < 1e-4
    normal_start = saved["canonical_full_manc_online_qpos"][0]
    frozen_start = saved["canonical_neural_state_frozen_qpos"][0]
    assert np.array_equal(normal_start, frozen_start)
    assert np.array_equal(saved["canonical_full_manc_online_initial_full_neural_rates"],
                          saved["canonical_neural_state_frozen_initial_full_neural_rates"])
    assert np.array_equal(saved["canonical_full_manc_online_initial_full_neural_rates"],
                          saved["canonical_DNg100_drive_removed_initial_full_neural_rates"])
    validation = {"date": "2026-09-24", "checked_body_trials": len(report["rows"]),
                  "reintegrated_full_manc_trials": 5, "maximum_neural_terminal_error_hz": max_neural_error,
                  "same_initial_body_and_neural_state_for_freeze": True,
                  "body_and_neural_traces_match_report": True, "passed": True,
                  "goal_complete": False}
    (OUT / "independent_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
