"""Test a live 23,532-cell official MANC rhythm as the walking phase source.

The downstream 56-actuator leg decoder and DNg100 tonic input remain explicit
engineering components. This experiment does not claim brain autonomy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.signal import find_peaks

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH, INVERSE_PATH, MODEL_TO_MM, ROOT, contact_state, steps_for_interval,
)
from calibrate_official_running_contact_compliance import make_model
from train_official_running_proprioceptive_policy import build_sequences
from train_official_running_endurance_policy import best_cycle, tile_cycle
import train_official_running_cpg_policy as cp
import train_official_running_cpg_residual_stabilizer as rs
from source_latest_vnc_hill_interface import compatible_neurons
from source_fullmanc_exact_ensemble import NEURONS, WEIGHTS, DATA, load_parameters

OUT = ROOT / "results/full-manc-phase-driven-walking-20260924"
BASE = ROOT / "results/official-running-cpg-recovery-20260924/checkpoint.pt"
RESIDUAL = ROOT / "results/official-running-cpg-residual-stabilizer-20260924/checkpoint.pt"
PARAMETER_ROW = 13
NEURAL_DT = .001
BODY_DT = 1 / 800


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FullMANCPhase:
    def __init__(self, matrix, tau, gain, threshold, cap, current, template_states,
                 readout_indices, e1_indices, motor_indices, body_period, initial_phase,
                 freeze=False, remove_descending_drive=False):
        self.matrix = matrix
        self.tau = tau
        self.gain = gain
        self.threshold = threshold
        self.cap = cap
        self.current = current
        self.template_states = template_states
        self.readout_indices = readout_indices
        self.e1_indices = e1_indices
        self.motor_indices = motor_indices
        self.body_period = body_period
        self.freeze = freeze
        self.remove_descending_drive = remove_descending_drive
        self.template_readout = template_states[:, readout_indices]
        self.scale = np.maximum(np.ptp(self.template_readout, axis=0), 1e-5)
        position = int(np.floor(initial_phase / body_period * len(template_states))) % len(template_states)
        self.rates = template_states[position].astype(np.float64).copy()
        self.remainder = 0.0
        self.neural_steps = 0

    def phase(self):
        difference = (self.template_readout - self.rates[self.readout_indices]) / self.scale
        nearest = int(np.argmin(np.mean(difference * difference, axis=1)))
        return int(np.floor(nearest / len(self.template_states) * self.body_period)) % self.body_period

    def step(self, external_current=None):
        if self.freeze:
            return
        self.remainder += BODY_DT
        while self.remainder + 1e-12 >= NEURAL_DT:
            total = (0.0 if self.remove_descending_drive else self.current) + self.matrix @ self.rates
            if external_current is not None:
                total = total + external_current
            active = np.maximum(self.cap * np.tanh((self.gain / self.cap) * (total - self.threshold)), 0.0)
            self.rates += NEURAL_DT * (active - self.rates) / self.tau
            self.remainder -= NEURAL_DT
            self.neural_steps += 1


def source_reference():
    neurons = compatible_neurons(pd.read_csv(NEURONS))
    pre_post = sparse.load_npz(WEIGHTS).tocsr()
    params = load_parameters()
    if len(neurons) != 23532 or pre_post.shape != (23532, 23532):
        raise RuntimeError("official full MANC dimensions changed")
    matrix = pre_post.T.tocsr() * .03
    tau = params["tau"][:, PARAMETER_ROW]
    gain = params["gain"][:, PARAMETER_ROW]
    threshold = params["threshold"][:, PARAMETER_ROW]
    cap = params["cap"][:, PARAMETER_ROW]
    current = params["input"][:, PARAMETER_ROW]
    readout_indices = np.asarray([
        int(neurons.index[neurons.type.eq(kind) & neurons.somaNeuromere.eq("T1") & neurons.somaSide.eq("LHS")][0])
        for kind in ("IN17A001", "INXXX466", "IN16B036")
    ])
    e1_indices = np.flatnonzero(neurons.type.eq("IN17A001"))
    motor_indices = np.asarray(params["motor_indices"]).astype(int).ravel()
    rates = np.zeros(len(neurons), dtype=np.float64)
    recent = []
    e1_trace = np.empty(2000, np.float32)
    for tick in range(2000):
        input_current = current if tick >= 20 else 0.0
        total = input_current + matrix @ rates
        active = np.maximum(cap * np.tanh((gain / cap) * (total - threshold)), 0.0)
        rates += NEURAL_DT * (active - rates) / tau
        e1_trace[tick] = rates[readout_indices[0]]
        if tick >= 1650:
            recent.append(rates.astype(np.float32).copy())
    peaks, _ = find_peaks(e1_trace[500:], prominence=.5, distance=50)
    peaks = peaks + 500
    if len(peaks) < 10 or peaks[-2] < 1650:
        raise RuntimeError("full MANC reference failed to enter a late stable E1 rhythm")
    left, right = int(peaks[-2]), int(peaks[-1])
    template_states = np.asarray(recent[left - 1650:right - 1650])
    if len(template_states) < 70 or len(template_states) > 130:
        raise RuntimeError("full MANC period outside expected range")
    return {
        "matrix": matrix, "tau": tau, "gain": gain, "threshold": threshold,
        "cap": cap, "current": current, "template_states": template_states,
        "readout_indices": readout_indices, "e1_indices": e1_indices,
        "motor_indices": motor_indices,
        "neural_period_ms": len(template_states),
        "readout_body_ids": neurons.bodyId.iloc[readout_indices].astype(int).tolist(),
        "e1_body_ids": neurons.bodyId.iloc[e1_indices].astype(int).tolist(),
        "neuron_count": len(neurons), "edge_count": int(matrix.nnz),
    }


def rollout(model, ground, claws, base_policy, residual_policy, stats, cap,
            qpos0, qvel0, initial_phase, body_period, command, neural_ref,
            freeze=False, remove_descending_drive=False, sensory_encoder=None,
            swapped_sensory_sides=False, sensory_strength=1.0,
            active_sensory_groups=None, perturbation=None):
    neural = FullMANCPhase(
        neural_ref["matrix"], neural_ref["tau"], neural_ref["gain"],
        neural_ref["threshold"], neural_ref["cap"], neural_ref["current"],
        neural_ref["template_states"], neural_ref["readout_indices"],
        neural_ref["e1_indices"], neural_ref["motor_indices"],
        body_period, initial_phase, freeze=freeze,
        remove_descending_drive=remove_descending_drive,
    )
    initial_full_neural_rates = neural.rates.astype(np.float32).copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    previous = np.zeros(model.nu)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
             for name in ("claw_T1_left", "claw_T1_right", "claw_T2_left",
                          "claw_T2_right", "claw_T3_left", "claw_T3_right")]
    frames = cp.FRAMES
    out = {
        "qpos": np.empty((frames, model.nq), np.float32),
        "qvel": np.empty((frames, model.nv), np.float32),
        "body_contact": np.empty(frames, bool),
        "foot_ground_contact": np.empty((frames, 6), bool),
        "foot_z_mm": np.empty((frames, 6), np.float32),
        "phase": np.empty(frames, np.int16),
        "e1_rates_hz": np.empty((frames, 6), np.float32),
        "motor_rates_hz": np.empty((frames, len(neural.motor_indices)), np.float32),
        "control": np.empty((frames, model.nu), np.float32),
    }
    if sensory_encoder is not None:
        out["sensory_features"] = np.empty((frames, 6, 5), np.float32)
        out["sensory_rate_hz"] = np.empty((frames, len(sensory_encoder.sensory_indices)), np.float32)
    if perturbation is not None:
        out["applied_joint_torque"] = np.zeros(frames, np.float32)
    for frame in range(frames):
        phase = neural.phase()
        contacts, body_contact, _ = contact_state(data, ground, claws)
        feature = cp.feature(data, contacts, previous, phase, body_period, command)
        base_action = cp.policy_action(base_policy, feature, stats, model)
        correction = rs.residual_action(residual_policy, feature, stats[:2], cap)
        action = np.clip(base_action + correction, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        out["qpos"][frame] = data.qpos
        out["qvel"][frame] = data.qvel
        out["body_contact"][frame] = body_contact
        out["foot_ground_contact"][frame] = contacts
        out["foot_z_mm"][frame] = data.site_xpos[sites, 2] * MODEL_TO_MM
        out["phase"][frame] = phase
        out["e1_rates_hz"][frame] = neural.rates[neural.e1_indices]
        out["motor_rates_hz"][frame] = neural.rates[neural.motor_indices]
        out["control"][frame] = action
        if sensory_encoder is not None:
            out["sensory_features"][frame] = sensory_encoder.features(data.qpos, data.qvel, contacts)
            out["sensory_rate_hz"][frame] = neural.rates[sensory_encoder.sensory_indices]
        if frame + 1 < frames:
            if perturbation is not None:
                pulse = (float(perturbation["torque"])
                         if perturbation["start_frame"] <= frame < perturbation["end_frame"]
                         else 0.0)
                data.qfrc_applied[int(perturbation["dof"])] = pulse
                out["applied_joint_torque"][frame] = pulse
            data.ctrl[:] = action
            previous = action
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
            if sensory_encoder is None:
                neural.step()
            else:
                next_contacts, _, _ = contact_state(data, ground, claws)
                sensory_current, _ = sensory_encoder.current(data.qpos, data.qvel,
                    next_contacts, swapped_sides=swapped_sensory_sides,
                    active_groups=active_sensory_groups)
                neural.step(sensory_current * sensory_strength)
    out["neural_steps"] = neural.neural_steps
    out["all_neural_rates_finite"] = bool(np.isfinite(neural.rates).all())
    out["initial_full_neural_rates"] = initial_full_neural_rates
    out["terminal_full_neural_rates"] = neural.rates.astype(np.float32).copy()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    neural_ref = source_reference()
    model, ground, claws, _ = make_model(.005)
    sequences = build_sequences(model, ground, claws, np.load(INVERSE_PATH), np.load(CONTACT_PATH))
    base_ck = torch.load(BASE, map_location="cpu", weights_only=False)
    residual_ck = torch.load(RESIDUAL, map_location="cpu", weights_only=False)
    source = next(s for s in sequences if s["key"] == base_ck["source_bout"])
    _, start, period, _, _ = best_cycle(source)
    canonical = tile_cycle(source, start, period)
    canonical["cycle_period"] = period
    q0, v0, _ = cp.cycle_from_phase(canonical, 0)
    template = canonical["reference"][:period, 7:]
    starts = [("canonical", q0, v0, 0)] + [
        (item["key"], item["reference"], item["qvel"], rs.phase_for(item["reference"][0], template))
        for item in sequences if item["split"] == "test"
    ]
    base_policy = cp.CPGPolicy(base_ck["input_size"], base_ck["output_size"])
    base_policy.load_state_dict(base_ck["state_dict"])
    base_policy.eval()
    residual_policy = rs.ResidualPolicy(residual_ck["input_size"], residual_ck["output_size"])
    residual_policy.load_state_dict(residual_ck["state_dict"])
    residual_policy.eval()
    stats = tuple(base_ck[k] for k in ("feature_mean", "feature_std", "target_mean", "target_std"))
    cap = np.asarray(residual_ck["cap"])
    payload, rows = {}, []
    for key, reference, velocity, phase in starts:
        for condition in (("full_manc_online", False, False),
                          ("neural_state_frozen", True, False),
                          ("DNg100_drive_removed", False, True)) if key == "canonical" else (("full_manc_online", False, False),):
            label, frozen, remove_drive = condition
            result = rollout(model, ground, claws, base_policy, residual_policy, stats, cap,
                             reference[0], velocity[0], phase, period,
                             base_ck["command_mm_s"], neural_ref,
                             freeze=frozen, remove_descending_drive=remove_drive)
            metric = rs.record({"key": key, "split": "paired", "reference": reference.copy()},
                               {**result, "tilt": 2 * (result["qpos"][:, 4] ** 2 + result["qpos"][:, 5] ** 2)})
            row = {
                "body_state": key, "condition": label,
                "distance_2s_mm": metric["distance_2s_mm"],
                "body_contact_fraction": metric["body_contact_fraction"],
                "cycles_per_leg": metric["cycles"],
                "neural_steps": result["neural_steps"],
                "neural_phase_cycles": float((np.unwrap(2 * np.pi * result["phase"] / period)[-1] -
                                                np.unwrap(2 * np.pi * result["phase"] / period)[0]) / (2 * np.pi)),
                "mean_motor_neuron_rate_hz": float(result["motor_rates_hz"].mean()),
                "source_motor_neurons_ever_above_1_hz": int(np.count_nonzero(result["motor_rates_hz"].max(axis=0) > 1.0)),
                "source_motor_neuron_count": int(result["motor_rates_hz"].shape[1]),
                "all_neural_rates_finite": result["all_neural_rates_finite"],
                "walking_gate_pass": bool(metric["finite"] and metric["distance_2s_mm"] >= 5 and
                                          metric["body_contact_fraction"] <= .05 and min(metric["cycles"]) >= 2),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            for name in ("qpos", "body_contact", "foot_ground_contact", "foot_z_mm", "phase", "e1_rates_hz", "motor_rates_hz", "control"):
                payload[f"{key}_{label}_{name}"] = result[name]
            for name in ("initial_full_neural_rates", "terminal_full_neural_rates"):
                payload[f"{key}_{label}_{name}"] = result[name]
    np.savez_compressed(OUT / "body_and_neural_rollouts.npz", **payload)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    data = payload["canonical_full_manc_online_e1_rates_hz"]
    for k, name in enumerate(("T2R", "T3L", "T3R", "T2L", "T1R", "T1L")):
        axes[0].plot(np.arange(cp.FRAMES) / 800, data[:, k], label=name, lw=.75)
    axes[0].set(xlabel="body time (s)", ylabel="E1 rate (Hz)", title="Six local E1 rhythms during physical walking")
    axes[0].legend(ncol=6, fontsize=8)
    axes[1].bar([f"{r['body_state']}\n{r['condition']}" for r in rows], [r["distance_2s_mm"] for r in rows])
    axes[1].axhline(5, ls="--", color="#b34c3f")
    axes[1].set(ylabel="forward movement in 2 s (mm)", title="Same-start physical gate")
    fig.savefig(OUT / "full-manc-neural-body-transfer.png", dpi=170)
    plt.close(fig)
    checks = {
        "full_official_manc_graph_loaded": neural_ref["neuron_count"] == 23532,
        "all_live_neural_states_finite": all(r["all_neural_rates_finite"] for r in rows),
        "neural_clock_matches_two_second_body": all(r["neural_steps"] == 2000 for r in rows if r["condition"] == "full_manc_online"),
        "DNg100_cut_uses_same_two_second_neural_clock": next(r["neural_steps"] for r in rows if r["condition"] == "DNg100_drive_removed") == 2000,
        "frozen_neural_control_has_zero_steps": next(r["neural_steps"] for r in rows if r["condition"] == "neural_state_frozen") == 0,
        "four_body_initial_states_and_two_controls_completed": len(rows) == 6,
    }
    report = {
        "date": "2026-09-24", "scope": "source full 23,532-cell MANC neural dynamics online with real 2-s free-body walking",
        "sources": {"neurons_sha256": sha256(NEURONS), "weights_sha256": sha256(WEIGHTS),
                    "official_neuron_params_sha256": sha256(DATA / "neuron_params.h5"),
                    "base_policy_sha256": sha256(BASE), "stabilizer_sha256": sha256(RESIDUAL)},
        "neural_source": {"parameter_row": PARAMETER_ROW, "neurons": neural_ref["neuron_count"],
                          "directed_edges": neural_ref["edge_count"], "E1_body_ids": neural_ref["e1_body_ids"],
                          "readout_body_ids": neural_ref["readout_body_ids"],
                          "late_reference_cycle_ms": neural_ref["neural_period_ms"]},
        "runtime_contract": "Every 1.25-ms body frame, exact official MANC rate equation advances for 1 or 2 one-ms neural steps; current T1L E1/E2/I1 state is decoded to a phase used by the unchanged engineering 56-actuator policy. Bilateral DNg100 input is still artificial.",
        "rows": rows, "checks": checks, "audit_passed": all(checks.values()),
        "four_start_walking_gate_passed": all(r["walking_gate_pass"] for r in rows if r["condition"] == "full_manc_online"),
        "brain_autonomous_walking_achieved": False, "goal_complete": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checks": checks, "four_start_walking_gate_passed": report["four_start_walking_gate_passed"], "goal_complete": False}, indent=2))
    if not report["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
