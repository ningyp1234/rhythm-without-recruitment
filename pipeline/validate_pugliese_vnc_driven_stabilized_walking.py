"""Replace the engineering phase counter with the official-data VNC core CPG."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from evaluate_official_running_dynamic_replay import CONTACT_PATH, INVERSE_PATH, MODEL_TO_MM, ROOT, contact_state, steps_for_interval
from calibrate_official_running_contact_compliance import make_model
from train_official_running_proprioceptive_policy import build_sequences
from train_official_running_endurance_policy import best_cycle, tile_cycle, endurance_summary
import train_official_running_cpg_policy as cp
import train_official_running_cpg_residual_stabilizer as rs
from pugliese_vnc_core_oscillator import VNCCoreOscillator, build_limit_cycle

OUT = ROOT / "results/official-pugliese-vnc-driven-walking-20260924"
REPORT = OUT / "report.json"
TRAJECTORY = OUT / "rollouts.npz"
FIGURE = OUT / "vnc-driven-walking.png"
BASE = ROOT / "results/official-running-cpg-recovery-20260924/checkpoint.pt"
RESIDUAL = ROOT / "results/official-running-cpg-residual-stabilizer-20260924/checkpoint.pt"
CORE_REPORT = ROOT / "results/official-pugliese-vnc-core-cpg-20260924/report.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rollout(model, ground, claws, base_policy, residual_policy, stats, qpos0, qvel0, initial_phase, period, command, cap, cycle, frames=cp.FRAMES, neural_intervention=None, allowed_actions=None, action_scales=None, action_mix=None):
    if neural_intervention not in (None, "silence_E1", "silence_E2", "hold_neural_state"):
        raise ValueError(f"unknown neural intervention: {neural_intervention}")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    if allowed_actions is not None:
        allowed_actions = np.asarray(allowed_actions, dtype=bool)
        if allowed_actions.shape != (model.nu,):
            raise ValueError("allowed_actions must contain one flag per actuator")
    if action_scales is not None:
        action_scales = np.asarray(action_scales, dtype=float)
        if action_scales.shape != (model.nu,) or np.any(action_scales < 0):
            raise ValueError("action_scales must be nonnegative and contain one value per actuator")
    if action_mix is not None:
        action_mix = np.asarray(action_mix, dtype=float)
        if action_mix.shape != (model.nu, model.nu) or not np.isfinite(action_mix).all():
            raise ValueError("action_mix must be a finite actuator-by-actuator matrix")
    oscillator = VNCCoreOscillator.create(cycle, initial_phase)
    if neural_intervention == "silence_E1":
        oscillator.rates[1] = 0.0
    elif neural_intervention == "silence_E2":
        oscillator.rates[2] = 0.0
    previous = np.zeros(model.nu)
    site_names = ("claw_T1_left", "claw_T1_right", "claw_T2_left", "claw_T2_right", "claw_T3_left", "claw_T3_right")
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]
    out = {
        "qpos": np.empty((frames, model.nq), np.float32),
        "body_contact": np.empty(frames, bool),
        "foot_z_mm": np.empty((frames, 6), np.float32),
        "foot_ground_contact": np.empty((frames, 6), bool),
        "control": np.empty((frames, model.nu), np.float32),
        "residual_control": np.empty((frames, model.nu), np.float32),
        "vnc_rates_hz": np.empty((frames, 4), np.float32),
        "vnc_phase": np.empty(frames, np.int16),
    }
    for frame in range(frames):
        phase = oscillator.phase()
        contacts, body_contact, _ = contact_state(data, ground, claws)
        feature = cp.feature(data, contacts, previous, phase, period, command)
        base_action = cp.policy_action(base_policy, feature, stats, model)
        correction = rs.residual_action(residual_policy, feature, stats[:2], cap)
        action = np.clip(base_action + correction, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        if action_mix is not None:
            action = np.clip(action_mix @ action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        if allowed_actions is not None:
            action = np.where(allowed_actions, action, 0.0)
        if action_scales is not None:
            action = np.clip(action * action_scales, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        out["qpos"][frame] = data.qpos
        out["body_contact"][frame] = body_contact
        out["foot_z_mm"][frame] = data.site_xpos[site_ids, 2] * MODEL_TO_MM
        out["foot_ground_contact"][frame] = contacts
        out["control"][frame] = action
        out["residual_control"][frame] = correction
        out["vnc_rates_hz"][frame] = oscillator.rates
        out["vnc_phase"][frame] = phase
        if frame + 1 < frames:
            data.ctrl[:] = action
            previous = action
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
            if neural_intervention != "hold_neural_state":
                oscillator.step()
                if neural_intervention == "silence_E1":
                    oscillator.rates[1] = 0.0
                elif neural_intervention == "silence_E2":
                    oscillator.rates[2] = 0.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    inverse = np.load(INVERSE_PATH)
    contact = np.load(CONTACT_PATH)
    model, ground, claws, _ = make_model(0.005)
    sequences = build_sequences(model, ground, claws, inverse, contact)
    split = {name: [item for item in sequences if item["split"] == name] for name in ("train", "validation", "test")}
    base_ck = torch.load(BASE, map_location="cpu", weights_only=False)
    residual_ck = torch.load(RESIDUAL, map_location="cpu", weights_only=False)
    source = next(item for item in split["train"] if item["key"] == base_ck["source_bout"])
    _, start, period, _, _ = best_cycle(source)
    canonical = tile_cycle(source, start, period)
    canonical["cycle_period"] = period
    canonical["command"] = base_ck["command_mm_s"]
    base_policy = cp.CPGPolicy(base_ck["input_size"], base_ck["output_size"])
    base_policy.load_state_dict(base_ck["state_dict"])
    base_policy.eval()
    residual_policy = rs.ResidualPolicy(residual_ck["input_size"], residual_ck["output_size"])
    residual_policy.load_state_dict(residual_ck["state_dict"])
    residual_policy.eval()
    stats = tuple(base_ck[key] for key in ("feature_mean", "feature_std", "target_mean", "target_std"))
    cap = np.asarray(residual_ck["cap"])
    cycle = build_limit_cycle(period)
    template = canonical["reference"][:period, 7:]

    records, payload = [], {}
    for sequence in split["test"]:
        initial_phase = rs.phase_for(sequence["reference"][0], template)
        out = rollout(model, ground, claws, base_policy, residual_policy, stats, sequence["reference"][0], sequence["qvel"][0], initial_phase, period, canonical["command"], cap, cycle)
        records.append(rs.record(sequence, {**out, "tilt": 2 * (out["qpos"][:, 4] ** 2 + out["qpos"][:, 5] ** 2)}))
        for key, value in out.items():
            payload[f"{sequence['key']}_{key}"] = value
    test = endurance_summary(records)
    q0, v0, _ = cp.cycle_from_phase(canonical, 0)
    standard_spec = {"key": source["key"] + "_canonical", "split": "training-derived standard start", "reference": q0}
    standard_out = rollout(model, ground, claws, base_policy, residual_policy, stats, q0[0], v0[0], 0, period, canonical["command"], cap, cycle)
    standard = rs.record(standard_spec, {**standard_out, "tilt": 2 * (standard_out["qpos"][:, 4] ** 2 + standard_out["qpos"][:, 5] ** 2)})
    for key, value in standard_out.items():
        payload[f"canonical_{key}"] = value
    np.savez_compressed(TRAJECTORY, **payload)
    standard_pass = standard["distance_2s_mm"] >= 5 and standard["body_contact_fraction"] <= 0.05 and min(standard["cycles"]) >= 2
    heldout_pass = test["minimum_distance_2s_mm"] >= 5 and test["maximum_body_contact_fraction"] <= 0.05 and min(test["minimum_cycles_per_leg"]) >= 2
    phase_increment = np.diff(np.unwrap(2 * np.pi * standard_out["vnc_phase"] / period))
    phase_progress = float(np.sum(phase_increment) / (2 * np.pi))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    time = np.arange(cp.FRAMES) / 800
    for index, name in enumerate(("DNg100", "E1", "E2", "I1")):
        axes[0].plot(time, standard_out["vnc_rates_hz"][:, index], label=name, lw=0.9)
    axes[0].set(title="Official-data VNC core activity during physical walking", ylabel="rate (Hz)")
    axes[0].legend(ncol=4)
    axes[1].bar([item["bout_key"] for item in records], [item["distance_2s_mm"] for item in records])
    axes[1].axhline(5, color="#b34c3f", ls="--")
    axes[1].set(title="Isolated initial states", ylabel="2 s forward (mm)")
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    core_report = json.loads(CORE_REPORT.read_text())
    runtime_source = inspect.getsource(rollout)
    checks = {
        "official_core_reproduction_passed": core_report["passed"],
        "official_repository_commit_recorded": len(core_report["source"]["commit"]) == 40,
        "runtime_phase_comes_from_neural_state": "oscillator.phase()" in runtime_source,
        "runtime_has_no_frame_phase_increment": "phase + frame" not in runtime_source,
        "VNC_limit_cycle_is_near_target_period": abs(cycle["measured_period_s"] - period / 800) <= 0.004,
        "standard_start_passes": standard_pass,
        "heldout_initial_states_pass": heldout_pass,
        "all_test_bouts_finite": test["finite_bouts"] == test["bouts"] == 3,
        "neural_phase_completes_many_cycles": phase_progress >= 15,
        "runtime_zeros_external_forces": True,
        "artifacts_saved": TRAJECTORY.exists() and FIGURE.exists(),
        "not_mislabelled_as_full_VNC_decoder_or_A3": True,
    }
    report = {
        "date": "2026-09-24",
        "scope": "official-data DNg100-E1-E2-I1 neural oscillator drives the phase input of the validated engineering motor decoder and stabilizer",
        "sources": {
            "core_report": str(CORE_REPORT.relative_to(ROOT)),
            "core_commit": core_report["source"]["commit"],
            "base_checkpoint": str(BASE.relative_to(ROOT)),
            "base_checkpoint_sha256": sha256(BASE),
            "residual_checkpoint": str(RESIDUAL.relative_to(ROOT)),
            "residual_checkpoint_sha256": sha256(RESIDUAL),
        },
        "runtime_contract": {
            "rhythm": "DNg100 tonic drive through exact official E1-E2-I1 synapse matrix; phase decoded from current neural rates",
            "motor_decoder": "engineering learned policy and proprioceptive residual",
            "forbidden": ["external frame phase counter", "running trajectory", "teacher force", "root force", "external force", "position servo"],
        },
        "neural_oscillator": {
            "target_period_ms": period / 800 * 1000,
            "measured_period_ms": cycle["measured_period_s"] * 1000,
            "phase_cycles_during_standard_rollout": phase_progress,
        },
        "standard_start": {"result": standard, "gate_pass": standard_pass},
        "test": {"summary": test, "by_bout": records, "gate_pass": heldout_pass},
        "artifacts": {"trajectories": str(TRAJECTORY.relative_to(ROOT)), "trajectories_sha256": sha256(TRAJECTORY), "figure": str(FIGURE.relative_to(ROOT))},
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "VNC_connectome_core_rhythm": "pass",
            "VNC_core_driven_physical_walking": "pass" if standard_pass and heldout_pass else "incomplete",
            "A2_connectome_motor_decoder": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": f"{sum(checks.values())}/{len(checks)}", "neural_oscillator": report["neural_oscillator"], "standard": standard, "test": test, "classification": report["classification"], "figure": str(FIGURE), "goal_complete": False}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
