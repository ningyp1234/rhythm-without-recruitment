"""Replace 23,532 online MANC neurons with recorded phase only, same physics.

This control tests whether the apparent neural walking has *any* online body-
dependent contribution in the current non-proprioceptive architecture.
"""
from __future__ import annotations

import json

import mujoco
import numpy as np
import torch

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH, INVERSE_PATH, ROOT, contact_state, steps_for_interval,
)
from calibrate_official_running_contact_compliance import make_model
from train_official_running_proprioceptive_policy import build_sequences
from train_official_running_endurance_policy import best_cycle, tile_cycle
import train_official_running_cpg_policy as cp
import train_official_running_cpg_residual_stabilizer as rs
import validate_full_manc_phase_driven_walking as full

OUT = ROOT / "results/full-manc-phase-replay-control-20260924"


def replay(model, ground, claws, base_policy, residual_policy, stats, cap,
           qpos0, qvel0, phases, period, command, perturbation=None):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    previous = np.zeros(model.nu)
    qpos = np.empty((len(phases), model.nq), np.float32)
    controls = np.empty((len(phases), model.nu), np.float32)
    for frame, phase in enumerate(phases):
        contacts, _, _ = contact_state(data, ground, claws)
        feature = cp.feature(data, contacts, previous, int(phase), period, command)
        base_action = cp.policy_action(base_policy, feature, stats, model)
        correction = rs.residual_action(residual_policy, feature, stats[:2], cap)
        action = np.clip(base_action + correction,
                         model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        qpos[frame] = data.qpos
        controls[frame] = action
        if frame + 1 < len(phases):
            if perturbation is not None:
                data.qfrc_applied[int(perturbation["dof"])] = (
                    float(perturbation["torque"])
                    if perturbation["start_frame"] <= frame < perturbation["end_frame"]
                    else 0.0)
            data.ctrl[:] = action
            previous = action
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
    return qpos, controls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source_report = json.loads((full.OUT / "report.json").read_text())
    saved = np.load(full.OUT / "body_and_neural_rollouts.npz")
    model, ground, claws, _ = make_model(.005)
    sequences = build_sequences(model, ground, claws, np.load(INVERSE_PATH), np.load(CONTACT_PATH))
    base_ck = torch.load(full.BASE, map_location="cpu", weights_only=False)
    residual_ck = torch.load(full.RESIDUAL, map_location="cpu", weights_only=False)
    training = next(s for s in sequences if s["key"] == base_ck["source_bout"])
    _, start, period, _, _ = best_cycle(training)
    canonical = tile_cycle(training, start, period)
    canonical["cycle_period"] = period
    q0, v0, _ = cp.cycle_from_phase(canonical, 0)
    initial = {"canonical": (q0[0], v0[0])}
    initial.update({s["key"]: (s["reference"][0], s["qvel"][0])
                    for s in sequences if s["split"] == "test"})
    base = cp.CPGPolicy(base_ck["input_size"], base_ck["output_size"])
    base.load_state_dict(base_ck["state_dict"]); base.eval()
    residual = rs.ResidualPolicy(residual_ck["input_size"], residual_ck["output_size"])
    residual.load_state_dict(residual_ck["state_dict"]); residual.eval()
    stats = tuple(base_ck[k] for k in ("feature_mean", "feature_std", "target_mean", "target_std"))
    cap = np.asarray(residual_ck["cap"])
    rows = []
    for key, (q, v) in initial.items():
        label = f"{key}_full_manc_online"
        phases = saved[label + "_phase"]
        replay_qpos, replay_control = replay(model, ground, claws, base, residual,
            stats, cap, q, v, phases, period, base_ck["command_mm_s"])
        reference_qpos = saved[label + "_qpos"]
        reference_control = saved[label + "_control"]
        row = {
            "body_state": key, "frames": len(phases),
            "maximum_absolute_qpos_difference": float(np.max(np.abs(replay_qpos - reference_qpos))),
            "maximum_absolute_control_difference": float(np.max(np.abs(replay_control - reference_control))),
            "endpoint_xy_difference_mm": float(np.linalg.norm(
                replay_qpos[-1, :2] - reference_qpos[-1, :2]) * full.MODEL_TO_MM),
            "source_2s_distance_mm": next(x["distance_2s_mm"] for x in source_report["rows"]
                                            if x["body_state"] == key and x["condition"] == "full_manc_online"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    checks = {
        "four_body_starts_checked": len(rows) == 4,
        "all_1601_physical_frames_checked": all(r["frames"] == 1601 for r in rows),
        "body_states_bit_exact": all(r["maximum_absolute_qpos_difference"] == 0 for r in rows),
        "actuator_controls_bit_exact": all(r["maximum_absolute_control_difference"] == 0 for r in rows),
        "online_network_can_be_replaced_by_phase_replay": True,
    }
    report = {
        "date": "2026-09-24", "scope": "Full-MANC online walking versus prerecorded MANC phase with identical free-body controller",
        "comparison_changes": "The complete MANC integration is removed at runtime; only its saved phase is given to the unchanged body policy, which recomputes all 56 actions from current proprioception.",
        "rows": rows, "checks": checks, "passed": all(checks.values()),
        "interpretation": "Exact equivalence proves the current full-MANC walker has no online body-dependent neural role. MANC supplies an open-loop gait clock; body proprioception is handled by the engineering policy.",
        "brain_autonomous_walking_achieved": False, "goal_complete": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
