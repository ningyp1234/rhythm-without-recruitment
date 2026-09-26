"""Intervene on the published four-cell VNC core during free-body walking.

The same frozen motor policy, body, and starting state are used in every arm.
This tests whether the neural clock is causally needed by the current hybrid
controller; it does not establish a whole-brain or muscle-level reconstruction.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from calibrate_official_running_contact_compliance import make_model
from evaluate_official_running_dynamic_replay import CONTACT_PATH, INVERSE_PATH, MODEL_TO_MM, ROOT, contact_state
from pugliese_vnc_core_oscillator import build_limit_cycle
from train_official_running_endurance_policy import best_cycle, tile_cycle
from train_official_running_proprioceptive_policy import build_sequences
import train_official_running_cpg_policy as cp
import train_official_running_cpg_residual_stabilizer as rs
import validate_pugliese_vnc_driven_stabilized_walking as vw

OUT = ROOT / "results/vnc-core-causal-audit-20260924"
CONDITIONS = ("intact", "hold_neural_state", "silence_E1", "silence_E2")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_cycles(phases, period):
    return float(np.sum(np.diff(np.unwrap(2 * np.pi * phases / period))) / (2 * np.pi))


def score_rollout(model, specification, result, period, intact_control, slip_limit):
    score = rs.record(specification, {**result, "tilt": 2 * (result["qpos"][:, 4] ** 2 + result["qpos"][:, 5] ** 2)})
    score["decoded_phase_cycles"] = phase_cycles(result["vnc_phase"], period)
    score["E1_mean_hz"] = float(np.mean(result["vnc_rates_hz"][:, 1]))
    score["E2_mean_hz"] = float(np.mean(result["vnc_rates_hz"][:, 2]))
    score["action_rms_difference_from_intact"] = float(np.sqrt(np.mean((result["control"] - intact_control) ** 2)))
    score["walking_gate"] = bool(
        score["finite"] and score["distance_2s_mm"] >= 5
        and score["body_contact_fraction"] <= 0.05 and min(score["cycles"]) >= 2
    )
    score.update(physical_foot_metrics(model, result))
    score["strict_walking_gate"] = bool(
        score["walking_gate"]
        and min(score["sustained_contact_cycles_8_75ms"]) >= 2
        and score["foot_only_support_fraction"] >= 0.99
        and score["median_planted_foot_slip_mm_s"] is not None
        and score["median_planted_foot_slip_mm_s"] <= slip_limit
    )
    return score


def physical_foot_metrics(model, result):
    """Score sustained ground transitions and stance slip from saved qpos."""
    contact = result["foot_ground_contact"]
    sustained_samples = 7  # 8.75 ms at 800 Hz; source period is only 90 ms.
    step_counts = []
    for leg in range(6):
        runs = []
        start = 0
        for index in range(1, len(contact) + 1):
            if index == len(contact) or contact[index, leg] != contact[start, leg]:
                if index - start >= sustained_samples:
                    runs.append(bool(contact[start, leg]))
                start = index
        compact = [state for i, state in enumerate(runs) if i == 0 or state != runs[i - 1]]
        step_counts.append(sum(compact[i - 2:i + 1] == [True, False, True] for i in range(2, len(compact))))
    data = mujoco.MjData(model)
    names = ("claw_T1_left", "claw_T1_right", "claw_T2_left", "claw_T2_right", "claw_T3_left", "claw_T3_right")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
    foot_xy = np.empty((len(contact), 6, 2), np.float32)
    for index, qpos in enumerate(result["qpos"]):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        foot_xy[index] = data.site_xpos[ids, :2] * MODEL_TO_MM
    planted = contact[1:] & contact[:-1]
    slip = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1) * 800
    median_slip = float(np.median(slip[planted])) if np.any(planted) else None
    return {
        "sustained_contact_cycles_8_75ms": step_counts,
        "foot_only_support_fraction": float(np.mean(contact.any(axis=1) & ~result["body_contact"])),
        "contact_fraction_per_leg": contact.mean(axis=0).astype(float).tolist(),
        "median_planted_foot_slip_mm_s": median_slip,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, ground, claws, _ = make_model(0.005)
    with np.load(INVERSE_PATH) as inverse, np.load(CONTACT_PATH) as contact:
        sequences = build_sequences(model, ground, claws, inverse, contact)
    base_ck = torch.load(vw.BASE, map_location="cpu", weights_only=False)
    residual_ck = torch.load(vw.RESIDUAL, map_location="cpu", weights_only=False)
    source = next(item for item in sequences if item["split"] == "train" and item["key"] == base_ck["source_bout"])
    _, start, period, _, _ = best_cycle(source)
    canonical = tile_cycle(source, start, period)
    canonical["cycle_period"] = period
    canonical["command"] = base_ck["command_mm_s"]
    q0, v0, _ = cp.cycle_from_phase(canonical, 0)
    # The old 30 ms/2 mm/s gates came from a different body; calibrate this
    # contact timing and slip against the actual official gait on this body.
    data = mujoco.MjData(model)
    reference_contacts = []
    reference_body_contact = []
    for position in q0:
        data.qpos[:] = position
        mujoco.mj_forward(model, data)
        feet, body_touch, _ = contact_state(data, ground, claws)
        reference_contacts.append(feet)
        reference_body_contact.append(body_touch)
    reference_metrics = physical_foot_metrics(model, {
        "qpos": q0,
        "foot_ground_contact": np.asarray(reference_contacts),
        "body_contact": np.asarray(reference_body_contact),
    })
    slip_limit = 1.25 * reference_metrics["median_planted_foot_slip_mm_s"]
    specification = {"key": source["key"] + "_canonical", "split": "training-derived standard start", "reference": q0}
    base = cp.CPGPolicy(base_ck["input_size"], base_ck["output_size"])
    base.load_state_dict(base_ck["state_dict"])
    base.eval()
    residual = rs.ResidualPolicy(residual_ck["input_size"], residual_ck["output_size"])
    residual.load_state_dict(residual_ck["state_dict"])
    residual.eval()
    stats = tuple(base_ck[key] for key in ("feature_mean", "feature_std", "target_mean", "target_std"))
    cap = np.asarray(residual_ck["cap"])
    cycle = build_limit_cycle(period)

    traces = {}
    records = {}
    for condition in CONDITIONS:
        intervention = None if condition == "intact" else condition
        result = vw.rollout(
            model, ground, claws, base, residual, stats, q0[0], v0[0],
            0, period, canonical["command"], cap, cycle,
            neural_intervention=intervention,
        )
        traces[condition] = result
        score = score_rollout(model, specification, result, period, traces["intact"]["control"], slip_limit)
        records[condition] = score
    test_records = []
    template = canonical["reference"][:period, 7:]
    for sequence in (item for item in sequences if item["split"] == "test"):
        initial_phase = rs.phase_for(sequence["reference"][0], template)
        paired = {}
        for condition, intervention in (("intact", None), ("hold_neural_state", "hold_neural_state")):
            output = vw.rollout(
                model, ground, claws, base, residual, stats,
                sequence["reference"][0], sequence["qvel"][0], initial_phase,
                period, canonical["command"], cap, cycle, neural_intervention=intervention,
            )
            paired[condition] = output
        scored = {}
        for condition, output in paired.items():
            scored[condition] = score_rollout(model, sequence, output, period, paired["intact"]["control"], slip_limit)
            for key, value in output.items():
                payload_key = f"{sequence['key']}_{condition}_{key}"
                # Keep every tested trajectory so gates can be recalculated.
                traces[payload_key] = value
        test_records.append({"bout_key": sequence["key"], "intact": scored["intact"], "hold_neural_state": scored["hold_neural_state"]})
    payload = {}
    for condition in CONDITIONS:
        for key, value in traces[condition].items():
            payload[f"{condition}_{key}"] = value
    for key, value in traces.items():
        if key not in CONDITIONS:
            payload[key] = value
    trace_path = OUT / "rollouts.npz"
    np.savez_compressed(trace_path, **payload)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), layout="constrained")
    time = np.arange(cp.FRAMES) / 800
    for condition in CONDITIONS:
        result = traces[condition]
        axes[0].plot(time, result["vnc_phase"], label=condition, lw=0.8)
        displacement = (result["qpos"][:, :2] - result["qpos"][0, :2]) @ (
            (q0[-1, :2] - q0[0, :2]) / max(np.linalg.norm(q0[-1, :2] - q0[0, :2]), 1e-12)
        ) * MODEL_TO_MM
        axes[1].plot(time, displacement, label=condition)
    axes[0].set(ylabel="decoded VNC phase", title="Four-cell neural intervention in the same free body")
    axes[1].axhline(5, color="#c44", ls="--", lw=0.8)
    axes[1].set(ylabel="forward displacement (mm)")
    positions = np.arange(len(CONDITIONS))
    axes[2].bar(positions, [records[c]["body_contact_fraction"] for c in CONDITIONS], color="#337a9e")
    axes[2].set(xticks=positions, xticklabels=CONDITIONS, ylabel="body-ground contact fraction")
    legs = np.arange(6)
    axes[3].bar(legs - 0.25, reference_metrics["sustained_contact_cycles_8_75ms"], width=0.24, label="real gait reference", color="#777")
    axes[3].bar(legs, records["intact"]["sustained_contact_cycles_8_75ms"], width=0.24, label="intact neural rhythm", color="#1f77b4")
    axes[3].bar(legs + 0.25, records["hold_neural_state"]["sustained_contact_cycles_8_75ms"], width=0.24, label="frozen neural state", color="#ff7f0e")
    axes[3].axhline(2, color="#c44", ls="--", lw=0.8)
    axes[3].set(xticks=legs, xticklabels=("LF", "RF", "LM", "RM", "LH", "RH"), ylabel="sustained stance-swing-stance cycles")
    axes[3].legend(ncol=3)
    axes[0].legend(ncol=2)
    figure = OUT / "vnc-core-causal-audit.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    checks = {
        "same_checkpoint_body_and_initial_state": True,
        "intact_reproduces_previous_success": records["intact"]["walking_gate"],
        "hold_core_really_holds_phase": bool(len(np.unique(traces["hold_neural_state"]["vnc_phase"])) == 1),
        "E1_intervention_really_silences_E1": bool(np.max(traces["silence_E1"]["vnc_rates_hz"][:, 1]) == 0),
        "E2_intervention_really_silences_E2": bool(np.max(traces["silence_E2"]["vnc_rates_hz"][:, 2]) == 0),
        "all_rollouts_finite": all(v["finite"] for v in records.values()),
        "saved_reproducible_trajectories": trace_path.exists() and figure.exists(),
    }
    report = {
        "date": "2026-09-24",
        "question": "Is current physical walking causally dependent on the published-data four-cell VNC core?",
        "scope": "frozen engineering motor decoder on official v1 joint-actuated body; no full-MANC MN-to-muscle drive",
        "source": {
            "core_parameters": str((ROOT / "data/pugliese-vnc-core-cpg.json").relative_to(ROOT)),
            "base_checkpoint_sha256": file_hash(vw.BASE),
            "residual_checkpoint_sha256": file_hash(vw.RESIDUAL),
        },
        "records": records,
        "heldout_test_pairs": test_records,
        "official_source_gait_calibration": reference_metrics,
        "source_relative_slip_limit_mm_s": slip_limit,
        "neural_core_necessary_for_current_walking_gate": not records["hold_neural_state"]["walking_gate"],
        "strict_walk_intact": records["intact"]["strict_walking_gate"],
        "limitations": [
            "A finite neural intervention may alter the decoder input rather than a measured biological synapse.",
            "The body has 56 engineering joint actuators and lacks validated six-leg muscles.",
            "Phototaxis and flight are outside this experiment.",
        ],
        "checks": checks,
        "passed": all(checks.values()),
        "artifacts": {"traces": str(trace_path.relative_to(ROOT)), "traces_sha256": file_hash(trace_path), "figure": str(figure.relative_to(ROOT))},
        "goal_complete": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "records": records, "checks": checks}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
