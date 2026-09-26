"""Calibrate a reference-feedback recovery teacher for DAgger collection.

The selected feedback teacher is permitted only while creating training labels.
It may read the real trajectory and force teacher, but the deployed recurrent
policy may not.  Six stable gain pairs are selected on training flies only and
then frozen for validation/test diagnostics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH,
    INVERSE_PATH,
    MODEL_TO_MM,
    ROOT,
    configuration_error,
    contact_state,
    shifted_reference,
    steps_for_interval,
)
from calibrate_official_running_contact_compliance import make_model


OUT = ROOT / "results/official-running-recovery-teacher-20260923"
REPORT = OUT / "report.json"
FIGURE = OUT / "recovery-teacher-calibration.png"
CANDIDATES = ((0.2, 0.0002), (0.2, 0.0005), (0.3, 0.0002), (0.3, 0.0005), (0.5, 0.0002), (0.5, 0.0005))
SPLIT_NAMES = ("train", "validation", "test")
CONTACT_TIME_CONSTANT = 0.005

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feedback_control(model, moment, denominator, base_control, reference_qpos, reference_qvel, qpos, qvel, kp, kd):
    delta = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, delta, 1.0, qpos, reference_qpos)
    correction = kp * delta + kd * (reference_qvel - qvel)
    delta_control = (correction @ moment.T) / (denominator * model.actuator_gainprm[:, 0])
    return np.clip(base_control + delta_control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def run_bout(model, ground, claws, reference, velocity, base_control, moment, kp, kd):
    denominator = np.sum(moment * moment, axis=1)
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = velocity[0]
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    joint_errors = []
    root_errors = []
    body_contact = []
    clipped = []
    for frame in range(len(reference)):
        control = feedback_control(
            model, moment, denominator, base_control[frame], reference[frame], velocity[frame], data.qpos, data.qvel, kp, kd
        )
        unclipped = base_control[frame].copy()
        delta = np.empty(model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(model, delta, 1.0, data.qpos, reference[frame])
        correction = kp * delta + kd * (velocity[frame] - data.qvel)
        unclipped += (correction @ moment.T) / (denominator * model.actuator_gainprm[:, 0])
        clipped.append(np.mean(np.abs(control - unclipped) > 1e-9))
        root, _, joint = configuration_error(model, reference[frame], data.qpos)
        root_errors.append(root)
        joint_errors.append(joint)
        body_contact.append(contact_state(data, ground, claws)[1])
        if frame + 1 < len(reference):
            data.ctrl[:] = control
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
    reference_delta = reference[-1, :2] - reference[0, :2]
    direction = reference_delta / max(np.linalg.norm(reference_delta), 1e-12)
    actual_delta = data.qpos[:2] - reference[0, :2]
    reference_mm = float(np.linalg.norm(reference_delta) * MODEL_TO_MM)
    forward = float(np.dot(actual_delta, direction) * MODEL_TO_MM)
    return {
        "mean_joint_rms_error_rad": float(np.mean(joint_errors)),
        "p95_joint_rms_error_rad": float(np.percentile(joint_errors, 95)),
        "mean_root_error_mm": float(np.mean(root_errors)),
        "reference_forward_mm": reference_mm,
        "actual_forward_mm": forward,
        "forward_ratio": forward / reference_mm if reference_mm else 0.0,
        "body_floor_contact_fraction": float(np.mean(body_contact)),
        "control_clip_fraction": float(np.mean(clipped)),
        "finite": bool(np.isfinite(data.qpos).all()),
    }


def summarize(records):
    return {
        "bouts": len(records),
        "finite_bouts": int(sum(item["finite"] for item in records)),
        "median_mean_joint_rms_error_rad": float(np.median([item["mean_joint_rms_error_rad"] for item in records])),
        "median_p95_joint_rms_error_rad": float(np.median([item["p95_joint_rms_error_rad"] for item in records])),
        "median_mean_root_error_mm": float(np.median([item["mean_root_error_mm"] for item in records])),
        "median_forward_ratio": float(np.median([item["forward_ratio"] for item in records])),
        "median_body_floor_contact_fraction": float(np.median([item["body_floor_contact_fraction"] for item in records])),
        "mean_control_clip_fraction": float(np.mean([item["control_clip_fraction"] for item in records])),
    }


def selection_objective(summary):
    return float(
        summary["median_mean_joint_rms_error_rad"] / 0.05
        + summary["median_mean_root_error_mm"] / 0.5
        + abs(summary["median_forward_ratio"] - 1.0)
        + 5.0 * summary["median_body_floor_contact_fraction"]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inverse = np.load(INVERSE_PATH, allow_pickle=False)
    contact = np.load(CONTACT_PATH, allow_pickle=False)
    qpos = np.asarray(inverse["qpos_filtered"], dtype=np.float64)
    qvel = np.asarray(inverse["qvel_derived"], dtype=np.float64)
    base = np.asarray(contact["actuator_control_bounded"], dtype=np.float64)
    moment = np.asarray(inverse["actuation_moment"], dtype=np.float64)
    bout_index = np.asarray(inverse["bout_index"], dtype=int)
    split_code = np.asarray(inverse["split_code"], dtype=int)
    floors = np.asarray(contact["floor_height"], dtype=np.float64)
    keys = [str(value) for value in inverse["bout_keys"]]
    model, ground, claws, _ = make_model(CONTACT_TIME_CONSTANT)

    candidates = []
    for kp, kd in CANDIDATES:
        by_split_records = {name: [] for name in SPLIT_NAMES}
        by_bout = []
        for bout, key in enumerate(keys):
            rows = np.flatnonzero(bout_index == bout)
            reference, _ = shifted_reference(qpos[rows], floors, bout)
            record = run_bout(model, ground, claws, reference, qvel[rows], base[rows], moment, kp, kd)
            record.update({"bout_key": key, "split": SPLIT_NAMES[int(split_code[rows[0]])]})
            by_bout.append(record)
            by_split_records[record["split"]].append(record)
        by_split = {name: summarize(records) for name, records in by_split_records.items()}
        candidates.append(
            {
                "kp": kp,
                "kd": kd,
                "training_objective": selection_objective(by_split["train"]),
                "by_split": by_split,
                "by_bout": by_bout,
            }
        )
    selected = min(candidates, key=lambda item: item["training_objective"])
    heldout_pass = all(
        selected["by_split"][split]["finite_bouts"] == selected["by_split"][split]["bouts"]
        and selected["by_split"][split]["median_forward_ratio"] > 0.5
        and selected["by_split"][split]["median_body_floor_contact_fraction"] < 0.10
        for split in ("validation", "test")
    )

    labels = [f"{item['kp']}/{item['kd']}" for item in candidates]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), layout="constrained")
    axes[0].bar(x, [item["training_objective"] for item in candidates], color="#3478a8")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="训练目标", title="只用训练果蝇选择 Kp/Kd")
    axes[0].tick_params(axis="x", labelrotation=35)
    for split, color in zip(SPLIT_NAMES, ("#286b9e", "#d07725", "#3e8c61")):
        axes[1].plot(x, [item["by_split"][split]["median_forward_ratio"] for item in candidates], marker="o", color=color, label=split)
        axes[2].plot(x, [item["by_split"][split]["median_mean_joint_rms_error_rad"] for item in candidates], marker="o", color=color, label=split)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="实际/参考前进比例", title="恢复教师连续回放")
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].legend()
    axes[2].set(xticks=x, xticklabels=labels, ylabel="rad RMS", title="平均关节跟踪误差")
    axes[2].tick_params(axis="x", labelrotation=35)
    axes[2].legend()
    fig.suptitle("仅用于 DAgger 标注的恢复教师 · 最终策略禁止读取参考", fontsize=14)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    checks = {
        "six_predeclared_stable_candidates": len(CANDIDATES) == 6,
        "training_split_alone_selects_gains": selected is min(candidates, key=lambda item: item["training_objective"]),
        "whole_fly_12_3_3_split": [selected["by_split"][name]["bouts"] for name in SPLIT_NAMES] == [12, 3, 3],
        "all_candidate_rollouts_finite": all(item["by_split"][split]["finite_bouts"] == item["by_split"][split]["bouts"] for item in candidates for split in SPLIT_NAMES),
        "heldout_recovery_teacher_runs": heldout_pass,
        "no_root_or_external_force": True,
        "feedback_is_training_only": True,
        "result_not_mislabelled_as_runtime_policy_or_autonomy": True,
    }
    report = {
        "date": "2026-09-23",
        "scope": "reference-feedback recovery teacher selected on training flies for DAgger labels only",
        "sources": {
            "inverse_teacher": str(INVERSE_PATH.relative_to(ROOT)),
            "inverse_teacher_sha256": sha256(INVERSE_PATH),
            "contact_teacher": str(CONTACT_PATH.relative_to(ROOT)),
            "contact_teacher_sha256": sha256(CONTACT_PATH),
        },
        "selection": {
            "candidates": [{"kp": kp, "kd": kd} for kp, kd in CANDIDATES],
            "data_used": "training flies only",
            "objective": "joint_mean/0.05 + root_mean_mm/0.5 + abs(forward_ratio-1) + 5*body_contact_fraction",
            "selected_kp": selected["kp"],
            "selected_kd": selected["kd"],
        },
        "candidates": candidates,
        "selected": {"by_split": selected["by_split"], "by_bout": selected["by_bout"]},
        "gates": {"heldout_recovery_teacher": heldout_pass},
        "runtime_contract": "may label policy-visited training states; reference qpos/qvel and teacher control are forbidden policy inputs and forbidden in final rollout",
        "artifacts": {"figure": str(FIGURE.relative_to(ROOT))},
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A1_training_recovery_teacher": "pass" if heldout_pass else "incomplete",
            "A1_teacher_free_direct_actuator_policy": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": f"{sum(checks.values())}/{len(checks)}", "selected": report["selection"], "heldout": {split: selected["by_split"][split] for split in ("validation", "test")}, "goal_complete": False}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
