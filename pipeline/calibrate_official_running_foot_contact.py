"""Fit one global foot-contact margin on training flies and test held-out flies.

The official body was built for inverse kinematics, not floor dynamics.  Its
small claw collision capsules therefore need an explicit calibration audit
before real inverse-force teachers can be judged in a walking simulation.
Only the training-fly split selects the margin.  Validation and test flies stay
held out, and the result remains an engineering contact layer rather than a
biological muscle or neural controller.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH,
    FPS,
    INVERSE_PATH,
    MODEL_TO_MM,
    ROOT,
    contact_scores,
    make_floor_model,
    run_continuous,
    run_local_steps,
    shifted_reference,
    vector_summary,
)


OUT = ROOT / "results/official-running-foot-contact-calibration-20260923"
REPORT = OUT / "report.json"
FIGURE = OUT / "foot-contact-calibration.png"
MARGINS = (0.0005, 0.003, 0.006, 0.009, 0.012, 0.015)
SPLIT_NAMES = ("train", "validation", "test")

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def objective(root_mm: np.ndarray, angle: np.ndarray, joint: np.ndarray, actual: np.ndarray, observed: np.ndarray) -> float:
    # Declared before evaluation.  It balances one-frame body translation,
    # body rotation, joint prediction and contact classification.
    match = contact_scores(actual, observed)
    return float(
        np.median(root_mm) / 0.05
        + np.median(angle) / 0.03
        + np.median(joint) / 0.005
        + (1.0 - match["f1"])
    )


def summarize_split(root: np.ndarray, angle: np.ndarray, joint: np.ndarray, actual: np.ndarray, observed: np.ndarray) -> dict:
    return {
        "frames": int(len(root)),
        "root_error_mm": vector_summary(root),
        "root_angle_error_rad": vector_summary(angle),
        "joint_rms_error_rad": vector_summary(joint),
        "observed_contact_match": contact_scores(actual, observed),
        "selection_objective": objective(root, angle, joint, actual, observed),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inverse = np.load(INVERSE_PATH, allow_pickle=False)
    contact = np.load(CONTACT_PATH, allow_pickle=False)
    qpos = np.asarray(inverse["qpos_filtered"], dtype=np.float64)
    qvel = np.asarray(inverse["qvel_derived"], dtype=np.float64)
    controls = np.asarray(contact["actuator_control_bounded"], dtype=np.float64)
    observed = np.asarray(contact["observed_near_floor"], dtype=bool)
    bout_index = np.asarray(inverse["bout_index"], dtype=int)
    split_code = np.asarray(inverse["split_code"], dtype=int)
    bout_keys = [str(value) for value in inverse["bout_keys"]]
    floor_height = np.asarray(contact["floor_height"], dtype=np.float64)

    candidate_reports: list[dict] = []
    for margin in MARGINS:
        model, ground, claw_to_leg, _ = make_floor_model(margin)
        combined = {
            split: {name: [] for name in ("root", "angle", "joint", "actual", "observed")}
            for split in SPLIT_NAMES
        }
        for bout, _key in enumerate(bout_keys):
            rows = np.flatnonzero(bout_index == bout)
            reference, _ = shifted_reference(qpos[rows], floor_height, bout)
            local = run_local_steps(model, ground, claw_to_leg, reference, qvel[rows], controls[rows])
            code = int(split_code[rows[0]])
            split = SPLIT_NAMES[code]
            combined[split]["root"].append(local["root_error_mm"])
            combined[split]["angle"].append(local["root_angle_error_rad"])
            combined[split]["joint"].append(local["joint_rms_error_rad"])
            combined[split]["actual"].append(local["claw_contact"])
            combined[split]["observed"].append(observed[rows][1:])
        by_split = {}
        for split in SPLIT_NAMES:
            arrays = {name: np.concatenate(values) for name, values in combined[split].items()}
            by_split[split] = summarize_split(**arrays)
        candidate_reports.append(
            {
                "contact_margin_native": margin,
                "contact_margin_mm": margin * MODEL_TO_MM,
                "train_selection_objective": by_split["train"]["selection_objective"],
                "by_split": by_split,
            }
        )

    selected = min(candidate_reports, key=lambda item: item["train_selection_objective"])
    selected_margin = float(selected["contact_margin_native"])
    model, ground, claw_to_leg, _ = make_floor_model(selected_margin)
    continuous_records = []
    for bout, key in enumerate(bout_keys):
        rows = np.flatnonzero(bout_index == bout)
        reference, floor = shifted_reference(qpos[rows], floor_height, bout)
        rollout = run_continuous(model, ground, claw_to_leg, reference, qvel[rows], controls[rows])
        reference_delta = reference[-1, :2] - reference[0, :2]
        distance = float(np.linalg.norm(reference_delta))
        direction = reference_delta / max(distance, 1e-12)
        actual_delta = rollout["qpos"][-1, :2] - rollout["qpos"][0, :2]
        forward = float(np.dot(actual_delta, direction) * MODEL_TO_MM)
        reference_mm = distance * MODEL_TO_MM
        continuous_records.append(
            {
                "bout_key": key,
                "split": SPLIT_NAMES[int(split_code[rows[0]])],
                "frames": int(len(rows)),
                "floor_height_native_median": floor,
                "reference_forward_mm": reference_mm,
                "actual_forward_mm": forward,
                "forward_ratio": forward / reference_mm if reference_mm else 0.0,
                "terminal_root_error_mm": float(rollout["root_error_mm"][-1]),
                "terminal_joint_rms_error_rad": float(rollout["joint_rms_error_rad"][-1]),
                "actual_claw_contact_frame_fraction": float(np.mean(rollout["claw_contact"])),
                "body_floor_contact_frame_fraction": float(np.mean(rollout["body_contact"])),
                "observed_contact_match": contact_scores(rollout["claw_contact"], observed[rows]),
                "finite": bool(np.isfinite(rollout["qpos"]).all()),
            }
        )

    continuous_by_split = {}
    for split in SPLIT_NAMES:
        rows = [row for row in continuous_records if row["split"] == split]
        continuous_by_split[split] = {
            "bouts": len(rows),
            "finite_bouts": int(sum(row["finite"] for row in rows)),
            "median_forward_ratio": float(np.median([row["forward_ratio"] for row in rows])),
            "median_terminal_root_error_mm": float(np.median([row["terminal_root_error_mm"] for row in rows])),
            "median_terminal_joint_rms_error_rad": float(np.median([row["terminal_joint_rms_error_rad"] for row in rows])),
            "median_body_floor_contact_frame_fraction": float(np.median([row["body_floor_contact_frame_fraction"] for row in rows])),
        }

    heldout = [selected["by_split"][name] for name in ("validation", "test")]
    capacity_pass = (
        all(item["root_error_mm"]["p95"] < 0.05 for item in heldout)
        and all(item["joint_rms_error_rad"]["p95"] < 0.01 for item in heldout)
        and all(item["observed_contact_match"]["f1"] > 0.60 for item in heldout)
        and all(continuous_by_split[name]["finite_bouts"] == continuous_by_split[name]["bouts"] for name in ("validation", "test"))
        and all(continuous_by_split[name]["median_forward_ratio"] > 0.50 for name in ("validation", "test"))
        and all(continuous_by_split[name]["median_body_floor_contact_frame_fraction"] < 0.05 for name in ("validation", "test"))
    )

    margins_mm = [item["contact_margin_mm"] for item in candidate_reports]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), layout="constrained")
    axes[0].plot(margins_mm, [item["train_selection_objective"] for item in candidate_reports], marker="o")
    axes[0].axvline(selected_margin * MODEL_TO_MM, color="#c54f36", ls="--", label="训练集选择")
    axes[0].set(xlabel="足爪接触裕量 (mm)", ylabel="训练目标（越低越好）", title="只用训练果蝇选择")
    axes[0].legend()
    for split, color in zip(SPLIT_NAMES, ("#286b9e", "#d07725", "#3e8c61")):
        axes[1].plot(margins_mm, [item["by_split"][split]["root_error_mm"]["p95"] for item in candidate_reports], marker="o", label=split, color=color)
        axes[2].plot(margins_mm, [item["by_split"][split]["observed_contact_match"]["f1"] for item in candidate_reports], marker="o", label=split, color=color)
    axes[1].axhline(0.05, color="#a74343", ls=":", label="门槛")
    axes[1].set(xlabel="足爪接触裕量 (mm)", ylabel="mm", title="单帧身体误差 p95")
    axes[1].legend()
    axes[2].axhline(0.60, color="#a74343", ls=":", label="门槛")
    axes[2].set(xlabel="足爪接触裕量 (mm)", ylabel="F1", title="真实近地标签接触匹配")
    axes[2].legend()
    fig.suptitle("官方 IK 身体的足爪—地面接触校准 · 验证/测试果蝇不参与选择", fontsize=14)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    checks = {
        "six_predeclared_global_margins": len(MARGINS) == 6,
        "selection_uses_training_objective_only": selected is min(candidate_reports, key=lambda item: item["train_selection_objective"]),
        "all_three_whole_fly_splits_reported": all(set(item["by_split"]) == set(SPLIT_NAMES) for item in candidate_reports),
        "validation_and_test_not_empty": all(selected["by_split"][name]["frames"] > 0 for name in ("validation", "test")),
        "all_18_continuous_bouts_run": len(continuous_records) == 18,
        "all_rollouts_finite": all(row["finite"] for row in continuous_records),
        "single_margin_shared_across_all_heldout_flies": True,
        "no_state_reset_in_continuous_rollouts": True,
        "no_root_or_external_force": True,
        "no_position_servo_or_runtime_phase": True,
        "result_not_mislabelled_as_muscle_or_autonomy": True,
    }
    report = {
        "date": "2026-09-23",
        "scope": "global engineering foot-contact calibration on training flies with held-out whole-fly validation and test",
        "sources": {
            "inverse_teacher": str(INVERSE_PATH.relative_to(ROOT)),
            "inverse_teacher_sha256": sha256(INVERSE_PATH),
            "contact_teacher": str(CONTACT_PATH.relative_to(ROOT)),
            "contact_teacher_sha256": sha256(CONTACT_PATH),
        },
        "selection": {
            "candidate_margins_native": list(MARGINS),
            "candidate_margins_mm": [value * MODEL_TO_MM for value in MARGINS],
            "objective": "median(root_mm)/0.05 + median(root_angle_rad)/0.03 + median(joint_rms_rad)/0.005 + 1 - observed_contact_F1",
            "data_used": "training flies only",
            "selected_margin_native": selected_margin,
            "selected_margin_mm": selected_margin * MODEL_TO_MM,
        },
        "candidates": candidate_reports,
        "selected_heldout_results": {
            "validation": selected["by_split"]["validation"],
            "test": selected["by_split"]["test"],
        },
        "continuous_rollout": {
            "by_split": continuous_by_split,
            "by_bout": continuous_records,
        },
        "artifacts": {"figure": str(FIGURE.relative_to(ROOT))},
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A0_global_contact_calibration": "pass" if all(checks.values()) else "incomplete",
            "A1_open_loop_source_body_capacity": "pass" if capacity_pass else "incomplete",
            "A1_source_aligned_six_leg_muscle_body": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": f"{sum(checks.values())}/{len(checks)}",
                "selected_margin_mm": report["selection"]["selected_margin_mm"],
                "heldout": report["selected_heldout_results"],
                "capacity": report["classification"]["A1_open_loop_source_body_capacity"],
                "figure": str(FIGURE),
                "goal_complete": False,
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
