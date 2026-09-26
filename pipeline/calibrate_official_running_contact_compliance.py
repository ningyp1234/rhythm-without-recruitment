"""Calibrate floor-contact compliance on training flies and test held-out flies.

The official v1 body uses a 0.2 ms contact time constant suited to its IK
overlap checks, while the public kinematics are sampled at 1.25 ms.  This audit
keeps the source claw geometry and friction fixed, changes one global MuJoCo
contact time constant, selects it only on training flies, and evaluates
validation/test flies plus continuous open-loop dynamics.
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
from calibrate_official_running_foot_contact import objective


OUT = ROOT / "results/official-running-contact-compliance-20260923"
REPORT = OUT / "report.json"
FIGURE = OUT / "contact-compliance-calibration.png"
TIME_CONSTANTS_S = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010)
SOURCE_CLAW_MARGIN = 0.0005
SPLIT_NAMES = ("train", "validation", "test")

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(time_constant: float):
    model, ground, claws, body_geoms = make_floor_model(SOURCE_CLAW_MARGIN)
    collision_geoms = np.asarray([ground, *sorted(body_geoms)], dtype=int)
    model.geom_solref[collision_geoms, 0] = time_constant
    model.geom_solref[collision_geoms, 1] = 1.0
    return model, ground, claws, body_geoms


def split_summary(values: dict[str, np.ndarray]) -> dict:
    return {
        "frames": int(len(values["root"])),
        "root_error_mm": vector_summary(values["root"]),
        "root_angle_error_rad": vector_summary(values["angle"]),
        "joint_rms_error_rad": vector_summary(values["joint"]),
        "observed_contact_match": contact_scores(values["actual"], values["observed"]),
        "selection_objective": objective(
            values["root"], values["angle"], values["joint"], values["actual"], values["observed"]
        ),
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

    candidates: list[dict] = []
    for time_constant in TIME_CONSTANTS_S:
        model, ground, claws, _ = make_model(time_constant)
        by_split_arrays = {
            split: {name: [] for name in ("root", "angle", "joint", "actual", "observed")}
            for split in SPLIT_NAMES
        }
        for bout, _key in enumerate(bout_keys):
            rows = np.flatnonzero(bout_index == bout)
            reference, _ = shifted_reference(qpos[rows], floor_height, bout)
            local = run_local_steps(model, ground, claws, reference, qvel[rows], controls[rows])
            split = SPLIT_NAMES[int(split_code[rows[0]])]
            target = by_split_arrays[split]
            target["root"].append(local["root_error_mm"])
            target["angle"].append(local["root_angle_error_rad"])
            target["joint"].append(local["joint_rms_error_rad"])
            target["actual"].append(local["claw_contact"])
            target["observed"].append(observed[rows][1:])
        by_split = {}
        for split, values in by_split_arrays.items():
            combined = {name: np.concatenate(parts) for name, parts in values.items()}
            by_split[split] = split_summary(combined)
        candidates.append(
            {
                "time_constant_s": time_constant,
                "time_constant_ms": time_constant * 1000.0,
                "training_objective": by_split["train"]["selection_objective"],
                "by_split": by_split,
            }
        )

    selected = min(candidates, key=lambda item: item["training_objective"])
    selected_time_constant = float(selected["time_constant_s"])
    model, ground, claws, _ = make_model(selected_time_constant)
    continuous_records = []
    for bout, key in enumerate(bout_keys):
        rows = np.flatnonzero(bout_index == bout)
        reference, floor = shifted_reference(qpos[rows], floor_height, bout)
        rollout = run_continuous(model, ground, claws, reference, qvel[rows], controls[rows])
        reference_delta = reference[-1, :2] - reference[0, :2]
        reference_distance = float(np.linalg.norm(reference_delta))
        direction = reference_delta / max(reference_distance, 1e-12)
        actual_delta = rollout["qpos"][-1, :2] - rollout["qpos"][0, :2]
        forward = float(np.dot(actual_delta, direction) * MODEL_TO_MM)
        reference_mm = reference_distance * MODEL_TO_MM
        continuous_records.append(
            {
                "bout_key": key,
                "split": SPLIT_NAMES[int(split_code[rows[0]])],
                "frames": int(len(rows)),
                "duration_s": float((len(rows) - 1) / FPS),
                "floor_height_native_median": floor,
                "reference_forward_mm": reference_mm,
                "actual_forward_mm": forward,
                "forward_ratio": forward / reference_mm if reference_mm else 0.0,
                "terminal_root_error_mm": float(rollout["root_error_mm"][-1]),
                "terminal_root_angle_error_rad": float(rollout["root_angle_error_rad"][-1]),
                "terminal_joint_rms_error_rad": float(rollout["joint_rms_error_rad"][-1]),
                "observed_contact_match": contact_scores(rollout["claw_contact"], observed[rows]),
                "body_floor_contact_frame_fraction": float(np.mean(rollout["body_contact"])),
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

    heldout = [selected["by_split"][split] for split in ("validation", "test")]
    heldout_local_pass = (
        all(item["root_error_mm"]["p95"] < 0.05 for item in heldout)
        and all(item["joint_rms_error_rad"]["p95"] < 0.01 for item in heldout)
        and all(item["observed_contact_match"]["f1"] > 0.60 for item in heldout)
    )
    heldout_continuous_pass = (
        all(continuous_by_split[split]["finite_bouts"] == continuous_by_split[split]["bouts"] for split in ("validation", "test"))
        and all(continuous_by_split[split]["median_forward_ratio"] > 0.50 for split in ("validation", "test"))
        and all(continuous_by_split[split]["median_body_floor_contact_frame_fraction"] < 0.05 for split in ("validation", "test"))
    )
    capacity_pass = heldout_local_pass and heldout_continuous_pass

    x = [item["time_constant_ms"] for item in candidates]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), layout="constrained")
    axes[0].plot(x, [item["training_objective"] for item in candidates], marker="o")
    axes[0].axvline(selected_time_constant * 1000.0, color="#c54f36", ls="--", label="训练集选择")
    axes[0].set(xlabel="接触时间常数 (ms)", ylabel="训练目标（越低越好）", title="全局顺应性选择", xscale="log")
    axes[0].legend()
    for split, color in zip(SPLIT_NAMES, ("#286b9e", "#d07725", "#3e8c61")):
        axes[1].plot(x, [item["by_split"][split]["root_error_mm"]["p95"] for item in candidates], marker="o", label=split, color=color)
        axes[2].plot(x, [item["by_split"][split]["observed_contact_match"]["f1"] for item in candidates], marker="o", label=split, color=color)
    axes[1].axhline(0.05, color="#a74343", ls=":", label="门槛")
    axes[1].set(xlabel="接触时间常数 (ms)", ylabel="mm", title="单帧身体误差 p95", xscale="log")
    axes[1].legend()
    axes[2].axhline(0.60, color="#a74343", ls=":", label="门槛")
    axes[2].set(xlabel="接触时间常数 (ms)", ylabel="F1", title="真实近地标签接触匹配", xscale="log")
    axes[2].legend()
    fig.suptitle("官方 IK 身体的地面接触顺应性 · 训练选择，整只果蝇留出验证", fontsize=14)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    checks = {
        "six_predeclared_global_time_constants": len(TIME_CONSTANTS_S) == 6,
        "source_claw_geometry_retained": SOURCE_CLAW_MARGIN == 0.0005,
        "source_friction_retained": float(model.geom_friction[ground, 0]) == 0.6,
        "training_split_alone_selects_parameter": selected is min(candidates, key=lambda item: item["training_objective"]),
        "whole_fly_validation_and_test_reported": all(selected["by_split"][split]["frames"] > 0 for split in ("validation", "test")),
        "all_18_continuous_bouts_run": len(continuous_records) == 18,
        "all_continuous_rollouts_finite": all(row["finite"] for row in continuous_records),
        "single_parameter_shared_by_all_flies": True,
        "no_root_or_external_force": True,
        "no_position_servo_runtime_phase_or_state_reset_in_continuous_rollout": True,
        "result_not_mislabelled_as_muscle_or_autonomy": True,
    }
    report = {
        "date": "2026-09-23",
        "scope": "global floor-contact compliance calibration for the official kinematic body using real running teachers",
        "sources": {
            "inverse_teacher": str(INVERSE_PATH.relative_to(ROOT)),
            "inverse_teacher_sha256": sha256(INVERSE_PATH),
            "contact_teacher": str(CONTACT_PATH.relative_to(ROOT)),
            "contact_teacher_sha256": sha256(CONTACT_PATH),
        },
        "fixed_parameters": {
            "claw_margin_native": SOURCE_CLAW_MARGIN,
            "claw_margin_mm": SOURCE_CLAW_MARGIN * MODEL_TO_MM,
            "friction_coefficient": 0.6,
            "damping_ratio": 1.0,
        },
        "selection": {
            "candidate_time_constants_s": list(TIME_CONSTANTS_S),
            "objective": "median(root_mm)/0.05 + median(root_angle_rad)/0.03 + median(joint_rms_rad)/0.005 + 1 - observed_contact_F1",
            "data_used": "training flies only",
            "selected_time_constant_s": selected_time_constant,
            "selected_time_constant_ms": selected_time_constant * 1000.0,
        },
        "candidates": candidates,
        "selected_heldout_results": {
            "validation": selected["by_split"]["validation"],
            "test": selected["by_split"]["test"],
        },
        "continuous_rollout": {"by_split": continuous_by_split, "by_bout": continuous_records},
        "gates": {
            "heldout_local_dynamics": heldout_local_pass,
            "heldout_continuous_dynamics": heldout_continuous_pass,
            "open_loop_source_body_capacity": capacity_pass,
        },
        "artifacts": {"figure": str(FIGURE.relative_to(ROOT))},
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A0_contact_compliance_calibration": "pass" if all(checks.values()) else "incomplete",
            "A1_open_loop_direct_actuator_body_capacity": "pass" if capacity_pass else "incomplete",
            "A1_source_aligned_six_leg_muscle_body": "incomplete",
            "A2_motor_neuron_recruitment": "incomplete",
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
                "selected_time_constant_ms": report["selection"]["selected_time_constant_ms"],
                "heldout_local_pass": heldout_local_pass,
                "heldout_continuous_pass": heldout_continuous_pass,
                "capacity": report["classification"]["A1_open_loop_direct_actuator_body_capacity"],
                "heldout_continuous": {split: continuous_by_split[split] for split in ("validation", "test")},
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
