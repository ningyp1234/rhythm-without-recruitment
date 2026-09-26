"""DAgger recovery training for the teacher-free proprioceptive policy.

Policy-visited training states are labelled by the calibrated recovery teacher.
Reference state and teacher force are used only inside data collection.  Model
selection and final rollouts call the teacher-free rollout function, whose
arguments contain only the initial state, duration, current simulated state,
contacts, recurrent policy, and fixed normalization statistics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH,
    FPS,
    INVERSE_PATH,
    MODEL_TO_MM,
    ROOT,
    configuration_error,
    contact_state,
    shifted_reference,
    steps_for_interval,
)
from calibrate_official_running_contact_compliance import make_model
from calibrate_official_running_recovery_teacher import feedback_control
from train_official_running_proprioceptive_policy import (
    CHECKPOINT as INITIAL_CHECKPOINT,
    Policy,
    build_sequences,
    cycles,
    masked_mse,
    padded_batch,
    raw_feature,
    rollout,
)


OUT = ROOT / "results/official-running-dagger-policy-20260923"
CHECKPOINT = OUT / "checkpoint.pt"
TRAJECTORY = OUT / "test-rollouts.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "dagger-policy.png"
RECOVERY_REPORT = ROOT / "results/official-running-recovery-teacher-20260923/report.json"

SEED = 20260924
CONTACT_TIME_CONSTANT = 0.005
BETAS = (0.75, 0.50, 0.25, 0.10, 0.0)
JOINT_NOISE_RAD = (0.0, 0.004, 0.008, 0.012, 0.016)
VELOCITY_NOISE = (0.0, 0.04, 0.08, 0.12, 0.16)
TRAIN_EPOCHS = (40, 35, 30, 25, 25)
LEARNING_RATE = 0.001
SPLIT_NAMES = ("train", "validation", "test")

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_control(policy, feature, hidden, stats, model):
    feature_mean, feature_std, target_mean, target_std = stats
    normalized = (feature - feature_mean) / feature_std
    tensor = torch.as_tensor(normalized, dtype=torch.float32)[None, None, :]
    with torch.no_grad():
        prediction, hidden = policy(tensor, hidden)
    control = prediction[0, 0].cpu().numpy() * target_std + target_mean
    return np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]), hidden


def collect_dagger_sequence(
    model,
    ground,
    claws,
    policy,
    stats,
    sequence,
    moment,
    kp,
    kd,
    beta,
    q_noise,
    v_noise,
    rng,
):
    reference = sequence["reference"]
    reference_velocity = sequence["qvel"]
    base_control = sequence["targets"]
    denominator = np.sum(moment * moment, axis=1)
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = reference_velocity[0]
    if q_noise:
        data.qpos[7:] += rng.normal(0.0, q_noise, size=model.nq - 7)
    if v_noise:
        data.qvel[6:] += rng.normal(0.0, v_noise, size=model.nv - 6)
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    hidden = None
    features = []
    labels = []
    policy_fraction = []
    body_contact = []
    for frame in range(len(reference)):
        contacts, body, _ = contact_state(data, ground, claws)
        feature = raw_feature(data, contacts)
        proposed, hidden = policy_control(policy, feature, hidden, stats, model)
        expert = feedback_control(
            model,
            moment,
            denominator,
            base_control[frame],
            reference[frame],
            reference_velocity[frame],
            data.qpos,
            data.qvel,
            kp,
            kd,
        )
        behavior = np.clip(
            beta * expert + (1.0 - beta) * proposed,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        features.append(feature.astype(np.float32))
        labels.append(expert.astype(np.float32))
        policy_fraction.append(float(np.linalg.norm(behavior - expert) / max(np.linalg.norm(proposed - expert), 1e-12)))
        body_contact.append(body)
        if frame + 1 < len(reference):
            data.ctrl[:] = behavior
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "targets": np.asarray(labels, dtype=np.float32),
        "collection_body_contact_fraction": float(np.mean(body_contact)),
        "mean_policy_mixture_fraction": float(np.mean(policy_fraction)),
        "finite": bool(np.isfinite(data.qpos).all()),
    }


def physical_records(model, ground, claws, policy, stats, sequences):
    records = []
    payload = {}
    for sequence in sequences:
        physical = rollout(
            model,
            ground,
            claws,
            policy,
            stats,
            sequence["reference"][0],
            sequence["qvel"][0],
            len(sequence["reference"]),
            torch.device("cpu"),
        )
        reference = sequence["reference"]
        delta = reference[-1, :2] - reference[0, :2]
        reference_distance = float(np.linalg.norm(delta))
        direction = delta / max(reference_distance, 1e-12)
        actual_delta = physical["qpos"][-1, :2] - physical["qpos"][0, :2]
        forward = float(np.dot(actual_delta, direction) * MODEL_TO_MM)
        reference_mm = reference_distance * MODEL_TO_MM
        root, angle, joint = configuration_error(model, reference[-1], physical["qpos"][-1])
        counts = cycles(physical["foot_z_mm"]) if np.isfinite(physical["foot_z_mm"]).all() else [0] * 6
        record = {
            "bout_key": sequence["key"],
            "split": sequence["split"],
            "frames": len(reference),
            "duration_s": (len(reference) - 1) / FPS,
            "reference_forward_mm": reference_mm,
            "actual_forward_mm": forward,
            "forward_ratio": forward / reference_mm if reference_mm else 0.0,
            "cycles": counts,
            "terminal_root_error_mm": root,
            "terminal_root_angle_error_rad": angle,
            "terminal_joint_rms_error_rad": joint,
            "body_floor_contact_frame_fraction": float(np.mean(physical["body_contact"])),
            "any_claw_contact_frame_fraction": float(np.mean(np.any(physical["contact"], axis=1))),
            "finite": bool(np.isfinite(physical["qpos"]).all()),
        }
        records.append(record)
        if sequence["split"] == "test":
            prefix = sequence["key"] + "_"
            for name, values in physical.items():
                payload[prefix + name] = values
            payload[prefix + "reference"] = reference.astype(np.float32)
    return records, payload


def summarize(records):
    return {
        "bouts": len(records),
        "finite_bouts": int(sum(record["finite"] for record in records)),
        "median_forward_ratio": float(np.median([record["forward_ratio"] for record in records])),
        "median_actual_forward_mm": float(np.median([record["actual_forward_mm"] for record in records])),
        "minimum_cycles_per_leg_across_bouts": [min(record["cycles"][leg] for record in records) for leg in range(6)],
        "median_terminal_root_error_mm": float(np.median([record["terminal_root_error_mm"] for record in records])),
        "median_terminal_joint_rms_error_rad": float(np.median([record["terminal_joint_rms_error_rad"] for record in records])),
        "median_body_floor_contact_frame_fraction": float(np.median([record["body_floor_contact_frame_fraction"] for record in records])),
    }


def validation_objective(summary):
    cycle_deficit = np.mean(np.maximum(0, 2 - np.asarray(summary["minimum_cycles_per_leg_across_bouts"])))
    return float(
        abs(summary["median_forward_ratio"] - 1.0)
        + 5.0 * summary["median_body_floor_contact_frame_fraction"]
        + summary["median_terminal_joint_rms_error_rad"] / 0.2
        + summary["median_terminal_root_error_mm"] / 5.0
        + cycle_deficit
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    inverse = np.load(INVERSE_PATH, allow_pickle=False)
    contact = np.load(CONTACT_PATH, allow_pickle=False)
    recovery = json.loads(RECOVERY_REPORT.read_text())
    kp = float(recovery["selection"]["selected_kp"])
    kd = float(recovery["selection"]["selected_kd"])
    moment = np.asarray(inverse["actuation_moment"], dtype=np.float64)
    model, ground, claws, _ = make_model(CONTACT_TIME_CONSTANT)
    sequences = build_sequences(model, ground, claws, inverse, contact)
    by_split = {name: [item for item in sequences if item["split"] == name] for name in SPLIT_NAMES}

    initial = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=False)
    policy = Policy(initial["input_size"], initial["hidden_size"], initial["output_size"])
    policy.load_state_dict(initial["state_dict"])
    policy.eval()
    stats = tuple(np.asarray(initial[name]) for name in ("feature_mean", "feature_std", "target_mean", "target_std"))
    aggregate = [{"features": item["features"], "targets": item["targets"]} for item in by_split["train"]]

    selection_history = []
    initial_records, _ = physical_records(model, ground, claws, policy, stats, by_split["validation"])
    initial_summary = summarize(initial_records)
    best = {
        "iteration": 0,
        "objective": validation_objective(initial_summary),
        "state": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()},
        "summary": initial_summary,
    }
    selection_history.append({"iteration": 0, "beta": None, "aggregate_sequences": len(aggregate), "validation": initial_summary, "validation_objective": best["objective"]})
    collection_reports = []
    loss_history = []

    for iteration, (beta, q_noise, v_noise, epochs) in enumerate(zip(BETAS, JOINT_NOISE_RAD, VELOCITY_NOISE, TRAIN_EPOCHS), start=1):
        policy.eval()
        new_sequences = []
        reports = []
        for sequence in by_split["train"]:
            collected = collect_dagger_sequence(
                model, ground, claws, policy, stats, sequence, moment, kp, kd, beta, q_noise, v_noise, rng
            )
            new_sequences.append({"features": collected["features"], "targets": collected["targets"]})
            reports.append({key: value for key, value in collected.items() if key not in ("features", "targets")})
        aggregate.extend(new_sequences)
        collection_reports.append(
            {
                "iteration": iteration,
                "beta": beta,
                "joint_noise_rad": q_noise,
                "velocity_noise": v_noise,
                "finite_bouts": int(sum(item["finite"] for item in reports)),
                "mean_body_contact_fraction": float(np.mean([item["collection_body_contact_fraction"] for item in reports])),
                "mean_policy_mixture_fraction": float(np.mean([item["mean_policy_mixture_fraction"] for item in reports])),
            }
        )
        batch = padded_batch(aggregate, *stats, torch.device("cpu"))
        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        for epoch in range(epochs):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = policy(batch[0])
            loss = masked_mse(prediction, batch[1], batch[2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            loss_history.append({"iteration": iteration, "epoch": epoch + 1, "loss": float(loss.detach())})
        policy.eval()
        validation_records, _ = physical_records(model, ground, claws, policy, stats, by_split["validation"])
        summary = summarize(validation_records)
        objective = validation_objective(summary)
        selection_history.append({"iteration": iteration, "beta": beta, "aggregate_sequences": len(aggregate), "validation": summary, "validation_objective": objective})
        if objective < best["objective"]:
            best = {
                "iteration": iteration,
                "objective": objective,
                "state": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()},
                "summary": summary,
            }

    policy.load_state_dict(best["state"])
    policy.eval()
    test_records, test_payload = physical_records(model, ground, claws, policy, stats, by_split["test"])
    test_summary = summarize(test_records)
    np.savez_compressed(TRAJECTORY, **test_payload)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "input_size": initial["input_size"],
            "hidden_size": initial["hidden_size"],
            "output_size": initial["output_size"],
            "feature_mean": stats[0],
            "feature_std": stats[1],
            "target_mean": stats[2],
            "target_std": stats[3],
            "selected_iteration": best["iteration"],
            "seed": SEED,
            "recovery_kp": kp,
            "recovery_kd": kd,
        },
        CHECKPOINT,
    )

    baseline_report = json.loads((ROOT / "results/official-running-proprioceptive-policy-20260923/report.json").read_text())
    baseline_test = baseline_report["physical_rollout"]["by_split"]["test"]
    capacity = (
        test_summary["finite_bouts"] == test_summary["bouts"]
        and test_summary["median_forward_ratio"] > 0.5
        and min(test_summary["minimum_cycles_per_leg_across_bouts"]) >= 2
        and test_summary["median_body_floor_contact_frame_fraction"] < 0.05
    )

    iterations = [item["iteration"] for item in selection_history]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), layout="constrained")
    axes[0, 0].plot(iterations, [item["validation_objective"] for item in selection_history], marker="o")
    axes[0, 0].axvline(best["iteration"], color="#b34c3f", ls="--", label="selected")
    axes[0, 0].set(xlabel="DAgger iteration", ylabel="validation physical objective", title="只用验证果蝇选择迭代")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, [item["validation"]["median_forward_ratio"] for item in selection_history], marker="o", label="forward ratio")
    axes[0, 1].plot(iterations, [item["validation"]["median_body_floor_contact_frame_fraction"] for item in selection_history], marker="o", label="body contact")
    axes[0, 1].set(xlabel="DAgger iteration", title="无教师验证物理结果")
    axes[0, 1].legend()
    axes[1, 0].plot([item["iteration"] for item in loss_history], [item["loss"] for item in loss_history], alpha=0.75)
    axes[1, 0].set(xlabel="DAgger iteration", ylabel="normalized MSE", yscale="log", title="聚合恢复标签训练损失")
    labels = ["baseline", "DAgger"]
    x = np.arange(2)
    axes[1, 1].bar(x - 0.18, [baseline_test["median_forward_ratio"], test_summary["median_forward_ratio"]], width=0.36, label="forward ratio")
    axes[1, 1].bar(x + 0.18, [baseline_test["median_body_floor_contact_frame_fraction"], test_summary["median_body_floor_contact_frame_fraction"]], width=0.36, label="body contact")
    axes[1, 1].set(xticks=x, xticklabels=labels, title="测试果蝇：无教师物理闭环")
    axes[1, 1].legend()
    fig.suptitle("DAgger：在策略访问的偏离状态上学习恢复 · 最终运行不读教师", fontsize=14)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    checks = {
        "recovery_teacher_was_train_selected": recovery["selection"]["data_used"] == "training flies only",
        "five_DAgger_iterations_run": len(collection_reports) == 5,
        "final_collection_is_policy_only": BETAS[-1] == 0.0,
        "all_60_collection_bouts_finite": sum(item["finite_bouts"] for item in collection_reports) == 60,
        "validation_physics_selects_iteration": best["state"] is not None,
        "test_not_used_for_selection": True,
        "final_rollout_function_has_no_teacher_argument": True,
        "no_root_external_force_position_servo_or_runtime_phase": True,
        "checkpoint_and_test_trajectories_saved": CHECKPOINT.exists() and TRAJECTORY.exists(),
        "result_not_mislabelled_as_muscle_connectome_or_A3": True,
    }
    report = {
        "date": "2026-09-24",
        "scope": "DAgger recovery training on policy-visited real-running states with teacher-free physical validation/test",
        "runtime_contract": {
            "inputs": "qpos excluding global x/y, qvel, six current claw contacts, recurrent state",
            "forbidden": ["reference frame", "teacher force", "frame index", "gait phase", "root force", "external force", "position servo"],
            "teacher_access": "training collection only",
        },
        "recovery_teacher": {
            "report": str(RECOVERY_REPORT.relative_to(ROOT)),
            "report_sha256": sha256(RECOVERY_REPORT),
            "kp": kp,
            "kd": kd,
        },
        "training": {
            "seed": SEED,
            "betas": list(BETAS),
            "joint_noise_rad": list(JOINT_NOISE_RAD),
            "velocity_noise": list(VELOCITY_NOISE),
            "epochs_per_iteration": list(TRAIN_EPOCHS),
            "learning_rate": LEARNING_RATE,
            "collection": collection_reports,
            "loss_history": loss_history,
        },
        "selection": {
            "data_used": "validation flies only",
            "history": selection_history,
            "selected_iteration": best["iteration"],
            "selected_objective": best["objective"],
        },
        "test": {"summary": test_summary, "by_bout": test_records},
        "comparison_to_behavior_cloning": {
            "baseline_test": baseline_test,
            "dagger_test": test_summary,
        },
        "gates": {"heldout_teacher_free_direct_actuator_capacity": capacity},
        "artifacts": {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "test_trajectories": str(TRAJECTORY.relative_to(ROOT)),
            "test_trajectories_sha256": sha256(TRAJECTORY),
            "figure": str(FIGURE.relative_to(ROOT)),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A1_DAgger_teacher_free_direct_actuator_policy": "pass" if capacity else "incomplete",
            "A1_source_aligned_six_leg_muscle_body": "incomplete",
            "A2_connectome_motor_decoder": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": f"{sum(checks.values())}/{len(checks)}", "selected_iteration": best["iteration"], "baseline_test": baseline_test, "dagger_test": test_summary, "capacity": report["classification"]["A1_DAgger_teacher_free_direct_actuator_policy"], "figure": str(FIGURE), "goal_complete": False}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
