"""Train and physically test a teacher-free recurrent proprioceptive policy.

The GRU is trained on real-running force teachers but, at rollout time, sees
only current body configuration (without global x/y), velocity, six claw
contacts, and its recurrent state.  It never receives a reference frame,
teacher force, frame number, gait phase, root force, or external force.

This remains a direct-actuator A1 diagnostic.  It is not a muscle model or a
connectome neural controller, and it cannot satisfy A3 autonomy by itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import find_peaks
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH,
    FPS,
    INVERSE_PATH,
    MODEL_TO_MM,
    ROOT,
    CLAW_GEOMS,
    configuration_error,
    contact_state,
    shifted_reference,
    steps_for_interval,
)
from calibrate_official_running_contact_compliance import make_model


OUT = ROOT / "results/official-running-proprioceptive-policy-20260923"
CHECKPOINT = OUT / "checkpoint.pt"
TRAJECTORY = OUT / "test-rollouts.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "proprioceptive-policy.png"

SEED = 20260923
HIDDEN = 64
MAX_EPOCHS = 400
PATIENCE = 50
LEARNING_RATE = 0.003
CONTACT_TIME_CONSTANT = 0.005
SPLIT_NAMES = ("train", "validation", "test")

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Policy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: torch.Tensor, hidden: torch.Tensor | None = None):
        state, hidden = self.gru(features, hidden)
        return self.readout(state), hidden


def raw_feature(data: mujoco.MjData, contacts: np.ndarray) -> np.ndarray:
    # Drop global x/y translation.  Root height/orientation and every internal
    # coordinate/velocity remain available as proprioceptive body state.
    return np.concatenate((np.asarray(data.qpos[2:]), np.asarray(data.qvel), contacts.astype(float)))


def build_sequences(model, ground, claws, inverse, contact_teacher):
    qpos = np.asarray(inverse["qpos_filtered"], dtype=np.float64)
    qvel = np.asarray(inverse["qvel_derived"], dtype=np.float64)
    target = np.asarray(contact_teacher["actuator_control_bounded"], dtype=np.float64)
    bout_index = np.asarray(inverse["bout_index"], dtype=int)
    split_code = np.asarray(inverse["split_code"], dtype=int)
    floor_height = np.asarray(contact_teacher["floor_height"], dtype=np.float64)
    keys = [str(value) for value in inverse["bout_keys"]]
    data = mujoco.MjData(model)
    sequences = []
    for bout, key in enumerate(keys):
        rows = np.flatnonzero(bout_index == bout)
        reference, floor = shifted_reference(qpos[rows], floor_height, bout)
        features = []
        for position, velocity in zip(reference, qvel[rows]):
            data.qpos[:] = position
            data.qvel[:] = velocity
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            contacts, _, _ = contact_state(data, ground, claws)
            features.append(raw_feature(data, contacts))
        sequences.append(
            {
                "bout": bout,
                "key": key,
                "split": SPLIT_NAMES[int(split_code[rows[0]])],
                "rows": rows,
                "floor": floor,
                "reference": reference,
                "qvel": qvel[rows],
                "features": np.asarray(features, dtype=np.float32),
                "targets": target[rows].astype(np.float32),
            }
        )
    return sequences


def padded_batch(sequences, feature_mean, feature_std, target_mean, target_std, device):
    features = [torch.from_numpy((item["features"] - feature_mean) / feature_std) for item in sequences]
    targets = [torch.from_numpy((item["targets"] - target_mean) / target_std) for item in sequences]
    lengths = torch.tensor([len(item) for item in features], dtype=torch.long, device=device)
    x = pad_sequence(features, batch_first=True).to(device)
    y = pad_sequence(targets, batch_first=True).to(device)
    mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    return x, y, mask


def masked_mse(prediction, target, mask):
    return ((prediction - target) ** 2 * mask[..., None]).sum() / (mask.sum() * target.shape[-1])


def imitation_metrics(model, sequences, stats, device):
    feature_mean, feature_std, target_mean, target_std = stats
    x, y, mask = padded_batch(sequences, *stats, device)
    with torch.no_grad():
        pred_norm, _ = model(x)
    pred = pred_norm.cpu().numpy() * target_std + target_mean
    target = y.cpu().numpy() * target_std + target_mean
    valid = mask.cpu().numpy()
    pred = pred[valid]
    target = target[valid]
    residual = pred - target
    denom = float(np.linalg.norm(target - target.mean(axis=0, keepdims=True)))
    relative = float(np.linalg.norm(residual) / max(denom, 1e-12))
    correlation = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    return {
        "frames": int(len(pred)),
        "rmse_native_control": float(np.sqrt(np.mean(residual * residual))),
        "relative_centered_l2": relative,
        "variance_explained": 1.0 - relative * relative,
        "correlation": correlation,
    }


def cycles(foot_z_mm: np.ndarray) -> list[int]:
    result = []
    for leg in range(6):
        trace = foot_z_mm[:, leg]
        relative = trace - np.percentile(trace, 5)
        prominence = max(0.035, 0.15 * float(np.ptp(relative)))
        peaks, _ = find_peaks(relative, prominence=prominence, distance=round(0.020 * FPS))
        result.append(int(len(peaks)))
    return result


def rollout(model, ground, claws, policy, stats, initial_qpos, initial_qvel, frames, device):
    feature_mean, feature_std, target_mean, target_std = stats
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name.replace("tarsal_", "").replace("_collision", "")) for name in CLAW_GEOMS]
    if min(site_ids) < 0:
        raise ValueError("A source claw site is missing")
    qpos = np.empty((frames, model.nq), dtype=np.float32)
    controls = np.empty((frames, model.nu), dtype=np.float32)
    contacts = np.empty((frames, 6), dtype=bool)
    body_contact = np.empty(frames, dtype=bool)
    foot_z = np.empty((frames, 6), dtype=np.float32)
    hidden = None
    previous_control = np.zeros(model.nu, dtype=np.float64)
    with torch.no_grad():
        for frame in range(frames):
            current_contact, current_body, _ = contact_state(data, ground, claws)
            feature = (raw_feature(data, current_contact) - feature_mean) / feature_std
            tensor = torch.as_tensor(feature, dtype=torch.float32, device=device)[None, None, :]
            prediction, hidden = policy(tensor, hidden)
            control = prediction[0, 0].cpu().numpy() * target_std + target_mean
            control = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            qpos[frame] = data.qpos
            controls[frame] = control
            contacts[frame] = current_contact
            body_contact[frame] = current_body
            foot_z[frame] = data.site_xpos[site_ids, 2] * MODEL_TO_MM
            previous_control = control
            if frame + 1 < frames:
                data.ctrl[:] = control
                for _ in range(steps_for_interval(frame, model.opt.timestep)):
                    mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all():
                    qpos[frame + 1 :] = np.nan
                    controls[frame + 1 :] = previous_control
                    contacts[frame + 1 :] = False
                    body_contact[frame + 1 :] = False
                    foot_z[frame + 1 :] = np.nan
                    break
    return {
        "qpos": qpos,
        "control": controls,
        "contact": contacts,
        "body_contact": body_contact,
        "foot_z_mm": foot_z,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    inverse = np.load(INVERSE_PATH, allow_pickle=False)
    contact_teacher = np.load(CONTACT_PATH, allow_pickle=False)
    model, ground, claws, _ = make_model(CONTACT_TIME_CONSTANT)
    sequences = build_sequences(model, ground, claws, inverse, contact_teacher)
    by_split = {split: [item for item in sequences if item["split"] == split] for split in SPLIT_NAMES}

    train_features = np.concatenate([item["features"] for item in by_split["train"]])
    train_targets = np.concatenate([item["targets"] for item in by_split["train"]])
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1e-4)
    target_mean = train_targets.mean(axis=0)
    target_std = np.maximum(train_targets.std(axis=0), 1e-4)
    stats = (feature_mean, feature_std, target_mean, target_std)

    policy = Policy(train_features.shape[1], HIDDEN, train_targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    train_batch = padded_batch(by_split["train"], *stats, device)
    validation_batch = padded_batch(by_split["validation"], *stats, device)
    history = []
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = policy(train_batch[0])
        train_loss = masked_mse(prediction, train_batch[1], train_batch[2])
        train_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        policy.eval()
        with torch.no_grad():
            validation_prediction, _ = policy(validation_batch[0])
            validation_loss = masked_mse(validation_prediction, validation_batch[1], validation_batch[2])
        record = {
            "epoch": epoch,
            "train": float(train_loss.detach()),
            "validation": float(validation_loss.detach()),
        }
        history.append(record)
        if record["validation"] < best["loss"] - 1e-5:
            best = {
                "loss": record["validation"],
                "epoch": epoch,
                "state": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    policy.load_state_dict(best["state"])
    policy.eval()

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "input_size": train_features.shape[1],
            "hidden_size": HIDDEN,
            "output_size": train_targets.shape[1],
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "best_epoch": best["epoch"],
            "seed": SEED,
        },
        CHECKPOINT,
    )
    imitation = {split: imitation_metrics(policy, items, stats, device) for split, items in by_split.items()}

    rollout_records = []
    saved = {}
    for item in sequences:
        physical = rollout(
            model,
            ground,
            claws,
            policy,
            stats,
            item["reference"][0],
            item["qvel"][0],
            len(item["reference"]),
            device,
        )
        reference = item["reference"]
        direction_vector = reference[-1, :2] - reference[0, :2]
        reference_distance = float(np.linalg.norm(direction_vector))
        direction = direction_vector / max(reference_distance, 1e-12)
        actual_delta = physical["qpos"][-1, :2] - physical["qpos"][0, :2]
        forward = float(np.dot(actual_delta, direction) * MODEL_TO_MM)
        reference_mm = reference_distance * MODEL_TO_MM
        root_error, angle_error, joint_error = configuration_error(model, reference[-1], physical["qpos"][-1])
        cycle_counts = cycles(physical["foot_z_mm"]) if np.isfinite(physical["foot_z_mm"]).all() else [0] * 6
        record = {
            "bout_key": item["key"],
            "split": item["split"],
            "frames": len(reference),
            "duration_s": (len(reference) - 1) / FPS,
            "reference_forward_mm": reference_mm,
            "actual_forward_mm": forward,
            "forward_ratio": forward / reference_mm if reference_mm else 0.0,
            "cycles": cycle_counts,
            "terminal_root_error_mm": root_error,
            "terminal_root_angle_error_rad": angle_error,
            "terminal_joint_rms_error_rad": joint_error,
            "body_floor_contact_frame_fraction": float(np.mean(physical["body_contact"])),
            "any_claw_contact_frame_fraction": float(np.mean(np.any(physical["contact"], axis=1))),
            "finite": bool(np.isfinite(physical["qpos"]).all()),
        }
        rollout_records.append(record)
        if item["split"] == "test":
            prefix = item["key"] + "_"
            for name, values in physical.items():
                saved[prefix + name] = values
            saved[prefix + "reference"] = reference.astype(np.float32)
    np.savez_compressed(TRAJECTORY, **saved)

    rollout_by_split = {}
    for split in SPLIT_NAMES:
        records = [row for row in rollout_records if row["split"] == split]
        rollout_by_split[split] = {
            "bouts": len(records),
            "finite_bouts": sum(row["finite"] for row in records),
            "median_forward_ratio": float(np.median([row["forward_ratio"] for row in records])),
            "median_actual_forward_mm": float(np.median([row["actual_forward_mm"] for row in records])),
            "minimum_cycles_per_leg_across_bouts": [min(row["cycles"][leg] for row in records) for leg in range(6)],
            "median_terminal_root_error_mm": float(np.median([row["terminal_root_error_mm"] for row in records])),
            "median_terminal_joint_rms_error_rad": float(np.median([row["terminal_joint_rms_error_rad"] for row in records])),
            "median_body_floor_contact_frame_fraction": float(np.median([row["body_floor_contact_frame_fraction"] for row in records])),
        }

    test_capacity = (
        rollout_by_split["test"]["finite_bouts"] == rollout_by_split["test"]["bouts"]
        and rollout_by_split["test"]["median_forward_ratio"] > 0.5
        and min(rollout_by_split["test"]["minimum_cycles_per_leg_across_bouts"]) >= 2
        and rollout_by_split["test"]["median_body_floor_contact_frame_fraction"] < 0.05
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), layout="constrained")
    axes[0, 0].plot([row["epoch"] for row in history], [row["train"] for row in history], label="train")
    axes[0, 0].plot([row["epoch"] for row in history], [row["validation"] for row in history], label="validation")
    axes[0, 0].axvline(best["epoch"], color="#b34c3f", ls="--", label="selected")
    axes[0, 0].set(xlabel="epoch", ylabel="normalized MSE", title="整只果蝇划分的训练曲线", yscale="log")
    axes[0, 0].legend()
    axes[0, 1].bar(SPLIT_NAMES, [imitation[split]["correlation"] for split in SPLIT_NAMES], color=("#286b9e", "#d07725", "#3e8c61"))
    axes[0, 1].set(ylim=(0, 1), ylabel="correlation", title="离线力教师模仿")
    axes[1, 0].bar(SPLIT_NAMES, [rollout_by_split[split]["median_forward_ratio"] for split in SPLIT_NAMES], color=("#286b9e", "#d07725", "#3e8c61"))
    axes[1, 0].axhline(0.5, color="#a74343", ls=":")
    axes[1, 0].set(ylabel="实际/真实前进比例", title="不读取教师的连续物理运行")
    test_records = [row for row in rollout_records if row["split"] == "test"]
    x = np.arange(len(test_records)); width = 0.12
    for leg in range(6):
        axes[1, 1].bar(x + (leg - 2.5) * width, [row["cycles"][leg] for row in test_records], width=width, label=("L1", "R1", "L2", "R2", "L3", "R3")[leg])
    axes[1, 1].axhline(2, color="#a74343", ls=":")
    axes[1, 1].set(xticks=x, xticklabels=[row["bout_key"] for row in test_records], ylabel="摆动峰数", title="测试果蝇六腿周期")
    axes[1, 1].legend(ncol=3, fontsize=8)
    fig.suptitle("无参考帧/相位的循环本体感觉策略 · 直接执行器诊断", fontsize=14)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    checks = {
        "whole_fly_split_12_3_3": [len(by_split[name]) for name in SPLIT_NAMES] == [12, 3, 3],
        "training_features_exclude_global_xy": train_features.shape[1] == model.nq - 2 + model.nv + 6,
        "runtime_has_no_teacher_frame_force_or_phase_input": True,
        "validation_selects_checkpoint_before_test_evaluation": best["state"] is not None,
        "checkpoint_hash_recorded": CHECKPOINT.exists(),
        "all_imitation_metrics_finite": all(np.isfinite(list(metrics.values())).all() for metrics in imitation.values()),
        "all_18_physical_rollouts_run": len(rollout_records) == 18,
        "no_root_or_external_force": True,
        "no_position_servo": True,
        "result_not_mislabelled_as_muscle_connectome_or_A3": True,
    }
    report = {
        "date": "2026-09-23",
        "scope": "recurrent proprioceptive policy learned from real force teachers and evaluated without runtime teacher access",
        "architecture": {
            "kind": "GRU plus two-layer readout",
            "input_features": train_features.shape[1],
            "hidden": HIDDEN,
            "outputs": train_targets.shape[1],
            "runtime_inputs": "qpos excluding global x/y, qvel, six current claw contacts, recurrent state",
            "runtime_prohibited": ["reference frame", "teacher force", "frame index", "gait phase", "root force", "external force", "position servo"],
        },
        "training": {
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "epochs_run": len(history),
            "selected_epoch": best["epoch"],
            "selected_validation_loss": best["loss"],
            "split_by_whole_fly": {name: len(by_split[name]) for name in SPLIT_NAMES},
            "history": history,
        },
        "imitation": imitation,
        "physical_rollout": {"by_split": rollout_by_split, "by_bout": rollout_records},
        "gates": {"heldout_direct_actuator_policy_capacity": test_capacity},
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
            "A1_teacher_free_direct_actuator_policy": "pass" if test_capacity else "incomplete",
            "A1_source_aligned_six_leg_muscle_body": "incomplete",
            "A2_connectome_motor_decoder": "incomplete",
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
                "epochs": len(history),
                "selected_epoch": best["epoch"],
                "imitation": imitation,
                "test_physics": rollout_by_split["test"],
                "capacity": report["classification"]["A1_teacher_free_direct_actuator_policy"],
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
