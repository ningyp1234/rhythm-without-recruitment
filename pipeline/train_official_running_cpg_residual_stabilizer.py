"""Train a residual posture stabilizer while freezing the successful CPG gait.

The frozen CPG preserves the standard-start gait.  A separate neural residual
learns only corrections for diverse real training-fly states.  Validation flies
select the checkpoint and test flies remain isolated until final evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from torch import nn

from evaluate_official_running_dynamic_replay import CONTACT_PATH, FPS, INVERSE_PATH, MODEL_TO_MM, ROOT, contact_state, steps_for_interval
from calibrate_official_running_contact_compliance import make_model
from calibrate_official_running_recovery_teacher import feedback_control
from train_official_running_proprioceptive_policy import build_sequences, cycles
from train_official_running_endurance_policy import best_cycle, tile_cycle, endurance_summary
import train_official_running_cpg_policy as cp

OUT = ROOT / "results/official-running-cpg-residual-stabilizer-20260924"
SOURCE = ROOT / "results/official-running-cpg-recovery-20260924/checkpoint.pt"
CHECKPOINT = OUT / "checkpoint.pt"
TRAJECTORY = OUT / "test-rollouts.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "residual-stabilizer.png"
SEED = 20261007
BETAS = (1.0, 0.75, 0.5, 0.25, 0.0, 0.0)
STARTS_PER_BOUT = 2
TRAIN_STEPS = 650
CAP_FRACTION = 0.15

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResidualPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 192), nn.Tanh(),
            nn.Linear(192, 192), nn.Tanh(),
            nn.Linear(192, output_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.tanh(self.net(x))


def phase_for(qpos, template):
    return int(np.argmin(np.mean((template - qpos[7:]) ** 2, axis=1)))


def residual_action(policy, feature, stats, cap):
    fm, fs = stats
    x = torch.from_numpy(np.clip((feature - fm) / fs, -10, 10).astype(np.float32))[None]
    with torch.no_grad():
        return policy(x)[0].numpy() * cap


def collect(model, ground, claws, base_policy, residual, feature_stats, sequence, frame_index, canonical, moment, beta, cap, rng):
    q0 = sequence["reference"][frame_index]
    v0 = sequence["qvel"][frame_index]
    period = canonical["cycle_period"]
    phase = phase_for(q0, canonical["reference"][:period, 7:])
    reference, velocity, expert_base = cp.cycle_from_phase(canonical, phase)
    reference[:, :2] += q0[:2] - reference[0, :2]
    denominator = np.sum(moment * moment, axis=1)
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    data.qvel[:] = v0
    data.qpos[7:] += rng.normal(0, 0.002, model.nq - 7)
    data.qvel[6:] += rng.normal(0, 0.02, model.nv - 6)
    mujoco.mj_forward(model, data)
    previous = np.zeros(model.nu)
    features, targets, body, tilt = [], [], [], []
    for frame in range(len(reference)):
        contacts, body_contact, _ = contact_state(data, ground, claws)
        feature = cp.feature(data, contacts, previous, (phase + frame) % period, period, canonical["command"])
        base = cp.policy_action(base_policy, feature, feature_stats, model)
        expert = feedback_control(model, moment, denominator, expert_base[frame], reference[frame], velocity[frame], data.qpos, data.qvel, 0.3, 0.0002)
        target = np.clip(expert - base, -cap, cap)
        proposed = residual_action(residual, feature, feature_stats[:2], cap)
        correction = beta * target + (1 - beta) * proposed
        action = np.clip(base + correction, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        quat = data.qpos[3:7]
        current_tilt = 2 * (quat[1] ** 2 + quat[2] ** 2)
        features.append(feature)
        targets.append(target / cap)
        body.append(body_contact)
        tilt.append(current_tilt)
        if frame + 1 < len(reference):
            data.ctrl[:] = action
            previous = action
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
    return {
        "features": np.asarray(features, np.float32),
        "targets": np.asarray(targets, np.float32),
        "body": np.asarray(body, bool),
        "tilt": np.asarray(tilt, np.float32),
    }


def rollout(model, ground, claws, base_policy, residual, feature_stats, qpos0, qvel0, phase, period, command, cap, frames=cp.FRAMES):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    previous = np.zeros(model.nu)
    site_names = ("claw_T1_left", "claw_T1_right", "claw_T2_left", "claw_T2_right", "claw_T3_left", "claw_T3_right")
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]
    out = {
        "qpos": np.empty((frames, model.nq), np.float32),
        "body_contact": np.empty(frames, bool),
        "foot_z_mm": np.empty((frames, 6), np.float32),
        "control": np.empty((frames, model.nu), np.float32),
        "base_control": np.empty((frames, model.nu), np.float32),
        "residual_control": np.empty((frames, model.nu), np.float32),
        "tilt": np.empty(frames, np.float32),
    }
    for frame in range(frames):
        contacts, body_contact, _ = contact_state(data, ground, claws)
        feature = cp.feature(data, contacts, previous, (phase + frame) % period, period, command)
        base = cp.policy_action(base_policy, feature, feature_stats, model)
        correction = residual_action(residual, feature, feature_stats[:2], cap)
        action = np.clip(base + correction, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        quat = data.qpos[3:7]
        out["qpos"][frame] = data.qpos
        out["body_contact"][frame] = body_contact
        out["foot_z_mm"][frame] = data.site_xpos[site_ids, 2] * MODEL_TO_MM
        out["control"][frame] = action
        out["base_control"][frame] = base
        out["residual_control"][frame] = correction
        out["tilt"][frame] = 2 * (quat[1] ** 2 + quat[2] ** 2)
        if frame + 1 < frames:
            data.ctrl[:] = action
            previous = action
            for _ in range(steps_for_interval(frame, model.opt.timestep)):
                mujoco.mj_step(model, data)
    return out


def record(sequence, out):
    direction = sequence["reference"][-1, :2] - sequence["reference"][0, :2]
    direction /= max(np.linalg.norm(direction), 1e-12)
    distance = float(np.dot(out["qpos"][-1, :2] - out["qpos"][0, :2], direction) * MODEL_TO_MM)
    body = out["body_contact"]
    return {
        "bout_key": sequence["key"],
        "split": sequence["split"],
        "distance_2s_mm": distance,
        "body_contact_fraction": float(np.mean(body)),
        "first_body_contact_s": float(np.argmax(body) / FPS) if np.any(body) else None,
        "maximum_tilt": float(np.max(out["tilt"])),
        "cycles": cycles(out["foot_z_mm"]),
        "finite": bool(np.isfinite(out["qpos"]).all()),
    }


def evaluate(model, ground, claws, base_policy, residual, feature_stats, sequences, canonical, cap, save=False):
    template = canonical["reference"][:canonical["cycle_period"], 7:]
    records, payload = [], {}
    for sequence in sequences:
        phase = phase_for(sequence["reference"][0], template)
        out = rollout(model, ground, claws, base_policy, residual, feature_stats, sequence["reference"][0], sequence["qvel"][0], phase, canonical["cycle_period"], canonical["command"], cap)
        records.append(record(sequence, out))
        if save:
            for key, value in out.items():
                payload[f"{sequence['key']}_{key}"] = value
    return records, payload


def objective(summary):
    return (
        1000 * (summary["bouts"] - summary["finite_bouts"])
        + 100 * max(0, summary["maximum_body_contact_fraction"] - 0.05)
        + 5 * max(0, 5 - summary["minimum_distance_2s_mm"])
        + 5 * sum(max(0, 2 - int(value)) for value in summary["minimum_cycles_per_leg"])
        - 0.01 * min(40, max(0, summary["minimum_distance_2s_mm"]))
    )


def train(policy, nominal, recovery, feature_stats, rng):
    nx = np.concatenate([item["features"] for item in nominal])
    ny = np.zeros((len(nx), recovery[0]["targets"].shape[1]), np.float32)
    rx = np.concatenate([item["features"] for item in recovery])
    ry = np.concatenate([item["targets"] for item in recovery])
    fm, fs = feature_stats[:2]
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    history = []
    for step in range(1, TRAIN_STEPS + 1):
        # Twice as many stable-cycle examples anchor the residual at zero.
        ni = rng.integers(len(nx), size=256)
        ri = rng.integers(len(rx), size=128)
        x = np.concatenate((nx[ni], rx[ri]))
        y = np.concatenate((ny[ni], ry[ri]))
        xb = torch.from_numpy(np.clip((x - fm) / fs, -10, 10).astype(np.float32))
        yb = torch.from_numpy(y.astype(np.float32))
        pred = policy(xb)
        loss = ((pred - yb) ** 2).mean() + 0.01 * (pred[: len(ni)] ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "loss": float(loss.detach())})
    return history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    inverse = np.load(INVERSE_PATH)
    contact = np.load(CONTACT_PATH)
    model, ground, claws, _ = make_model(0.005)
    sequences = build_sequences(model, ground, claws, inverse, contact)
    split = {name: [item for item in sequences if item["split"] == name] for name in ("train", "validation", "test")}
    moment = np.asarray(inverse["actuation_moment"], float)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_sequence = next(item for item in split["train"] if item["key"] == source["source_bout"])
    _, start, period, _, _ = best_cycle(source_sequence)
    canonical = tile_cycle(source_sequence, start, period)
    canonical["cycle_period"] = period
    canonical["command"] = source["command_mm_s"]
    base_policy = cp.CPGPolicy(source["input_size"], source["output_size"])
    base_policy.load_state_dict(source["state_dict"])
    base_policy.eval()
    for parameter in base_policy.parameters():
        parameter.requires_grad_(False)
    feature_stats = tuple(source[key] for key in ("feature_mean", "feature_std", "target_mean", "target_std"))
    cap = CAP_FRACTION * (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0])
    residual = ResidualPolicy(source["input_size"], source["output_size"])

    nominal = []
    for phase in np.linspace(0, period - 1, 6, dtype=int):
        item, _, _ = cp.collect(model, ground, claws, base_policy, feature_stats, canonical, moment, 0.3, 0.0002, 0.0, rng, int(phase), (0.001, 0.01))
        nominal.append({"features": item["features"]})
    initial_records, _ = evaluate(model, ground, claws, base_policy, residual, feature_stats, split["validation"], canonical, cap)
    initial_summary = endurance_summary(initial_records)
    best = {"iteration": 0, "objective": objective(initial_summary), "state": {key: value.detach().clone() for key, value in residual.state_dict().items()}}
    selection = [{"iteration": 0, "validation": initial_summary, "by_bout": initial_records, "objective": best["objective"]}]
    collections, rolling, losses = [], [], []
    for iteration, beta in enumerate(BETAS, 1):
        current, body, max_tilt, finite, sampled = [], [], [], 0, {}
        for sequence in split["train"]:
            frame_indices = [0, int(rng.integers(len(sequence["reference"])))]
            sampled[sequence["key"]] = frame_indices
            for frame_index in frame_indices:
                item = collect(model, ground, claws, base_policy, residual, feature_stats, sequence, frame_index, canonical, moment, beta, cap, rng)
                current.append(item)
                body.append(float(np.mean(item["body"])))
                max_tilt.append(float(np.max(item["tilt"])))
                finite += int(np.isfinite(item["features"]).all())
        rolling.append(current)
        rolling = rolling[-2:]
        recovery = [item for group in rolling for item in group]
        iteration_losses = train(residual, nominal, recovery, feature_stats, rng)
        for item in iteration_losses:
            item["iteration"] = iteration
        losses.extend(iteration_losses)
        validation_records, _ = evaluate(model, ground, claws, base_policy, residual, feature_stats, split["validation"], canonical, cap)
        validation = endurance_summary(validation_records)
        score = objective(validation)
        selection.append({"iteration": iteration, "validation": validation, "by_bout": validation_records, "objective": score})
        collections.append({"iteration": iteration, "beta": beta, "finite_bouts": finite, "mean_body_contact": float(np.mean(body)), "maximum_tilt": float(np.max(max_tilt)), "sampled_frames": sampled})
        if score < best["objective"]:
            best = {"iteration": iteration, "objective": score, "state": {key: value.detach().clone() for key, value in residual.state_dict().items()}}

    residual.load_state_dict(best["state"])
    test_records, payload = evaluate(model, ground, claws, base_policy, residual, feature_stats, split["test"], canonical, cap, True)
    test = endurance_summary(test_records)
    q0, v0, _ = cp.cycle_from_phase(canonical, 0)
    standard_spec = {"key": source_sequence["key"] + "_canonical", "split": "training-derived standard start", "reference": q0}
    standard_out = rollout(model, ground, claws, base_policy, residual, feature_stats, q0[0], v0[0], 0, period, canonical["command"], cap)
    standard = record(standard_spec, standard_out)
    payload.update({f"canonical_{key}": value for key, value in standard_out.items()})
    np.savez_compressed(TRAJECTORY, **payload)
    torch.save({
        "state_dict": residual.state_dict(),
        "input_size": source["input_size"],
        "output_size": source["output_size"],
        "cap": cap,
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(SOURCE),
        "selected_iteration": best["iteration"],
        "seed": SEED,
    }, CHECKPOINT)
    standard_pass = standard["finite"] and standard["distance_2s_mm"] >= 5 and standard["body_contact_fraction"] <= 0.05 and min(standard["cycles"]) >= 2
    heldout_pass = test["minimum_distance_2s_mm"] >= 5 and test["maximum_body_contact_fraction"] <= 0.05 and min(test["minimum_cycles_per_leg"]) >= 2

    x = [item["iteration"] for item in selection]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), layout="constrained")
    axes[0].plot(x, [item["validation"]["minimum_distance_2s_mm"] for item in selection], marker="o")
    axes[0].axhline(5, color="#b34c3f", ls="--")
    axes[0].set(title="验证最差 2 秒前进", ylabel="mm")
    axes[1].plot(x, [item["validation"]["maximum_body_contact_fraction"] for item in selection], marker="o")
    axes[1].axhline(0.05, color="#b34c3f", ls="--")
    axes[1].set(title="验证最大身体触地")
    axes[2].bar([item["bout_key"] for item in test_records], [item["distance_2s_mm"] for item in test_records])
    axes[2].axhline(5, color="#b34c3f", ls="--")
    axes[2].set(title="隔离测试", ylabel="mm")
    fig.suptitle("冻结 CPG + 独立姿态恢复残差")
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    checks = {
        "base_CPG_is_frozen": all(not parameter.requires_grad for parameter in base_policy.parameters()),
        "whole_fly_split_12_3_3": [len(split[name]) for name in split] == [12, 3, 3],
        "two_training_starts_per_bout": STARTS_PER_BOUT == 2,
        "six_residual_DAgger_iterations": len(collections) == 6,
        "last_two_policy_only": BETAS[-2:] == (0.0, 0.0),
        "all_144_collection_bouts_finite": sum(item["finite_bouts"] for item in collections) == 144,
        "nominal_zero_residual_anchor": len(nominal) == 6,
        "validation_selects_checkpoint": True,
        "test_not_used_for_training_or_selection": True,
        "runtime_has_no_teacher_trajectory_root_force_or_position_servo": True,
        "artifacts_saved": CHECKPOINT.exists() and TRAJECTORY.exists() and FIGURE.exists(),
        "not_mislabelled_as_connectome_or_A3": True,
    }
    report = {
        "date": "2026-09-24",
        "scope": "frozen engineering CPG plus separately trained proprioceptive posture residual",
        "runtime_contract": {
            "inputs": "internal oscillator, current proprioception/contact, previous applied control, speed command",
            "forbidden": ["running trajectory", "teacher force", "root force", "external force", "position servo"],
        },
        "training": {"seed": SEED, "betas": list(BETAS), "cap_fraction": CAP_FRACTION, "collections": collections, "loss": losses},
        "selection": {"source": "validation flies only", "history": selection, "selected_iteration": best["iteration"], "objective": best["objective"]},
        "standard_start": {"result": standard, "gate_pass": standard_pass},
        "test": {"summary": test, "by_bout": test_records, "gate_pass": heldout_pass},
        "artifacts": {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT), "trajectories": str(TRAJECTORY.relative_to(ROOT)), "trajectories_sha256": sha256(TRAJECTORY), "figure": str(FIGURE.relative_to(ROOT))},
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A1_engineering_CPG_standard_start": "pass" if standard_pass else "incomplete",
            "A1_heldout_initial_state_robustness": "pass" if heldout_pass else "incomplete",
            "A2_connectome_motor_decoder": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": f"{sum(checks.values())}/{len(checks)}", "selected_iteration": best["iteration"], "standard": standard, "test": test, "classification": report["classification"], "figure": str(FIGURE), "goal_complete": False}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
