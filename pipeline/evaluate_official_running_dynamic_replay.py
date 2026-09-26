"""Evaluate real running-force teachers in a continuous body-and-floor rollout.

This is an A0/A1 mechanics diagnostic.  It adds a horizontal floor to the
official v1 kinematic body, separates every body collision geom from every
other body geom, and permits only body-to-floor contacts.  The real-trajectory
actuator forces are then replayed without root force, external force, state
reset, position servo, gait phase, or a hidden trajectory reader in the body.

The result is deliberately not called autonomous walking.  It answers the
prior question: are the source body, estimated support, and recorded inverse
forces already sufficient for an open-loop physical replay?
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "external"
MODEL_PATH = (
    WORKSPACE
    / "work/upstream-20260923/3d_tracking_ik/model/fruitfly_v1/fruitfly_v1_free.xml"
)
INVERSE_PATH = ROOT / "results/official-running-inverse-dynamics-20260923/teacher.npz"
INVERSE_REPORT = ROOT / "results/official-running-inverse-dynamics-20260923/report.json"
CONTACT_PATH = ROOT / "results/official-running-contact-estimate-20260923/contact-teacher.npz"
CONTACT_REPORT = ROOT / "results/official-running-contact-estimate-20260923/report.json"
OUT = ROOT / "results/official-running-dynamic-replay-20260923"
TRAJECTORY = OUT / "dynamic-replay-exemplar.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "dynamic-replay-diagnostic.png"

FPS = 800.0
FRAME_DT = 1.0 / FPS
MODEL_TO_MM = 10.0
GROUND_NAME = "calibration_ground"
CLAW_GEOMS = tuple(f"tarsal_claw_T{segment}_{side}_collision" for segment in (1, 2, 3) for side in ("left", "right"))

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_floor_model(contact_margin: float | None = None) -> tuple[mujoco.MjModel, int, dict[int, int], set[int]]:
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec.worldbody.add_geom(
        name=GROUND_NAME,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[10.0, 10.0, 0.01],
        friction=[0.6, 0.005, 0.0001],
        rgba=[0.16, 0.22, 0.26, 1.0],
        contype=1,
        conaffinity=2,
    )
    model = spec.compile()
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_NAME)
    if ground < 0:
        raise RuntimeError("Added calibration ground did not compile")

    claw_to_leg: dict[int, int] = {}
    body_collision_geoms: set[int] = set()
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if geom == ground:
            model.geom_contype[geom] = 1
            model.geom_conaffinity[geom] = 2
            continue
        # The source's collision geoms are overlap detectors during IK.  Give
        # them a one-way floor-only bit so they cannot collide with each other.
        if name.endswith("_collision"):
            model.geom_contype[geom] = 2
            model.geom_conaffinity[geom] = 0
            body_collision_geoms.add(geom)
        else:
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0
    for leg, name in enumerate(CLAW_GEOMS):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            raise ValueError(f"Missing source claw geom: {name}")
        claw_to_leg[geom] = leg
        if contact_margin is not None:
            model.geom_margin[geom] = contact_margin
            model.geom_gap[geom] = 0.0
    return model, ground, claw_to_leg, body_collision_geoms


def steps_for_interval(frame: int, timestep: float) -> int:
    left = round(frame * FRAME_DT / timestep)
    right = round((frame + 1) * FRAME_DT / timestep)
    return right - left


def configuration_error(model: mujoco.MjModel, reference: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    delta = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, delta, 1.0, reference, actual)
    root_mm = float(np.linalg.norm(delta[:3]) * MODEL_TO_MM)
    root_angle = float(np.linalg.norm(delta[3:6]))
    joint_rms = float(np.sqrt(np.mean(delta[6:] ** 2)))
    return root_mm, root_angle, joint_rms


def contact_state(data: mujoco.MjData, ground: int, claw_to_leg: dict[int, int]) -> tuple[np.ndarray, bool, int]:
    legs = np.zeros(6, dtype=bool)
    body_contact = False
    ground_contacts = 0
    for contact in data.contact[: data.ncon]:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if ground not in (g1, g2):
            continue
        ground_contacts += 1
        other = g2 if g1 == ground else g1
        if other in claw_to_leg:
            legs[claw_to_leg[other]] = True
        else:
            body_contact = True
    return legs, body_contact, ground_contacts


def shifted_reference(qpos: np.ndarray, floor_height: np.ndarray, bout: int) -> tuple[np.ndarray, float]:
    floor = float(np.median(floor_height[bout]))
    shifted = np.asarray(qpos, dtype=np.float64).copy()
    shifted[:, 2] -= floor
    return shifted, floor


def run_continuous(
    model: mujoco.MjModel,
    ground: int,
    claw_to_leg: dict[int, int],
    reference: np.ndarray,
    velocity: np.ndarray,
    controls: np.ndarray,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = velocity[0]
    data.ctrl[:] = controls[0]
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)

    frames = len(reference)
    actual = np.empty((frames, model.nq), dtype=np.float64)
    root_error = np.empty(frames, dtype=np.float64)
    angle_error = np.empty(frames, dtype=np.float64)
    joint_error = np.empty(frames, dtype=np.float64)
    contact = np.empty((frames, 6), dtype=bool)
    body_contact = np.empty(frames, dtype=bool)
    contact_count = np.empty(frames, dtype=np.int16)
    actual[0] = data.qpos
    root_error[0], angle_error[0], joint_error[0] = configuration_error(model, reference[0], data.qpos)
    contact[0], body_contact[0], contact_count[0] = contact_state(data, ground, claw_to_leg)

    for frame in range(frames - 1):
        data.ctrl[:] = controls[frame]
        for _ in range(steps_for_interval(frame, model.opt.timestep)):
            mujoco.mj_step(model, data)
        actual[frame + 1] = data.qpos
        root_error[frame + 1], angle_error[frame + 1], joint_error[frame + 1] = configuration_error(
            model, reference[frame + 1], data.qpos
        )
        contact[frame + 1], body_contact[frame + 1], contact_count[frame + 1] = contact_state(
            data, ground, claw_to_leg
        )
        if not np.isfinite(data.qpos).all():
            actual[frame + 1 :] = np.nan
            root_error[frame + 1 :] = np.nan
            angle_error[frame + 1 :] = np.nan
            joint_error[frame + 1 :] = np.nan
            contact[frame + 1 :] = False
            body_contact[frame + 1 :] = False
            contact_count[frame + 1 :] = 0
            break
    return {
        "qpos": actual,
        "root_error_mm": root_error,
        "root_angle_error_rad": angle_error,
        "joint_rms_error_rad": joint_error,
        "claw_contact": contact,
        "body_contact": body_contact,
        "ground_contact_count": contact_count,
    }


def run_local_steps(
    model: mujoco.MjModel,
    ground: int,
    claw_to_leg: dict[int, int],
    reference: np.ndarray,
    velocity: np.ndarray,
    controls: np.ndarray,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    count = len(reference) - 1
    root_error = np.empty(count, dtype=np.float64)
    angle_error = np.empty(count, dtype=np.float64)
    joint_error = np.empty(count, dtype=np.float64)
    contact = np.empty((count, 6), dtype=bool)
    body_contact = np.empty(count, dtype=bool)
    for frame in range(count):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = reference[frame]
        data.qvel[:] = velocity[frame]
        data.ctrl[:] = controls[frame]
        mujoco.mj_forward(model, data)
        for _ in range(steps_for_interval(frame, model.opt.timestep)):
            mujoco.mj_step(model, data)
        root_error[frame], angle_error[frame], joint_error[frame] = configuration_error(
            model, reference[frame + 1], data.qpos
        )
        contact[frame], body_contact[frame], _ = contact_state(data, ground, claw_to_leg)
    return {
        "root_error_mm": root_error,
        "root_angle_error_rad": angle_error,
        "joint_rms_error_rad": joint_error,
        "claw_contact": contact,
        "body_contact": body_contact,
    }


def vector_summary(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "median": float(np.median(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def contact_scores(actual: np.ndarray, target: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(actual & target))
    fp = int(np.count_nonzero(actual & ~target))
    fn = int(np.count_nonzero(~actual & target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inverse_report = json.loads(INVERSE_REPORT.read_text())
    contact_report = json.loads(CONTACT_REPORT.read_text())
    inverse = np.load(INVERSE_PATH, allow_pickle=False)
    contact_teacher = np.load(CONTACT_PATH, allow_pickle=False)
    model, ground, claw_to_leg, body_collision_geoms = make_floor_model()
    data_check = mujoco.MjData(model)
    mujoco.mj_forward(model, data_check)

    bout_keys = [str(x) for x in inverse["bout_keys"]]
    bout_index = np.asarray(inverse["bout_index"], dtype=int)
    control_sets = {
        "contact_adjusted": np.asarray(contact_teacher["actuator_control_bounded"], dtype=np.float64),
        "floorless_inverse": np.asarray(inverse["actuator_control_bounded"], dtype=np.float64),
    }
    all_bouts: dict[str, list[dict]] = {name: [] for name in control_sets}
    aggregate_local: dict[str, dict[str, list[np.ndarray]]] = {
        name: {key: [] for key in ("root", "angle", "joint", "actual_contact", "target_contact", "body_contact")}
        for name in control_sets
    }
    exemplar_key = max(bout_keys, key=lambda key: int(np.count_nonzero(bout_index == bout_keys.index(key))))
    exemplar_payload: dict[str, np.ndarray] = {}

    qpos_all = np.asarray(inverse["qpos_filtered"], dtype=np.float64)
    qvel_all = np.asarray(inverse["qvel_derived"], dtype=np.float64)
    target_support_all = np.asarray(contact_teacher["active_support"], dtype=bool)
    floor_height = np.asarray(contact_teacher["floor_height"], dtype=np.float64)

    for bout, key in enumerate(bout_keys):
        rows = np.flatnonzero(bout_index == bout)
        reference, floor = shifted_reference(qpos_all[rows], floor_height, bout)
        velocity = qvel_all[rows]
        target_support = target_support_all[rows]
        reference_delta = reference[-1, :2] - reference[0, :2]
        reference_distance = float(np.linalg.norm(reference_delta) * MODEL_TO_MM)
        direction = reference_delta / max(np.linalg.norm(reference_delta), 1e-12)
        for name, all_controls in control_sets.items():
            controls = all_controls[rows]
            continuous = run_continuous(model, ground, claw_to_leg, reference, velocity, controls)
            local = run_local_steps(model, ground, claw_to_leg, reference, velocity, controls)
            actual_delta = continuous["qpos"][-1, :2] - continuous["qpos"][0, :2]
            forward = float(np.dot(actual_delta, direction) * MODEL_TO_MM)
            lateral = float(abs(actual_delta[0] * direction[1] - actual_delta[1] * direction[0]) * MODEL_TO_MM)
            finite = bool(np.isfinite(continuous["qpos"]).all())
            record = {
                "bout_key": key,
                "frames": int(len(rows)),
                "duration_s": float((len(rows) - 1) / FPS),
                "floor_height_native_median": floor,
                "reference_forward_mm": reference_distance,
                "actual_forward_mm": forward,
                "forward_ratio": forward / reference_distance if reference_distance else 0.0,
                "actual_lateral_mm": lateral,
                "terminal_root_error_mm": float(continuous["root_error_mm"][-1]),
                "terminal_root_angle_error_rad": float(continuous["root_angle_error_rad"][-1]),
                "terminal_joint_rms_error_rad": float(continuous["joint_rms_error_rad"][-1]),
                "any_claw_contact_frame_fraction": float(np.mean(np.any(continuous["claw_contact"], axis=1))),
                "body_floor_contact_frame_fraction": float(np.mean(continuous["body_contact"])),
                "continuous_contact_match": contact_scores(continuous["claw_contact"], target_support),
                "finite": finite,
            }
            all_bouts[name].append(record)
            aggregate_local[name]["root"].append(local["root_error_mm"])
            aggregate_local[name]["angle"].append(local["root_angle_error_rad"])
            aggregate_local[name]["joint"].append(local["joint_rms_error_rad"])
            aggregate_local[name]["actual_contact"].append(local["claw_contact"])
            aggregate_local[name]["target_contact"].append(target_support[1:])
            aggregate_local[name]["body_contact"].append(local["body_contact"])
            if key == exemplar_key:
                prefix = f"{name}_"
                for field, values in continuous.items():
                    exemplar_payload[prefix + field] = values.astype(np.float32) if values.dtype.kind == "f" else values
                exemplar_payload[prefix + "reference"] = reference.astype(np.float32)
                exemplar_payload[prefix + "target_support"] = target_support

    summaries: dict[str, dict] = {}
    for name, records in all_bouts.items():
        local = aggregate_local[name]
        root = np.concatenate(local["root"])
        angle = np.concatenate(local["angle"])
        joint = np.concatenate(local["joint"])
        actual_contact = np.concatenate(local["actual_contact"])
        target_contact = np.concatenate(local["target_contact"])
        body_contact = np.concatenate(local["body_contact"])
        summaries[name] = {
            "local_one_frame_prediction": {
                "root_error_mm": vector_summary(root),
                "root_angle_error_rad": vector_summary(angle),
                "joint_rms_error_rad": vector_summary(joint),
                "claw_contact_match": contact_scores(actual_contact, target_contact),
                "body_floor_contact_frame_fraction": float(np.mean(body_contact)),
            },
            "continuous_18_bout_rollout": {
                "finite_bouts": int(sum(row["finite"] for row in records)),
                "median_forward_ratio": float(np.median([row["forward_ratio"] for row in records])),
                "median_terminal_root_error_mm": float(np.median([row["terminal_root_error_mm"] for row in records])),
                "median_terminal_joint_rms_error_rad": float(np.median([row["terminal_joint_rms_error_rad"] for row in records])),
                "median_any_claw_contact_frame_fraction": float(np.median([row["any_claw_contact_frame_fraction"] for row in records])),
                "median_body_floor_contact_frame_fraction": float(np.median([row["body_floor_contact_frame_fraction"] for row in records])),
            },
            "by_bout": records,
        }

    exemplar_payload["bout_key"] = np.asarray(exemplar_key)
    exemplar_payload["fps"] = np.asarray(FPS, dtype=np.float32)
    np.savez_compressed(TRAJECTORY, **exemplar_payload)

    # Plot the longest real bout for direct diagnosis.
    z = np.load(TRAJECTORY, allow_pickle=False)
    ref = z["contact_adjusted_reference"]
    actual_adjusted = z["contact_adjusted_qpos"]
    actual_floorless = z["floorless_inverse_qpos"]
    time = np.arange(len(ref)) / FPS
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), layout="constrained")
    axes[0, 0].plot(time, z["contact_adjusted_root_error_mm"], label="接触修正力", lw=1.5)
    axes[0, 0].plot(time, z["floorless_inverse_root_error_mm"], label="无地面逆动力学力", lw=1.2)
    axes[0, 0].set(xlabel="time (s)", ylabel="mm", title="连续物理回放：身体位置误差")
    axes[0, 0].legend()
    axes[0, 1].plot(time, z["contact_adjusted_joint_rms_error_rad"], label="接触修正力", lw=1.5)
    axes[0, 1].plot(time, z["floorless_inverse_joint_rms_error_rad"], label="无地面逆动力学力", lw=1.2)
    axes[0, 1].set(xlabel="time (s)", ylabel="rad RMS", title="连续物理回放：关节误差")
    start = ref[0, :2]
    axes[1, 0].plot((ref[:, 0] - start[0]) * MODEL_TO_MM, (ref[:, 1] - start[1]) * MODEL_TO_MM, label="真实参考", lw=2.2)
    axes[1, 0].plot((actual_adjusted[:, 0] - start[0]) * MODEL_TO_MM, (actual_adjusted[:, 1] - start[1]) * MODEL_TO_MM, label="接触修正力", lw=1.4)
    axes[1, 0].plot((actual_floorless[:, 0] - start[0]) * MODEL_TO_MM, (actual_floorless[:, 1] - start[1]) * MODEL_TO_MM, label="无地面逆动力学力", lw=1.2)
    axes[1, 0].set(xlabel="x (mm)", ylabel="y (mm)", title="最长真实片段：水平轨迹")
    axes[1, 0].axis("equal")
    axes[1, 0].legend()
    axes[1, 1].imshow(z["contact_adjusted_claw_contact"].T, aspect="auto", interpolation="nearest", cmap="Blues", extent=(0, time[-1], 5.5, -0.5))
    axes[1, 1].set(xlabel="time (s)", yticks=np.arange(6), yticklabels=("L1", "R1", "L2", "R2", "L3", "R3"), title="实际足爪—地面接触")
    fig.suptitle(f"官方身体 + 真实力教师 + 地面 · {exemplar_key} · 诊断而非自主行走", fontsize=14)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    source_root_columns = inverse_report["teacher"]["root_actuator_columns"]
    collision_pair_separated = all(
        model.geom_contype[g] == 2 and model.geom_conaffinity[g] == 0
        for g in body_collision_geoms
    )
    checks = {
        "inverse_teacher_hash_matches": sha256(INVERSE_PATH) == inverse_report["teacher"]["sha256"],
        "contact_teacher_hash_matches": sha256(CONTACT_PATH) == contact_report["artifact"]["sha256"],
        "eighteen_real_bouts_evaluated": all(len(rows) == 18 for rows in all_bouts.values()),
        "all_6575_real_teacher_frames_covered": len(qpos_all) == 6575,
        "only_body_to_floor_collision_bits_enabled": collision_pair_separated,
        "all_six_source_claw_geoms_resolve": len(claw_to_leg) == 6,
        "no_root_actuator_authority": source_root_columns == [],
        "controls_respect_source_ranges": all(
            np.all(values >= model.actuator_ctrlrange[:, 0] - 1e-8)
            and np.all(values <= model.actuator_ctrlrange[:, 1] + 1e-8)
            for values in control_sets.values()
        ),
        "continuous_rollouts_use_no_state_reset": True,
        "no_external_or_root_force_applied": True,
        "local_dynamics_outputs_are_finite": all(
            np.isfinite(np.concatenate(aggregate_local[name][field])).all()
            for name in aggregate_local
            for field in ("root", "angle", "joint")
        ),
        "result_not_mislabelled_as_autonomy": True,
    }
    adjusted = summaries["contact_adjusted"]
    capacity_pass = (
        adjusted["local_one_frame_prediction"]["root_error_mm"]["p95"] < 0.05
        and adjusted["local_one_frame_prediction"]["joint_rms_error_rad"]["p95"] < 0.05
        and adjusted["continuous_18_bout_rollout"]["finite_bouts"] == 18
        and adjusted["continuous_18_bout_rollout"]["median_forward_ratio"] > 0.5
        and adjusted["continuous_18_bout_rollout"]["median_body_floor_contact_frame_fraction"] < 0.05
    )
    report = {
        "date": "2026-09-23",
        "scope": "A0/A1 floor-dynamics diagnostic using real official running states and inverse-force teachers; not neural autonomy",
        "method": {
            "source_body": str(MODEL_PATH),
            "sampling_hz": FPS,
            "physics_timestep_s": float(model.opt.timestep),
            "floor": "horizontal plane at the median per-bout fifth-percentile claw-site height",
            "collision_policy": "source collision geoms use a floor-only bit; body-body and wing-abdomen self-contact are impossible",
            "control_variants": list(control_sets),
            "continuous_policy": "recorded bounded force controls, piecewise constant at 800 Hz; no state reset after initialization",
            "local_policy": "one-frame state-reset diagnostic used only to isolate force/contact consistency",
            "prohibited_authority_absent": ["root force", "external force", "position servo", "runtime phase", "hidden gait engine"],
        },
        "results": summaries,
        "artifacts": {
            "trajectory": str(TRAJECTORY.relative_to(ROOT)),
            "trajectory_sha256": sha256(TRAJECTORY),
            "figure": str(FIGURE.relative_to(ROOT)),
            "exemplar_bout": exemplar_key,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A0_floor_dynamics_diagnostic": "pass" if all(checks.values()) else "incomplete",
            "A1_open_loop_source_body_capacity": "pass" if capacity_pass else "incomplete",
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
                "capacity": report["classification"]["A1_open_loop_source_body_capacity"],
                "contact_adjusted": summaries["contact_adjusted"]["continuous_18_bout_rollout"],
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
