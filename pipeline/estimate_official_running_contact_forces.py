"""Estimate support forces missing from the public 800 Hz kinematic dataset.

The source has no force-plate measurements.  This script therefore produces an
explicitly engineering, friction-constrained estimate.  It never labels the
result as measured contact force or neural autonomy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "external"
MODEL = (
    WORKSPACE
    / "work/upstream-20260923/3d_tracking_ik/model/fruitfly_v1/fruitfly_v1_free.xml"
)
ID_RESULTS = ROOT / "results/official-running-inverse-dynamics-20260923"
ID_TEACHER = ID_RESULTS / "teacher.npz"
ID_REPORT = ID_RESULTS / "report.json"
OUT = ROOT / "results/official-running-contact-estimate-20260923"
CONTACT_TEACHER = OUT / "contact-teacher.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "contact-force-estimate.png"

LEGS = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
FPS = 800.0
MODEL_TO_MM = 10.0
NEAR_FLOOR_THRESHOLD_MM = 0.12
FRICTION_COEFFICIENT = 0.6
MINIMUM_SUPPORT_FEET = 3
ROOT_RESIDUAL_ADAPT_THRESHOLD = 0.01
RIDGE = 1e-6
LEG_ACTUATOR_START = 8

plt.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def solve_support(
    root_target: np.ndarray,
    jacobians: list[np.ndarray],
    initial_active: np.ndarray,
    height_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit a pyramidal Coulomb support force, adding feet only if required."""
    basis = np.asarray(
        [
            [FRICTION_COEFFICIENT, -FRICTION_COEFFICIENT, 0.0, 0.0],
            [0.0, 0.0, FRICTION_COEFFICIENT, -FRICTION_COEFFICIENT],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    active = list(dict.fromkeys(int(value) for value in initial_active))
    imputed = 0
    for foot in height_order:
        if len(active) >= MINIMUM_SUPPORT_FEET:
            break
        if int(foot) not in active:
            active.append(int(foot))
            imputed += 1

    while True:
        blocks = [jacobians[foot][:, :6].T @ basis for foot in active]
        coefficient = np.concatenate(blocks, axis=1)
        row_scale = np.maximum(np.linalg.norm(coefficient, axis=1), 1e-7)
        normalized = coefficient / row_scale[:, None]
        target = root_target / row_scale
        # A tiny Tikhonov term selects a stable low-force point from the
        # underdetermined friction pyramid without hiding root-wrench error.
        augmented = np.vstack(
            (normalized, np.sqrt(RIDGE) * np.eye(normalized.shape[1]))
        )
        augmented_target = np.concatenate(
            (target, np.zeros(normalized.shape[1]))
        )
        solution = lsq_linear(
            augmented,
            augmented_target,
            bounds=(0, np.inf),
            tol=1e-8,
            lsmr_tol="auto",
            max_iter=50,
        ).x
        residual = coefficient @ solution - root_target
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(root_target), 1e-12))
        remaining = [int(foot) for foot in height_order if int(foot) not in active]
        if relative <= ROOT_RESIDUAL_ADAPT_THRESHOLD or not remaining:
            break
        active.append(remaining[0])
        imputed += 1

    force = np.zeros((6, 3), dtype=np.float64)
    for index, foot in enumerate(active):
        force[foot] = basis @ solution[4 * index : 4 * (index + 1)]
    return force, np.asarray(active, dtype=np.int8), relative, imputed


def metrics(target: np.ndarray, residual: np.ndarray) -> dict:
    relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
    return {
        "target_rms_native": float(np.sqrt(np.mean(target * target))),
        "residual_rms_native": float(np.sqrt(np.mean(residual * residual))),
        "relative_residual_l2": relative,
        "norm_coverage": 1.0 - relative,
        "energy_explained_fraction": 1.0 - relative * relative,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inverse_report = json.loads(ID_REPORT.read_text())
    teacher = np.load(ID_TEACHER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    site_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"claw_{leg}"
            )
            for leg in LEGS
        ],
        dtype=np.int32,
    )
    if np.any(site_ids < 0):
        raise ValueError("Pinned source model does not expose all six claw sites")

    qpos = teacher["qpos_filtered"].astype(np.float64)
    qfrc = teacher["qfrc_inverse"].astype(np.float64)
    bout_index = teacher["bout_index"].astype(int)
    split_code = teacher["split_code"].astype(int)
    moment = teacher["actuation_moment"].astype(np.float64)
    leg_dofs = teacher["leg_dofs"].astype(int)
    frames = len(qpos)
    foot_position = np.empty((frames, 6, 3), dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        foot_position[frame] = data.site_xpos[site_ids]

    floors = np.empty((len(teacher["bout_keys"]), 6), dtype=np.float64)
    for bout in range(len(floors)):
        floors[bout] = np.percentile(foot_position[bout_index == bout, :, 2], 5, axis=0)
    relative_height = foot_position[:, :, 2] - floors[bout_index]
    observed = relative_height <= NEAR_FLOOR_THRESHOLD_MM / MODEL_TO_MM

    contact_force = np.zeros((frames, 6, 3), dtype=np.float64)
    active = np.zeros((frames, 6), dtype=bool)
    generalized_contact = np.zeros((frames, model.nv), dtype=np.float64)
    relative_root_residual = np.empty(frames, dtype=np.float64)
    imputed_count = np.empty(frames, dtype=np.int8)
    solver_failures = 0
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        jacobians = []
        for site in site_ids:
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(model, data, jacp, jacr, int(site))
            jacobians.append(jacp)
        initial = np.flatnonzero(observed[frame])
        order = np.argsort(relative_height[frame])
        force, selected, relative, imputed = solve_support(
            qfrc[frame, :6], jacobians, initial, order
        )
        if not np.isfinite(force).all():
            solver_failures += 1
            force[:] = 0
        contact_force[frame] = force
        active[frame, selected] = True
        relative_root_residual[frame] = relative
        imputed_count[frame] = imputed
        for foot in selected:
            generalized_contact[frame] += jacobians[int(foot)].T @ force[int(foot)]
        if (frame + 1) % 1000 == 0:
            print(f"contact frames {frame + 1}/{frames}", flush=True)

    root_residual = qfrc[:, :6] - generalized_contact[:, :6]
    actuator_demand = qfrc - generalized_contact
    denominator = np.sum(moment * moment, axis=1)
    actuator_force = (actuator_demand @ moment.T) / denominator
    gain = model.actuator_gainprm[:, 0]
    actuator_control = actuator_force / gain
    bounded_control = np.clip(
        actuator_control,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    bounded_force = bounded_control * gain
    projection = actuator_force @ moment
    bounded_projection = bounded_force @ moment
    leg_target = actuator_demand[:, leg_dofs]
    leg_residual = (actuator_demand - projection)[:, leg_dofs]
    bounded_leg_residual = (actuator_demand - bounded_projection)[:, leg_dofs]
    low = model.actuator_ctrlrange[LEG_ACTUATOR_START:, 0]
    high = model.actuator_ctrlrange[LEG_ACTUATOR_START:, 1]
    clipped = (actuator_control[:, LEG_ACTUATOR_START:] < low) | (
        actuator_control[:, LEG_ACTUATOR_START:] > high
    )
    normal = contact_force[:, :, 2]
    tangential = np.linalg.norm(contact_force[:, :, :2], axis=2)
    friction_utilization = np.divide(
        tangential,
        FRICTION_COEFFICIENT * normal,
        out=np.zeros_like(tangential),
        where=normal > 1e-10,
    )
    nonzero_force = normal > 1e-10
    imputed = active & ~observed

    split_results = {}
    for code, name in enumerate(("train", "validation", "test")):
        mask = split_code == code
        split_results[name] = {
            "teacher_frames": int(mask.sum()),
            "root_wrench": metrics(qfrc[mask, :6], root_residual[mask]),
            "leg_projection": metrics(leg_target[mask], leg_residual[mask]),
            "bounded_leg_projection": metrics(
                leg_target[mask], bounded_leg_residual[mask]
            ),
            "imputed_support_foot_fraction": float(imputed[mask].mean()),
        }

    np.savez_compressed(
        CONTACT_TEACHER,
        foot_position=foot_position.astype(np.float32),
        floor_height=floors.astype(np.float32),
        observed_near_floor=observed,
        active_support=active,
        imputed_support=imputed,
        contact_force_estimate=contact_force.astype(np.float32),
        generalized_contact_force=generalized_contact.astype(np.float32),
        root_wrench_residual=root_residual.astype(np.float32),
        relative_root_residual=relative_root_residual.astype(np.float32),
        actuator_force_contact_adjusted=actuator_force.astype(np.float32),
        actuator_control_contact_adjusted=actuator_control.astype(np.float32),
        actuator_control_bounded=bounded_control.astype(np.float32),
        generalized_actuator_projection=projection.astype(np.float32),
        bounded_generalized_actuator_projection=bounded_projection.astype(np.float32),
        bout_index=bout_index.astype(np.int16),
        local_frame=teacher["local_frame"],
        split_code=split_code.astype(np.int8),
        site_ids=site_ids,
        site_names=np.asarray([f"claw_{leg}" for leg in LEGS]),
        friction_coefficient=np.asarray(FRICTION_COEFFICIENT, dtype=np.float32),
        near_floor_threshold_mm=np.asarray(
            NEAR_FLOOR_THRESHOLD_MM, dtype=np.float32
        ),
        source_inverse_teacher_sha256=np.asarray(sha256(ID_TEACHER)),
    )

    root_metrics = metrics(qfrc[:, :6], root_residual)
    leg_metrics = metrics(leg_target, leg_residual)
    bounded_leg_metrics = metrics(leg_target, bounded_leg_residual)
    checks = {
        "source_inverse_teacher_was_validated": inverse_report["passed"],
        "all_estimates_finite": all(
            np.isfinite(value).all()
            for value in (
                contact_force,
                generalized_contact,
                root_residual,
                actuator_control,
            )
        ),
        "all_frames_solved": solver_failures == 0,
        "minimum_three_support_feet": int(active.sum(axis=1).min()) >= 3,
        "normal_forces_nonnegative": float(normal.min()) >= -1e-10,
        "friction_pyramid_respected": float(friction_utilization.max()) <= 1.0 + 1e-6,
        "root_wrench_relative_l2_below_1pct": root_metrics[
            "relative_residual_l2"
        ]
        < 0.01,
        "root_wrench_p95_frame_residual_below_1pct": float(
            np.percentile(relative_root_residual, 95)
        )
        < 0.01,
        "bounded_leg_projection_relative_l2_below_20pct": bounded_leg_metrics[
            "relative_residual_l2"
        ]
        < 0.20,
        "no_root_actuator_authority": not np.any(moment[:, :6]),
        "support_forces_are_marked_estimated": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "date": "2026-09-23",
        "scope": (
            "Engineering support-force reconstruction from official real kinematics, "
            "source-body Jacobians and source friction coefficient; no force-plate "
            "measurement and no claim of biological ground-reaction-force truth."
        ),
        "sources": {
            "inverse_teacher": {
                "file": str(ID_TEACHER.relative_to(ROOT)),
                "sha256": sha256(ID_TEACHER),
            },
            "model": str(MODEL.relative_to(WORKSPACE)),
            "model_commit": "defdb66932a17e41c8852f3b4c5056bee74046eb",
            "friction_coefficient_from_source_xml": FRICTION_COEFFICIENT,
        },
        "method": {
            "observed_contact_proxy": f"per-bout/per-foot height <= 5th percentile + {NEAR_FLOOR_THRESHOLD_MM} mm",
            "minimum_support": (
                "If fewer than three near-floor feet are observed, add the lowest "
                "feet and mark them imputed; add another only if root residual exceeds 1%."
            ),
            "force_model": "four-ray pyramidal Coulomb cone per active foot; nonnegative ray weights",
            "objective": "weighted root-wrench least squares plus 1e-6 minimum-force ridge",
            "contact_generalized_force": "sum of source claw-site translational Jacobian transpose times estimated Cartesian foot force",
            "leg_teacher": "qfrc_inverse minus estimated contact generalized force, projected onto the 48 source leg actuator columns",
            "collision_policy": "IK-only wing/abdomen detector collisions disabled, as in the preceding inverse-dynamics teacher",
        },
        "artifact": {
            "file": str(CONTACT_TEACHER.relative_to(ROOT)),
            "sha256": sha256(CONTACT_TEACHER),
            "frames": frames,
            "legs": list(LEGS),
        },
        "results": {
            "observed_near_floor_foot_fraction": float(observed.mean()),
            "active_support_foot_fraction": float(active.mean()),
            "imputed_support_foot_fraction": float(imputed.mean()),
            "frames_requiring_any_imputed_support": int(np.count_nonzero(imputed_count)),
            "maximum_imputed_feet_in_frame": int(imputed_count.max()),
            "support_feet_per_frame": {
                str(count): int(np.count_nonzero(active.sum(axis=1) == count))
                for count in range(3, 7)
            },
            "root_wrench": root_metrics,
            "root_frame_relative_residual": {
                "median": float(np.median(relative_root_residual)),
                "p95": float(np.percentile(relative_root_residual, 95)),
                "p99": float(np.percentile(relative_root_residual, 99)),
                "maximum": float(relative_root_residual.max()),
            },
            "leg_projection": leg_metrics,
            "bounded_leg_projection": bounded_leg_metrics,
            "leg_control_clip_fraction": float(clipped.mean()),
            "normal_support_force_native": {
                "mean_total_per_frame": float(normal.sum(axis=1).mean()),
                "p95_total_per_frame": float(np.percentile(normal.sum(axis=1), 95)),
                "maximum_single_foot": float(normal.max()),
            },
            "friction_utilization_on_nonzero_forces": {
                "median": float(np.median(friction_utilization[nonzero_force])),
                "p95": float(np.percentile(friction_utilization[nonzero_force], 95)),
                "maximum": float(friction_utilization.max()),
            },
            "by_split": split_results,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A0_engineering_contact_force_teacher": "pass"
            if all(checks.values())
            else "fail",
            "measured_ground_reaction_force": "unavailable",
            "A1_source_aligned_six_leg_muscle_body": "incomplete",
            "A2_motor_neuron_recruitment": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "limits": [
            "No force plate or pressure sensor was published with the kinematics; every contact force here is an optimization estimate.",
            "Near-floor contact is inferred from height. Added support feet are flagged per frame and remain an engineering assumption.",
            "A 0.6 pyramidal friction cone comes from the source XML, not a measured surface in these recordings.",
            "The result supplies joint-force targets for body reconstruction; it does not reveal muscle activation or motor-neuron recruitment.",
        ],
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    exemplar = bout_index == 0
    time = teacher["local_frame"][exemplar] / FPS
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].imshow(
        observed[exemplar].T,
        aspect="auto",
        interpolation="nearest",
        extent=(time[0], time[-1], 5.5, -0.5),
        cmap="Blues",
    )
    axes[0, 0].set(
        title="足端近地观测（高度代理）",
        xlabel="时间 / s",
        yticks=np.arange(6),
        yticklabels=("L1", "R1", "L2", "R2", "L3", "R3"),
    )
    axes[0, 1].imshow(
        active[exemplar].T + imputed[exemplar].T,
        aspect="auto",
        interpolation="nearest",
        extent=(time[0], time[-1], 5.5, -0.5),
        cmap="YlOrRd",
        vmin=0,
        vmax=2,
    )
    axes[0, 1].set(
        title="动力学支撑集（深色为补选）",
        xlabel="时间 / s",
        yticks=np.arange(6),
        yticklabels=("L1", "R1", "L2", "R2", "L3", "R3"),
    )
    for leg in range(6):
        axes[1, 0].plot(time, normal[exemplar, leg], lw=0.9, label=LEGS[leg])
    axes[1, 0].set(
        title="估计的六足法向支撑力",
        xlabel="时间 / s",
        ylabel="MuJoCo 原生力单位",
    )
    axes[1, 0].legend(fontsize=7, ncol=3)
    axes[1, 1].plot(time, relative_root_residual[exemplar] * 100, color="#b54b4b")
    axes[1, 1].axhline(1, ls="--", color="black", lw=0.8)
    axes[1, 1].set(
        title="根部外力重建残差",
        xlabel="时间 / s",
        ylabel="逐帧相对残差 / %",
    )
    axes[2, 0].bar(
        ["训练", "验证", "测试"],
        [split_results[name]["root_wrench"]["norm_coverage"] * 100 for name in ("train", "validation", "test")],
        color="#4b91b5",
    )
    axes[2, 0].set_ylim(95, 100)
    axes[2, 0].set(title="按来源个体隔离的根部外力覆盖", ylabel="L2 范数覆盖率 / %")
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.02,
        0.97,
        "结论\n\n"
        f"根部外力总体残差：{root_metrics['relative_residual_l2']*100:.3f}%\n"
        f"需要补选支撑的帧：{np.count_nonzero(imputed_count)}/{frames}\n"
        f"接触修正后腿部限幅残差：{bounded_leg_metrics['relative_residual_l2']*100:.2f}%\n\n"
        "蓝图来自真实运动学和来源身体雅可比；\n"
        "接触力本身是摩擦约束优化估计，不是实测。\n"
        "下一关仍是用真实肌肉和运动神经元产生这些力。",
        va="top",
        fontsize=12,
    )
    figure.suptitle("真实六足奔跑的缺失接触力重建", fontsize=16)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            {
                "passed": report["passed"],
                "frames": frames,
                "root_relative_residual": root_metrics["relative_residual_l2"],
                "bounded_leg_relative_residual": bounded_leg_metrics[
                    "relative_residual_l2"
                ],
                "frames_with_imputed_support": int(np.count_nonzero(imputed_count)),
                "artifact": str(CONTACT_TEACHER),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
