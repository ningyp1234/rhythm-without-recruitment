"""Derive source-body inverse-dynamics teachers from real 800 Hz running bouts.

This stage does not infer muscles, motor-neuron recruitment, or contact forces.
It converts the public source kinematics into the generalized forces required by
the pinned source MuJoCo body.  Root wrench is kept separate because the source
IK model has no floor and therefore cannot measure or identify ground reaction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "external"
DATA = ROOT / "data/official-3d-kinematics-20260916"
SUBSET = DATA / "running-reference-subset.h5"
MANIFEST = DATA / "manifest.json"
UPSTREAM = WORKSPACE / "work/upstream-20260923/3d_tracking_ik"
MODEL = UPSTREAM / "model/fruitfly_v1/fruitfly_v1_free.xml"
OUT = ROOT / "results/official-running-inverse-dynamics-20260923"
TEACHER = OUT / "teacher.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "inverse-dynamics-teacher.png"

FPS = 800.0
DT = 1.0 / FPS
FILTER_WINDOW = 9
FILTER_POLYORDER = 3
EDGE_FRAMES = 8
MODEL_TO_MM = 10.0
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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence(node) -> list[str]:
    if isinstance(node, h5py.Dataset):
        values = np.atleast_1d(node[()])
        return [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    return [
        node[key][()].decode()
        if isinstance(node[key][()], bytes)
        else str(node[key][()])
        for key in sorted(node.keys(), key=lambda key: int(key))
    ]


def dense_actuator_moment(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Expand MuJoCo's sparse actuator moment matrix to ``(nu, nv)``."""
    moment = np.zeros((model.nu, model.nv), dtype=np.float64)
    for actuator in range(model.nu):
        address = int(data.moment_rowadr[actuator])
        count = int(data.moment_rownnz[actuator])
        columns = data.moment_colind[address : address + count]
        moment[actuator, columns] = data.actuator_moment[address : address + count]
    return moment


def smooth_trajectory(
    model: mujoco.MjModel, qpos: np.ndarray, window: int = FILTER_WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth measured coordinates and derive quaternion-safe velocity/acceleration.

    A 9-frame (11.25 ms) cubic Savitzky-Golay filter removes frame-scale IK
    jitter.  Quaternion signs are made continuous and normalized.  Velocity is
    then computed with MuJoCo's manifold-aware finite difference, filtered once
    more, and differentiated by the same declared filter.  Edge frames are not
    exported as teacher samples.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    signed = qpos.copy()
    for index in range(1, len(signed)):
        if np.dot(signed[index - 1, 3:7], signed[index, 3:7]) < 0:
            signed[index, 3:7] *= -1
    filtered = savgol_filter(
        signed, window, FILTER_POLYORDER, axis=0, mode="interp"
    )
    filtered[:, 3:7] /= np.linalg.norm(filtered[:, 3:7], axis=1, keepdims=True)

    velocity = np.empty((len(filtered), model.nv), dtype=np.float64)
    for index in range(len(filtered)):
        left = max(0, index - 1)
        right = min(len(filtered) - 1, index + 1)
        mujoco.mj_differentiatePos(
            model,
            velocity[index],
            (right - left) * DT,
            filtered[left],
            filtered[right],
        )
    velocity = savgol_filter(
        velocity, window, FILTER_POLYORDER, axis=0, mode="interp"
    )
    acceleration = savgol_filter(
        velocity,
        window,
        FILTER_POLYORDER,
        deriv=1,
        delta=DT,
        axis=0,
        mode="interp",
    )
    return filtered, velocity, acceleration


def norm_metrics(target: np.ndarray, residual: np.ndarray) -> dict:
    target_norm = float(np.linalg.norm(target))
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / target_norm if target_norm else 0.0
    return {
        "target_rms_native": float(np.sqrt(np.mean(target * target))),
        "residual_rms_native": float(np.sqrt(np.mean(residual * residual))),
        "relative_residual_l2": relative,
        "norm_coverage": 1.0 - relative,
        "energy_explained_fraction": 1.0 - relative * relative,
    }


def dof_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    free_labels = ("x", "y", "z", "rx", "ry", "rz")
    for joint in range(model.njnt):
        start = int(model.jnt_dofadr[joint])
        end = int(model.jnt_dofadr[joint + 1]) if joint + 1 < model.njnt else model.nv
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f"joint_{joint}"
        count = end - start
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            names.extend(f"{name}:{label}" for label in free_labels)
        elif count == 1:
            names.append(name)
        else:
            names.extend(f"{name}:{axis}" for axis in range(count))
    if len(names) != model.nv:
        raise ValueError(f"DOF naming produced {len(names)} names for nv={model.nv}")
    return names


def derive_bout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    moment: np.ndarray,
    body_ids: np.ndarray,
    qpos_source: np.ndarray,
    xpos_source: np.ndarray,
    window: int = FILTER_WINDOW,
    collect_pose_error: bool = True,
) -> dict:
    qpos, qvel, qacc = smooth_trajectory(model, qpos_source, window)
    qfrc = np.empty((len(qpos), model.nv), dtype=np.float64)
    constraint = np.empty_like(qfrc)
    pose_errors: list[np.ndarray] = []
    moment_error = 0.0
    external_drive_error = 0.0
    maximum_contacts = 0
    maximum_constraints = 0
    constraint_type_counts: dict[int, int] = {}
    for index in range(len(qpos)):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        data.qacc[:] = qacc[index]
        data.ctrl[:] = 0
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_inverse(model, data)
        qfrc[index] = data.qfrc_inverse
        constraint[index] = data.qfrc_constraint
        maximum_contacts = max(maximum_contacts, int(data.ncon))
        maximum_constraints = max(maximum_constraints, int(data.nefc))
        for constraint_type in np.asarray(data.efc_type[: data.nefc], dtype=int):
            key = int(constraint_type)
            constraint_type_counts[key] = constraint_type_counts.get(key, 0) + 1
        if index in (0, len(qpos) // 2, len(qpos) - 1):
            moment_error = max(
                moment_error,
                float(np.max(np.abs(dense_actuator_moment(model, data) - moment))),
            )
        external_drive_error = max(
            external_drive_error,
            float(np.max(np.abs(data.qfrc_applied))),
            float(np.max(np.abs(data.xfrc_applied))),
        )
        if collect_pose_error:
            pose_errors.append(
                np.linalg.norm(data.xpos[body_ids] - xpos_source[index], axis=1)
                * MODEL_TO_MM
            )
    denominator = np.sum(moment * moment, axis=1)
    actuator_force = (qfrc @ moment.T) / denominator
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
    return {
        "qpos": qpos,
        "qvel": qvel,
        "qacc": qacc,
        "qfrc": qfrc,
        "constraint": constraint,
        "actuator_force": actuator_force,
        "actuator_control": actuator_control,
        "bounded_control": bounded_control,
        "projection": projection,
        "bounded_projection": bounded_projection,
        "pose_errors": np.concatenate(pose_errors) if pose_errors else np.empty(0),
        "moment_error": moment_error,
        "external_drive_error": external_drive_error,
        "maximum_contacts": maximum_contacts,
        "maximum_constraints": maximum_constraints,
        "constraint_type_counts": constraint_type_counts,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    native_model = mujoco.MjModel.from_xml_path(str(MODEL))
    native_data = mujoco.MjData(native_model)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    source_collision_geoms = int(
        np.count_nonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    # The source XML itself says these collision geoms make overlap detectable
    # while IK only runs forward kinematics and never reaches the contact
    # solver.  They therefore cannot be treated as a validated dynamics model.
    # Disable only collision generation in this analysis copy; mass, inertia,
    # joints, limits, passive forces and actuator moments stay source-exact.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    moment = dense_actuator_moment(model, data)
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        or f"actuator_{index}"
        for index in range(model.nu)
    ]
    leg_dofs = np.flatnonzero(np.any(moment[LEG_ACTUATOR_START:] != 0, axis=0))
    root_columns = np.flatnonzero(np.any(moment[:, :6] != 0, axis=0))
    gram = moment @ moment.T
    off_diagonal = gram - np.diag(np.diag(gram))

    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "qpos",
            "qvel",
            "qacc",
            "qfrc",
            "constraint",
            "actuator_force",
            "actuator_control",
            "bounded_control",
            "projection",
            "bounded_projection",
            "bout_index",
            "local_frame",
            "split_code",
        )
    }
    pose_errors: list[np.ndarray] = []
    bout_reports: list[dict] = []
    maximum_constraint = 0.0
    maximum_moment_error = 0.0
    maximum_external_drive = 0.0
    maximum_contacts = 0
    maximum_constraints = 0
    constraint_type_counts: dict[int, int] = {}
    native_contact_frames = 0
    native_contact_samples = 0
    native_maximum_contacts = 0
    native_inverse_delta_sumsq = 0.0
    native_inverse_delta_count = 0
    native_inverse_delta_maximum = 0.0
    native_contact_pair_counts: dict[str, int] = {}
    split_codes = {"train": 0, "validation": 1, "test": 2}

    with h5py.File(SUBSET, "r") as source:
        bout_keys = sequence(source["subset/bout_keys"])
        splits = sequence(source["subset/splits"])
        body_names = sequence(source["info/names_xpos"])
        body_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in body_names
            ],
            dtype=np.int32,
        )
        if np.any(body_ids < 0):
            raise ValueError("At least one source body name is absent from the pinned model")
        sensitivity_sources: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        for bout_index, (key, split) in enumerate(zip(bout_keys, splits)):
            group = source[key]
            qpos_source = np.asarray(group["qpos"], dtype=np.float64)
            xpos_source = np.asarray(group["xpos"], dtype=np.float64)
            result = derive_bout(
                model, data, moment, body_ids, qpos_source, xpos_source
            )
            valid = np.arange(EDGE_FRAMES, len(qpos_source) - EDGE_FRAMES)
            for name in (
                "qpos",
                "qvel",
                "qacc",
                "qfrc",
                "constraint",
                "actuator_force",
                "actuator_control",
                "bounded_control",
                "projection",
                "bounded_projection",
            ):
                arrays[name].append(result[name][valid])
            arrays["bout_index"].append(np.full(len(valid), bout_index, dtype=np.int16))
            arrays["local_frame"].append(valid.astype(np.int32))
            arrays["split_code"].append(
                np.full(len(valid), split_codes[split], dtype=np.int8)
            )
            pose_errors.append(result["pose_errors"])
            maximum_constraint = max(
                maximum_constraint, float(np.max(np.abs(result["constraint"])))
            )
            maximum_moment_error = max(maximum_moment_error, result["moment_error"])
            maximum_external_drive = max(
                maximum_external_drive, result["external_drive_error"]
            )
            maximum_contacts = max(maximum_contacts, result["maximum_contacts"])
            maximum_constraints = max(
                maximum_constraints, result["maximum_constraints"]
            )
            for constraint_type, count in result["constraint_type_counts"].items():
                constraint_type_counts[constraint_type] = (
                    constraint_type_counts.get(constraint_type, 0) + count
                )
            # Audit what would happen if the IK-only source collision geoms were
            # accidentally allowed to enter inverse dynamics.
            for frame in valid:
                native_data.qpos[:] = result["qpos"][frame]
                native_data.qvel[:] = result["qvel"][frame]
                native_data.qacc[:] = result["qacc"][frame]
                native_data.ctrl[:] = 0
                native_data.qfrc_applied[:] = 0
                native_data.xfrc_applied[:] = 0
                mujoco.mj_inverse(native_model, native_data)
                native_contact_samples += 1
                if native_data.ncon:
                    native_contact_frames += 1
                native_maximum_contacts = max(
                    native_maximum_contacts, int(native_data.ncon)
                )
                delta = native_data.qfrc_inverse - result["qfrc"][frame]
                native_inverse_delta_sumsq += float(np.dot(delta, delta))
                native_inverse_delta_count += len(delta)
                native_inverse_delta_maximum = max(
                    native_inverse_delta_maximum, float(np.max(np.abs(delta)))
                )
                for contact_index in range(native_data.ncon):
                    contact = native_data.contact[contact_index]
                    first = mujoco.mj_id2name(
                        native_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ) or f"geom_{int(contact.geom1)}"
                    second = mujoco.mj_id2name(
                        native_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ) or f"geom_{int(contact.geom2)}"
                    pair = " | ".join(sorted((first, second)))
                    native_contact_pair_counts[pair] = (
                        native_contact_pair_counts.get(pair, 0) + 1
                    )
            leg_target = result["qfrc"][valid][:, leg_dofs]
            leg_residual = (
                result["qfrc"][valid] - result["projection"][valid]
            )[:, leg_dofs]
            bounded_residual = (
                result["qfrc"][valid] - result["bounded_projection"][valid]
            )[:, leg_dofs]
            leg_controls = result["actuator_control"][valid, LEG_ACTUATOR_START:]
            low = model.actuator_ctrlrange[LEG_ACTUATOR_START:, 0]
            high = model.actuator_ctrlrange[LEG_ACTUATOR_START:, 1]
            bout_reports.append(
                {
                    "bout_key": key,
                    "split": split,
                    "source_frames": len(qpos_source),
                    "teacher_frames": len(valid),
                    "leg_projection": norm_metrics(leg_target, leg_residual),
                    "bounded_leg_projection": norm_metrics(
                        leg_target, bounded_residual
                    ),
                    "leg_control_clip_fraction": float(
                        np.mean((leg_controls < low) | (leg_controls > high))
                    ),
                }
            )
            if split not in sensitivity_sources:
                sensitivity_sources[split] = (qpos_source, xpos_source, key)

    combined = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    all_pose_errors = np.concatenate(pose_errors)
    leg_target = combined["qfrc"][:, leg_dofs]
    leg_residual = (combined["qfrc"] - combined["projection"])[:, leg_dofs]
    bounded_leg_residual = (
        combined["qfrc"] - combined["bounded_projection"]
    )[:, leg_dofs]
    leg_control = combined["actuator_control"][:, LEG_ACTUATOR_START:]
    leg_low = model.actuator_ctrlrange[LEG_ACTUATOR_START:, 0]
    leg_high = model.actuator_ctrlrange[LEG_ACTUATOR_START:, 1]
    clipped = (leg_control < leg_low) | (leg_control > leg_high)

    split_reports = {}
    for split, code in split_codes.items():
        mask = combined["split_code"] == code
        target = combined["qfrc"][mask][:, leg_dofs]
        residual = (
            combined["qfrc"][mask] - combined["projection"][mask]
        )[:, leg_dofs]
        bounded_residual = (
            combined["qfrc"][mask] - combined["bounded_projection"][mask]
        )[:, leg_dofs]
        split_reports[split] = {
            "teacher_frames": int(mask.sum()),
            "source_flies": int(
                len({row["bout_key"] for row in bout_reports if row["split"] == split})
            ),
            "leg_projection": norm_metrics(target, residual),
            "bounded_leg_projection": norm_metrics(target, bounded_residual),
        }

    # Filter sensitivity is deliberately computed on one real bout from every
    # held-out partition, never on fabricated or interpolated poses.
    sensitivity = []
    baseline_force: dict[str, np.ndarray] = {}
    for split, (qpos_source, xpos_source, key) in sensitivity_sources.items():
        for window in (7, 9, 13):
            result = derive_bout(
                model,
                data,
                moment,
                np.empty(0, dtype=np.int32),
                qpos_source,
                xpos_source,
                window=window,
                collect_pose_error=False,
            )
            pad = max(EDGE_FRAMES, window)
            valid = slice(pad, len(qpos_source) - pad)
            forces = result["actuator_force"][valid, LEG_ACTUATOR_START:]
            if window == FILTER_WINDOW:
                baseline_force[split] = forces
            sensitivity.append(
                {
                    "split": split,
                    "bout_key": key,
                    "window_frames": window,
                    "window_ms": window / FPS * 1000,
                    "leg_force_rms_native": float(np.sqrt(np.mean(forces * forces))),
                }
            )
        base = baseline_force[split]
        for row in sensitivity:
            if row["split"] != split:
                continue
            window = row["window_frames"]
            other = derive_bout(
                model,
                data,
                moment,
                np.empty(0, dtype=np.int32),
                qpos_source,
                xpos_source,
                window=window,
                collect_pose_error=False,
            )["actuator_force"]
            pad = max(EDGE_FRAMES, 13)
            base_common = derive_bout(
                model,
                data,
                moment,
                np.empty(0, dtype=np.int32),
                qpos_source,
                xpos_source,
                window=FILTER_WINDOW,
                collect_pose_error=False,
            )["actuator_force"][pad:-pad, LEG_ACTUATOR_START:]
            other_common = other[pad:-pad, LEG_ACTUATOR_START:]
            row["correlation_with_9_frame_teacher"] = float(
                np.corrcoef(base_common.ravel(), other_common.ravel())[0, 1]
            )

    # Independent equation check on deterministic saved samples.
    inverse_identity_error = 0.0
    full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    for index in np.linspace(0, len(combined["qpos"]) - 1, 17, dtype=int):
        data.qpos[:] = combined["qpos"][index]
        data.qvel[:] = combined["qvel"][index]
        data.qacc[:] = combined["qacc"][index]
        data.ctrl[:] = 0
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_inverse(model, data)
        mujoco.mj_fullM(model, full_mass, data.qM)
        reconstructed = (
            full_mass @ data.qacc
            + data.qfrc_bias
            - data.qfrc_passive
            - data.qfrc_constraint
        )
        inverse_identity_error = max(
            inverse_identity_error,
            float(np.max(np.abs(reconstructed - data.qfrc_inverse))),
        )

    string_dtype = np.str_
    np.savez_compressed(
        TEACHER,
        qpos_filtered=combined["qpos"].astype(np.float32),
        qvel_derived=combined["qvel"].astype(np.float32),
        qacc_derived=combined["qacc"].astype(np.float32),
        qfrc_inverse=combined["qfrc"].astype(np.float32),
        qfrc_constraint=combined["constraint"].astype(np.float32),
        actuator_force_unbounded=combined["actuator_force"].astype(np.float32),
        actuator_control_unbounded=combined["actuator_control"].astype(np.float32),
        actuator_control_bounded=combined["bounded_control"].astype(np.float32),
        generalized_force_projection=combined["projection"].astype(np.float32),
        bounded_generalized_force_projection=combined["bounded_projection"].astype(
            np.float32
        ),
        bout_index=combined["bout_index"],
        local_frame=combined["local_frame"],
        split_code=combined["split_code"],
        split_names=np.asarray(["train", "validation", "test"], dtype=string_dtype),
        bout_keys=np.asarray([row["bout_key"] for row in bout_reports], dtype=string_dtype),
        actuator_names=np.asarray(actuator_names, dtype=string_dtype),
        dof_names=np.asarray(dof_names(model), dtype=string_dtype),
        actuation_moment=moment.astype(np.float32),
        leg_dofs=leg_dofs.astype(np.int16),
        source_subset_sha256=np.asarray(sha256(SUBSET), dtype=string_dtype),
        source_model_commit=np.asarray(
            "defdb66932a17e41c8852f3b4c5056bee74046eb", dtype=string_dtype
        ),
        filter_window_frames=np.asarray(FILTER_WINDOW, dtype=np.int16),
        filter_polyorder=np.asarray(FILTER_POLYORDER, dtype=np.int16),
        sampling_hz=np.asarray(FPS, dtype=np.float32),
    )

    projection_metrics = norm_metrics(leg_target, leg_residual)
    bounded_metrics = norm_metrics(leg_target, bounded_leg_residual)
    per_actuator_clip = {
        name: float(clipped[:, offset].mean())
        for offset, name in enumerate(actuator_names[LEG_ACTUATOR_START:])
    }
    checks = {
        "all_teacher_arrays_finite": all(
            np.isfinite(combined[name]).all()
            for name in (
                "qpos",
                "qvel",
                "qacc",
                "qfrc",
                "actuator_force",
                "actuator_control",
            )
        ),
        "real_source_subset_hash_matches_manifest": sha256(SUBSET)
        == manifest["subset"]["sha256"],
        "all_18_real_bouts_contribute": len(bout_reports) == 18
        and len(np.unique(combined["bout_index"])) == 18,
        "whole_fly_splits_preserved": {
            split: sum(row["split"] == split for row in bout_reports)
            for split in split_codes
        }
        == {"train": 12, "validation": 3, "test": 3},
        "source_body_actuator_rank_56": int(np.linalg.matrix_rank(moment)) == 56,
        "leg_actuator_rank_48": int(
            np.linalg.matrix_rank(moment[LEG_ACTUATOR_START:, leg_dofs])
        )
        == 48,
        "no_actuator_has_root_authority": len(root_columns) == 0,
        "actuator_columns_are_orthogonal": float(np.max(np.abs(off_diagonal)))
        < 1e-12,
        "actuator_moment_constant_on_source_poses": maximum_moment_error < 1e-12,
        "no_hidden_applied_or_external_force": maximum_external_drive == 0.0,
        "analysis_copy_disables_ik_only_collisions": source_collision_geoms > 0
        and not np.any(model.geom_contype)
        and not np.any(model.geom_conaffinity),
        "analysis_has_no_contacts": maximum_contacts == 0,
        "native_ik_collision_audit_detects_self_overlap": native_contact_frames > 0
        and native_maximum_contacts > 0
        and native_inverse_delta_maximum > 0,
        "joint_limit_constraint_is_explicitly_saved": maximum_constraints > 0
        and maximum_constraint > 0
        and "constraint" in combined,
        "inverse_equation_reconstructs": inverse_identity_error < 1e-8,
        "filtered_pose_remains_source_close": float(np.percentile(all_pose_errors, 95))
        < 0.1,
        "unbounded_leg_projection_relative_residual_below_10pct": projection_metrics[
            "relative_residual_l2"
        ]
        < 0.10,
        "bounded_leg_projection_relative_residual_below_20pct": bounded_metrics[
            "relative_residual_l2"
        ]
        < 0.20,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "date": "2026-09-23",
        "scope": (
            "Inverse-dynamics teacher from real official 800 Hz running kinematics on "
            "the pinned official v1 body; not a muscle model, ground-force measurement, "
            "motor-neuron controller, or autonomous simulation."
        ),
        "source": {
            "paper": "https://www.biorxiv.org/content/10.64898/2026.05.03.722293v2",
            "dataset_folder": "https://drive.google.com/drive/folders/1flBiyFmJYWPA6EIN4Xoh_2Lc5VT2_AoV?usp=sharing",
            "subset_file": str(SUBSET.relative_to(ROOT)),
            "subset_sha256": sha256(SUBSET),
            "model_file": str(MODEL.relative_to(WORKSPACE)),
            "model_commit": "defdb66932a17e41c8852f3b4c5056bee74046eb",
        },
        "method": {
            "sampling_hz": FPS,
            "filter": {
                "kind": "cubic Savitzky-Golay on qpos, quaternion normalization, MuJoCo central manifold difference, second Savitzky-Golay on qvel and derivative",
                "window_frames": FILTER_WINDOW,
                "window_ms": FILTER_WINDOW / FPS * 1000,
                "polyorder": FILTER_POLYORDER,
                "discarded_edge_frames_per_side": EDGE_FRAMES,
            },
            "inverse_equation": "qfrc_inverse = M(q) qacc + qfrc_bias - qfrc_passive - qfrc_constraint",
            "projection": "orthogonal least-squares projection of generalized inverse force onto actuator moment columns",
            "bounded_projection": "same actuator force clipped through the source XML ctrlrange and gainprm",
            "root_wrench_interpretation": "required unmeasured environment wrench in a floorless IK model; not an identified ground-reaction force",
            "collision_policy": "all source collision geoms disabled in the inverse-dynamics analysis copy because the source XML explicitly says they are overlap detectors for forward-kinematics IK and are not inert in dynamics",
            "units": "native source MuJoCo units; length model unit is 0.1 mm according to the source geometry scaling",
        },
        "teacher": {
            "file": str(TEACHER.relative_to(ROOT)),
            "sha256": sha256(TEACHER),
            "source_bouts": len(bout_reports),
            "source_frames": int(
                sum(row["source_frames"] for row in bout_reports)
            ),
            "teacher_frames": int(len(combined["qpos"])),
            "discarded_edge_frames": int(
                sum(row["source_frames"] - row["teacher_frames"] for row in bout_reports)
            ),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "leg_actuators": model.nu - LEG_ACTUATOR_START,
            "leg_dofs_touched": int(len(leg_dofs)),
            "leg_actuator_rank": int(
                np.linalg.matrix_rank(moment[LEG_ACTUATOR_START:, leg_dofs])
            ),
            "root_actuator_columns": root_columns.tolist(),
        },
        "results": {
            "leg_projection": projection_metrics,
            "bounded_leg_projection": bounded_metrics,
            "leg_control_clip_fraction": float(clipped.mean()),
            "per_leg_actuator_clip_fraction": per_actuator_clip,
            "root_required_wrench_rms_native": float(
                np.sqrt(np.mean(combined["qfrc"][:, :6] ** 2))
            ),
            "root_required_wrench_peak_abs_native": float(
                np.max(np.abs(combined["qfrc"][:, :6]))
            ),
            "filtered_pose_body_error_mm": {
                "rms": float(np.sqrt(np.mean(all_pose_errors * all_pose_errors))),
                "p95": float(np.percentile(all_pose_errors, 95)),
                "maximum": float(np.max(all_pose_errors)),
            },
            "maximum_constraint_force_native": maximum_constraint,
            "maximum_simultaneous_constraints": maximum_constraints,
            "maximum_contacts": maximum_contacts,
            "constraint_type_counts": {
                str(key): value for key, value in sorted(constraint_type_counts.items())
            },
            "native_ik_collision_audit": {
                "source_collision_geoms": source_collision_geoms,
                "teacher_frames_audited": native_contact_samples,
                "frames_with_self_contact": native_contact_frames,
                "self_contact_frame_fraction": native_contact_frames
                / native_contact_samples,
                "maximum_simultaneous_contacts": native_maximum_contacts,
                "inverse_force_delta_rms_native": float(
                    np.sqrt(native_inverse_delta_sumsq / native_inverse_delta_count)
                ),
                "inverse_force_delta_peak_abs_native": native_inverse_delta_maximum,
                "top_contact_pairs": dict(
                    sorted(
                        native_contact_pair_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:20]
                ),
                "interpretation": "These are internal mesh/overlap-detector contacts in a model with no floor plane; using them as walking contact would corrupt the teacher.",
            },
            "inverse_identity_max_abs_error": inverse_identity_error,
            "by_split": split_reports,
            "by_bout": bout_reports,
            "filter_sensitivity": sensitivity,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "classification": {
            "A0_real_kinematic_reference": "pass",
            "A0_source_body_inverse_dynamics_teacher": "pass"
            if all(checks.values())
            else "fail",
            "A1_source_aligned_six_leg_muscle_body": "incomplete",
            "A2_motor_neuron_to_muscle_recruitment": "incomplete",
            "A3_autonomous_walking": "incomplete",
        },
        "limits": [
            "The public running file supplies kinematics, not measured leg muscle activation, motor-neuron recruitment, or force-plate contact forces.",
            "The pinned public v1 body has direct joint/tendon actuators, no muscle activation states, and no floor geom.",
            "The native IK XML generates internal self-contacts in dynamics even though its own comment says contact was inert during IK; the teacher disables those detector collisions and records a separate contamination audit.",
            "Source joint-limit constraints can be active; their generalized force is saved separately and is not labelled as a ground reaction.",
            "Root wrench is retained only as a missing-environment diagnostic and is never exported as an allowable locomotion actuator.",
            "The 48 leg controls are teacher targets for the next muscle and neural mapping stage; replaying them would be imitation, not autonomy.",
        ],
        "goal_complete": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # A compact visual audit focused on the held-out evidence and remaining gap.
    exemplar = int(np.flatnonzero(combined["bout_index"] == 0)[0])
    exemplar_mask = combined["bout_index"] == 0
    local = combined["local_frame"][exemplar_mask]
    time = local / FPS
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    chosen = [8, 11, 13, 32, 35, 37, 48, 51, 53]
    for actuator in chosen:
        axes[0, 0].plot(
            time,
            combined["actuator_force"][exemplar_mask, actuator],
            lw=0.9,
            label=actuator_names[actuator],
        )
    axes[0, 0].set_title("真实奔跑的腿部逆动力学教师力（示例）")
    axes[0, 0].set_xlabel("时间 / s")
    axes[0, 0].set_ylabel("MuJoCo 原生力单位")
    axes[0, 0].legend(fontsize=7, ncol=3)

    axes[0, 1].plot(time, combined["qfrc"][exemplar_mask, 0:3])
    axes[0, 1].set_title("无地面模型所需的根部平移外力（仅诊断）")
    axes[0, 1].set_xlabel("时间 / s")
    axes[0, 1].set_ylabel("原生广义力")
    axes[0, 1].legend(["x", "y", "z"], fontsize=8)

    split_labels = list(split_codes)
    unbounded = [
        split_reports[split]["leg_projection"]["norm_coverage"] * 100
        for split in split_labels
    ]
    bounded = [
        split_reports[split]["bounded_leg_projection"]["norm_coverage"] * 100
        for split in split_labels
    ]
    x = np.arange(len(split_labels))
    axes[1, 0].bar(x - 0.18, unbounded, 0.36, label="无限幅投影")
    axes[1, 0].bar(x + 0.18, bounded, 0.36, label="按源模型限幅")
    axes[1, 0].set_xticks(x, ["训练", "验证", "测试"])
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_ylabel("L2 范数覆盖率 / %")
    axes[1, 0].set_title("按来源个体隔离的数据分组")
    axes[1, 0].legend()

    clip_rate = clipped.mean(axis=0) * 100
    axes[1, 1].bar(np.arange(len(clip_rate)), clip_rate, color="#d46a40")
    axes[1, 1].set_title("48 个腿部执行器的控制限幅率")
    axes[1, 1].set_xlabel("腿部执行器编号")
    axes[1, 1].set_ylabel("超出原 XML ctrlrange / %")

    axes[2, 0].hist(all_pose_errors, bins=80, color="#3e91b5")
    axes[2, 0].axvline(
        np.percentile(all_pose_errors, 95), color="#d46a40", label="95 分位"
    )
    axes[2, 0].set_title("11.25 ms 滤波后的身体点偏差")
    axes[2, 0].set_xlabel("相对原始来源姿态 / mm")
    axes[2, 0].set_ylabel("身体点数")
    axes[2, 0].legend()

    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.02,
        0.96,
        "结论\n\n"
        f"18 段真实奔跑，{len(combined['qpos'])} 个教师帧\n"
        f"腿部无约束残差：{projection_metrics['relative_residual_l2']*100:.2f}%\n"
        f"源控制限幅后残差：{bounded_metrics['relative_residual_l2']*100:.2f}%\n"
        f"腿部控制超限帧比例：{clipped.mean()*100:.2f}%\n\n"
        "这证明可从真实六足运动得到稳定的关节力教师。\n"
        "它没有提供肌肉、运动神经元募集或实测地面力，\n"
        "因此仍不能称为大脑自主行走。",
        va="top",
        fontsize=12,
    )
    figure.suptitle("官方真实六足奔跑 → 来源身体逆动力学桥接", fontsize=16)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            {
                "passed": report["passed"],
                "teacher_frames": report["teacher"]["teacher_frames"],
                "leg_relative_residual": projection_metrics[
                    "relative_residual_l2"
                ],
                "bounded_leg_relative_residual": bounded_metrics[
                    "relative_residual_l2"
                ],
                "leg_control_clip_fraction": float(clipped.mean()),
                "teacher": str(TEACHER),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
