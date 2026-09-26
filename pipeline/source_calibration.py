"""Run the source-aligned FlyGym 2.1 / FlyMimic calibration seam.

This is an A0/A1 diagnostic for a tethered left-front leg.  It does not run
the MaleCNS, create a six-leg controller, or claim autonomous locomotion.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "external"
FLYGYM = WORKSPACE / "work/flygym"
FLYGYM_SRC = FLYGYM / "src"
OUT = ROOT / "results/source-calibration-20260922"
XML_SHA256 = "04f6070d6733940357be005ca72c02ba0d9455538ff018da70c74de7458e9531"
FLYGYM_RELEASE_COMMIT = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
FLYGYM_HEAD = "38c8ec61034cd59bc5ba0de20688d4a3c0000d60"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def source_xml() -> Path:
    candidates = [
        ROOT / "data/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml",
        WORKSPACE / "work/closed-loop/final-extracted/fly-brain/data/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml",
        WORKSPACE / "work/release-20260915/fly-brain/data/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml",
    ]
    for path in candidates:
        mesh = path.parent / "meshes/stl/Thorax.stl"
        if path.is_file() and mesh.is_file() and sha(path) == XML_SHA256:
            return path.resolve()
    raise FileNotFoundError("No complete, hash-matched FlyMimic XML + mesh set")


def rollout(env, policy):
    obs, _ = env.reset(seed=0)
    rewards = []
    started = time.perf_counter()
    for _ in range(224):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return np.asarray(rewards), time.perf_counter() - started


def summary(values: np.ndarray) -> dict:
    return {
        "steps": int(len(values)),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    if str(FLYGYM_SRC) not in sys.path:
        sys.path.insert(0, str(FLYGYM_SRC))
    import gymnasium
    import mujoco
    import pandas as pd
    from flygym_demo.muscle_imitation import (
        ImitationConfig,
        MoCapDataset,
        make_imitation_env,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    xml = source_xml()
    local_head = git_head(FLYGYM)
    if local_head != FLYGYM_HEAD:
        raise RuntimeError(f"FlyGym source drift: {local_head}")
    dataset = MoCapDataset.default()
    clip = dataset.load("0002")
    arrays = {
        name: dataset.clip_dir / folder / "0002.npy"
        for name, folder in {
            "qpos": "qpos", "qvel": "qvel", "xipos": "xipos", "xivel": "xivel"
        }.items()
    }

    build_started = time.perf_counter()
    env = make_imitation_env(
        xml_path=xml,
        config=ImitationConfig(clip="0002", test=True),
        dataset=dataset,
    )
    build_seconds = time.perf_counter() - build_started
    model, data = env.sim.mj_model, env.sim.mj_data

    # A0: impose each recorded state and independently recompute forward
    # kinematics.  This checks registration/data plumbing, not muscle control.
    direct_reward, xpos_error = [], []
    for frame in range(clip.n_frames):
        env.sim.reset()
        data.qpos[env._tracked_qposadrs] = clip.qpos[frame]
        data.qvel[env._tracked_qveladrs] = clip.qvel[frame]
        mujoco.mj_forward(model, data)
        per_body = np.linalg.norm(
            clip.xipos[frame] - data.xpos[env._tracked_body_ids], axis=1
        )
        xpos_error.append(per_body)
        xpos_reward = np.exp(-5.0 * np.mean(per_body))
        direct_reward.append((1.0 + 1.0 + xpos_reward) / 3.0)
    direct_reward = np.asarray(direct_reward)
    xpos_error = np.asarray(xpos_error)

    # A1 smoke controls.  These demonstrate the size of the active-control gap;
    # none is a fitted policy and none is allowed to satisfy a biological gate.
    policies = {
        "zero": lambda _obs: np.zeros(env.n_muscles, np.float32),
        "constant_0.1": lambda _obs: np.full(env.n_muscles, .1, np.float32),
    }
    rngs = {seed: np.random.default_rng(seed) for seed in (0, 1, 2)}
    for seed, rng in rngs.items():
        policies[f"random_{seed}"] = lambda _obs, rng=rng: rng.uniform(
            .05, .6, env.n_muscles
        ).astype(np.float32)
    traces, wall = {}, {}
    for name, policy in policies.items():
        traces[name], wall[name] = rollout(env, policy)
    env.close()

    frame_count = min(map(len, traces.values()))
    table = {
        "time_s": np.arange(frame_count) * .002,
        "direct_state_replay": direct_reward[1:frame_count + 1],
        "fk_mean_body_error_mm": xpos_error[1:frame_count + 1].mean(axis=1),
    }
    table.update({name: values[:frame_count] for name, values in traces.items()})
    pd.DataFrame(table).to_csv(OUT / "traces.csv", index=False)

    report = {
        "scope": "Source-aligned A0/A1 calibration; tethered left-front leg only",
        "not_claimed": [
            "six-leg walking", "free-body support", "MaleCNS control",
            "neural autonomy", "biological parameter identification",
        ],
        "provenance": {
            "flygym": {
                "repository": "https://github.com/NeLy-EPFL/flygym",
                "release": "v2.1.0",
                "release_commit": FLYGYM_RELEASE_COMMIT,
                "tested_local_commit": local_head,
                "license": "Apache-2.0",
            },
            "flymimic": {
                "repository": "https://github.com/gizemozd/FlyMimic",
                "commit": "9ea1131626cd76f7203b74076ef8f0e9cab30bef",
                "license": "Apache-2.0",
            },
            "xml": {"path": str(xml), "sha256": sha(xml)},
            "mocap": {
                "clip": "0002", "control_rate_hz": 500,
                "files": {name: {"path": str(path), "sha256": sha(path)}
                          for name, path in arrays.items()},
            },
        },
        "runtime": {
            "python": sys.version.split()[0], "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__, "build_seconds": build_seconds,
            "rollout_wall_seconds": wall,
        },
        "model": {
            "bodies": int(model.nbody), "joints": int(model.njnt),
            "actuators": int(model.nu), "activation_states": int(model.na),
            "muscles": int(len(env.muscle_names)),
            "muscle_names": env.muscle_names,
            "physics_timestep_s": float(model.opt.timestep),
            "tracked_joint_dofs": int(clip.qpos.shape[1]),
            "clip_frames": int(clip.n_frames),
            "clip_duration_s": float((clip.n_frames - 1) / 500),
        },
        "A0_registration": {
            "status": "pass",
            "criterion": "all direct-state replay rewards >= 0.90 and finite",
            "direct_state_reward": summary(direct_reward),
            "body_position_error_mm": {
                "samples": int(xpos_error.size),
                "mean": float(xpos_error.mean()),
                "median": float(np.median(xpos_error)),
                "p95": float(np.percentile(xpos_error, 95)),
                "maximum": float(xpos_error.max()),
            },
        },
        "A1_muscle_smoke": {
            "status": "incomplete",
            "controls": {name: summary(values) for name, values in traces.items()},
            "interpretation": (
                "The official source-aligned environment runs locally. Zero, constant and random "
                "activations are negative controls, not a learned controller or capacity proof."
            ),
        },
        "hypothesis_update": {
            "H1_cross_body_calibration": "still plausible; source-aligned registration passed but six-leg transfer was not tested",
            "H2_fixed_reference_demand": "unresolved; active-control gap measured, trajectory feasibility not yet established",
            "H3_motor_unit_pooling_filter": "not tested in this source-body seam",
            "H4_full_network_context": "not tested in this source-body seam",
        },
        "next_gate": (
            "Fit a source-leg muscle controller on clip 0002 with held-out segments, then compare "
            "the same force/length/recruitment audit before any six-leg parameter transfer."
        ),
        "goal_complete": False,
    }
    if not (np.isfinite(direct_reward).all() and direct_reward.min() >= .90):
        report["A0_registration"]["status"] = "fail"
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    t = table["time_s"]
    axes[0].plot(t, table["direct_state_replay"], label="A0 direct state replay", lw=2)
    axes[0].plot(t, table["zero"], label="A1 zero activation")
    axes[0].plot(t, table["random_0"], label="A1 random seed 0", alpha=.8)
    axes[0].set(ylabel="tracking reward", ylim=(0, 1.02), title="Source-aligned FlyMimic calibration")
    axes[0].legend(loc="best")
    axes[1].plot(t, table["fk_mean_body_error_mm"], color="#7c3aed")
    axes[1].set(xlabel="model time (s)", ylabel="mean body-position error (mm)",
                title="Motion-data / model registration")
    fig.savefig(OUT / "source-calibration.png", dpi=160)
    plt.close(fig)
    print(json.dumps({
        "A0": report["A0_registration"]["status"],
        "A1": report["A1_muscle_smoke"]["status"],
        "direct_reward_mean": report["A0_registration"]["direct_state_reward"]["mean"],
        "zero_reward_mean": report["A1_muscle_smoke"]["controls"]["zero"]["mean"],
        "random_reward_means": [report["A1_muscle_smoke"]["controls"][f"random_{s}"]["mean"] for s in (0, 1, 2)],
        "goal_complete": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
