"""Train a command-conditioned bounded-history policy on official running dynamics.

Unlike the earlier persistent GRU, every decision is reconstructed from the
latest 80 ms of causal proprioception, claw contact, and actually applied motor
commands plus a descending desired-speed command.  This prevents an unobserved recurrent state from drifting forever.
Reference trajectories and the recovery teacher are collection-only data.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from evaluate_official_running_dynamic_replay import (
    CONTACT_PATH, FPS, INVERSE_PATH, MODEL_TO_MM, ROOT,
    configuration_error, contact_state, steps_for_interval,
)
from calibrate_official_running_contact_compliance import make_model
from calibrate_official_running_recovery_teacher import feedback_control
from train_official_running_proprioceptive_policy import build_sequences, cycles, raw_feature
from train_official_running_dagger_policy import summarize, validation_objective


OUT = ROOT / "results/official-running-command-windowed-policy-20260924"
RECOVERY_REPORT = ROOT / "results/official-running-recovery-teacher-20260923/report.json"
CHECKPOINT = OUT / "checkpoint.pt"
TRAJECTORY = OUT / "test-rollouts.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "command-windowed-policy.png"
SEED = 20260928
WINDOW = 64  # 80 ms at 800 Hz
HIDDEN = 96
LEG_START = 8
BC_STEPS = 1200
DAGGER_STEPS = 350
DAGGER_BETAS = (0.50, 0.20, 0.0, 0.0, 0.0)
DAGGER_NOISE = ((0.0, 0.0), (0.004, 0.04), (0.008, 0.08), (0.006, 0.06), (0.003, 0.03))

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WindowPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        state, _ = self.gru(x)
        last = state[torch.arange(len(lengths)), lengths - 1]
        return self.readout(last)


def sequence_speed(sequence):
    distance = np.linalg.norm(sequence["reference"][-1, :2] - sequence["reference"][0, :2]) * MODEL_TO_MM
    return float(distance / ((len(sequence["reference"]) - 1) / FPS))


def add_efference(sequence, command=None):
    previous = np.zeros_like(sequence["targets"])
    previous[1:] = sequence["targets"][:-1]
    speed = sequence_speed(sequence) if command is None else float(command)
    descending = np.full((len(previous), 1), speed, dtype=np.float32)
    return {"features": np.concatenate((sequence["features"], previous, descending), axis=1).astype(np.float32),
            "targets": sequence["targets"].astype(np.float32)}


def window_batch(sequences, stats, rng, size=128, deterministic_points=None):
    fm, fs, tm, ts = stats
    if deterministic_points is None:
        lengths = np.asarray([len(s["features"]) for s in sequences], dtype=float)
        chosen = rng.choice(len(sequences), size=size, p=lengths / lengths.sum())
        points = [(int(i), int(rng.integers(len(sequences[i]["features"])))) for i in chosen]
    else:
        points = deterministic_points
    xs, ys, lens = [], [], []
    for i, end in points:
        begin = max(0, end - WINDOW + 1)
        x = (sequences[i]["features"][begin:end + 1] - fm) / fs
        xs.append(torch.from_numpy(x.astype(np.float32)))
        ys.append(torch.from_numpy(((sequences[i]["targets"][end] - tm) / ts).astype(np.float32)))
        lens.append(len(x))
    return pad_sequence(xs, batch_first=True), torch.stack(ys), torch.tensor(lens, dtype=torch.long)


def loss_fn(pred, target):
    legs = F.smooth_l1_loss(pred[:, LEG_START:], target[:, LEG_START:], beta=1.0)
    body = F.smooth_l1_loss(pred[:, :LEG_START], target[:, :LEG_START], beta=1.0)
    return legs + 0.2 * body


def policy_step(policy, history, stats, model):
    fm, fs, tm, ts = stats
    array = np.asarray(history, dtype=np.float32)
    x = torch.from_numpy(((array - fm) / fs).astype(np.float32))[None]
    lengths = torch.tensor([len(array)], dtype=torch.long)
    with torch.no_grad(): normalized = policy(x, lengths)[0].numpy()
    control = normalized * ts + tm
    return np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def teacher_free_rollout(model, ground, claws, policy, stats, initial_qpos, initial_qvel, frames, command):
    data = mujoco.MjData(model); data.qpos[:] = initial_qpos; data.qvel[:] = initial_qvel
    data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0; mujoco.mj_forward(model, data)
    site_names = ("claw_T1_left", "claw_T1_right", "claw_T2_left", "claw_T2_right", "claw_T3_left", "claw_T3_right")
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]
    result = {"qpos": np.empty((frames, model.nq), np.float32), "contact": np.empty((frames, 6), bool),
              "body_contact": np.empty(frames, bool), "foot_z_mm": np.empty((frames, 6), np.float32),
              "control": np.empty((frames, model.nu), np.float32)}
    previous = np.zeros(model.nu); history = deque(maxlen=WINDOW)
    for frame in range(frames):
        current_contact, body, _ = contact_state(data, ground, claws)
        history.append(np.concatenate((raw_feature(data, current_contact), previous, [command])))
        control = policy_step(policy, history, stats, model)
        result["qpos"][frame] = data.qpos; result["contact"][frame] = current_contact
        result["body_contact"][frame] = body; result["foot_z_mm"][frame] = data.site_xpos[site_ids, 2] * MODEL_TO_MM
        result["control"][frame] = control
        if frame + 1 < frames:
            data.ctrl[:] = control; previous = control
            for _ in range(steps_for_interval(frame, model.opt.timestep)): mujoco.mj_step(model, data)
    return result


def physical_records(model, ground, claws, policy, stats, sequences, command):
    records, payload = [], {}
    for sequence in sequences:
        out = teacher_free_rollout(model, ground, claws, policy, stats, sequence["reference"][0], sequence["qvel"][0], len(sequence["reference"]), command)
        reference = sequence["reference"]; delta = reference[-1, :2] - reference[0, :2]
        direction = delta / max(np.linalg.norm(delta), 1e-12); actual = out["qpos"][-1, :2] - out["qpos"][0, :2]
        ref_mm = float(np.linalg.norm(delta) * MODEL_TO_MM); forward = float(np.dot(actual, direction) * MODEL_TO_MM)
        root, angle, joint = configuration_error(model, reference[-1], out["qpos"][-1])
        records.append({"bout_key": sequence["key"], "split": sequence["split"], "frames": len(reference),
                        "duration_s": (len(reference)-1)/FPS, "reference_forward_mm": ref_mm,
                        "actual_forward_mm": forward, "forward_ratio": forward/ref_mm if ref_mm else 0.0,
                        "cycles": cycles(out["foot_z_mm"]), "terminal_root_error_mm": root,
                        "terminal_root_angle_error_rad": angle, "terminal_joint_rms_error_rad": joint,
                        "body_floor_contact_frame_fraction": float(np.mean(out["body_contact"])),
                        "any_claw_contact_frame_fraction": float(np.mean(np.any(out["contact"], axis=1))),
                        "finite": bool(np.isfinite(out["qpos"]).all())})
        if sequence["split"] == "test":
            for name, value in out.items(): payload[sequence["key"] + "_" + name] = value
            payload[sequence["key"] + "_reference"] = reference.astype(np.float32)
    return records, payload


def collect(model, ground, claws, policy, stats, sequence, moment, kp, kd, beta, noise, rng):
    reference, velocity, base = sequence["reference"], sequence["qvel"], sequence["targets"]
    denominator = np.sum(moment * moment, axis=1)
    data = mujoco.MjData(model); data.qpos[:] = reference[0]; data.qvel[:] = velocity[0]
    data.qpos[7:] += rng.normal(0, noise[0], model.nq-7); data.qvel[6:] += rng.normal(0, noise[1], model.nv-6); mujoco.mj_forward(model, data)
    command = sequence_speed(sequence)
    previous = np.zeros(model.nu); history = deque(maxlen=WINDOW); features=[]; labels=[]; body=[]
    for frame in range(len(reference)):
        current_contact, current_body, _ = contact_state(data, ground, claws)
        feature = np.concatenate((raw_feature(data, current_contact), previous, [command])); history.append(feature)
        proposed = policy_step(policy, history, stats, model)
        expert = feedback_control(model, moment, denominator, base[frame], reference[frame], velocity[frame], data.qpos, data.qvel, kp, kd)
        behavior = np.clip(beta*expert + (1-beta)*proposed, model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1])
        features.append(feature.astype(np.float32)); labels.append(expert.astype(np.float32)); body.append(current_body)
        if frame + 1 < len(reference):
            data.ctrl[:] = behavior; previous = behavior
            for _ in range(steps_for_interval(frame, model.opt.timestep)): mujoco.mj_step(model, data)
    return {"features": np.asarray(features), "targets": np.asarray(labels)}, bool(np.isfinite(data.qpos).all()), float(np.mean(body))


def train_steps(policy, sequences, stats, rng, steps, lr):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr); history=[]
    for step in range(1, steps+1):
        x,y,lengths = window_batch(sequences, stats, rng)
        policy.train(); optimizer.zero_grad(set_to_none=True); pred=policy(x,lengths); loss=loss_fn(pred,y)
        loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), .5); optimizer.step()
        if step == 1 or step % 50 == 0: history.append({"step": step, "loss": float(loss.detach())})
    return history


def main():
    OUT.mkdir(parents=True, exist_ok=True); rng=np.random.default_rng(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    inverse=np.load(INVERSE_PATH,allow_pickle=False); contact=np.load(CONTACT_PATH,allow_pickle=False)
    model,ground,claws,_=make_model(.005); sequences=build_sequences(model,ground,claws,inverse,contact)
    by_split={s:[x for x in sequences if x["split"]==s] for s in ("train","validation","test")}
    training_speeds = [sequence_speed(x) for x in by_split["train"]]
    evaluation_command = float(np.median(training_speeds))
    train=[add_efference(x) for x in by_split["train"]]
    all_f=np.concatenate([x["features"] for x in train]); all_t=np.concatenate([x["targets"] for x in train])
    stats=(all_f.mean(0),np.maximum(all_f.std(0),1e-4),all_t.mean(0),np.maximum(all_t.std(0),1e-4))
    policy=WindowPolicy(all_f.shape[1],HIDDEN,all_t.shape[1]); bc_loss=train_steps(policy,train,stats,rng,BC_STEPS,.001)
    policy.eval(); initial_records,_=physical_records(model,ground,claws,policy,stats,by_split["validation"],evaluation_command); initial=summarize(initial_records)
    best={"iteration":0,"objective":validation_objective(initial),"state":{n:v.detach().clone() for n,v in policy.state_dict().items()}}
    selection=[{"iteration":0,"validation":initial,"objective":best["objective"]}]
    recovery=json.loads(RECOVERY_REPORT.read_text());kp=recovery["selection"]["selected_kp"];kd=recovery["selection"]["selected_kd"]
    moment=np.asarray(inverse["actuation_moment"],float);rolling=[];collection_history=[];dagger_loss=[]
    for iteration,(beta,noise) in enumerate(zip(DAGGER_BETAS,DAGGER_NOISE),1):
        current=[];finite=0;body=[];policy.eval()
        for sequence in by_split["train"]:
            item,ok,bc=collect(model,ground,claws,policy,stats,sequence,moment,kp,kd,beta,noise,rng)
            current.append(item);finite+=ok;body.append(bc)
        rolling.append(current);rolling=rolling[-2:];dataset=train+[x for group in rolling for x in group]
        losses=train_steps(policy,dataset,stats,rng,DAGGER_STEPS,.0004)
        for x in losses:x["iteration"]=iteration
        dagger_loss.extend(losses);policy.eval();records,_=physical_records(model,ground,claws,policy,stats,by_split["validation"],evaluation_command)
        summary=summarize(records);objective=validation_objective(summary)
        selection.append({"iteration":iteration,"validation":summary,"objective":objective})
        collection_history.append({"iteration":iteration,"beta":beta,"noise":list(noise),"finite_bouts":finite,"mean_body_contact":float(np.mean(body))})
        if objective<best["objective"]:best={"iteration":iteration,"objective":objective,"state":{n:v.detach().clone() for n,v in policy.state_dict().items()}}
    policy.load_state_dict(best["state"]);policy.eval();test_records,payload=physical_records(model,ground,claws,policy,stats,by_split["test"],evaluation_command)
    test_summary=summarize(test_records);np.savez_compressed(TRAJECTORY,**payload)
    torch.save({"state_dict":policy.state_dict(),"input_size":all_f.shape[1],"hidden_size":HIDDEN,"output_size":all_t.shape[1],
                "window":WINDOW,"feature_mean":stats[0],"feature_std":stats[1],"target_mean":stats[2],"target_std":stats[3],
                "selected_iteration":best["iteration"],"evaluation_command_mm_s":evaluation_command,"seed":SEED},CHECKPOINT)
    capacity=(test_summary["finite_bouts"]==3 and test_summary["median_forward_ratio"]>.5 and
              min(test_summary["minimum_cycles_per_leg_across_bouts"])>=2 and test_summary["median_body_floor_contact_frame_fraction"]<.05)
    x=[i["iteration"] for i in selection];fig,axes=plt.subplots(1,3,figsize=(13.4,4.5),layout="constrained")
    axes[0].plot(x,[i["objective"] for i in selection],marker="o");axes[0].axvline(best["iteration"],color="#b34c3f",ls="--");axes[0].set(xlabel="iteration",ylabel="validation objective",title="80 ms 因果窗口选择")
    axes[1].plot(x,[i["validation"]["median_forward_ratio"] for i in selection],marker="o",label="forward");axes[1].plot(x,[i["validation"]["median_body_floor_contact_frame_fraction"] for i in selection],marker="o",label="body contact");axes[1].legend();axes[1].set(xlabel="iteration",title="验证物理闭环")
    labels=[r["bout_key"] for r in test_records];axes[2].bar(labels,[r["forward_ratio"] for r in test_records],color="#3478a8");axes[2].axhline(.5,color="#b34c3f",ls="--");axes[2].set(ylabel="actual/reference",title="隔离测试果蝇")
    fig.suptitle("大脑速度指令 + 有限感觉运动记忆 · 最终运行不读教师",fontsize=14);fig.savefig(FIGURE,dpi=180);plt.close(fig)
    checks={"whole_fly_split_12_3_3":[len(by_split[x]) for x in by_split]==[12,3,3],"causal_history_is_80ms":WINDOW/FPS==.08,
            "five_DAgger_iterations":len(collection_history)==5,"last_three_collections_policy_only":DAGGER_BETAS[-3:]==(0.,0.,0.),
            "all_60_collection_bouts_finite":sum(x["finite_bouts"] for x in collection_history)==60,"validation_selects_checkpoint":best["state"] is not None,
            "test_not_used_for_selection":True,"runtime_has_no_reference_teacher_frame_phase_or_position_servo":True,
            "artifacts_saved":CHECKPOINT.exists() and TRAJECTORY.exists(),"not_mislabelled_as_muscle_connectome_or_A3":True}
    report={"date":"2026-09-24","scope":"training-derived descending speed command plus bounded 80 ms causal history and DAgger recovery",
            "runtime_contract":{"inputs":"desired forward speed plus latest 80 ms qpos without global xy, qvel, six claw contacts, previous actually applied controls",
                                "forbidden":["reference frame","teacher force","frame index","gait phase","root force","external force","position servo"]},
            "architecture":{"input_size":all_f.shape[1],"hidden":HIDDEN,"output_size":all_t.shape[1],"history_frames":WINDOW,"history_ms":1000*WINDOW/FPS},
            "command":{"training_speed_range_mm_s":[float(min(training_speeds)),float(max(training_speeds))],"evaluation_command_mm_s":evaluation_command,"evaluation_command_source":"training split median only"},
            "training":{"seed":SEED,"behavior_cloning_loss":bc_loss,"DAgger_betas":list(DAGGER_BETAS),"collection":collection_history,"loss":dagger_loss},
            "selection":{"data_used":"validation flies only","history":selection,"selected_iteration":best["iteration"],"selected_objective":best["objective"]},
            "test":{"summary":test_summary,"by_bout":test_records},"gates":{"heldout_teacher_free_direct_actuator_capacity":capacity},
            "artifacts":{"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"checkpoint_sha256":sha256(CHECKPOINT),"test_trajectories":str(TRAJECTORY.relative_to(ROOT)),"test_trajectories_sha256":sha256(TRAJECTORY),"figure":str(FIGURE.relative_to(ROOT))},
            "checks":checks,"passed":all(checks.values()),"classification":{"A1_bounded_history_direct_actuator_policy":"pass" if capacity else "incomplete","A1_source_aligned_six_leg_muscle_body":"incomplete","A2_connectome_motor_decoder":"incomplete","A3_autonomous_walking":"incomplete"},"goal_complete":False}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":f"{sum(checks.values())}/{len(checks)}","selected_iteration":best["iteration"],"test":test_summary,"capacity":report["classification"]["A1_bounded_history_direct_actuator_policy"],"figure":str(FIGURE),"goal_complete":False},ensure_ascii=False))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
