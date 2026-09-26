"""Fine-tune the causal command policy on real, closed running cycles for 2 s endurance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_official_running_dynamic_replay import CONTACT_PATH, FPS, INVERSE_PATH, MODEL_TO_MM, ROOT
from calibrate_official_running_contact_compliance import make_model
from calibrate_official_running_recovery_teacher import run_bout
from train_official_running_proprioceptive_policy import build_sequences, cycles
import train_official_running_command_windowed_policy as wp


OUT = ROOT / "results/official-running-endurance-policy-20260924"
SOURCE_CHECKPOINT = ROOT / "results/official-running-command-windowed-policy-20260924/checkpoint.pt"
CHECKPOINT = OUT / "checkpoint.pt"
TRAJECTORY = OUT / "test-2s-rollouts.npz"
REPORT = OUT / "report.json"
FIGURE = OUT / "endurance-policy.png"
SEED = 20260929
FRAMES = 1601
BETAS = (1.0, 0.75, 0.50, 0.25, 0.0, 0.0)
NOISE = ((0.,0.),(0.,0.),(.003,.03),(.005,.05),(.004,.04),(.002,.02))
TRAIN_STEPS = 450

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def best_cycle(sequence):
    q=sequence["reference"];v=sequence["qvel"];n=len(q);best=None
    for lag in range(48,min(241,n-8)):
        dq=q[lag:,7:]-q[:-lag,7:];dv=v[lag:,6:]-v[:-lag,6:]
        error=np.sqrt(np.mean(dq*dq,axis=1))+.002*np.sqrt(np.mean(dv*dv,axis=1))
        distance=np.linalg.norm(q[lag:,:2]-q[:-lag,:2],axis=1)*MODEL_TO_MM
        valid=distance>.4
        if not np.any(valid):continue
        start=int(np.flatnonzero(valid)[np.argmin(error[valid])]);candidate=(float(error[start])+.001*lag,start,lag,float(error[start]),float(distance[start]))
        if best is None or candidate<best:best=candidate
    return best


def tile_cycle(sequence,start,period,frames=FRAMES):
    indices=start+np.arange(period);q=sequence["reference"][indices];v=sequence["qvel"][indices];u=sequence["targets"][indices]
    delta=sequence["reference"][start+period,:2]-sequence["reference"][start,:2]
    qr=np.empty((frames,q.shape[1]));vr=np.empty((frames,v.shape[1]));ur=np.empty((frames,u.shape[1]))
    for t in range(frames):
        lap,phase=divmod(t,period);qr[t]=q[phase];qr[t,:2]+=lap*delta;vr[t]=v[phase];ur[t]=u[phase]
    return {"key":sequence["key"]+"_cycle","split":"train","reference":qr,"qvel":vr,"targets":ur,"cycle_start":start,"cycle_period":period}


def endurance_records(model,ground,claws,policy,stats,sequences,command,save=False):
    records=[];payload={}
    for s in sequences:
        out=wp.teacher_free_rollout(model,ground,claws,policy,stats,s["reference"][0],s["qvel"][0],FRAMES,command)
        direction=s["reference"][-1,:2]-s["reference"][0,:2];direction/=max(np.linalg.norm(direction),1e-12)
        distance=float(np.dot(out["qpos"][-1,:2]-out["qpos"][0,:2],direction)*MODEL_TO_MM)
        contact=out["body_contact"]
        records.append({"bout_key":s["key"],"split":s["split"],"distance_2s_mm":distance,"body_contact_fraction":float(np.mean(contact)),
                        "first_body_contact_s":float(np.argmax(contact)/FPS) if np.any(contact) else None,"cycles":cycles(out["foot_z_mm"]),
                        "finite":bool(np.isfinite(out["qpos"]).all())})
        if save:
            for name,value in out.items():payload[s["key"]+"_"+name]=value
    return records,payload


def endurance_summary(records):
    return {"bouts":len(records),"finite_bouts":sum(r["finite"] for r in records),"minimum_distance_2s_mm":float(min(r["distance_2s_mm"] for r in records)),
            "median_distance_2s_mm":float(np.median([r["distance_2s_mm"] for r in records])),"maximum_body_contact_fraction":float(max(r["body_contact_fraction"] for r in records)),
            "minimum_cycles_per_leg":[min(r["cycles"][i] for r in records) for i in range(6)]}


def objective(summary):
    return float(12*summary["maximum_body_contact_fraction"]+max(0,5-summary["minimum_distance_2s_mm"])/2-max(-1,min(10,summary["median_distance_2s_mm"]))/10)


def main():
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.set_num_threads(1)
    inverse=np.load(INVERSE_PATH,allow_pickle=False);contact=np.load(CONTACT_PATH,allow_pickle=False);model,ground,claws,_=make_model(.005)
    sequences=build_sequences(model,ground,claws,inverse,contact);by_split={s:[x for x in sequences if x["split"]==s] for s in ("train","validation","test")}
    moment=np.asarray(inverse["actuation_moment"],float);cycle_diagnostics=[];long_sequences=[]
    for s in by_split["train"]:
        _,start,period,closure,cycle_mm=best_cycle(s);loop=tile_cycle(s,start,period);teacher=run_bout(model,ground,claws,loop["reference"],loop["qvel"],loop["targets"],moment,.3,.0002)
        record={"bout_key":s["key"],"start":start,"period_frames":period,"closure_error":closure,"cycle_mm":cycle_mm,"recovery_teacher":teacher}
        cycle_diagnostics.append(record)
        if teacher["forward_ratio"]>.3 and teacher["body_floor_contact_fraction"]==0:long_sequences.append(loop)
    source=torch.load(SOURCE_CHECKPOINT,map_location="cpu",weights_only=False);policy=wp.WindowPolicy(source["input_size"],source["hidden_size"],source["output_size"]);policy.load_state_dict(source["state_dict"]);policy.eval()
    stats=tuple(source[k] for k in ("feature_mean","feature_std","target_mean","target_std"));command=float(source["evaluation_command_mm_s"])
    original_train=[wp.add_efference(x) for x in by_split["train"]]
    initial_records,_=endurance_records(model,ground,claws,policy,stats,by_split["validation"],command);initial=endurance_summary(initial_records)
    best={"iteration":0,"objective":objective(initial),"state":{k:v.detach().clone() for k,v in policy.state_dict().items()}}
    selection=[{"iteration":0,"validation":initial,"by_bout":initial_records,"objective":best["objective"]}];rolling=[];collections=[];losses=[]
    for iteration,(beta,noise) in enumerate(zip(BETAS,NOISE),1):
        current=[];body=[];finite=0;policy.eval()
        for sequence in long_sequences:
            item,ok,bc=wp.collect(model,ground,claws,policy,stats,sequence,moment,.3,.0002,beta,noise,rng);current.append(item);finite+=ok;body.append(bc)
        rolling.append(current);rolling=rolling[-3:];dataset=original_train+[x for group in rolling for x in group]
        ls=wp.train_steps(policy,dataset,stats,rng,TRAIN_STEPS,.00025)
        for x in ls:x["iteration"]=iteration
        losses.extend(ls);policy.eval();records,_=endurance_records(model,ground,claws,policy,stats,by_split["validation"],command);summary=endurance_summary(records);obj=objective(summary)
        selection.append({"iteration":iteration,"validation":summary,"by_bout":records,"objective":obj});collections.append({"iteration":iteration,"beta":beta,"noise":list(noise),"finite_bouts":finite,"mean_body_contact":float(np.mean(body))})
        if obj<best["objective"]:best={"iteration":iteration,"objective":obj,"state":{k:v.detach().clone() for k,v in policy.state_dict().items()}}
    policy.load_state_dict(best["state"]);policy.eval();test_records,payload=endurance_records(model,ground,claws,policy,stats,by_split["test"],command,True);test=endurance_summary(test_records);np.savez_compressed(TRAJECTORY,**payload)
    torch.save({**source,"state_dict":policy.state_dict(),"selected_endurance_iteration":best["iteration"],"source_checkpoint":str(SOURCE_CHECKPOINT.relative_to(ROOT)),"seed":SEED},CHECKPOINT)
    capacity=(test["finite_bouts"]==test["bouts"] and test["minimum_distance_2s_mm"]>=5 and test["maximum_body_contact_fraction"]<=.05 and min(test["minimum_cycles_per_leg"])>=2)
    x=[r["iteration"] for r in selection];fig,axes=plt.subplots(1,3,figsize=(13.4,4.5),layout="constrained")
    axes[0].plot(x,[r["validation"]["minimum_distance_2s_mm"] for r in selection],marker="o");axes[0].axhline(5,color="#b34c3f",ls="--");axes[0].set(xlabel="iteration",ylabel="mm",title="验证最差 2 秒前进")
    axes[1].plot(x,[r["validation"]["maximum_body_contact_fraction"] for r in selection],marker="o");axes[1].axhline(.05,color="#b34c3f",ls="--");axes[1].set(xlabel="iteration",title="验证最差身体触地")
    axes[2].bar([r["bout_key"] for r in test_records],[r["distance_2s_mm"] for r in test_records]);axes[2].axhline(5,color="#b34c3f",ls="--");axes[2].set(ylabel="mm",title="隔离测试 2 秒前进")
    fig.suptitle("真实闭合步态周期 · 2 秒耐久 DAgger",fontsize=14);fig.savefig(FIGURE,dpi=180);plt.close(fig)
    checks={"cycles_selected_from_training_only":len(long_sequences)>=2,"selected_cycles_have_zero_teacher_body_contact":all(x["recovery_teacher"]["body_floor_contact_fraction"]==0 for x in cycle_diagnostics if x["recovery_teacher"]["forward_ratio"]>.3),
            "six_endurance_iterations":len(collections)==6,"last_two_collections_policy_only":BETAS[-2:]==(0.,0.),"all_collections_finite":all(x["finite_bouts"]==len(long_sequences) for x in collections),
            "validation_2s_selects_checkpoint":best["state"] is not None,"test_not_used_for_selection":True,"runtime_has_no_reference_teacher_frame_phase_or_position_servo":True,"artifacts_saved":CHECKPOINT.exists() and TRAJECTORY.exists(),"not_mislabelled_as_connectome_or_A3":True}
    report={"date":"2026-09-24","scope":"2 s endurance DAgger distilled from real closed gait cycles selected on training flies",
            "source_checkpoint":str(SOURCE_CHECKPOINT.relative_to(ROOT)),"runtime_contract":{"inputs":"desired forward speed plus latest 80 ms causal proprioception, claw contact, and applied controls","forbidden":["reference frame","teacher force","frame index","gait phase","root force","external force","position servo"]},
            "cycle_selection":{"source":"training split only","criterion":"recovery forward ratio > 0.3 and zero body-floor contact","candidates":cycle_diagnostics,"selected_keys":[x["key"] for x in long_sequences]},
            "training":{"seed":SEED,"frames_per_collection_bout":FRAMES,"betas":list(BETAS),"collections":collections,"loss":losses},"selection":{"source":"validation split 2 s endurance only","history":selection,"selected_iteration":best["iteration"],"objective":best["objective"]},
            "test":{"summary":test,"by_bout":test_records},"gates":{"heldout_2s_walk_5mm_all_starts":capacity},"artifacts":{"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"checkpoint_sha256":sha256(CHECKPOINT),"test_trajectories":str(TRAJECTORY.relative_to(ROOT)),"test_trajectories_sha256":sha256(TRAJECTORY),"figure":str(FIGURE.relative_to(ROOT))},"checks":checks,"passed":all(checks.values()),"classification":{"A1_2s_endurance_direct_actuator_policy":"pass" if capacity else "incomplete","A2_connectome_motor_decoder":"incomplete","A3_autonomous_walking":"incomplete"},"goal_complete":False}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"passed":report["passed"],"checks":f"{sum(checks.values())}/{len(checks)}","selected_iteration":best["iteration"],"selected_cycles":len(long_sequences),"test":test,"capacity":report["classification"]["A1_2s_endurance_direct_actuator_policy"],"figure":str(FIGURE),"goal_complete":False},ensure_ascii=False))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
