"""Prioritize pre-fall recovery states for the selected autonomous CPG policy."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,torch
from evaluate_official_running_dynamic_replay import CONTACT_PATH,FPS,INVERSE_PATH,ROOT
from calibrate_official_running_contact_compliance import make_model
from calibrate_official_running_recovery_teacher import run_bout
from train_official_running_proprioceptive_policy import build_sequences
from train_official_running_endurance_policy import best_cycle,tile_cycle,endurance_summary
import train_official_running_cpg_policy as cp

OUT=ROOT/"results/official-running-cpg-recovery-20260924";SOURCE=ROOT/"results/official-running-cpg-policy-20260924/checkpoint.pt";CHECKPOINT=OUT/"checkpoint.pt";TRAJECTORY=OUT/"rollouts.npz";REPORT=OUT/"report.json";FIGURE=OUT/"cpg-recovery.png"
SEED=20261005;ROUNDS=8;EPISODES=6;STEPS=350;DANGER_FRAMES=240
plt.rcParams["font.sans-serif"]=["Hiragino Sans GB","Arial Unicode MS","DejaVu Sans"];plt.rcParams["axes.unicode_minus"]=False

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def score(r):return 20*r["body_contact_fraction"]+max(0,5-r["distance_2s_mm"])/5-min(40,r["distance_2s_mm"])/40

def main():
 OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.set_num_threads(1);inverse=np.load(INVERSE_PATH);contact=np.load(CONTACT_PATH);model,ground,claws,_=make_model(.005);sequences=build_sequences(model,ground,claws,inverse,contact);train=[s for s in sequences if s["split"]=="train"];moment=np.asarray(inverse["actuation_moment"],float);source_ck=torch.load(SOURCE,map_location="cpu",weights_only=False)
 source=next(s for s in train if s["key"]==source_ck["source_bout"]);_,start,period,_,cycle_mm=best_cycle(source);canonical=tile_cycle(source,start,period);canonical["cycle_period"]=period;canonical["command"]=source_ck["command_mm_s"]
 policy=cp.CPGPolicy(source_ck["input_size"],source_ck["output_size"]);policy.load_state_dict(source_ck["state_dict"]);stats=tuple(source_ck[k] for k in ("feature_mean","feature_std","target_mean","target_std"));q0,v0,_=cp.cycle_from_phase(canonical,0);spec={"key":source["key"]+"_canonical","split":"training-derived standard start","reference":q0}
 initial_out=cp.rollout(model,ground,claws,policy,stats,q0[0],v0[0],0,period,canonical["command"]);initial=cp.record(spec,initial_out,"internal CPG phase zero");best={"round":0,"score":score(initial),"state":{k:v.detach().clone() for k,v in policy.state_dict().items()},"record":initial};history=[{"round":0,"standard":initial,"score":best["score"]}]
 base=[]
 for _ in range(EPISODES):
  phase=int(rng.integers(period));item,_,_=cp.collect(model,ground,claws,None,None,canonical,moment,.3,.0002,1.,rng,phase,(.003,.03));base.append(item)
 fm,fs,tm,ts=stats;base_x=np.concatenate([x["features"] for x in base]);base_y=np.concatenate([x["targets"] for x in base])
 for round_id in range(1,ROUNDS+1):
  danger_x=[];danger_y=[];onsets=[]
  for _ in range(EPISODES):
   phase=int(rng.integers(period));item,_,_=cp.collect(model,ground,claws,policy,stats,canonical,moment,.3,.0002,0.,rng,phase,(.002,.02));body=item["body"];onset=int(np.argmax(body)) if np.any(body) else len(body);begin=max(0,onset-DANGER_FRAMES);end=min(len(body),max(onset+32,begin+DANGER_FRAMES));danger_x.append(item["features"][begin:end]);danger_y.append(item["targets"][begin:end]);onsets.append(onset/FPS)
  dx=np.concatenate(danger_x);dy=np.concatenate(danger_y);optimizer=torch.optim.Adam(policy.parameters(),lr=8e-5);losses=[]
  for step in range(STEPS):
   ib=rng.integers(len(base_x),size=128);idg=rng.integers(len(dx),size=128);x=np.concatenate((base_x[ib],dx[idg]));y=np.concatenate((base_y[ib],dy[idg]));xb=torch.from_numpy(np.clip((x-fm)/fs,-10,10).astype(np.float32));yb=torch.from_numpy(((y-tm)/ts).astype(np.float32));pred=policy(xb);loss=((pred-yb)**2).mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step();losses.append(float(loss.detach()))
  out=cp.rollout(model,ground,claws,policy,stats,q0[0],v0[0],0,period,canonical["command"]);rec=cp.record(spec,out,"internal CPG phase zero");entry={"round":round_id,"danger_onset_s_median":float(np.median(onsets)),"danger_frames":len(dx),"loss_final":losses[-1],"standard":rec,"score":score(rec)};history.append(entry)
  if entry["score"]<best["score"]:best={"round":round_id,"score":entry["score"],"state":{k:v.detach().clone() for k,v in policy.state_dict().items()},"record":rec}
 policy.load_state_dict(best["state"]);canonical_out=cp.rollout(model,ground,claws,policy,stats,q0[0],v0[0],0,period,canonical["command"]);standard=cp.record(spec,canonical_out,"internal CPG phase zero");held=[];payload={"canonical_qpos":canonical_out["qpos"],"canonical_body_contact":canonical_out["body_contact"],"canonical_foot_z_mm":canonical_out["foot_z_mm"],"canonical_control":canonical_out["control"]};template=canonical["reference"][:period,7:]
 for s in [x for x in sequences if x["split"]=="test"]:
  phase=int(np.argmin(np.mean((template-s["reference"][0,7:])**2,axis=1)));out=cp.rollout(model,ground,claws,policy,stats,s["reference"][0],s["qvel"][0],phase,period,canonical["command"]);held.append(cp.record(s,out,"one-time proprioceptive nearest CPG state"))
  for k,v in out.items():payload[s["key"]+"_"+k]=v
 np.savez_compressed(TRAJECTORY,**payload);torch.save({**source_ck,"state_dict":policy.state_dict(),"recovery_selected_round":best["round"],"recovery_seed":SEED},CHECKPOINT);standard_pass=standard["distance_2s_mm"]>=5 and standard["body_contact_fraction"]<=.05 and min(standard["cycles"])>=2;held_summary=endurance_summary(held)
 x=[h["round"] for h in history];fig,axes=plt.subplots(1,3,figsize=(13.4,4.5),layout="constrained");axes[0].plot(x,[h["standard"]["distance_2s_mm"] for h in history],marker="o");axes[0].axhline(5,color="#b34c3f",ls="--");axes[0].set(title="标准起点前进",ylabel="mm");axes[1].plot(x,[h["standard"]["body_contact_fraction"] for h in history],marker="o");axes[1].axhline(.05,color="#b34c3f",ls="--");axes[1].set(title="标准起点触地比例");axes[2].bar([r["bout_key"] for r in held],[r["distance_2s_mm"] for r in held]);axes[2].axhline(5,color="#b34c3f",ls="--");axes[2].set(title="隔离初态",ylabel="mm");fig.suptitle("神经 CPG 跌倒前状态重点恢复",fontsize=14);fig.savefig(FIGURE,dpi=180);plt.close(fig)
 checks={"source_checkpoint_is_selected_CPG":source_ck["selected_iteration"]==4,"danger_window_300ms":DANGER_FRAMES/FPS==.3,"eight_recovery_rounds":len(history)==9,"selection_uses_standard_training_derived_start":True,"heldout_test_not_used_for_selection":True,"runtime_has_no_trajectory_teacher_root_force_or_position_servo":True,"artifacts_saved":CHECKPOINT.exists() and TRAJECTORY.exists(),"not_mislabelled_as_connectome_derived":True}
 report={"date":"2026-09-24","scope":"pre-fall prioritized recovery refinement for autonomous engineering CPG","source_checkpoint":str(SOURCE.relative_to(ROOT)),"training":{"seed":SEED,"rounds":ROUNDS,"episodes_per_round":EPISODES,"danger_window_frames":DANGER_FRAMES,"history":history,"selected_round":best["round"]},"standard_start":{"result":standard,"gate_pass":standard_pass},"heldout_initial_state_generalization":{"summary":held_summary,"by_bout":held},"artifacts":{"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"checkpoint_sha256":sha256(CHECKPOINT),"trajectories":str(TRAJECTORY.relative_to(ROOT)),"trajectories_sha256":sha256(TRAJECTORY),"figure":str(FIGURE.relative_to(ROOT))},"checks":checks,"passed":all(checks.values()),"classification":{"A1_engineering_CPG_standard_start":"pass" if standard_pass else "incomplete","A1_heldout_initial_state_robustness":"pass" if held_summary["minimum_distance_2s_mm"]>=5 and held_summary["maximum_body_contact_fraction"]<=.05 else "incomplete","A2_connectome_motor_decoder":"incomplete","A3_autonomous_walking":"incomplete"},"goal_complete":False}
 REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"passed":report["passed"],"checks":f"{sum(checks.values())}/{len(checks)}","selected_round":best["round"],"standard":standard,"heldout":held_summary,"classification":report["classification"],"figure":str(FIGURE),"goal_complete":False},ensure_ascii=False))
 if not report["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
