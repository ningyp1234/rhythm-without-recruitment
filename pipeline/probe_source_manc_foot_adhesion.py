"""Bound the missing foot-grip outlet using original-author MANC rates.

Adhesion is NeuroMechFly's abstract tarsus actuator, not a reconstructed
long-tendon/claw. All conditions use identical 78 Hill channels and source
MANC rates. Contact-gated adhesion is an external reflex control bound;
LTM-gated adhesion is an uncalibrated MN-to-actuator transduction hypothesis.
Neither condition can establish brain-autonomous walking by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

from homologous_hill_body import HomologousHillBody
from probe_author_motor_pool_aggregation import original_rates, pool_trace
from source_calibration import ROOT
from source_fullmanc_exact_ensemble import NEURONS
from source_latest_vnc_hill_interface import compatible_neurons
from walking_benchmark import score
from walking_body import LEGS

OUT = ROOT / "results/source-manc-foot-adhesion-20260924"
ROWS = (117, 2)
MODES = ("absent", "released", "always_on", "contact_gated", "ltm_rate_gated")


def ltm_commands(neurons, selected, source_rates, types=("ltm MN",), pooling="mean"):
    commands = np.zeros((6, 200))
    cell_count = []
    for li, (_, segment, side, _) in enumerate(LEGS):
        group = np.flatnonzero(
            neurons.superclass.eq("vnc_motor")
            & neurons.somaNeuromere.eq(segment)
            & neurons.side.eq(side)
            & neurons.type.str.strip().isin(types)
        )
        positions = np.searchsorted(selected, group)
        assert np.all(positions < len(selected)) and np.array_equal(selected[positions], group)
        cell_count.append(len(group))
        if len(group):
            samples = source_rates[positions, 10:2001:10]
            if pooling == "mean":rates = samples.mean(axis=0)
            elif pooling == "max":rates = samples.max(axis=0)
            elif pooling == "sum":rates = samples.sum(axis=0)
            else:raise ValueError(pooling)
            commands[li] = 1 - np.exp(-np.maximum(rates, 0) / 100)
    return commands, cell_count


def run_condition(body, pooled, grip_trace, mode):
    body.reset()
    m, d = body.sim.mj_model, body.sim.mj_data
    nstep = round(0.01 / body.sim.timestep)
    assert abs(nstep * body.sim.timestep - 0.01) < 1e-12
    rows, feet, contacts, applied = [], [], [], []
    previous_contact = np.zeros(6)
    for tick in range(200):
        rates = pooled[:, tick]
        body.hill_rates = rates.copy()
        body.hill_excitation = 1 - np.exp(-np.maximum(rates, 0) / 100)
        d.ctrl[:] = 0
        d.ctrl[body.hill_actuators] = body.hill_excitation
        if mode == "always_on":
            grip = np.ones(6)
        elif mode == "contact_gated":
            grip = previous_contact
        elif mode == "ltm_rate_gated":
            grip = grip_trace[:, tick]
        else:
            grip = np.zeros(6)
        if len(body.adhesion_actuators):
            d.ctrl[body.adhesion_actuators] = grip
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        for _ in range(nstep):
            mujoco.mj_step(m, d)
        state = body.telemetry()
        previous_contact = np.asarray(state["contact"], dtype=float)
        p = state["position"]
        rows.append(dict(t=(tick+1)*.01, x=p[0], y=p[1], z=p[2],
                         upright=state["upright"],
                         nonfoot_load_fraction=state["nonfoot_load_fraction"]))
        feet.append(state["feet"])
        contacts.append(state["contact"])
        applied.append(grip.copy())
    result = score(rows, feet, contacts)
    result.update(physics_warnings=int(d.warning.number.sum()),
                  zero_native_joint_position_gain=bool(np.all(m.actuator_gainprm[:42] == 0)),
                  no_applied_force=bool(np.all(d.qfrc_applied == 0) and np.all(d.xfrc_applied == 0)),
                  adhesion_mean_control=float(np.mean(applied)),
                  adhesion_nonzero_fraction=float(np.mean(np.asarray(applied) > 0)),
                  peak_hill_excitation=float(np.max(1-np.exp(-np.maximum(pooled,0)/100))))
    return result, dict(position=np.asarray([[r["x"],r["y"],r["z"]] for r in rows]),
                        feet=np.asarray(feet), contact=np.asarray(contacts),
                        grip_control=np.asarray(applied))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    neurons = compatible_neurons(pd.read_csv(NEURONS))
    plain = HomologousHillBody(neurons, stiffness=32, experimental_adhesion=False)
    adhesion = HomologousHillBody(neurons, stiffness=32, experimental_adhesion=True)
    assert len(plain.hill_actuators) == len(adhesion.hill_actuators) == 78
    assert len(adhesion.adhesion_actuators) == 6
    results = []
    traces = {}
    for row in ROWS:
        selected, source_rates = original_rates(row)
        pooled = pool_trace(plain, selected, source_rates, "mean")
        grip, counts = ltm_commands(neurons, selected, source_rates)
        assert pooled.shape == (78, 200) and grip.shape == (6, 200)
        for mode in MODES:
            body = plain if mode == "absent" else adhesion
            result, arrays = run_condition(body, pooled, grip, mode)
            result.update(source_row=row, grip_mode=mode, ltm_cell_count_by_leg=counts,
                          ltm_control_hypothesis=(mode == "ltm_rate_gated"),
                          outside_neural_grip_control=(mode in ("always_on", "contact_gated")))
            results.append(result)
            traces[(row, mode)] = arrays
            print(row, mode, result["forward_mm"], result["steps_per_leg"], result["walking_pass"], flush=True)
        # Merely adding inactive body-target adhesion actuators changes the
        # compiled solver's numerical path during passive settling. Compare
        # active conditions to the released case within that same compiled
        # model; retain absent as the original-body historical baseline.
        released_repeat, repeat_arrays = run_condition(adhesion, pooled, grip, "released")
        assert np.array_equal(traces[(row,"released")]["position"], repeat_arrays["position"]), \
            "Same compiled model must replay deterministically"
        np.savez_compressed(OUT / f"row{row}_traces.npz", **{
            f"{mode}_{key}": arr for mode in MODES for key, arr in traces[(row,mode)].items()
        }, ltm_source_control=grip, hill_pooled_source_rates=pooled)
    checks = {
        "all_original_source_traces": len(results) == len(ROWS)*len(MODES),
        "all_physics_finite": all(r["physics_warnings"] == 0 for r in results),
        "all_native_position_gains_zero": all(r["zero_native_joint_position_gain"] for r in results),
        "all_applied_body_forces_zero": all(r["no_applied_force"] for r in results),
        "released_differs_from_original_compiled_baseline": any(not np.array_equal(traces[(row,"absent")]["position"], traces[(row,"released")]["position"]) for row in ROWS),
        "adhesion_comparisons_use_same_compiled_model": all(r["grip_mode"] != "absent" or r["adhesion_mean_control"] == 0 for r in results),
        "source_ltm_trace_nonzero": any(np.max(traces[(row,"ltm_rate_gated")]["grip_control"]) > 0 for row in ROWS),
    }
    report = dict(date="2026-09-24", scope="Original published full-MANC DNg100 source-rate open-loop replay into 78 experimental Hill muscles; optional abstract tarsal adhesion sensitivity",
                  source="Pugliese 2026 Zenodo 22260924; locally saved original-author 1ms 32-row blocks",
                  neuroMechFly_adhesion="FlyGym v2 add_leg_adhesion default gain 40, control 0..1; not a measured LTM/claw reconstruction",
                  rows=list(ROWS), conditions=list(MODES), results=results, checks=checks,
                  passed=all(checks.values()), goal_complete=False)
    (OUT/"report.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, row in zip(axes, ROWS):
        for mode in MODES:
            ax.plot(np.arange(200)*.01+.01, traces[(row,mode)]["position"][:,0], label=mode)
        ax.set_ylabel(f"row {row} forward x (mm)")
        ax.grid(alpha=.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("source replay time (s)")
    fig.tight_layout()
    fig.savefig(OUT/"forward_paths.png",dpi=160)
    plt.close(fig)
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
