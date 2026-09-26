"""Source-graph screen of experimentally motivated DNg100/DNg97 input modes.

No synaptic weights, cell parameters, sensory gains or body settings are fit.
The same 380 model-current units per stimulated DN are a matched artificial
comparison, not a measured neural current or autonomous brain command.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from cpg_context_diagnostic import rhythm
from evaluate_official_running_dynamic_replay import ROOT
from source_fullmanc_exact_ensemble import NEURONS, WEIGHTS, load_parameters

OUT = ROOT / "results/full-manc-descending-codrive-screen-20260924"
LEGS = (("T1", "LHS", "T1_left"), ("T1", "RHS", "T1_right"),
        ("T2", "LHS", "T2_left"), ("T2", "RHS", "T2_right"),
        ("T3", "LHS", "T3_left"), ("T3", "RHS", "T3_right"))


def ids(n, type_name):
    return np.flatnonzero(n.type.eq(type_name).to_numpy())


def motor_groups(n):
    groups = {}
    for segment, side, name in LEGS:
        base = n["class"].eq("motor neuron") & n.somaNeuromere.eq(segment) & n.somaSide.eq(side)
        groups[name] = {
            "flex": np.flatnonzero((base & n["motor module"].eq("tibia flex")).to_numpy()),
            "extend": np.flatnonzero((base & n["motor module"].eq("tibia extend")).to_numpy()),
        }
        if not groups[name]["flex"].size or not groups[name]["extend"].size:
            raise ValueError("missing tibia antagonist pool: " + name)
    return groups


def summarize_group(trace, members, selected_lookup):
    values = trace[50:, [selected_lookup[int(v)] for v in members]]
    return {"neurons": len(members), "ever_over_1_hz": int((values.max(axis=0) > 1).sum()),
        "maximum_hz": float(values.max()),
        "pool_range_hz": float(np.ptp(values.mean(axis=1))),
        "pool_mean_hz": float(values.mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = pd.read_csv(NEURONS).fillna("")
    p = load_parameters()
    tau, gain, threshold, cap, author_input = (p[key].astype(np.float32)
        for key in ("tau", "gain", "threshold", "cap", "input"))
    pre_post = sparse.load_npz(WEIGHTS).tocsr()
    matrix = (pre_post.T.tocsr() * np.float32(.03)).astype(np.float32)
    g100, g97 = ids(n, "DNg100"), ids(n, "DNg97")
    e1, inhibitor = ids(n, "IN17A001"), ids(n, "IN12B003")
    assert (len(g100), len(g97), len(e1), len(inhibitor), tau.shape[1]) == (2, 2, 6, 6, 32)
    assert all(n.instance.iloc[g100[i]].endswith(("_L", "_R")[i]) and
               n.instance.iloc[g97[i]].endswith(("_L", "_R")[i]) for i in range(2))
    assert np.array_equal(author_input[g100], np.full((2, 32), 380, np.float32))
    assert np.count_nonzero(author_input) == 2 * 32
    groups = motor_groups(n)
    selected = np.unique(np.concatenate((g100, g97, e1, inhibitor,
        *[part for group in groups.values() for part in group.values()])))
    lookup = {int(source): column for column, source in enumerate(selected)}
    no_drive = np.zeros_like(author_input)
    g97_only = no_drive.copy(); g97_only[g97] = 380
    co = author_input.copy(); co[g97] = 380
    source_target = np.maximum(cap[g100] * np.tanh((gain[g100] / cap[g100]) *
        (author_input[g100] - threshold[g100])), 0)
    if not np.all(source_target < .99 * cap[g97]):
        raise ValueError("DNg100 isolated target exceeds DNg97 cap")
    matched_g97_current = (threshold[g97] + cap[g97] / gain[g97] *
        np.arctanh(source_target / cap[g97]))
    matched_only = no_drive.copy(); matched_only[g97] = matched_g97_current
    matched_co = author_input.copy(); matched_co[g97] = matched_g97_current
    drives = {"DNg100_author": author_input,
              "DNg97_equal_current": g97_only,
              "DNg100_plus_DNg97_equal_current": co,
              "DNg97_rate_matched": matched_only,
              "DNg100_plus_DNg97_rate_matched": matched_co}
    arrays, rows = {}, []
    for condition, drive in drives.items():
        rates = np.zeros_like(tau, np.float32)
        trace = np.empty((200, len(selected), 32), np.float32)
        for step in range(2000):
            current = drive if step >= 20 else 0.0
            total = current + matrix @ rates
            activation = np.maximum(cap * np.tanh((gain / cap) *
                (total - threshold)), 0.0)
            rates += np.float32(.001) * (activation - rates) / tau
            if (step + 1) % 10 == 0:
                trace[(step + 1) // 10 - 1] = rates[selected]
        arrays[condition + "_rates_10ms"] = trace
        for row in range(32):
            counts = [int(rhythm(trace[50:, lookup[int(index)], row], dt=.01)[
                "diagnostic_sustained_rhythm"]) for index in e1]
            legs = {}
            for leg, parts in groups.items():
                flex = summarize_group(trace[:, :, row], parts["flex"], lookup)
                extend = summarize_group(trace[:, :, row], parts["extend"], lookup)
                legs[leg] = {"flex": flex, "extend": extend,
                    "both_pools_recruited_variable": bool(
                        flex["ever_over_1_hz"] >= 1 and extend["ever_over_1_hz"] >= 1 and
                        flex["pool_range_hz"] >= .5 and extend["pool_range_hz"] >= .5)}
            item = {"condition": condition, "parameter_row": row,
                "six_E1_sustained_count": sum(counts),
                "E1_sustained_by_leg": counts,
                "tibia_motor_by_leg": legs,
                "legs_with_both_antagonist_pools": sum(v[
                    "both_pools_recruited_variable"] for v in legs.values()),
                "six_E1_plus_left_front_tibia_gate": bool(sum(counts) == 6 and
                    legs["T1_left"]["both_pools_recruited_variable"]),
                "six_E1_plus_all_six_tibia_gate": bool(sum(counts) == 6 and all(
                    v["both_pools_recruited_variable"] for v in legs.values())),
                "IN12B003_max_hz": [float(trace[50:, lookup[int(index)], row].max())
                    for index in inhibitor],
                "DNg100_max_hz": [float(trace[50:, lookup[int(index)], row].max())
                    for index in g100],
                "DNg97_max_hz": [float(trace[50:, lookup[int(index)], row].max())
                    for index in g97]}
            rows.append(item)
        print(json.dumps({"condition": condition,
            "joint_gate_rows": [r["parameter_row"] for r in rows if
                r["condition"] == condition and r["six_E1_plus_left_front_tibia_gate"]],
            "all_six_leg_gate_rows": [r["parameter_row"] for r in rows if
                r["condition"] == condition and r["six_E1_plus_all_six_tibia_gate"]]}), flush=True)
    arrays["selected_neuron_indices"] = selected
    arrays["selected_body_ids"] = n.bodyId.iloc[selected].astype(int).to_numpy()
    arrays["rate_matched_DNg97_current"] = matched_g97_current
    np.savez_compressed(OUT / "three_drives_all_32_parameter_rows.npz", **arrays)
    direct = {"DNg100_to_T1_left_tibia_flex_synapse_count": int(pre_post[g100, :][:, groups[
        "T1_left"]["flex"]].sum()),
        "DNg97_to_T1_left_tibia_flex_synapse_count": int(pre_post[g97, :][:, groups[
        "T1_left"]["flex"]].sum())}
    baseline = json.loads((ROOT / "results/full-manc-motor-recruitment-audit-20260924/official_32row_tibia_screen.json").read_text())
    author_rows = [r for r in rows if r["condition"] == "DNg100_author"]
    checks = {
        "author_source_input_only_two_DNg100_cells": True,
        "all_32_author_rows_reproduced_E1": all(a["six_E1_sustained_count"] == b[
            "sustained_rhythmic_E1_count"] for a, b in zip(author_rows, baseline["rows"])),
        "all_32_author_rows_reproduced_left_front_tibia": all(a[
            "tibia_motor_by_leg"]["T1_left"]["flex"]["ever_over_1_hz"] == b[
            "left_front_tibia_flexor_ever_over_1_hz"] and a[
            "tibia_motor_by_leg"]["T1_left"]["extend"]["ever_over_1_hz"] == b[
            "left_front_tibia_extensor_ever_over_1_hz"] for a, b in zip(author_rows, baseline["rows"])),
        "all_five_input_conditions_run_on_same_32_parameter_columns": len(rows) == 160,
        "all_rate_matched_currents_below_380": bool(np.all(matched_g97_current < 380)),
    }
    report = {"date": "2026-09-24",
        "scope": "Evidence-motivated DNg100, DNg97 or combined tonic input in unchanged source full MANC model, all 32 author parameter columns; equal-current and isolated-rate-matched artificial controls; neural screen only",
        "source_paper": "https://www.biorxiv.org/content/10.64898/2026.04.29.721658v2.full",
        "input_model_current_per_stimulated_cell": 380,
        "rate_match_method": "For each author parameter column and anatomical side, invert DNg97 single-neuron source nonlinearity to match the DNg100 isolated 380-current rate; no network/body outcomes used to choose current.",
        "rate_matched_DNg97_current_row13": matched_g97_current[:, 13].astype(float).tolist(),
        "input_current_is_physically_calibrated": False,
        "graph_neurons": len(n), "directed_pairs": int(pre_post.nnz),
        "source_neuron_body_ids": {"DNg100": n.bodyId.iloc[g100].astype(int).tolist(),
            "DNg97": n.bodyId.iloc[g97].astype(int).tolist()},
        "structural_prior": direct, "rows": rows, "checks": checks,
        "passed": all(checks.values()),
        "brain_autonomous_walking_achieved": False,
        "goal_complete": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": checks, "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
