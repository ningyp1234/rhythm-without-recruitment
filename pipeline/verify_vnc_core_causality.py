"""Independent checks on saved neural-intervention and physical-contact traces."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from evaluate_official_running_dynamic_replay import ROOT

OUT = ROOT / "results/vnc-core-causal-audit-20260924"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sustained_contact_steps(contact, leg, minimum_frames=7):
    trace = contact[:, leg]
    boundaries = np.r_[0, np.flatnonzero(trace[1:] != trace[:-1]) + 1, len(trace)]
    states = [bool(trace[a]) for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a >= minimum_frames]
    states = [value for index, value in enumerate(states) if index == 0 or value != states[index - 1]]
    return sum(states[index - 2:index + 1] == [True, False, True] for index in range(2, len(states)))


def main():
    report = json.loads((OUT / "report.json").read_text())
    replay = json.loads((OUT / "causal-walking-replay.json").read_text())
    page = (ROOT / "vnc_core_causal_replay.html").read_text()
    trace_path = OUT / "rollouts.npz"
    with np.load(trace_path) as trace:
        intact_contact = trace["intact_foot_ground_contact"]
        frozen_contact = trace["hold_neural_state_foot_ground_contact"]
        intact_steps = [sustained_contact_steps(intact_contact, leg) for leg in range(6)]
        frozen_steps = [sustained_contact_steps(frozen_contact, leg) for leg in range(6)]
        checks = {
            "trace_hash_matches_report": digest(trace_path) == report["artifacts"]["traces_sha256"],
            "intact_neural_phase_changes": len(np.unique(trace["intact_vnc_phase"])) > 20,
            "frozen_neural_phase_constant": len(np.unique(trace["hold_neural_state_vnc_phase"])) == 1,
            "saved_intact_contact_steps_match_report": intact_steps == report["records"]["intact"]["sustained_contact_cycles_8_75ms"],
            "saved_frozen_contact_steps_match_report": frozen_steps == report["records"]["hold_neural_state"]["sustained_contact_cycles_8_75ms"],
            "intact_has_all_six_sustained_leg_cycles": min(intact_steps) >= 2,
            "frozen_lacks_three_leg_cycles": frozen_steps[:3] == [0, 0, 0],
            "intact_and_frozen_both_move_far": report["records"]["intact"]["distance_2s_mm"] >= 5 and report["records"]["hold_neural_state"]["distance_2s_mm"] >= 5,
            "simple_gate_false_positive_detected": report["records"]["hold_neural_state"]["walking_gate"] and not report["records"]["hold_neural_state"]["strict_walking_gate"],
            "intact_passes_source_calibrated_standard_gate": report["records"]["intact"]["strict_walking_gate"],
            "frozen_slip_above_source_relative_limit": report["records"]["hold_neural_state"]["median_planted_foot_slip_mm_s"] > report["source_relative_slip_limit_mm_s"],
            "source_calibration_has_six_steps": min(report["official_source_gait_calibration"]["sustained_contact_cycles_8_75ms"]) >= 2,
            "heldout_intact_two_of_three_strict": sum(item["intact"]["strict_walking_gate"] for item in report["heldout_test_pairs"]) == 2,
            "heldout_frozen_zero_of_three_strict": sum(item["hold_neural_state"]["strict_walking_gate"] for item in report["heldout_test_pairs"]) == 0,
            "global_goal_stays_open": report["goal_complete"] is False,
            "replay_uses_exact_audited_trace": replay["source_sha256"] == digest(trace_path),
            "replay_intact_final_contact_steps_match": replay["episodes"][0]["cumulative_cycles"][-1] == intact_steps,
            "replay_frozen_final_contact_steps_match": replay["episodes"][1]["cumulative_cycles"][-1] == frozen_steps,
            "browser_page_labels_real_contact_steps_and_limit": "六腿接触步数" in page and "全脑自主行走未完成" in page,
        }
    output = {"passed": all(checks.values()), "checks": checks, "total": len(checks), "successful": sum(checks.values())}
    (OUT / "independent_validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
