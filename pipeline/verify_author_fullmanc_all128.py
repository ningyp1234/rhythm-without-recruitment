"""Independently verify the four-run original-data and physical-gate audit."""
from __future__ import annotations

import hashlib
import json
import zlib

import h5py
import numpy as np
import pandas as pd

from source_calibration import ROOT


RUNS = (30662613, 33195857, 33195876, 33195882)
DATA = ROOT / "data/pugliese-zenodo-22260924/full-manc-all-runs"
OLD_DATA = ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882"
OUT = ROOT / "results/author-fullmanc-128-walking-20260924"
OLD = ROOT / "results/row21-body-diagnostic-20260924"
FIGURE_COUNTS = ROOT / "external/Pugliese_2026/figures/DNg100_Stim_fullManc-hyak-30662613-33195857-33195876-33195882/nMnsActive_allLegs.csv"


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest_and_crc(path):
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
    return digest.hexdigest(), f"{crc:08x}"


def main() -> None:
    manifest = load(DATA / "manifest.json")
    old_manifest = load(OLD_DATA / "author_trace_manifest.json")
    neural = load(OUT / "author_fullmanc_128_neural_screen.json")
    mapping = load(OUT / "author_fullmanc_128_motor_mapping.json")
    physical = load(OUT / "author_fullmanc_128_body_screen.json")
    old_physical = load(OLD / "author_fullmanc_32_body_screen.json")
    published = pd.read_csv(FIGURE_COUNTS)
    checks = {}
    checks["published_run_order_and_128_rows"] = (manifest["figure_run_order"] == list(RUNS)
        and [x["run_id"] for x in neural["runs"]] == list(RUNS)
        and len(neural["rows"]) == len(mapping["rows"]) == len(physical["rows"]) == 128)
    checks["four_original_trajectory_shapes"] = all(x["source_tensor_shape"] == [1,32,23532,2001] for x in neural["runs"])
    checks["all_15_new_zip_members_crc_verified_by_fetch"] = (
        len(manifest["files"]) == 15 and all(x["zip_crc_verified"] for x in manifest["files"]))
    source_records = [next(x for x in manifest["files"] if x["run_id"] == run_id and x["member"].endswith("_Rs.npz"))
                      for run_id in RUNS[:-1]]
    checks["original_source_hashes_and_CRC_reverified"] = True
    for record in source_records:
        path = ROOT / record["relative_file"]
        hash_now, crc_now = digest_and_crc(path)
        checks["original_source_hashes_and_CRC_reverified"] &= (
            path.stat().st_size == record["size"] and hash_now == record["sha256"] and crc_now == record["crc32"])
    old_path = OLD_DATA / "DNg100_Stim_fullManc_Rs.npz"
    hash_now, crc_now = digest_and_crc(old_path)
    checks["old_original_source_hash_and_CRC_reverified"] = (hash_now == old_manifest["sha256"] and crc_now == old_manifest["crc32"])
    checks["four_two_DNg100_380_inputs"] = True
    for run_id in RUNS:
        path = (OLD_DATA if run_id == 33195882 else DATA / str(run_id)) / "neuron_params.h5"
        with h5py.File(path) as h:
            c = h["input_currents"][0]
            checks["four_two_DNg100_380_inputs"] &= bool(c.shape == (32,23532)
                and np.all(np.count_nonzero(c, axis=1) == 2)
                and np.all(c[:,59] == 380) and np.all(c[:,282] == 380))
    checks["figure4_motor_counts_exact_all_four"] = all(x["published_figure4_active_motor_counts_match"] for x in neural["runs"])
    checks["neural_six_E1_and_six_tibia_rows_as_recorded"] = (
        neural["six_E1_rows"] == [36,38,39,45,72] and neural["six_tibia_rows"] == [2,117]
        and neural["joint_gate_rows"] == [])
    checks["exact_author_row117_mapping_corrected"] = (mapping["author_global_row117_exact"]["active_annotated_leg_motor_neurons"] == 189
        and mapping["author_global_row117_exact"]["mapped_active_annotated_leg_motor_neurons"] == 154
        and mapping["author_global_row117_exact"]["unmapped_active_annotated_leg_motor_neurons"] == 35)
    checks["source_sparse_samples_equal_saved_selected_traces"] = True
    for block, run_id in enumerate(RUNS):
        source = (OLD_DATA if run_id == 33195882 else DATA / str(run_id)) / "DNg100_Stim_fullManc_Rs.npz"
        selected_file = (OLD / "author_fullmanc_32_selected_neurons.npz" if run_id == 33195882
                         else OUT / f"run{run_id}_selected_738_neurons.npz")
        with np.load(source) as sparse, np.load(selected_file) as selected_npz:
            coords, values = sparse["coords"], sparse["data"]
            selected = selected_npz["selected_indices"]
            trace = selected_npz["rates_1ms"]
        lookup = {int(v): i for i, v in enumerate(selected)}
        mask = np.flatnonzero((coords[0] == 0) & np.isin(coords[2], selected))
        sample = mask[np.linspace(0, len(mask)-1, 100, dtype=int)]
        checks["source_sparse_samples_equal_saved_selected_traces"] &= all(
            trace[int(coords[1,i]),lookup[int(coords[2,i])],int(coords[3,i])] == values[i] for i in sample)
        del coords, values, trace
    checks["all_256_physics_trials_no_warnings"] = all(r[b]["physics_warnings"] == 0 for r in physical["rows"] for b in ("hill","torque"))
    checks["all_256_physics_trials_below_5mm"] = all(r[b]["forward_mm"] < 5 for r in physical["rows"] for b in ("hill","torque"))
    checks["all_256_physics_trials_fail_full_gait"] = (physical["hill_walking_pass_rows"] == []
        and physical["torque_walking_pass_rows"] == [])
    checks["old_32_rows_replayed_identically"] = all(
        abs(physical["rows"][96+i][body]["forward_mm"] - old_physical["rows"][i][body]["forward_mm"]) < 1e-6
        for i in range(32) for body in ("hill", "torque"))
    checks["no_autonomous_behavior_claim"] = neural["autonomous_walking_achieved"] is False and physical["autonomous_walking_achieved"] is False
    report = {"source": "https://zenodo.org/records/22260924",
              "scope": "Four original ZIP member hashes/CRC, published Figure 4 cross-check, source sparse samples, 128 neural and 256 body outcomes",
              "checks": checks, "checks_passed": sum(checks.values()),
              "checks_total": len(checks), "passed": all(checks.values()), "goal_complete": False}
    (OUT / "independent_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["checks_passed"], "total": report["checks_total"],
                      "failed": [k for k,v in checks.items() if not v]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
