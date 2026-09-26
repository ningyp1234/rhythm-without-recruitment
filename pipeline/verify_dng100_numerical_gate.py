"""Cross-solver audit of the one apparent DNg100 antagonist-gate hit."""
from __future__ import annotations

import json
import h5py
import numpy as np
import pandas as pd

from screen_official_dng100_t1_grid import OUT, TABLE, evaluate, make_pools
from validate_official_dng100_t1_candidate import BASE, DATA, GLU, INH


def main():
    n = pd.read_csv(TABLE).fillna("")
    pools = make_pools(n)
    left = int(np.flatnonzero(n.bodyId.eq(10707).to_numpy())[0])
    right = int(np.flatnonzero(n.bodyId.eq(10690).to_numpy())[0])
    diffrax = np.load(OUT / "source_diffrax_rows10_78.npz")
    rk = np.load(OUT / "candidate_adaptive_rows10_78.npz")
    convergence = np.load(OUT / "candidate_numerical_convergence.npz")
    indexes = diffrax["parameter_rows"]
    full_lookup = {j: j for j in range(len(n))}
    rk_lookup = {int(v): i for i, v in enumerate(rk["selected_indices"])}
    convergence_lookup = {int(v): i for i, v in enumerate(convergence["selected_indices"])}
    evaluations = []
    for col, row in enumerate(indexes):
        for method, trace, lookup in [
            ("source_Diffrax_Dopri5", diffrax["rates_10ms"], full_lookup),
            ("independent_scipy_RK45", rk["rates_10ms"], rk_lookup),
            *[(f"fixed_Euler_dt_{dt:g}", convergence[f"dt_{dt:g}_rates_10ms"], convergence_lookup)
              for dt in (.001, .0005, .00025, .000125)],
        ]:
            item = evaluate(trace, lookup, left, right, pools, col)
            evaluations.append({"parameter_row": int(row), "method": method,
                                "left_E1_sustained": item["E1_left_sustained"],
                                "left_antagonist_gate": item["left_joint_gate"],
                                "left_tibia_correlation": item["sides"]["LHS"]["tibia_correlation"]})
    with h5py.File(BASE / "neuron_params.h5") as baseline, h5py.File(DATA) as candidate:
        original, archived = baseline["W"][:], candidate["W"][:]
        glutamate = n.predictedNt.eq("glutamate").to_numpy()
        expected = original.copy()
        expected[glutamate] *= np.float32(GLU / INH)
        weight_exact = bool(np.array_equal(expected, archived))
    by = {(x["parameter_row"], x["method"]): x for x in evaluations}
    checks = {
        "source_archived_W_matches_source_glutamate_ratio_exactly": weight_exact,
        "source_Diffrax_all_finite": bool(np.isfinite(diffrax["rates_10ms"]).all()),
        "one_ms_Euler_false_positive_both_rows": all(by[(int(row), "fixed_Euler_dt_0.001")]["left_antagonist_gate"] for row in indexes),
        "all_three_refined_Euler_steps_fail_both_rows": all(not by[(int(row), f"fixed_Euler_dt_{dt:g}")]["left_antagonist_gate"]
                                                   for row in indexes for dt in (.0005, .00025, .000125)),
        "independent_RK45_fails_both_rows": all(not by[(int(row), "independent_scipy_RK45")]["left_antagonist_gate"] for row in indexes),
        "source_Diffrax_fails_both_rows": all(not by[(int(row), "source_Diffrax_Dopri5")]["left_antagonist_gate"] for row in indexes),
    }
    report = {"source": "https://zenodo.org/records/22260924",
              "source_code": "https://github.com/smpuglie/Pugliese_2026",
              "scope": "Fixed-step candidate was screened on author MANC T1 graph; exact author checkpoint re-solved with source Diffrax Dopri5, SciPy RK45, and three finer fixed steps. Numerical gate is not stable.",
              "parameter_rows": [int(v) for v in indexes], "evaluations": evaluations,
              "checks": checks, "passed": all(checks.values()),
              "numerically_validated_antagonist_gate": False,
              "body_walking_achieved": False}
    (OUT / "numerical_gate_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "passed": report["passed"]}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
