"""Re-check every E1-rhythm count quoted in the manuscript with both detectors."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_128 import rhythm_peak, rhythm_spectral  # noqa: E402  (re-runs module-level analysis once)
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results"
E1 = [10072, 10498, 10558, 10559, 10690, 10707]
out = {}
def counts(traces_10ms=None, traces_1ms=None):
    pk = sp = None
    if traces_10ms is not None:
        pk = int(sum(rhythm_peak(x[50:]) for x in traces_10ms))
        sp = int(sum(rhythm_spectral(x[50:], dt=.01)[0] for x in traces_10ms))
    if traces_1ms is not None:
        sp1 = int(sum(rhythm_spectral(x[500:])[0] for x in traces_1ms))
        pk1 = int(sum(rhythm_peak(x[500::10]) for x in traces_1ms))
        return {"peak_from_1ms": pk1, "spectral_1ms": sp1, **({"peak_10ms": pk, "spectral_10ms": sp} if pk is not None else {})}
    return {"peak_10ms": pk, "spectral_10ms": sp}
z = np.load(R / "fullmanc-premotor-source-solver-20260924/selected_neural_traces.npz")
ids = list(z["selected_body_ids"]); e = [ids.index(b) for b in E1]
out["premotor_injection"] = {str(c): counts(z["rates_10ms"][i][e]) for i, c in enumerate(z["case_names"])}
z = np.load(R / "fullmanc-premotor-source-solver-20260924/fixed_step_dose_screen.npz")
sel = list(z["selected_indices"]); zz = np.load(R / "fullmanc-premotor-source-solver-20260924/selected_neural_traces.npz")
# map body ids via premotor selected ids (same 741 selection)
assert list(zz["selected_indices"]) == sel
out["euler_0p5ms_dose_screen"] = {f"dose_{float(d):g}": counts(z["rates_10ms"][i][e]) for i, d in enumerate(z["doses"])}
for name in ["unclamped_source", "no_sensory", "FeCO_plus_load", "FeCO_plus_load_outgoing_cut", "legacy_after_first_microstep", "aligned_before_first_microstep"]:
    zz = np.load(R / f"manc-adaptive-physical-feedback-20260924/{name}.npz")
    x = zz["E1_rates_1ms"]
    # E1_rates_1ms has 2000 samples starting at t=1 ms
    out[f"closed_loop_{name}"] = {"spectral_1ms": int(sum(rhythm_spectral(v[499:])[0] for v in x)),
                                  "peak_from_1ms": int(sum(rhythm_peak(v[499::10]) for v in x))}
zc = np.load(R / "full-manc-descending-codrive-screen-20260924/three_drives_all_32_parameter_rows.npz")
cid = list(zc["selected_body_ids"]); ce = [cid.index(b) for b in E1]
arch = json.load(open(ROOT / "paper/verification/robustness_128.json"))["rows"]
cmp = []
for row in range(32):
    tr = zc["DNg100_author_rates_10ms"][:, ce, row].T  # 200 samples from 10 ms
    a = arch[96 + row]
    cmp.append({"row": row, "euler1ms_peak": int(sum(rhythm_peak(v[49:]) for v in tr)), "euler1ms_spectral": int(sum(rhythm_spectral(v[49:], dt=.01)[0] for v in tr)),
                "archived_peak": a["E1_peak10ms"], "archived_spectral": a["E1_spectral"]})
out["codrive_DNg100_vs_archived_run33195882"] = cmp
json.dump(out, open(ROOT / "paper/verification/e1_detector_recheck.json", "w"), indent=1)
for k, v in out.items():
    if k.startswith("codrive"):
        agree = sum(c["euler1ms_spectral"] == c["archived_spectral"] for c in v)
        print(k, "rows where Euler-1ms spectral == archived spectral:", agree, "/32;", "row13:", v[13])
        print("   mismatches:", [(c["row"], c["euler1ms_spectral"], c["archived_spectral"]) for c in v if c["euler1ms_spectral"] != c["archived_spectral"]])
    else:
        print(k, v)
