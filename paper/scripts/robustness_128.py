"""Robustness of the rhythm/recruitment dissociation across the 128 author-saved full-MANC runs.

Uses only the archived author rates (1 ms) already extracted to NPZ; no simulation.
(1) reproduces the original 10-ms peak-based E1 classification;
(2) re-classifies E1 with an independent 1-ms spectral detector (no minimum peak spacing);
(3) sweeps the tibia-recruitment thresholds;
(4) tests the graded association (any E1 rhythm vs any antagonistic tibia output) with Fisher's exact test.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "paper" / "verification"
OUT.mkdir(parents=True, exist_ok=True)
FILES = [(30662613, R / "author-fullmanc-128-walking-20260924/run30662613_selected_738_neurons.npz"),
         (33195857, R / "author-fullmanc-128-walking-20260924/run33195857_selected_738_neurons.npz"),
         (33195876, R / "author-fullmanc-128-walking-20260924/run33195876_selected_738_neurons.npz"),
         (33195882, R / "row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz")]
LEGS = (("lf", "T1", "L"), ("lm", "T2", "L"), ("lh", "T3", "L"), ("rf", "T1", "R"), ("rm", "T2", "R"), ("rh", "T3", "R"))

n = pd.read_csv(ROOT / "data/cpg/manc-neurons-20251006.csv.gz").fillna("")
n["side"] = n.somaSide.str[:1]
e1_idx = np.flatnonzero(n.type.eq("IN17A001").to_numpy())
mn_idx = np.flatnonzero(n["class"].eq("motor neuron").to_numpy())
assert len(e1_idx) == 6 and len(mn_idx) == 732

def rhythm_peak(x, dt=.01):  # identical to cpg_context_diagnostic.rhythm
    x = np.asarray(x, float); amp = float(np.ptp(x))
    peaks, _ = find_peaks(x, prominence=max(.5, .1 * amp), distance=10)
    per = np.diff(peaks) * dt
    cv = float(np.std(per) / np.mean(per)) if len(per) else None
    first, last = float(np.ptp(x[:len(x)//3])), float(np.ptp(x[-len(x)//3:]))
    return bool(amp >= 2 and len(peaks) >= 5 and cv is not None and cv <= .2 and last >= .5 * max(first, 1e-12))

def rhythm_spectral(x, conc_min=0.3, fmin=2., fmax=50., dt=.001):
    x = np.asarray(x, float); amp = float(np.ptp(x))
    if amp < 2: return False, None, 0.0
    t = np.arange(len(x)); y = x - np.polyval(np.polyfit(t, x, 1), t)
    w = np.hanning(len(y)); P = np.abs(np.fft.rfft(y * w))**2; f = np.fft.rfftfreq(len(y), dt)
    band = (f >= 1) & (f <= 100); fb, Pb = f[band], P[band]
    sel = (fb >= fmin) & (fb <= fmax)
    fpk = float(fb[sel][np.argmax(Pb[sel])])
    conc = float(Pb[np.abs(fb - fpk) <= max(1.5, .15 * fpk)].sum() / Pb.sum())
    k = len(x) // 3
    sustained = np.ptp(x[-k:]) >= .5 * max(np.ptp(x[:k]), 1e-12)
    return bool(conc >= conc_min and sustained), fpk, conc

rows = []
for run_i, (run_id, path) in enumerate(FILES):
    z = np.load(path)
    ids = z["selected_indices"]; lk = {int(v): i for i, v in enumerate(ids)}
    rates = z["rates_1ms"]
    e1 = [lk[int(i)] for i in e1_idx]
    pools = {}
    for leg, neu, side in LEGS:
        for mod in ("tibia flex", "tibia extend"):
            idx = np.flatnonzero((n.somaNeuromere.eq(neu) & n.side.eq(side) & n["motor module"].eq(mod)).to_numpy())
            pools[(leg, mod)] = [lk[int(i)] for i in idx]
    mods = n.loc[mn_idx, "motor module"].to_numpy()
    mn_local = np.array([lk[int(i)] for i in mn_idx])
    for row in range(32):
        tr = rates[row]
        post = tr[:, 500:]
        rec = {"global_row": run_i * 32 + row, "run_id": run_id, "parameter_row": row}
        rec["E1_peak10ms"] = int(sum(rhythm_peak(tr[i, 500::10]) for i in e1))
        spec = [rhythm_spectral(tr[i, 500:]) for i in e1]
        rec["E1_spectral"] = int(sum(s[0] for s in spec))
        rec["E1_spectral_freq_hz"] = [s[1] for s in spec]
        rec["E1_spectral_conc"] = [round(s[2], 3) for s in spec]
        for c in (0.2, 0.5):
            rec[f"E1_spectral_conc{c}"] = int(sum(rhythm_spectral(tr[i, 500:], conc_min=c)[0] for i in e1))
        rec["E1_mean_hz"] = float(post[e1].mean()); rec["E1_ptp_hz"] = float(np.ptp(post[e1], axis=1).mean())
        rec["MN_active_over_1hz"] = int((post[mn_local].max(axis=1) > 1).sum())
        act = post[mn_local].max(axis=1) > 1
        rec["active_modules"] = pd.Series(mods[act]).value_counts().to_dict()
        tib_idx = sum((pools[(l, m)] for l, _, _ in LEGS for m in ("tibia flex", "tibia extend")), [])
        rec["tibia_MN_active_over_1hz"] = int((tr[tib_idx, 500::10].max(axis=1) > 1).sum())
        for thr in (0.1, 1., 5.):
            for rng in (0.1, 0.5, 2.):
                cnt = 0
                for leg, _, _ in LEGS:
                    ok = True
                    for mod in ("tibia flex", "tibia extend"):
                        v = tr[pools[(leg, mod)], 500::10]
                        ok &= bool((v.max(axis=1) > thr).sum() >= 1 and np.ptp(v.mean(axis=0)) >= rng)
                    cnt += ok
                rec[f"tib_{thr}_{rng}"] = cnt
        rows.append(rec)

orig = json.load(open(R / "author-fullmanc-128-walking-20260924/author_fullmanc_128_neural_screen.json"))["rows"]
repro_e1 = all(o["sustained_E1_count"] == r["E1_peak10ms"] for o, r in zip(orig, rows))
repro_tib = all(o["six_leg_tibia_both_count"] == r["tib_1.0_0.5"] for o, r in zip(orig, rows))

def assoc(e1key, tibkey):
    e = np.array([r[e1key] for r in rows]); t = np.array([r[tibkey] for r in rows])
    table = [[int(((e >= 1) & (t >= 1)).sum()), int(((e >= 1) & (t == 0)).sum())],
             [int(((e == 0) & (t >= 1)).sum()), int(((e == 0) & (t == 0)).sum())]]
    _, p = fisher_exact(table)
    rho, prho = spearmanr(e, t)
    return {"six_E1": int((e == 6).sum()), "six_tibia": int((t == 6).sum()), "joint_six_six": int(((e == 6) & (t == 6)).sum()),
            "six_E1_with_any_tibia": int(((e == 6) & (t >= 1)).sum()), "any_tibia_rows": int((t >= 1).sum()),
            "any_tibia_rows_with_any_E1": int(((t >= 1) & (e >= 1)).sum()),
            "table_[anyE1&anyTib, anyE1&noTib],[noE1&anyTib, noE1&noTib]": table, "fisher_p_two_sided": float(p),
            "spearman_rho": float(rho), "spearman_p": float(prho)}

summary = {"reproduces_original_E1_counts_128": repro_e1, "reproduces_original_tibia_counts_128": repro_tib, "conditions": {}}
for e1key in ("E1_peak10ms", "E1_spectral", "E1_spectral_conc0.2", "E1_spectral_conc0.5"):
    for thr in (0.1, 1., 5.):
        for rng in (0.1, 0.5, 2.):
            summary["conditions"][f"{e1key}|active>{thr}Hz|range>={rng}Hz"] = assoc(e1key, f"tib_{thr}_{rng}")
# hypergeometric expectation of 6/6 overlap under independence (original gates)
from math import comb
summary["six_six_overlap_prob_zero_under_independence"] = comb(128 - 5, 2) / comb(128, 2)
rh = [r for r in rows if r["E1_peak10ms"] == 6]
summary["rhythmic_rows"] = [{k: r[k] for k in ("global_row", "E1_spectral", "E1_spectral_freq_hz", "E1_mean_hz", "E1_ptp_hz", "MN_active_over_1hz", "tibia_MN_active_over_1hz", "active_modules")} for r in rh]
rec = [r for r in rows if r["tib_1.0_0.5"] == 6]
summary["recruiting_rows"] = [{k: r[k] for k in ("global_row", "E1_peak10ms", "E1_spectral", "E1_mean_hz", "E1_ptp_hz", "MN_active_over_1hz", "tibia_MN_active_over_1hz")} for r in rec]
e1n = np.array([r["E1_peak10ms"] for r in rows]); mna = np.array([r["MN_active_over_1hz"] for r in rows]); tb = np.array([r["tib_1.0_0.5"] for r in rows])
summary["MN_active_by_E1_count"] = {int(k): [int(np.median(mna[e1n == k])), int(mna[e1n == k].min()), int(mna[e1n == k].max())] for k in range(7) if (e1n == k).any()}
summary["MN_active_rows_any_tibia_vs_none_median"] = [float(np.median(mna[tb >= 1])), float(np.median(mna[tb == 0]))]
summary["spectral_vs_peak_agreement_rows"] = int(sum(r["E1_spectral"] == r["E1_peak10ms"] for r in rows))
json.dump({"summary": summary, "rows": rows}, open(OUT / "robustness_128.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:9000])
