"""Full-MANC integrator convergence check against the authors' archived adaptive solutions.

Run 33195882 (bilateral DNg100 = 380, pulse 0.02-1.999 s, T = 2 s, prune_network = false),
parameter rows 26 and 30 (E1 classification differs between 1-ms Euler and the archive) and row 13 (control).
Integrates the authors' rate equation with forward Euler at 1, 0.5, 0.25, 0.1 ms and with SciPy RK45
(rtol 2e-6, atol 5e-9, max_step 1 ms), then compares the six E1 traces with the archived rates.
Usage: python fullmanc_integrator_convergence.py [euler1|euler0.5|euler0.25|euler0.1|rk45_26|rk45_30|summary|all]
"""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy import sparse
from scipy.integrate import solve_ivp
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882"
OUT = ROOT / "paper/verification"
ROWS = [13, 26, 30]
E1 = [10072, 10498, 10558, 10559, 10690, 10707]
n = pd.read_csv(ROOT / "data/cpg/manc-neurons-20251006.csv.gz")
ids = n.bodyId.to_numpy() if "bodyId" in n.columns else n.iloc[:, 0].to_numpy()
e1i = [int(np.flatnonzero(ids == b)[0]) for b in E1]
with h5py.File(D / "neuron_params.h5", "r") as h:
    tau = h["tau"][ROWS].T.astype(np.float64); gain = h["a"][ROWS].T.astype(np.float64)
    thr = h["threshold"][ROWS].T.astype(np.float64); cap = h["fr_cap"][ROWS].T.astype(np.float64)
    inp = h["input_currents"][0][ROWS].T.astype(np.float64)
M = (sparse.load_npz(ROOT / "data/cpg/manc-pre-post-20251006.npz").T.tocsr() * 0.03).astype(np.float64)
za = np.load(ROOT / "results/row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz")
ae = [list(za["selected_body_ids"]).index(b) for b in E1]
arch = {r: za["rates_1ms"][r][ae].astype(np.float64) for r in ROWS}   # 6 x 2001, t = 0..2 s
def f_rate(t, R):
    cur = inp if 0.02 <= t <= 1.999 else 0.0
    act = np.maximum(cap * np.tanh((gain / cap) * (cur + M @ R - thr)), 0.0)
    return (act - R) / tau
JOB = sys.argv[1] if len(sys.argv) > 1 else "all"
RES = OUT / "fullmanc_integrator_convergence.json"
res = json.load(open(RES)) if RES.exists() else {"rows": ROWS, "euler": {}, "rk45": {}}
for dt in [d for d in (1e-3, 5e-4, 2.5e-4, 1e-4) if JOB in ("all", f"euler{d*1e3:g}")]:
    t0 = time.time(); R = np.zeros_like(tau); steps = int(round(2.0 / dt)); every = int(round(1e-3 / dt))
    tr = np.zeros((len(ROWS), 6, 2001)); k = 0
    for s in range(steps):
        t = s * dt
        R = R + dt * f_rate(t, R)
        if (s + 1) % every == 0:
            k += 1; tr[:, :, k] = R[e1i].T
    res["euler"][f"{dt*1e3:g}ms"] = {str(r): {"rms_vs_archive_hz": float(np.sqrt(((tr[i, :, 500:] - arch[r][:, 500:])**2).mean())),
                                              "max_abs_vs_archive_hz": float(np.abs(tr[i, :, 500:] - arch[r][:, 500:]).max())}
                                     for i, r in enumerate(ROWS)}
    np.save(OUT / f"conv_euler_{dt*1e3:g}ms.npy", tr.astype(np.float32))
    json.dump(res, open(RES, "w"), indent=1); print("euler", dt, round(time.time() - t0, 1), json.dumps(res["euler"][f"{dt*1e3:g}ms"]), flush=True)
for i, r in enumerate([r for r in (26, 30) if JOB in ("all", f"rk45_{r}")]):
    j = ROWS.index(r); t0 = time.time()
    sl = lambda t, y, j=j: ((np.maximum(cap[:, j] * np.tanh((gain[:, j] / cap[:, j]) * ((inp[:, j] if 0.02 <= t <= 1.999 else 0.0) + M @ y - thr[:, j])), 0.0) - y) / tau[:, j])
    sol = solve_ivp(sl, (0, 2.0), np.zeros(tau.shape[0]), method="RK45", rtol=2e-6, atol=5e-9, max_step=1e-3, t_eval=np.arange(2001) * 1e-3)
    y = sol.y[e1i]
    res["rk45"][str(r)] = {"rms_vs_archive_hz": float(np.sqrt(((y[:, 500:] - arch[r][:, 500:])**2).mean())),
                           "max_abs_vs_archive_hz": float(np.abs(y[:, 500:] - arch[r][:, 500:]).max()), "nfev": int(sol.nfev), "seconds": round(time.time() - t0, 1)}
    np.save(OUT / f"conv_rk45_row{r}.npy", y.astype(np.float32))
    print("rk45", r, json.dumps(res["rk45"][str(r)]), flush=True)
if JOB in ("all", "summary"):
    # E1 rhythm classification (spectral detector of robustness_128.py) for every saved trajectory
    from scipy.signal import find_peaks
    _src = open(Path(__file__).with_name("robustness_128.py")).read(); _ns = {}
    exec(_src[_src.index("def rhythm_peak"):_src.index("rows = []")], {"np": np, "find_peaks": find_peaks}, _ns)
    _rs = _ns["rhythm_spectral"]
    counts = {}
    for j, r in enumerate(ROWS):
        rec = {"archive": int(sum(_rs(v[500:])[0] for v in arch[r]))}
        for tag in ("1", "0.5", "0.25", "0.1"):
            f = OUT / f"conv_euler_{tag}ms.npy"
            if f.exists(): rec[f"euler_{tag}ms"] = int(sum(_rs(v[500:])[0] for v in np.load(f)[j]))
        f = OUT / f"conv_rk45_row{r}.npy"
        if f.exists(): rec["rk45"] = int(sum(_rs(v[500:])[0] for v in np.load(f)))
        counts[str(r)] = rec
    res["E1_spectral_counts"] = counts
    print("E1 spectral counts", json.dumps(counts), flush=True)
json.dump(res, open(RES, "w"), indent=1)
print("saved", JOB, flush=True)
