"""Regenerate manuscript figures (v2) from saved results and verification outputs. No new simulation
except what is stored in paper/verification (integrator convergence, detector re-analysis)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"; V = ROOT / "paper" / "verification"; OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 7.5, "legend.fontsize": 6.3,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "pdf.fonttype": 42})
C_RHY, C_MOT, C_GREY, C_ACC, C_DARK = "#2F6DB5", "#C8553D", "#8C8C8C", "#2A9D8F", "#333333"
load = lambda p: json.load(open(p))
def tag(ax, s, x=-0.2, y=1.08): ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
src = open(Path(__file__).with_name("robustness_128.py")).read(); ns = {}
exec(src[src.index("def rhythm_peak"):src.index("rows = []")], {"np": np, "find_peaks": find_peaks}, ns)
rhythm_spectral = ns["rhythm_spectral"]
nt = pd.read_csv(ROOT / "data/cpg/manc-neurons-20251006.csv.gz").fillna(""); nt["side"] = nt.somaSide.str[:1]
E1 = [10072, 10498, 10558, 10559, 10690, 10707]  # IN17A001; 10707 = left T1
def pool_ids(neu, side, mod):
    return nt.bodyId[(nt.somaNeuromere.eq(neu) & nt.side.eq(side) & nt["motor module"].eq(mod)).to_numpy()].tolist()
LF_FLEX, LF_EXT = pool_ids("T1", "L", "tibia flex"), pool_ids("T1", "L", "tibia extend")
rob = load(V / "robustness_128.json"); rows = rob["rows"]
e1s = np.array([r["E1_spectral"] for r in rows]); e1p = np.array([r["E1_peak10ms"] for r in rows])
tib = np.array([r["tib_1.0_0.5"] for r in rows]); mna = np.array([r["MN_active_over_1hz"] for r in rows])
S = {}

# ---------------- Figure 2 ----------------
body = load(R / "author-fullmanc-128-walking-20260924/author_fullmanc_128_body_screen.json")
hill = np.array([r["hill"]["forward_mm"] for r in body["rows"]]); torq = np.array([r["torque"]["forward_mm"] for r in body["rows"]])
grid = np.zeros((7, 7), int)
for a, b in zip(e1s, tib): grid[b, a] += 1
arch = {0: R / "author-fullmanc-128-walking-20260924/run30662613_selected_738_neurons.npz",
        1: R / "author-fullmanc-128-walking-20260924/run33195857_selected_738_neurons.npz",
        2: R / "author-fullmanc-128-walking-20260924/run33195876_selected_738_neurons.npz",
        3: R / "row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz"}
def archived(g):
    z = np.load(arch[g // 32]); ids = list(z["selected_body_ids"]); tr = z["rates_1ms"][g % 32]
    return tr, ids
hyper = mna > 100
S["fig2"] = dict(grid_rows_tib_cols_E1=grid.tolist(), six_E1_spectral=int((e1s == 6).sum()), six_E1_peak=int((e1p == 6).sum()),
                 six_tib=int((tib == 6).sum()), any_tib=int((tib >= 1).sum()), joint=int(((e1s == 6) & (tib == 6)).sum()),
                 sixE1_with_any_tib=int(((e1s == 6) & (tib >= 1)).sum()), low_regime_n=int((~hyper).sum()), hyper_n=int(hyper.sum()),
                 low_regime_MN_max=int(mna[~hyper].max()), hyper_MN_min=int(mna[hyper].min()), hyper_MN_max=int(mna[hyper].max()),
                 hyper_any_tib=int((tib[hyper] >= 1).sum()), low_any_tib=int((tib[~hyper] >= 1).sum()),
                 hyper_E1_max=int(e1s[hyper].max()), low_sixE1=int((e1s[~hyper] == 6).sum()),
                 hill_max=float(hill.max()), torque_max=float(torq.max()), hill_min=float(hill.min()), torque_min=float(torq.min()),
                 hill_neg=int((hill < 0).sum()), torque_neg=int((torq < 0).sum()), hill_median=float(np.median(hill)), torque_median=float(np.median(torq)))
fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 0.62, 0.62], hspace=0.62, wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
m = np.ma.masked_where(grid == 0, grid)
ax.imshow(m, origin="lower", cmap="Blues", vmin=0, vmax=grid.max() * 1.25)
for i in range(7):
    for j in range(7):
        if grid[i, j]: ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=6.5, color="white" if grid[i, j] > 30 else "black")
ax.add_patch(plt.Rectangle((5.5, 5.5), 1, 1, fill=False, ec=C_MOT, lw=1.4, ls="--")); ax.text(6, 6, "0", ha="center", va="center", color=C_MOT, fontsize=7.5, fontweight="bold")
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xlabel("hemisegments with sustained E1 rhythm"); ax.set_ylabel("legs with antagonistic\ntibia modulation")
ax.set_title("128 author-saved runs"); tag(ax, "A", x=-0.3)
for k, (g, title) in enumerate([(45, "Set 45: rhythm, no tibia output"), (117, "Set 117: tibia output, no rhythm")]):
    tr, ids = archived(g); t = np.arange(tr.shape[1]) * 1e-3
    sub = gs[0, 1 + k].subgridspec(2, 1, hspace=0.15)
    a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1)
    for b in E1: a1.plot(t, tr[ids.index(b)], color=C_RHY, lw=0.35, alpha=0.35)
    a1.plot(t, tr[ids.index(10707)], color=C_RHY, lw=0.8)
    a1.set_ylabel("E1 (Hz)"); a1.set_title(title, fontsize=7.2); plt.setp(a1.get_xticklabels(), visible=False)
    a2.plot(t, tr[[ids.index(b) for b in LF_FLEX]].mean(axis=0), color=C_MOT, lw=0.8, label="LF flexor pool")
    a2.plot(t, tr[[ids.index(b) for b in LF_EXT]].mean(axis=0), color=C_ACC, lw=0.8, label="LF extensor pool")
    a2.set_ylabel("MN (Hz)"); a2.set_xlabel("time (s)"); a2.set_xlim(0, 2)
    if k == 0: a2.set_ylim(-1, 10); a2.legend(frameon=False, loc="upper left", fontsize=5.8); tag(a1, "B", x=-0.42, y=1.18)
ax = fig.add_subplot(gs[1, 0:2])
rng = np.random.default_rng(1)
xj = e1s + rng.uniform(-0.18, 0.18, len(e1s))
ax.scatter(xj[tib == 0], np.maximum(mna[tib == 0], 0.8), s=9, color=C_RHY, alpha=0.7, lw=0, label="no leg with antagonistic tibia output")
ax.scatter(xj[tib >= 1], np.maximum(mna[tib >= 1], 0.8), s=16, color=C_MOT, marker="D", lw=0, label="≥1 leg with antagonistic tibia output")
ax.set_yscale("log"); ax.axhspan(38, 289, color=C_GREY, alpha=0.12, lw=0)
ax.text(3.0, 105, "no sets between 38 and 289 active MNs", ha="center", fontsize=6.3, color=C_DARK)
ax.set_xticks(range(7)); ax.set_xlabel("hemisegments with sustained E1 rhythm"); ax.set_ylabel("motor neurons >1 Hz (of 732)")
ax.set_ylim(0.6, 1200); ax.legend(frameon=False, loc="upper right", fontsize=6.2)
ax.set_title("Two regimes: low-activity rhythmic vs hyperactive recruiting"); tag(ax, "C", x=-0.12)
ax = fig.add_subplot(gs[1, 2])
x = np.arange(len(hill))
ax.plot(x, np.sort(hill)[::-1], "o", ms=2.0, color=C_RHY, label="78-muscle body")
ax.plot(x, np.sort(torq)[::-1], "s", ms=1.8, color=C_GREY, label="torque body")
ax.set_yscale("symlog", linthresh=1e-3); ax.axhline(5, color=C_MOT, ls="--", lw=0.9); ax.axhline(0, color="k", lw=0.5)
ax.text(len(x) * 0.02, 7.5, "5 mm gate", color=C_MOT, fontsize=6.2)
ax.set_ylim(-0.5, 30); ax.set_xlabel("set (sorted per body)"); ax.set_ylabel("displacement in 2 s (mm)")
ax.legend(frameon=False, loc="lower left", fontsize=5.8); ax.set_title("Body replay: 0/256"); tag(ax, "D", x=-0.42)
fig.savefig(OUT / "fig2_rhythm_vs_recruitment.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig2_rhythm_vs_recruitment.png", dpi=220, bbox_inches="tight")

# ---------------- Figure 3 ----------------
mi = load(R / "author-motor-input-margins-20260924/report.json")
mrows, peak = [], []
for c in mi["cases"]:
    mrows.append(c["global_row"]); peak.append(max(x["max_drive_minus_threshold"] for x in c["cells"]))
me1 = [int(e1s[g]) for g in mrows]
pm = load(R / "fullmanc-premotor-source-solver-20260924/report.json")
zp = np.load(R / "fullmanc-premotor-source-solver-20260924/selected_neural_traces.npz"); pids = list(zp["selected_body_ids"])
cases = [str(c) for c in zp["case_names"]]
pinfo = []
for i, c in enumerate(pm["conditions"]):
    tr = zp["rates_10ms"][cases.index(c["case"])]
    e1n = int(sum(rhythm_spectral(tr[pids.index(b)][50:], dt=.01)[0] for b in E1))
    L = c["neural"]["legs"]["lf"]
    pinfo.append(dict(case=c["case"], E1_spectral=e1n, E1_peak_reported=c["neural"]["six_E1_sustained_count"],
                      flex_active=L["tibia flex"]["active_over_1_hz"], ext_active=L["tibia extend"]["active_over_1_hz"],
                      flex_range=L["tibia flex"]["mean_pool_range_hz"], flex_mean=L["tibia flex"]["mean_pool_rate_hz"],
                      named=c["neural"]["named_cell_mean_hz_after_0p5s"]))
ad = load(R / "manc-adaptive-physical-feedback-20260924/adaptive_tibia_threshold_margins.json")
recs = {r["case"]: r["rows"] for r in ad["records"]}
S["fig3"] = dict(rows=mrows, peak_margin=peak, E1_spectral=me1, premotor=pinfo,
                 adaptive={k: [(r["leg"], r["motor_module"], r["peak_margin"], r["active_cells_over_1hz"]) for r in v] for k, v in recs.items()})
fig = plt.figure(figsize=(7.2, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.35], wspace=0.5)
ax = fig.add_subplot(gs[0])
ax.bar(range(len(mrows)), peak, color=[C_RHY if e == 6 else C_MOT for e in me1])
ax.set_yscale("symlog", linthresh=10); ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-12, 2e4)
ax.set_xticks(range(len(mrows))); ax.set_xticklabels(mrows); ax.set_xlabel("author parameter set")
ax.set_ylabel("max(input − threshold), 17 LF tibia MNs")
ax.legend(handles=[Patch(color=C_RHY, label="E1 rhythmic in 6/6"), Patch(color=C_MOT, label="E1 rhythmic in ≤1/6")], frameon=False, fontsize=6, loc="upper center")
ax.set_title("Rhythmic sets:\nMNs below threshold"); tag(ax, "A", x=-0.32)
sub = gs[1].subgridspec(2, 1, hspace=0.18)
a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1)
t = (np.arange(zp["rates_10ms"].shape[2])) * 0.01
base = zp["rates_10ms"][cases.index("author_DNg100_only")]; hi = zp["rates_10ms"][cases.index("IN21A004_12650_plus160")]
a1.plot(t, base[pids.index(10707)], color=C_GREY, lw=0.8, label="baseline")
a1.plot(t, hi[pids.index(10707)], color=C_RHY, lw=0.8, label="+160 → IN21A004")
a1.set_ylabel("LF E1 (Hz)"); a1.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=1); plt.setp(a1.get_xticklabels(), visible=False); a1.set_ylim(top=a1.get_ylim()[1] * 1.9)
a2.plot(t, base[[pids.index(b) for b in LF_FLEX]].mean(axis=0), color=C_GREY, lw=0.8)
a2.plot(t, hi[[pids.index(b) for b in LF_FLEX]].mean(axis=0), color=C_MOT, lw=0.9, label="LF flexor pool")
a2.plot(t, hi[[pids.index(b) for b in LF_EXT]].mean(axis=0), color=C_ACC, lw=0.9, label="LF extensor pool")
a2.set_ylabel("MN (Hz)"); a2.set_xlabel("time (s)"); a2.set_xlim(0, 2); a2.set_ylim(-0.3, 4.2); a2.legend(frameon=False, fontsize=5.6, loc="center right", ncol=1)
a1.set_title("Premotor current:\ntonic flexor recruitment"); tag(a1, "B", x=-0.3, y=1.3)
ax = fig.add_subplot(gs[2])
fb, nsr = recs["FeCO_plus_load"], recs["no_sensory"]
lab = [f'{r["leg"]} {"flx" if "flex" in r["motor_module"] else "ext"}' for r in fb]; xx = np.arange(len(fb))
ax.bar(xx - 0.2, [r["peak_margin"] for r in nsr], 0.4, color=C_GREY, label="no sensory input")
ax.bar(xx + 0.2, [r["peak_margin"] for r in fb], 0.4, color=C_RHY, label="FeCO + load feedback")
ax.set_yscale("symlog", linthresh=10); ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-60, 2e3)
ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=6.2)
ax.set_ylabel("peak(input − threshold)"); ax.axvspan(-0.5, 7.5, color=C_MOT, alpha=0.06)
ax.text(3.5, 6, "T1–T2: 8/8 pools\nnever cross threshold", ha="center", fontsize=6.2, color=C_MOT)
ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.set_title("Closed body–VNC loop\n(adaptive solver, 86 tibia MNs)"); tag(ax, "C", x=-0.2)
fig.savefig(OUT / "fig3_subthreshold_mechanism.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig3_subthreshold_mechanism.png", dpi=220, bbox_inches="tight")

# ---------------- Figure 4 ----------------
pr = load(R / "full-manc-phase-replay-control-20260924/report.json")
cv = np.load(R / "official-dng100-t1-grid-20260924/candidate_numerical_convergence.npz")
dx = np.load(R / "official-dng100-t1-grid-20260924/source_diffrax_rows10_78.npz")
ad45 = np.load(R / "official-dng100-t1-grid-20260924/candidate_adaptive_rows10_78.npz")
i = list(cv["selected_body_ids"]).index(10707); gi = int(cv["selected_indices"][i]); j = list(ad45["selected_body_ids"]).index(10707)
vc = load(R / "vnc-core-causal-audit-20260924/report.json"); it, hd = vc["records"]["intact"], vc["records"]["hold_neural_state"]
conv = load(V / "fullmanc_integrator_convergence.json")
S["fig4"] = dict(phase_replay=[(r["body_state"], r["source_2s_distance_mm"], r["maximum_absolute_qpos_difference"], r["maximum_absolute_control_difference"]) for r in pr["rows"]],
                 intact=dict(d=it["distance_2s_mm"], peaks=it["cycles"], strict=it["sustained_contact_cycles_8_75ms"], slip=it["median_planted_foot_slip_mm_s"]),
                 frozen=dict(d=hd["distance_2s_mm"], peaks=hd["cycles"], strict=hd["sustained_contact_cycles_8_75ms"], slip=hd["median_planted_foot_slip_mm_s"]),
                 convergence=conv, detector_counts=dict(peak=np.bincount(e1p, minlength=7).tolist(), spectral=np.bincount(e1s, minlength=7).tolist()))
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2)); axs = axs.ravel()
ax = axs[0]
d = [r["source_2s_distance_mm"] for r in pr["rows"]]; xx = np.arange(4)
ax.bar(xx - 0.2, d, 0.4, color=C_RHY, label="online 23,532-neuron MANC"); ax.bar(xx + 0.2, d, 0.4, color="none", ec=C_MOT, hatch="///", label="replayed phase only")
ax.set_xticks(xx); ax.set_xticklabels(["1", "2", "3", "4"]); ax.set_xlabel("initial body state"); ax.set_ylabel("displacement in 2 s (mm)"); ax.set_ylim(0, 60)
ax.text(1.5, 52, "max |Δ| over 1,601 frames = 0\n(body state and 56 controls)", ha="center", fontsize=6)
ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.84)); ax.set_title("Control 1: online network\nvs replayed phase"); tag(ax, "A")
ax = axs[1]; tt = (np.arange(200) + 1) * 0.01
ax.plot(tt, cv["dt_0.001_rates_10ms"][:, i, 0], color=C_MOT, lw=0.9, label="Euler 1 ms")
ax.plot(tt, cv["dt_0.000125_rates_10ms"][:, i, 0], color=C_GREY, lw=1.1, label="Euler 0.125 ms")
ax.plot(tt, ad45["rates_10ms"][:, j, 0], color=C_ACC, lw=1.0, ls=":", label="RK45")
ax.plot(tt, dx["rates_10ms"][:, gi, 0], color=C_RHY, lw=1.1, ls="--", label="Dopri5 (authors)")
ax.set_xlabel("time (s)"); ax.set_ylabel("left E1 (Hz)"); ax.set_xlim(0, 2); ax.set_ylim(-5, 300)
ax.legend(frameon=False, fontsize=5.5, loc="upper left"); ax.set_title("Control 2a: T1 sub-network\n(setting 90, row 10)"); tag(ax, "B")
ax = axs[2]
za = np.load(arch[3]); aid = list(za["selected_body_ids"]); t1 = np.arange(2001) * 1e-3
e1eu1 = np.load(V / "conv_euler_1ms.npy")[1, 5]; e1eu01 = np.load(V / "conv_euler_0.1ms.npy")[1, 5]; e1rk = np.load(V / "conv_rk45_row26.npy")[5]
ax.plot(t1, e1eu1, color=C_MOT, lw=0.7, label="Euler 1 ms")
ax.plot(t1, e1eu01, color=C_GREY, lw=0.7, label="Euler 0.1 ms")
ax.plot(t1, za["rates_1ms"][26][aid.index(10707)], color=C_RHY, lw=1.3, label="authors' archive")
ax.plot(t1, e1rk, color=C_ACC, lw=0.7, ls="--", label="RK45 (ours)")
ax.set_xlim(0.5, 1.2); ax.set_xlabel("time (s)"); ax.set_ylabel("left-front E1 (Hz)")
ax.legend(frameon=False, fontsize=5.5, loc="upper right", ncol=2); ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
ax.set_title("Control 2b: full MANC\n(run 33195882, set 26)"); tag(ax, "C")
ax = axs[3]
legs = ["LF", "RF", "LM", "RM", "LH", "RH"]; xx = np.arange(6); w = 0.2
ax.bar(xx - 1.5 * w, it["cycles"], w, color=C_RHY, alpha=0.45, label="intact: toe-height peaks")
ax.bar(xx - 0.5 * w, it["sustained_contact_cycles_8_75ms"], w, color=C_RHY, label="intact: contact cycles")
ax.bar(xx + 0.5 * w, hd["cycles"], w, color=C_MOT, alpha=0.45, label="frozen: toe-height peaks")
ax.bar(xx + 1.5 * w, hd["sustained_contact_cycles_8_75ms"], w, color=C_MOT, label="frozen: contact cycles")
ax.set_xticks(xx); ax.set_xticklabels(legs); ax.set_ylabel("cycles in 2 s"); ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=5.5, loc="upper left"); ax.set_title("Control 3: kinematic vs\ncontact-cycle gait gate"); tag(ax, "D")
ax = axs[4]
g = 42; tr, ids = archived(g); k = ids.index(10707)
xs = tr[k, 500::10]; ts = 0.5 + np.arange(len(xs)) * 0.01
pk, _ = find_peaks(xs, prominence=max(.5, .1 * np.ptp(xs)), distance=10)
ok, fpk, conc = rhythm_spectral(tr[k, 500:])
ax.plot(np.arange(2001) * 1e-3, tr[k], color=C_RHY, lw=0.6, alpha=0.5, label="1-ms archived rate")
ax.plot(ts, xs, ".", ms=2.2, color=C_DARK, label="10-ms samples")
ax.plot(ts[pk], xs[pk], "v", ms=4, color=C_MOT, label="peaks (≥100 ms apart)")
ax.set_xlim(0.8, 1.6); ax.set_xlabel("time (s)"); ax.set_ylabel("left-front E1 (Hz)"); ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
ivl = np.diff(pk) * 10
ax.set_title("Control 4: detector resolution\n(set 42, left-front E1)"); tag(ax, "E")
ax.text(0.02, 0.02, f"spectral peak {fpk:.1f} Hz (concentration {conc:.2f})\npeak intervals {int(ivl.min())}–{int(ivl.max())} ms, CV {np.std(ivl)/np.mean(ivl):.2f}", transform=ax.transAxes, fontsize=5.6, va="bottom")
ax.legend(frameon=False, fontsize=5.4, loc="upper right")
S["fig4"]["detector_example"] = dict(set=g, cell=10707, spectral_hz=fpk, concentration=conc, peak_intervals_ms=[int(v) for v in ivl], cv=float(np.std(ivl) / np.mean(ivl)))
ax = axs[5]
xx = np.arange(7)
ax.bar(xx - 0.2, np.bincount(e1p, minlength=7), 0.4, color=C_GREY, label="peak-based (inherited)")
ax.bar(xx + 0.2, np.bincount(e1s, minlength=7), 0.4, color=C_RHY, label="spectral (1 ms)")
ax.set_xticks(xx); ax.set_xlabel("hemisegments with sustained E1 rhythm"); ax.set_ylabel("parameter sets (of 128)")
ax.legend(frameon=False, fontsize=5.8, loc="upper left"); ax.set_title("Control 4: rhythmic sets\nper detector"); tag(ax, "F")
fig.tight_layout(h_pad=1.6, w_pad=1.0)
fig.savefig(OUT / "fig4_validation_controls.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig4_validation_controls.png", dpi=220, bbox_inches="tight")
json.dump(S, open(OUT / "figure_source_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: (v if k != "fig4" else {kk: vv for kk, vv in v.items() if kk in ("detector_example", "detector_counts")}) for k, v in S.items()}, default=float)[:5000])
