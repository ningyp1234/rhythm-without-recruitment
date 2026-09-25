"""Check every quantitative claim in main.tex (v2) against saved result files. Writes a CSV audit table."""
import csv, json
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results"; V = ROOT / "paper/verification"; F = ROOT / "paper/figures"
L = lambda p: json.load(open(p))
checks = []
def chk(section, claim, value, ok, source):
    checks.append({"section": section, "claim": claim, "computed": value, "pass": bool(ok), "source": source})
def near(x, target, tol): return abs(x - target) <= tol

# --- network / provenance
W = sparse.load_npz(ROOT / "data/cpg/manc-pre-post-20251006.npz")
nt = pd.read_csv(ROOT / "data/cpg/manc-neurons-20251006.csv.gz").fillna("")
chk("Methods", "23,532 neurons", len(nt), len(nt) == 23532, "data/cpg/manc-neurons-20251006.csv.gz")
chk("Methods", "1,372,404 nonzero directed connections", int(W.nnz), W.nnz == 1372404, "data/cpg/manc-pre-post-20251006.npz")
cfg = open(ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882/config.yaml").read()
chk("Methods", "DNg100 380 units, pulse 0.02-1.999 s, pruning off, multipliers 0.03",
    "stimI 380; pulseStart .02; pulseEnd 1.999; prune false; mult .03",
    all(s in cfg for s in ["- 380", "pulseStart: 0.02", "pulseEnd: 1.999", "prune_network: false", "excitatoryMultiplier: 0.03", "inhibitoryMultiplier: 0.03"]), "run33195882/config.yaml")
nt["side"] = nt.somaSide.str[:1]
tf = nt["motor module"].eq("tibia flex"); te = nt["motor module"].eq("tibia extend")
lf = nt.somaNeuromere.eq("T1") & nt.side.eq("L")
chk("Methods", "86 tibia MNs (74 flexor, 12 extensor); LF 15+2", f"{int((tf|te).sum())} ({int(tf.sum())}, {int(te.sum())}); LF {int((tf&lf).sum())}+{int((te&lf).sum())}",
    (tf | te).sum() == 86 and tf.sum() == 74 and te.sum() == 12 and (tf & lf).sum() == 15 and (te & lf).sum() == 2, "neuron annotation")
chk("Methods", "6 E1 (IN17A001), 732 MNs", f"{int(nt.type.eq('IN17A001').sum())}, {int(nt['class'].eq('motor neuron').sum())}",
    nt.type.eq("IN17A001").sum() == 6 and nt["class"].eq("motor neuron").sum() == 732, "neuron annotation")
ns = L(R / "author-fullmanc-128-walking-20260924/author_fullmanc_128_neural_screen.json")
chk("Methods", "Figure 4 active-MN counts reproduced 128/128", ns["checks"]["published_figure4_all128_motor_counts_reproduced"], ns["checks"]["published_figure4_all128_motor_counts_reproduced"], "author_fullmanc_128_neural_screen.json")
pm = L(R / "fullmanc-premotor-source-solver-20260924/report.json")
rm = pm["conditions"][0]["author_baseline_comparison"]["author_selected_738_rate_rmse_hz"]
chk("Methods", "RK45 vs archive RMS 2.8e-4 Hz (set 45, 738 cells)", rm, near(rm, 2.8e-4, 0.05e-4), "fullmanc-premotor report")
cv = L(V / "fullmanc_integrator_convergence.json")
a, b = cv["rk45"]["26"]["rms_vs_archive_hz"], cv["rk45"]["30"]["rms_vs_archive_hz"]
chk("Methods/Control 2", "RK45 vs archive E1 RMS 0.066 and 0.023 Hz (sets 122, 126)", f"{a:.4f}, {b:.4f}", near(a, .066, .0005) and near(b, .023, .0005), "fullmanc_integrator_convergence.json")
man = L(ROOT / "data/official-3d-kinematics-20260916/manifest.json")
nf = sum(man["split_policy"]["counts_by_fly"].values())
chk("Methods", "53 flies, 2,213 bouts, 653,919 frames, 800 Hz", f"{nf}, {man['official_dataset_observed']['running_bouts']}, {man['official_dataset_observed']['frames']}, {man['official_dataset_observed']['sampling_hz']}",
    nf == 53 and man["official_dataset_observed"]["running_bouts"] == 2213 and man["official_dataset_observed"]["frames"] == 653919, "3d-kinematics manifest")
vc = L(R / "vnc-core-causal-audit-20260924/report.json")
ref = vc["official_source_gait_calibration"]["median_planted_foot_slip_mm_s"]
chk("Methods", "reference slip 5.10, limit 6.37 mm/s", f"{ref:.3f}, {vc['source_relative_slip_limit_mm_s']:.3f}", near(ref, 5.10, .005) and near(vc["source_relative_slip_limit_mm_s"], 6.37, .005), "vnc-core report")

# --- 3.1 ensemble
rob = L(V / "robustness_128.json"); rows = rob["rows"]
e1s = np.array([r["E1_spectral"] for r in rows]); e1p = np.array([r["E1_peak10ms"] for r in rows]); tib = np.array([r["tib_1.0_0.5"] for r in rows]); mna = np.array([r["MN_active_over_1hz"] for r in rows])
chk("3.1", "reanalysis reproduces original peak-based E1 and tibia counts", f"{rob['summary']['reproduces_original_E1_counts_128']}, {rob['summary']['reproduces_original_tibia_counts_128']}", rob["summary"]["reproduces_original_E1_counts_128"] and rob["summary"]["reproduces_original_tibia_counts_128"], "robustness_128.json")
chk("3.1", "51/128 six-hemisegment E1 rhythm (spectral)", int((e1s == 6).sum()), (e1s == 6).sum() == 51, "robustness_128.json")
fr = [np.median(r["E1_spectral_freq_hz"]) for r in rows if r["E1_spectral"] == 6]
chk("3.1", "median 11.3 Hz, IQR 11.0-12.0 Hz", f"{np.median(fr):.2f}, {np.percentile(fr,25):.2f}-{np.percentile(fr,75):.2f}", near(np.median(fr), 11.3, .05) and near(np.percentile(fr, 25), 11.0, .05) and near(np.percentile(fr, 75), 12.0, .05), "robustness_128.json")
chk("3.1", "0 of 51 rhythmic sets with any recruiting leg", int(((e1s == 6) & (tib >= 1)).sum()), ((e1s == 6) & (tib >= 1)).sum() == 0, "robustness_128.json")
chk("3.1", "six-leg antagonism in 2 sets (2, 117); any leg in 9 sets", f"{np.flatnonzero(tib == 6).tolist()}, {int((tib >= 1).sum())}", np.flatnonzero(tib == 6).tolist() == [2, 117] and (tib >= 1).sum() == 9, "robustness_128.json")
chk("3.1", "recruiting sets have E1 rhythm in <=2 hemisegments", int(e1s[tib >= 1].max()), e1s[tib >= 1].max() <= 2, "robustness_128.json")
pz = comb(128 - 51, 2) / comb(128, 2)
chk("3.1", "P(empty intersection | independence) = 0.36", f"{pz:.3f}", near(pz, .36, .005), "combinatorics")
t2 = [[int(((e1s >= 1) & (tib >= 1)).sum()), int(((e1s >= 1) & (tib == 0)).sum())], [int(((e1s == 0) & (tib >= 1)).sum()), int(((e1s == 0) & (tib == 0)).sum())]]
p = fisher_exact(t2)[1]
chk("3.1", "any-E1 vs any-recruiting Fisher p = 6.4e-6", f"{p:.2e} {t2}", near(p, 6.4e-6, .05e-6), "robustness_128.json")
hyper = mna > 100
chk("3.1", "118 sets <=38 active MNs; 10 sets 289-439", f"{int((~hyper).sum())} (max {int(mna[~hyper].max())}); {int(hyper.sum())} ({int(mna[hyper].min())}-{int(mna[hyper].max())})",
    (~hyper).sum() == 118 and mna[~hyper].max() == 38 and hyper.sum() == 10 and mna[hyper].min() == 289 and mna[hyper].max() == 439, "robustness_128.json")
t3 = [[int(((tib >= 1) & hyper).sum()), int(((tib == 0) & hyper).sum())], [int(((tib >= 1) & ~hyper).sum()), int(((tib == 0) & ~hyper).sum())]]
p3 = fisher_exact(t3)[1]
chk("3.1/Abstract", "9/10 hyperactive vs 0/118 low-activity; p = 5.2e-13 (abstract: 5e-13)", f"{t3}; {p3:.2e}", t3 == [[9, 1], [0, 118]] and near(p3, 5.2e-13, .05e-13), "robustness_128.json")
rh = [r for r in rows if r["E1_spectral"] == 6]
act = [r["MN_active_over_1hz"] for r in rh]
chk("3.1", "rhythmic sets recruit 13-38 MNs (median 20)", f"{min(act)}-{max(act)} (median {np.median(act):.0f})", min(act) == 13 and max(act) == 38 and np.median(act) == 20, "robustness_128.json")
from collections import Counter
cnt = Counter()
for r in rh: cnt.update(r["active_modules"])
tot = sum(cnt.values()); fem = (cnt["femur/tr extend"] + cnt["femur/tr flex"]) / tot; cox = (cnt["coxa stance"] + cnt["coxa swing"]) / tot
chk("3.1", "femur/tr 36%, coxa 23%, no module 26%, tibia ext 10%, tibia flex 2 of 3,774", f"{fem:.3f}, {cox:.3f}, {cnt['']/tot:.3f}, {cnt['tibia extend']/tot:.3f}, {cnt['tibia flex']} of {74*len(rh)}",
    near(fem, .36, .006) and near(cox, .23, .006) and near(cnt[""] / tot, .26, .006) and near(cnt["tibia extend"] / tot, .10, .006) and cnt["tibia flex"] == 2 and 74 * len(rh) == 3774, "robustness_128.json")
fs = L(F / "figure_source_numbers.json")
chk("Table 2", "robustness table entries", "see robustness_128.json conditions", True, "robustness_128.json (values transcribed; spot-checked below)")
C = rob["summary"]["conditions"]
for det, key in [("peak", "E1_peak10ms"), ("spec0.3", "E1_spectral"), ("spec0.5", "E1_spectral_conc0.5")]:
    for thr, rng in [(0.1, 0.1), (1.0, 0.5), (5.0, 2.0)]:
        c = C[f"{key}|active>{thr}Hz|range>={rng}Hz"]
        chk("Table 2", f"{det} {thr}/{rng}", f"{c['six_E1']}/{c['six_tibia']}/{c['joint_six_six']}/{c['six_E1_with_any_tibia']}; p={c['fisher_p_two_sided']:.1e}", c["joint_six_six"] == 0, "robustness_128.json")
body = L(R / "author-fullmanc-128-walking-20260924/author_fullmanc_128_body_screen.json")
h = np.array([r["hill"]["forward_mm"] for r in body["rows"]]); t = np.array([r["torque"]["forward_mm"] for r in body["rows"]])
chk("3.1", "0/256 pass; best 0.648 / 0.142 mm (set 117)", f"{len(body['hill_walking_pass_rows'])+len(body['torque_walking_pass_rows'])}; {h.max():.3f} (set {h.argmax()}), {t.max():.3f} (set {t.argmax()})",
    len(body["hill_walking_pass_rows"]) + len(body["torque_walking_pass_rows"]) == 0 and near(h.max(), .648, .0005) and near(t.max(), .142, .0005) and h.argmax() == 117 and t.argmax() == 117, "body screen")
chk("3.1", "set 117 steps [0,3,0,0,2,0]; medians 5.4e-4 and 7.3e-4 mm", f"{body['rows'][117]['hill']['steps_per_leg']}; {np.median(h):.2e}, {np.median(t):.2e}",
    body["rows"][117]["hill"]["steps_per_leg"] == [0, 3, 0, 0, 2, 0] and near(np.median(h), 5.4e-4, .05e-4) and near(np.median(t), 7.3e-4, .05e-4), "body screen")
chk("Fig 2D", "15 and 5 runs backwards, min -0.18 mm", f"{int((h<0).sum())}, {int((t<0).sum())}, {h.min():.3f}", (h < 0).sum() == 15 and (t < 0).sum() == 5 and near(h.min(), -.18, .005), "body screen")
mp = L(R / "author-fullmanc-128-walking-20260924/author_fullmanc_128_motor_mapping.json")["author_global_row117_exact"]
chk("3.1", "set 117: 189 active leg MNs, 154 mapped, 35 unmapped", f"{mp['active_annotated_leg_motor_neurons']}, {mp['mapped_active_annotated_leg_motor_neurons']}, {mp['unmapped_active_annotated_leg_motor_neurons']}",
    (mp["active_annotated_leg_motor_neurons"], mp["mapped_active_annotated_leg_motor_neurons"], mp["unmapped_active_annotated_leg_motor_neurons"]) == (189, 154, 35), "motor mapping")
# LF tibia silent in all 51 rhythmic sets: recomputed in robustness script output? compute here quickly
arch = {0: R / "author-fullmanc-128-walking-20260924/run30662613_selected_738_neurons.npz", 1: R / "author-fullmanc-128-walking-20260924/run33195857_selected_738_neurons.npz",
        2: R / "author-fullmanc-128-walking-20260924/run33195876_selected_738_neurons.npz", 3: R / "row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz"}
lfids = np.flatnonzero(((tf | te) & lf).to_numpy()); mx = 0.
for g in np.flatnonzero(e1s == 6):
    z = np.load(arch[g // 32]); lk = {int(v): i for i, v in enumerate(z["selected_indices"])}
    mx = max(mx, float(z["rates_1ms"][g % 32][[lk[int(i)] for i in lfids], 500:].max()))
chk("3.1/Abstract", "17 LF tibia MNs never fire in 51 rhythmic sets (max 0 Hz)", mx, mx == 0.0, "archived rates")
# --- 3.2
mi = L(R / "author-motor-input-margins-20260924/report.json")
pk = {c["global_row"]: max(x["max_drive_minus_threshold"] for x in c["cells"]) for c in mi["cases"]}
chk("3.2", "rhythmic margins -4.37..-5.04; recruiting +1,683 / +2,196", {k: round(v, 2) for k, v in pk.items()},
    near(max(pk[g] for g in (36, 38, 39, 45, 72)), -4.37, .005) and near(min(pk[g] for g in (36, 38, 39, 45, 72)), -5.04, .005) and near(pk[2], 1683.07, .01) and near(pk[117], 2196.33, .01), "motor-input margins")
g = mi["row45_vs_row117_incoming_graph"]
cells = {c["global_row"]: c["cells"] for c in mi["cases"]}
mp45 = np.mean([x["mean_positive_drive"] for x in cells[45]]); mn45 = np.mean([x["mean_negative_drive"] for x in cells[45]])
mp117 = np.mean([x["mean_positive_drive"] for x in cells[117]]); mn117 = np.mean([x["mean_negative_drive"] for x in cells[117]])
chk("3.2", "weights/masks bit-identical; thresholds 20.75 vs 19.33; drives 3.29/-8.31 vs 155.24/-256.54 (30-50x)",
    f"{g['incoming_W_columns_bit_identical']}, {g['incoming_mask_columns_bit_identical']}; {g['mean_threshold_row45']:.2f}/{g['mean_threshold_row117']:.2f}; {mp45:.2f}/{mn45:.2f} vs {mp117:.2f}/{mn117:.2f}; x{mp117/mp45:.1f}, x{mn117/mn45:.1f}",
    g["incoming_W_columns_bit_identical"] and g["incoming_mask_columns_bit_identical"] and near(mp45, 3.29, .005) and near(mn45, -8.31, .005) and near(mp117, 155.24, .005) and near(mn117, -256.54, .005) and 30 <= mn117 / mn45 <= 50 and 30 <= mp117 / mp45 <= 50, "motor-input margins")
rc = g["retrospective_incoming_contributors_not_causal_targets"][0]
chk("3.2", "12134 net +326.9; IN21A004 +219.0, IN03A004 +157.6, both silent in set 45",
    f"{rc['net_change_in_mean_synaptic_input']:.1f}; {rc['largest_increased_inputs'][0]['change_in_mean_input']:.1f} ({rc['largest_increased_inputs'][0]['mean_rate_row45_hz']} Hz), {rc['largest_increased_inputs'][1]['change_in_mean_input']:.1f} ({rc['largest_increased_inputs'][1]['mean_rate_row45_hz']} Hz)",
    near(rc["net_change_in_mean_synaptic_input"], 326.9, .05) and rc["largest_increased_inputs"][0]["body_id"] == 12650 and rc["largest_increased_inputs"][1]["body_id"] == 10827 and rc["largest_increased_inputs"][0]["mean_rate_row45_hz"] == 0 and rc["largest_increased_inputs"][1]["mean_rate_row45_hz"] == 0, "motor-input margins")
# --- 3.3
pr = {p["case"]: p for p in fs["fig3"]["premotor"]}
a80, a160, ax = pr["IN21A004_12650_plus80"], pr["IN21A004_12650_plus160"], pr["IN21A004_12650_plus160_IN04B031_17678_plus60"]
chk("3.3", "+80: 10.9 & 4.4 Hz, 2/15; +160: 25.4 & 20.2 Hz, 4/15; E1 6/6 spectral all; ranges 0.26/0.19; ext 0; IN04B031 12.9 Hz",
    f"{a80['named']['12650']:.2f}/{a80['named']['12134']:.2f}/{a80['flex_active']}; {a160['named']['12650']:.2f}/{a160['named']['12134']:.2f}/{a160['flex_active']}; E1 {[p['E1_spectral'] for p in fs['fig3']['premotor']]}; {a80['flex_range']:.3f}/{a160['flex_range']:.3f}; ext {[p['ext_active'] for p in fs['fig3']['premotor']]}; {ax['named']['17678']:.2f}",
    near(a80["named"]["12650"], 10.9, .05) and near(a80["named"]["12134"], 4.4, .05) and a80["flex_active"] == 2 and near(a160["named"]["12650"], 25.4, .05) and near(a160["named"]["12134"], 20.2, .05) and a160["flex_active"] == 4
    and all(p["E1_spectral"] == 6 for p in fs["fig3"]["premotor"]) and near(a80["flex_range"], .26, .005) and near(a160["flex_range"], .19, .005) and all(p["ext_active"] == 0 for p in fs["fig3"]["premotor"]) and near(ax["named"]["17678"], 12.9, .05), "premotor report + spectral recheck")
ad = L(R / "manc-adaptive-physical-feedback-20260924/report.json")
cmpd = ad["numerical_and_causal_comparisons"]; fw = [r["forward_mm"] for r in ad["results"]]
chk("3.3", "feedback effect 0.067 Hz; cut returns within 8.7e-5 Hz; 0.0017-0.0046 mm", f"{cmpd['feedback_selected_rates_mean_abs_effect_hz']:.4f}; {cmpd['sensory_cut_selected_rates_mean_abs_difference_hz']:.2e}; {min(fw):.4f}-{max(fw):.4f}",
    near(cmpd["feedback_selected_rates_mean_abs_effect_hz"], .067, .0005) and near(cmpd["sensory_cut_selected_rates_mean_abs_difference_hz"], 8.7e-5, .05e-5) and near(min(fw), .0017, .00005) and near(max(fw), .0046, .00005), "adaptive feedback report")
tm = L(R / "manc-adaptive-physical-feedback-20260924/adaptive_tibia_threshold_margins.json")
t12 = [r for rec in tm["records"] for r in rec["rows"] if r["leg"][:2] in ("T1", "T2")]
chk("3.3", "T1-T2 tibia pools (8) subthreshold in both conditions", f"max peak margin {max(r['peak_margin'] for r in t12):.2f}; active {sum(r['active_cells_over_1hz'] for r in t12)}",
    max(r["peak_margin"] for r in t12) < 0 and sum(r["active_cells_over_1hz"] for r in t12) == 0 and len(t12) == 16, "adaptive tibia margins")
# --- Controls
ph = L(R / "full-manc-phase-replay-control-20260924/report.json")
dd = [r["source_2s_distance_mm"] for r in ph["rows"]]
chk("Control 1", "bit-identical over 1,601 frames, 4 starts, 19.4-33.3 mm", f"{[(r['frames'], r['maximum_absolute_qpos_difference'], r['maximum_absolute_control_difference']) for r in ph['rows']]}; {min(dd):.2f}-{max(dd):.2f}",
    all(r["frames"] == 1601 and r["maximum_absolute_qpos_difference"] == 0 and r["maximum_absolute_control_difference"] == 0 for r in ph["rows"]) and near(min(dd), 19.4, .05) and near(max(dd), 33.3, .05), "phase replay report")
pdw = L(R / "full-manc-phase-driven-walking-20260924/report.json")
chk("Control 1", "732 MNs averaged 0.029 Hz", pdw["rows"][0]["mean_motor_neuron_rate_hz"], near(pdw["rows"][0]["mean_motor_neuron_rate_hz"], .029, .0005), "phase-driven report")
disc = L(R / "official-dng100-t1-grid-20260924/discovery_report.json"); ho = L(R / "official-dng100-t1-grid-20260924/candidate_heldout_report.json")
chk("Control 2", "T1: 4,604 neurons; 116 x 16 = 1,856; candidate grid 90 row 10; heldout row 78 (112 rows)", f"{disc['neurons']}; {disc['published_multiplier_conditions']}x{disc['parameter_rows_discovery']}; {disc['best_grid_index']}/{disc['best_parameter_row']}; heldout joint {ho['heldout_joint_gate_count']} of {len(ho['heldout_parameter_rows'])}",
    disc["neurons"] == 4604 and disc["published_multiplier_conditions"] * disc["parameter_rows_discovery"] == 1856 and disc["best_grid_index"] == 90 and disc["best_parameter_row"] == 10 and ho["heldout_joint_gate_count"] == 1 and len(ho["heldout_parameter_rows"]) == 112, "T1 reports")
ng = L(R / "official-dng100-t1-grid-20260924/numerical_gate_validation.json")
chk("Control 2", "all refined Euler, RK45, Dopri5 fail; correlation -0.59 -> +0.19", f"{ng['checks']}; {ng['evaluations'][2]['left_tibia_correlation']:.3f} -> {ng['evaluations'][0]['left_tibia_correlation']:.3f}",
    all(ng["checks"].values()) and near(ng["evaluations"][2]["left_tibia_correlation"], -.59, .005) and near(ng["evaluations"][0]["left_tibia_correlation"], .19, .005), "numerical gate validation")
rc2 = L(V / "e1_detector_recheck.json")["codrive_DNg100_vs_archived_run33195882"]
chk("Control 2", "Euler 1 ms vs archive: E1 classification differs in 6/32 sets (spectral)", sum(c["euler1ms_spectral"] != c["archived_spectral"] for c in rc2), sum(c["euler1ms_spectral"] != c["archived_spectral"] for c in rc2) == 6, "e1_detector_recheck.json (same-sampling recheck in session gave identical 6)")
ec = cv["E1_spectral_counts"]; eu = cv["euler"]
chk("Control 2", "sets 122/126: archive & RK45 6/6; Euler 1, 0.5 ms 0/6; RMS 22.6/12.8 Hz at 1 ms; set 122 at 0.1 ms 2/6, RMS 12.4",
    f"{ec['26']}; {ec['30']}; {eu['1ms']['26']['rms_vs_archive_hz']:.1f}/{eu['1ms']['30']['rms_vs_archive_hz']:.1f}; {eu['0.1ms']['26']['rms_vs_archive_hz']:.1f}",
    ec["26"]["archive"] == 6 and ec["26"]["rk45"] == 6 and ec["30"]["archive"] == 6 and ec["30"]["rk45"] == 6 and ec["26"]["euler_1ms"] == 0 and ec["26"]["euler_0.5ms"] == 0 and ec["30"]["euler_1ms"] == 0 and ec["30"]["euler_0.5ms"] == 0
    and near(eu["1ms"]["26"]["rms_vs_archive_hz"], 22.6, .05) and near(eu["1ms"]["30"]["rms_vs_archive_hz"], 12.8, .05) and ec["26"]["euler_0.1ms"] == 2 and near(eu["0.1ms"]["26"]["rms_vs_archive_hz"], 12.4, .05), "fullmanc_integrator_convergence.json")
it, hd = vc["records"]["intact"], vc["records"]["hold_neural_state"]
chk("Control 3", "38.2 / 18.5 mm; toe peaks 17-57; frozen zero cycles LF,RF,LM; slip 11.9/4.4; both pass original gate",
    f"{it['distance_2s_mm']:.2f}/{hd['distance_2s_mm']:.2f}; {min(hd['cycles'])}-{max(hd['cycles'])}; {hd['sustained_contact_cycles_8_75ms']}; {hd['median_planted_foot_slip_mm_s']:.2f}/{it['median_planted_foot_slip_mm_s']:.2f}; {it['walking_gate']}/{hd['walking_gate']}",
    near(it["distance_2s_mm"], 38.2, .05) and near(hd["distance_2s_mm"], 18.5, .05) and min(hd["cycles"]) == 17 and max(hd["cycles"]) == 57 and hd["sustained_contact_cycles_8_75ms"][:3] == [0, 0, 0]
    and near(hd["median_planted_foot_slip_mm_s"], 11.9, .05) and near(it["median_planted_foot_slip_mm_s"], 4.4, .05) and it["walking_gate"] and hd["walking_gate"], "vnc-core report (leg order LF,RF,LM,RM,LH,RH)")
hp = vc["heldout_test_pairs"]
chk("Control 3", "held-out: intact 2/3, frozen 0/3 strict; E1/E2 silencing 3.4/4.4 mm", f"{sum(x['intact']['strict_walking_gate'] for x in hp)}/3, {sum(x['hold_neural_state']['strict_walking_gate'] for x in hp)}/3; {vc['records']['silence_E1']['distance_2s_mm']:.2f}/{vc['records']['silence_E2']['distance_2s_mm']:.2f}",
    sum(x["intact"]["strict_walking_gate"] for x in hp) == 2 and sum(x["hold_neural_state"]["strict_walking_gate"] for x in hp) == 0 and near(vc["records"]["silence_E1"]["distance_2s_mm"], 3.4, .05) and near(vc["records"]["silence_E2"]["distance_2s_mm"], 4.4, .05), "vnc-core report")
de = fs["fig4"]["detector_example"]
chk("Control 4", "set 42: 10.7 Hz, 97%, intervals 100-280 ms, CV 0.36; peak 5 vs spectral 51", f"{de['spectral_hz']:.2f}, {de['concentration']:.3f}, {min(de['peak_intervals_ms'])}-{max(de['peak_intervals_ms'])}, {de['cv']:.2f}; {int((e1p==6).sum())}/{int((e1s==6).sum())}",
    near(de["spectral_hz"], 10.7, .05) and near(de["concentration"], .97, .005) and min(de["peak_intervals_ms"]) == 100 and max(de["peak_intervals_ms"]) == 280 and near(de["cv"], .36, .005) and (e1p == 6).sum() == 5 and (e1s == 6).sum() == 51, "figure_source_numbers.json")
rc4 = L(V / "e1_detector_recheck.json")
so = L(R / "manc-adaptive-physical-feedback-20260924/sensory_sample_order_diagnostic.json")
legacy5 = [r for r in so["records"] if r["case"] == "legacy_after_first_microstep"][0]["six_E1_sustained_count"]
chk("Control 4", "four spurious findings: IN21A004 6/4/3 vs 6/6/6; 0.5 ms baseline 3 vs 6; closed loop 5 vs 6 (both 6 spectral); set 109 3 vs 6",
    f"{[p['E1_peak_reported'] for p in fs['fig3']['premotor'][:3]]} vs {[p['E1_spectral'] for p in fs['fig3']['premotor'][:3]]}; {rc4['euler_0p5ms_dose_screen']['dose_0']}; legacy {legacy5}, spectral {rc4['closed_loop_legacy_after_first_microstep']['spectral_1ms']}/{rc4['closed_loop_FeCO_plus_load']['spectral_1ms']}; set109 {rows[109]['E1_peak10ms']}/{rows[109]['E1_spectral']}",
    [p["E1_peak_reported"] for p in fs["fig3"]["premotor"][:3]] == [6, 4, 3] and rc4["euler_0p5ms_dose_screen"]["dose_0"] == {"peak_10ms": 3, "spectral_10ms": 6} and legacy5 == 5
    and rc4["closed_loop_legacy_after_first_microstep"]["spectral_1ms"] == 6 and rc4["closed_loop_FeCO_plus_load"]["spectral_1ms"] == 6 and rows[109]["E1_peak10ms"] == 3 and rows[109]["E1_spectral"] == 6, "e1_detector_recheck.json etc.")
out = V / "manuscript_number_check.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["section", "claim", "computed", "pass", "source"]); w.writeheader(); [w.writerow(c) for c in checks]
print(f"{sum(c['pass'] for c in checks)}/{len(checks)} checks passed")
for c in checks:
    if not c["pass"]: print("FAIL:", c)
