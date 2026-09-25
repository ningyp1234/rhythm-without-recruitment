# Provenance of result files

Each directory in `results/` was produced by the author's original simulation pipeline. The pipeline is not included in this repository and is available from the corresponding author upon reasonable request. The table names the original scripts, in execution order, so that each file can be traced. The paper's analyses in `paper/scripts/` read only the files included here.

| `results/` directory | Procedure (original scripts) | Used in the paper for |
|---|---|---|
| `author-fullmanc-128-walking-20260924` | Extraction of the authors' archived rates for all 128 parameter sets; neural screen; MN-to-muscle mapping; physical replay in two six-legged MuJoCo bodies (`fetch_author_fullmanc_remaining_runs.py`, `screen_author_fullmanc_all128.py`, `audit_author_fullmanc128_motor_mapping.py`, `replay_author_fullmanc_all128_bodies.py`, `verify_author_fullmanc_all128.py`) | Ensemble screen, MN-output body replays (§3.1; Fig. 2) |
| `row21-body-diagnostic-20260924` | Extraction of the archived rates of run 33195882 (`screen_author_fullmanc_32.py`) | §3.1; Controls 2b and 4 |
| `author-motor-input-margins-20260924` | Threshold margins of the 17 left-front tibia MNs from the archived rates and same-run weights (`audit_author_motor_input_margins.py`) | §3.2; Fig. 3A |
| `fullmanc-premotor-source-solver-20260924` | Adaptive (RK45) re-integration of set 45 with premotor current injection; fixed-step dose screen, shown only as a detector/numerics example (`probe_fullmanc_premotor_source_solver.py`, `screen_fullmanc_in21a004_dose.py`, `verify_fullmanc_premotor_source_solver.py`) | §3.3; Fig. 3B; Control 4 |
| `manc-adaptive-physical-feedback-20260924` | Closed body–VNC loop with FeCO and load feedback; sampling-order diagnostic; tibia threshold margins (`test_manc_adaptive_physical_feedback.py`, `diagnose_manc_sensory_sample_order.py`, `audit_adaptive_tibia_threshold_margins.py`, `verify_manc_adaptive_physical_feedback.py`) | §3.3; Fig. 3C; Control 4 |
| `full-manc-phase-driven-walking-20260924`, `full-manc-phase-replay-control-20260924` | Hybrid walker driven online by the full MANC versus the replayed phase (`validate_full_manc_phase_driven_walking.py`, `verify_full_manc_phase_walking.py`, `verify_full_manc_phase_replay_equivalence.py`) | Control 1 (Fig. 4A) |
| `official-dng100-t1-grid-20260924` | T1 sub-network multiplier screen and convergence tests with Euler, RK45 and the authors' Diffrax Dopri5 solver (`screen_official_dng100_t1_grid.py`, `validate_official_dng100_t1_candidate.py`, `dng100_candidate_numerical_convergence.py`, `validate_dng100_candidate_adaptive.py`, `source_dng100_diffrax_exact.py`, `verify_dng100_numerical_gate.py`, `screen_dng100_author_ranked_adaptive.py`, `check_dng100_high_motor_grid_adaptive.py`) | Control 2a (Fig. 4B) |
| `full-manc-descending-codrive-screen-20260924` | Fixed-step (1 ms Euler) screen of descending drives over 32 parameter sets (`screen_manc_dng100_dng97_codrive.py`) | Control 2b: Euler versus archive comparison; Control 4 |
| `vnc-core-causal-audit-20260924` | Hybrid walker driven by the four-cell E1/E2/I1 core; intact, frozen and silenced conditions; contact-cycle gait metrics (`audit_vnc_core_causality.py`, `verify_vnc_core_causality.py`) | Control 3 (Fig. 4D) |

## Analyses added for the paper (`paper/scripts/`, included)

| Script | Output (`paper/verification/`) |
|---|---|
| `robustness_128.py` | `robustness_128.json`: spectral and peak-based E1 detectors, tibia threshold sweep, Fisher tests, activity regimes |
| `e1_detector_recheck.py` | `e1_detector_recheck.json`: every E1 count quoted in the paper under both detectors |
| `fullmanc_integrator_convergence.py` | `fullmanc_integrator_convergence.json`, `conv_*.npy`: Euler 1–0.1 ms and RK45 versus the authors' archive, full network (re-integrated from scratch) |
| `make_figures_v2.py` | Figures 2–4 and `../figures/figure_source_numbers.json` |
| `verify_manuscript_numbers.py` | `manuscript_number_check.csv`: 53 checks of the numbers in `main.tex` |

## Known caveats of the original pipeline (documented in the paper)

- **Rhythm detector.** The original peak-based rhythm detector enforces a 100 ms minimum peak spacing, so it undercounts the ~11 Hz E1 rhythm. The paper uses a spectral detector and reports both (Table 2, Control 4).
- **Fixed-step integration.** Several original screens used 1 ms or 0.5 ms fixed-step Euler integration. These results are used only where the paper says so, and never as positive evidence (Control 2).
