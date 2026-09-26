# Simulation pipeline

This directory contains the code that produced the result files in [`../results/`](../results/). It has 69 Python files, about 11,400 lines:

- the 28 scripts listed in [`../PROVENANCE.md`](../PROVENANCE.md), which produced the paper's results;
- 6 scripts that produced inputs of those scripts from public third-party data: one downloads an author trace, the other five build the hybrid walker's training data and controller;
- the 35 modules they import.

These are the files that were run, with one exception. Path definitions were changed so that the code runs from this repository: 18 lines in 14 files, listed in [`PATH_CHANGES.diff`](PATH_CHANGES.diff). No computational code was changed.

Five imported modules belong to other experiments of the original project: `brain.py`, `conductance.py`, `conductance_fast.py`, `activity_view.py` and `vision_bridge.py`. `walking_benchmark.py` imports them, but no script here executes them, and their data are not included. Several other modules are imported only for shared functions; their own `main()` programs are not needed. For the same reason, some data that those programs read are not included: `data/neurons.csv.gz`, `data/synapse_counts.npz`, `data/vision/`, the older MANC files `data/cpg/manc-neurons.csv.gz`, `data/cpg/manc-pre-post.npz` and `data/cpg/front-neurons.csv.gz`, and `results/cpg-walking/`, `results/cpg-context-diagnostic/` and `results/walking/`.

## Scripts and the results they produced

Run every command from the repository root, for example `python pipeline/screen_author_fullmanc_all128.py`. `verify_vnc_core_causality.py` also reads `vnc_core_causal_replay.html`, a small replay viewer at the repository root. Outputs are written to `results/` and overwrite the files included there; use `git diff --stat results/` to compare.

| Result directory | Scripts, in order | Used in the paper for |
|---|---|---|
| `author-fullmanc-128-walking-20260924` | `fetch_author_fullmanc_remaining_runs`, `screen_author_fullmanc_all128`, `audit_author_fullmanc128_motor_mapping`, `replay_author_fullmanc_all128_bodies`, `verify_author_fullmanc_all128` | Ensemble screen and motor-neuron-output body replays (§3.1, Fig. 2) |
| `row21-body-diagnostic-20260924` | `fetch_fullmanc_author_trace`, `screen_author_fullmanc_32` | §3.1; Controls 2b and 4 |
| `author-motor-input-margins-20260924` | `audit_author_motor_input_margins` | §3.2; Fig. 3A |
| `fullmanc-premotor-source-solver-20260924` | `probe_fullmanc_premotor_source_solver`, `screen_fullmanc_in21a004_dose`, `verify_fullmanc_premotor_source_solver` | §3.3; Fig. 3B; Control 4 |
| `manc-adaptive-physical-feedback-20260924` | `test_manc_adaptive_physical_feedback`, `diagnose_manc_sensory_sample_order`, `audit_adaptive_tibia_threshold_margins`, `verify_manc_adaptive_physical_feedback` | §3.3; Fig. 3C; Control 4 |
| `full-manc-phase-driven-walking-20260924`, `full-manc-phase-replay-control-20260924` | `validate_full_manc_phase_driven_walking`, `verify_full_manc_phase_walking`, `verify_full_manc_phase_replay_equivalence` | Control 1 (Fig. 4A) |
| `official-dng100-t1-grid-20260924` | `screen_official_dng100_t1_grid`, `validate_official_dng100_t1_candidate`, `dng100_candidate_numerical_convergence`, `validate_dng100_candidate_adaptive`, `source_dng100_diffrax_exact`, `verify_dng100_numerical_gate`, `screen_dng100_author_ranked_adaptive`, `check_dng100_high_motor_grid_adaptive` | Control 2a (Fig. 4B) |
| `full-manc-descending-codrive-screen-20260924` | `screen_manc_dng100_dng97_codrive` | Control 2b; Control 4 |
| `vnc-core-causal-audit-20260924` | `audit_vnc_core_causality`, `verify_vnc_core_causality` | Control 3 (Fig. 4D) |

### Hybrid walker (Controls 1 and 3)

The hybrid walker was trained on 18 bouts of the public whole-body 3D kinematics dataset of Ispizua, Abe et al. The trained controllers and the teacher data derived from the dataset are included, so Controls 1 and 3 run without retraining. To rebuild them from the dataset:

1. `extract_official_running_subset` → `data/official-3d-kinematics-20260916/running-reference-subset.h5`. This file is not included.
2. `derive_official_running_inverse_dynamics` → `results/official-running-inverse-dynamics-20260923/teacher.npz`.
3. `estimate_official_running_contact_forces` → `results/official-running-contact-estimate-20260923/contact-teacher.npz`.
4. Calibration: `calibrate_official_running_foot_contact`, `calibrate_official_running_contact_compliance`, `calibrate_official_running_recovery_teacher`.
5. Training stages: `train_official_running_proprioceptive_policy` → `train_official_running_dagger_policy` → `train_official_running_command_windowed_policy` → `train_official_running_endurance_policy` → `train_official_running_cpg_policy` → `refine_official_running_cpg_recovery` → `train_official_running_cpg_residual_stabilizer`. `export_cpg_walking_replay` exports the replay used for checking.

This order follows the checkpoint that each script loads. Training is stochastic, so retrained controllers will not match the included checkpoints bit for bit.

## Setup

1. **Python.** Use Python 3.11 or later; the original runs recorded Python 3.12.14. Some scripts use syntax introduced in Python 3.11.
2. **Packages.** `pip install -r pipeline/requirements.txt`.
3. **External code.** `bash pipeline/setup_external.sh`. It clones two repositories into `external/work/`, at the commits used in the original runs, and installs FlyGym in editable mode:
   - FlyGym at `38c8ec61034cd59bc5ba0de20688d4a3c0000d60` (Apache-2.0);
   - `elliottabe/3d_tracking_ik` at `defdb66932a17e41c8852f3b4c5056bee74046eb` (MIT), which provides the `fruitfly_v1` body.
4. **Third-party data not redistributed here.**
   - Pugliese et al., Zenodo record 22260924 (CC BY 4.0). Run `python pipeline/fetch_pugliese_zenodo_subset.py`, `python pipeline/fetch_pugliese_fullmanc_subset.py`, `python pipeline/fetch_fullmanc_author_trace.py` and `python pipeline/fetch_author_fullmanc_remaining_runs.py`. They download only the archive members needed, using HTTP range requests, and check each member's CRC32.
   - 3D kinematics dataset. This is needed only to rebuild the hybrid walker. Download `Full_running_dataset.h5` (3,303,870,746 bytes; SHA-256 `221d930b5755c163de843fec2a953fa1b9dc6dd95e95b643231327274a5408ee`) from the authors' public data folder linked in [`../data/official-3d-kinematics-20260916/manifest.json`](../data/official-3d-kinematics-20260916/manifest.json). Place it in `data/official-3d-kinematics-20260916/raw/`, then run `extract_official_running_subset`. The expected SHA-256 of the subset is `817f46cec0121c5fe07bad6bfcf0919e0630a854817937752c179bbc98858815`.

## Environment recorded in the original runs

The result files record these versions: Python 3.12.14, MuJoCo 3.9.0, PyTorch 2.14.0, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Gymnasium 1.3.0 and FlyGym 2.1.0 (source commit above). One report records NumPy 1.26.4. The versions of JAX, Diffrax, Numba, h5py and Matplotlib were not recorded. Runtimes were not benchmarked on a clean installation. The body replays, closed-loop runs and adaptive integrations are the slowest steps.

## Caveats documented in the paper

- Several screens use fixed-step Euler integration. They are used only where the paper says so, and never as positive evidence (Control 2).
- `cpg_context_diagnostic.rhythm` is the original peak-based rhythm detector. It undercounts the ~11 Hz E1 rhythm (Control 4). The paper's spectral detector is in `../paper/scripts/`.

## What was re-run for this release

See [`VERIFICATION.md`](VERIFICATION.md).
