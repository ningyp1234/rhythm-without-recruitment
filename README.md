# Rhythm without recruitment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22957834.svg)](https://doi.org/10.5281/zenodo.22957834)

Code, derived data and verification outputs for the preprint

> **Yupeng Ning.** *Rhythm without recruitment: a falsification-first audit of a connectome-constrained ventral nerve cord model as a controller for a simulated* Drosophila *body.* Preprint (2026). Beijing ZHYU Tech Corp. Correspondence: ningyp@gmail.com

The manuscript source and PDF are in [`paper/`](paper/). Each GitHub release is archived on Zenodo; the concept DOI [10.5281/zenodo.22957834](https://doi.org/10.5281/zenodo.22957834) always resolves to the latest version.

## What this repository is

The study asks whether the published firing-rate model of the full male adult nerve cord (MANC; 23,532 neurons, 1,372,404 signed connections) by Pugliese et al. can drive a simulated six-legged *Drosophila* body. The answer, within the published parameter ensemble, is no.

- Six-leg E1 premotor rhythm appears in 51 of the 128 author-saved parameter sets.
- None of those 51 sets shows antagonistic tibia flexor/extensor output in any leg.
- The ensemble splits into a low-activity rhythmic regime and a hyperactive recruiting regime.
- The paper also documents four pitfalls that produced false positives or negatives during the project: phase-replay equivalence, fixed-step integration, kinematic gait gates, and rhythm-detector resolution.

This repository lets anyone re-derive every number and figure in the paper from the saved result files. **It does not claim connectome-driven walking.**

## Quick start

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt

# Re-derive every quantitative claim in the manuscript (53 checks; writes paper/verification/manuscript_number_check.csv)
python paper/scripts/verify_manuscript_numbers.py

# Regenerate Figures 2-4 and paper/figures/figure_source_numbers.json
python paper/scripts/make_figures_v2.py

# Re-run the ensemble re-analysis (spectral vs. peak detector, threshold sweep, Fisher tests)
python paper/scripts/robustness_128.py
python paper/scripts/e1_detector_recheck.py

# Full-network integrator comparison (Control 2b); each job is independent
python paper/scripts/fullmanc_integrator_convergence.py euler1      # ~10 s
python paper/scripts/fullmanc_integrator_convergence.py euler0.5    # ~20 s
python paper/scripts/fullmanc_integrator_convergence.py euler0.25   # ~35 s
python paper/scripts/fullmanc_integrator_convergence.py euler0.1    # ~90 s
python paper/scripts/fullmanc_integrator_convergence.py rk45_26     # ~50 s
python paper/scripts/fullmanc_integrator_convergence.py rk45_30     # ~45 s
python paper/scripts/fullmanc_integrator_convergence.py summary     # E1 rhythm classification of all saved traces
```

Timings were measured on a laptop CPU. No GPU is needed.

## Layout

| Path | Contents |
|---|---|
| `paper/` | `main.tex`, `references.bib`, compiled PDF, figures (PDF/PNG) and `figure_source_numbers.json` |
| `paper/scripts/` | The analysis and verification scripts used for the paper (see Quick start) |
| `paper/verification/` | Their outputs: re-analysis JSON, integrator traces (`conv_*.npy`) and the 53-item number check (`manuscript_number_check.csv`) |
| `results/` | The saved result files read by the scripts, with the same directory names as in the original project |
| `data/` | Small inputs: MANC neuron table and signed weights (2025-10-06 export), parameters and configuration of author run 33195882, and provenance manifests with hashes |

## Provenance and scope

- **Where the result files come from.** The files in `results/` were produced by the author's original simulation pipeline. That pipeline also covers MuJoCo body models and experiments outside the scope of this paper, and it is not included in this repository. It is available from the corresponding author upon reasonable request. [`PROVENANCE.md`](PROVENANCE.md) lists, for each result directory, the procedure and the original scripts that produced it.
- **What you can reproduce here.** Every number and figure in the paper can be re-derived with `paper/scripts/` from the files in this repository.
- **Integrator comparison.** `paper/scripts/fullmanc_integrator_convergence.py` re-integrates the full MANC network from the included parameters and weights, so Control 2b is reproduced from scratch rather than from saved results.
- **Path cleaning.** Absolute local paths have been removed from all text files.

## Data sources and licences

This repository's own code is released under the MIT License (see [`LICENSE`](LICENSE)). Third-party data keep their original licences; see [`DATA_LICENSES.md`](DATA_LICENSES.md). Key upstream sources:

- Pugliese S. M. et al., *Connectome simulations identify a central pattern generator circuit for fly walking*, bioRxiv (2025), doi:10.1101/2025.09.12.675944. Code: github.com/smpuglie/Pugliese_2026 (MIT). Data: Zenodo record 22260924 (CC BY 4.0).
- MANC connectome: Takemura et al. (2024) and Marin et al. (2024), *eLife* (CC BY 4.0).
- Whole-body 3D kinematics: Ispizua et al. (2026), bioRxiv, doi:10.64898/2026.05.03.722293. Only metadata is included here.

## Use of AI

- The original simulations, analyses and independent re-computations were executed by an autonomous LLM-based coding agent.
- The audit, the re-analyses in `paper/scripts/` and the drafting of the manuscript were assisted by an AI model (Claude, Anthropic).
- The author defined the study, verified the reported numbers against the saved outputs, and takes full responsibility for the content.

## Citation

Please cite the preprint. To cite the code and data, use the Zenodo concept DOI [10.5281/zenodo.22957834](https://doi.org/10.5281/zenodo.22957834), or the version-specific DOI shown on the Zenodo record. Machine-readable metadata are in [`CITATION.cff`](CITATION.cff).
