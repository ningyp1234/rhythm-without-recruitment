# Third-party data

The MIT licence in `LICENSE` covers only the code written for this project. The files below derive from third-party sources and remain under their original licences. Please cite the original works when reusing them.

| Files in this repository | Source | Licence |
|---|---|---|
| `data/pugliese-zenodo-22260924/**` (parameters, configuration, manifests); `results/author-fullmanc-128-walking-20260924/run*_selected_738_neurons.npz` and `results/row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz` (subsets of the authors' archived firing rates) | Pugliese S. M., Chou G. M., Abe E. T. T., Turcu D., Lancaster J. K., Tuthill J. C., Brunton B. W. *Connectome simulations identify a central pattern generator circuit for fly walking.* bioRxiv (2025), doi:10.1101/2025.09.12.675944. Simulation archive: Zenodo record 22260924 | CC BY 4.0 (data) |
| Rate equation re-implemented in `paper/scripts/fullmanc_integrator_convergence.py` and used in `pipeline/`; three input files in `external/Pugliese_2026/` (see its `NOTICE.md`) | Pugliese et al. code, github.com/smpuglie/Pugliese_2026 (commit 10e7661) | MIT |
| `data/cpg/` (MANC neuron annotations and signed connectivity: the 2025-10-06 export used by Pugliese et al. and a 2026-05-22 export); `data/proprioception/` (proprioceptive sensory neuron annotations) | MANC connectome: Takemura S. et al., *eLife* 13:RP97769 (2024); Marin E. C. et al., *eLife* 13:RP97766 (2024); Cheong H. S. J. et al., *eLife* 13:RP96084 (2024); Janelia FlyEM | CC BY 4.0 |
| `data/official-3d-kinematics-20260916/` (manifest and bout index); `results/official-running-*` (inverse-dynamics teacher, contact estimates, trained hybrid-walker controllers and their reports, derived from 18 bouts of the dataset) | Ispizua J. I., Abe E. T. T., et al. *Whole-body 3D kinematics of freely behaving Drosophila.* bioRxiv (2026), doi:10.64898/2026.05.03.722293 | CC BY 4.0 (article licence; the public data folder states no separate licence) |
| `data/musculoskeletal/` (musculoskeletal model and meshes; the original `LICENSE` is included) | FlyMimic, github.com/gizemozd/FlyMimic (commit 9ea1131) | Apache 2.0 |

The pipeline in `pipeline/` also uses the following tools. They are installed or downloaded separately, and their licences apply to those tools, not to this repository:

- NeuroMechFly/FlyGym, commit 38c8ec61 (Apache 2.0)
- `3d_tracking_ik` (Ispizua, Abe et al.), commit defdb669, which provides the `fruitfly_v1` body (MIT)
- MuJoCo (Apache 2.0), PyTorch (BSD-3-Clause), JAX and Diffrax (Apache 2.0)

`pipeline/brain.py` uses the neuron constants of the Shiu et al. whole-brain model (MIT). No script in this repository executes it.
