# Third-party data

The MIT licence in `LICENSE` covers only the code written for this project. The files below derive from third-party sources and remain under their original licences. Please cite the original works when reusing them.

| Files in this repository | Source | Licence |
|---|---|---|
| `data/pugliese-zenodo-22260924/**` (parameters, configuration, manifests); `results/author-fullmanc-128-walking-20260924/run*_selected_738_neurons.npz` and `results/row21-body-diagnostic-20260924/author_fullmanc_32_selected_neurons.npz` (subsets of the authors' archived firing rates) | Pugliese S. M., Chou G. M., Abe E. T. T., Turcu D., Lancaster J. K., Tuthill J. C., Brunton B. W. *Connectome simulations identify a central pattern generator circuit for fly walking.* bioRxiv (2025), doi:10.1101/2025.09.12.675944. Simulation archive: Zenodo record 22260924 | CC BY 4.0 (data) |
| Rate equation re-implemented in `paper/scripts/fullmanc_integrator_convergence.py` | Pugliese et al. code, github.com/smpuglie/Pugliese_2026 (commit 10e7661) | MIT |
| `data/cpg/manc-neurons-20251006.csv.gz`, `data/cpg/manc-pre-post-20251006.npz` (MANC neuron annotations and signed connectivity, 2025-10-06 export used by Pugliese et al.) | MANC connectome: Takemura S. et al., *eLife* 13:RP97769 (2024); Marin E. C. et al., *eLife* 13:RP97766 (2024); Cheong H. S. J. et al., *eLife* 13:RP96084 (2024); Janelia FlyEM | CC BY 4.0 |
| `data/official-3d-kinematics-20260916/manifest.json` (metadata only; no kinematic data) | Ispizua J. I., Abe E. T. T., et al. *Whole-body 3D kinematics of freely behaving Drosophila.* bioRxiv (2026), doi:10.64898/2026.05.03.722293 | CC BY 4.0 (article) |

The original simulation pipeline, which is not included here, additionally used the following tools. Their licences apply to those tools, not to this repository:

- FlyMimic (Apache 2.0)
- NeuroMechFly/FlyGym (Apache 2.0)
- FlyBody (Apache 2.0)
- FlyVis (MIT)
- MuJoCo (Apache 2.0)
