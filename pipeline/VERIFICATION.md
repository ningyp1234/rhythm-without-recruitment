# What was re-run for this release

Checked on 2026-09-26 on a Linux aarch64 machine. The environment was Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, MuJoCo 3.9.0, Gymnasium 1.3.0, JAX 0.11.2, Diffrax 0.7.2, Numba 0.67.0 and FlyGym at commit `38c8ec61` (editable install). Two things were not available:

- PyTorch: its wheel for this platform needs several gigabytes of CUDA packages.
- Network access to Zenodo from this machine.

## Results

1. **Syntax.** All 69 files compile with Python 3.12.14.
2. **External code.** `setup_external.sh` checked out FlyGym and `3d_tracking_ik` at the pinned commits, and found the `fruitfly_v1` body. FlyGym installed, and `flygym` and `flygym_demo` import.
3. **Imports.** Each module was imported from the repository root. 57 of 69 import without error. The other 12 need PyTorch and were not tested: the hybrid-walker training scripts and the Control 1 and 3 scripts.
4. **Verification scripts.** Four scripts were re-run against the included results:

| Script | Result | Regenerated output vs. included file |
|---|---|---|
| `verify_dng100_numerical_gate.py` | passed | Every decision is identical. 7 correlation values differ in the last one or two digits (relative difference below 10⁻¹⁵), which is platform floating-point rounding. The included file was kept. |
| `verify_fullmanc_premotor_source_solver.py` | passed | byte-identical |
| `verify_manc_adaptive_physical_feedback.py` | passed | byte-identical |
| `verify_vnc_core_causality.py` | 19 of 19 checks | byte-identical |

## Not re-run here

- The four `fetch_*` scripts, because Zenodo was not reachable.
- Every script that needs the fetched archives, or that takes longer than a few minutes: the screens, body replays and adaptive integrations.
- The 12 scripts that need PyTorch.

The paper's own analysis scripts (`../paper/scripts/`) do not depend on this directory. They were re-run, and all 53 manuscript checks pass.
