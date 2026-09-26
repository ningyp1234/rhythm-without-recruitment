"""Run the author's DNg100 T1 checkpoint with its Diffrax Dopri5 solver.

Use the isolated work/diffrax-env environment, which has JAX and Diffrax.
This is a source-numerics confirmation of two fixed rows, not a gait test.
"""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/pugliese-zenodo-22260924/dng100-t1-seed145-baseline/neuron_params_candidate_grid90.h5"
OUT = ROOT / "results/official-dng100-t1-grid-20260924/source_diffrax_rows10_78.npz"
ROWS = (10, 78)


def rhs(t, rates, args):
    current, tau, gain, threshold, cap, matrix = args
    pulse = (t >= .02) & (t <= 1.999)
    total = current * pulse + jnp.dot(matrix, rates)
    active = jnp.maximum(cap * jnp.tanh((gain / cap) * (total - threshold)), 0)
    return (active - rates) / tau


def main():
    with h5py.File(DATA) as h:
        original = h["W"][:]
        params = {key: h[key][list(ROWS)] for key in ("tau", "a", "threshold", "fr_cap")}
        drives = h["input_currents"][0, list(ROWS)]
    matrix = jnp.asarray(np.maximum(original.T, 0) * .045 + np.minimum(original.T, 0) * .01333,
                         dtype=jnp.float32)
    traces = []
    stats = []
    for col, row in enumerate(ROWS):
        args = (jnp.asarray(drives[col]), jnp.asarray(params["tau"][col]),
                jnp.asarray(params["a"][col]), jnp.asarray(params["threshold"][col]),
                jnp.asarray(params["fr_cap"][col]), matrix)
        sol = diffeqsolve(ODETerm(rhs), Dopri5(), 0., 2., .001,
                         jnp.zeros(original.shape[0], dtype=jnp.float32), args=args,
                         saveat=SaveAt(ts=jnp.arange(201) / 100),
                         stepsize_controller=PIDController(rtol=2e-6, atol=5e-9),
                         max_steps=100000, throw=False)
        result = np.asarray(sol.ys)
        print("Diffrax row", row, "result", str(sol.result), "stats", sol.stats,
              "finite", np.isfinite(result).all(), flush=True)
        traces.append(result[1:].astype(np.float32))
        stats.append({k: int(v) if isinstance(v, (int, np.integer)) else str(v) for k, v in sol.stats.items()})
    np.savez_compressed(OUT, parameter_rows=np.asarray(ROWS), rates_10ms=np.stack(traces, axis=2),
                        solver="diffrax.Dopri5 PIDController rtol=2e-6 atol=5e-9 initial_dt=.001",
                        jax_version=jax.__version__)
    print("saved", OUT, "JAX", jax.__version__, "Diffrax", __import__("diffrax").__version__, flush=True)


if __name__ == "__main__":
    main()
