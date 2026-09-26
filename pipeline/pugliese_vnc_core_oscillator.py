"""Runtime state for the official-data DNg100→E1–E2–I1 VNC core CPG."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from evaluate_official_running_dynamic_replay import FPS, ROOT

DATA = ROOT / "data/pugliese-vnc-core-cpg.json"


def load_parameters():
    source = json.loads(DATA.read_text())
    weighted_w = np.asarray(source["weighted_matrix_post_by_pre"], dtype=float)
    params = {key: np.asarray(value, dtype=float) for key, value in source["deterministic_mean_parameters"].items()}
    current = np.asarray([source["stimulus"]["current"], 0.0, 0.0, 0.0])
    return source, weighted_w, params, current


def derivative(rates, weighted_w, params, current, tau_scale):
    total = current + weighted_w @ rates
    activation = np.maximum(
        params["fr_cap"]
        * np.tanh((params["a"] / params["fr_cap"]) * (total - params["threshold"])),
        0.0,
    )
    return (activation - rates) / (params["tau"] * tau_scale)


def build_limit_cycle(period_frames):
    source, weighted_w, params, current = load_parameters()
    native_period_ms = 62.0
    target_period_s = period_frames / FPS
    tau_scale = target_period_s / (native_period_ms / 1000.0)

    def ode(_, rates):
        return derivative(rates, weighted_w, params, current, tau_scale)

    time = np.arange(0.0, 3.0 + 0.0005, 0.001)
    solution = solve_ivp(ode, (0.0, 3.0), np.zeros(4), t_eval=time, rtol=2e-8, atol=1e-10)
    peaks, _ = find_peaks(solution.y[2], prominence=0.5, distance=max(10, int(target_period_s / 0.001 * 0.7)))
    if len(peaks) < 3:
        raise RuntimeError("VNC core did not converge to a limit cycle")
    left, right = peaks[-2], peaks[-1]
    source_phase = np.linspace(time[left], time[right], period_frames, endpoint=False)
    template = np.vstack([np.interp(source_phase, time, trace) for trace in solution.y]).T
    measured_period_s = float(time[right] - time[left])
    return {
        "source": source,
        "weighted_w": weighted_w,
        "params": params,
        "current": current,
        "tau_scale": tau_scale,
        "template": template,
        "measured_period_s": measured_period_s,
    }


@dataclass
class VNCCoreOscillator:
    weighted_w: np.ndarray
    params: dict
    current: np.ndarray
    tau_scale: float
    template: np.ndarray
    rates: np.ndarray

    @classmethod
    def create(cls, cycle, phase):
        return cls(
            cycle["weighted_w"],
            cycle["params"],
            cycle["current"],
            cycle["tau_scale"],
            cycle["template"],
            cycle["template"][phase].copy(),
        )

    def phase(self):
        scale = np.maximum(np.ptp(self.template[:, 1:], axis=0), 1e-6)
        distance = np.mean(((self.template[:, 1:] - self.rates[1:]) / scale) ** 2, axis=1)
        return int(np.argmin(distance))

    def step(self):
        dt = 1.0 / FPS
        f = lambda state: derivative(state, self.weighted_w, self.params, self.current, self.tau_scale)
        k1 = f(self.rates)
        k2 = f(self.rates + 0.5 * dt * k1)
        k3 = f(self.rates + 0.5 * dt * k2)
        k4 = f(self.rates + dt * k3)
        self.rates = np.maximum(self.rates + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0.0)
        return self.phase()
