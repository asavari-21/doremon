"""
Domain-randomized market simulator.

Calibrated to TRAINING returns only (mean, vol, correlation per asset), then
every episode draws a fresh set of market parameters (see config.DR_SPEC) and
simulates a 3-regime (calm / normal / stress) Markov-switching market with
fat tails (Student-t), a randomly-signed AR(1) term (momentum or reversal),
shifted/shrunk drifts, and randomized correlation and transaction costs.

The point: the agent never sees the same market twice, so it can't memorise
which stock did well or what volatility looked like in the training years.

IMPORTANT LIMIT: this is a stylised simulator, not a true model of the NSE.
DR only helps if the real market lies inside the range you randomize over.
"""
from dataclasses import dataclass

import numpy as np

import config
from env import Episode

BOUNDS = {
    "vol_scale": (0.2, 5.0), "p_stay": (0.5, 0.999), "tail_df": (2.6, 60.0),
    "corr_blend": (0.0, 0.95), "equicorr": (0.0, 0.95), "cost": (0.0, 0.02),
    "stress_vol": (1.0, 6.0), "phi": (-0.3, 0.3),
}
CALM_VOL, CALM_DRIFT = 0.75, 0.05      # fixed calm-regime settings
START_PROBS = [0.3, 0.6, 0.1]          # calm / normal / stress


@dataclass
class Calibration:
    mu_ann: np.ndarray     # (N,) annual log-return drift
    sigma_d: np.ndarray    # (N,) daily log-return vol
    corr: np.ndarray       # (N, N)


def calibrate(R_train: np.ndarray) -> Calibration:
    lr = np.log1p(R_train)
    return Calibration(
        mu_ann=lr.mean(0) * config.TRADING_DAYS,
        sigma_d=lr.std(0, ddof=1),
        corr=np.corrcoef(lr.T),
    )


class DomainRandomizer:
    def __init__(self, calib: Calibration, strength: float = 1.0):
        self.c = calib
        self.strength = float(strength)

    def sample_params(self, rng) -> dict:
        p = {}
        for name, (nom, lo, hi) in config.DR_SPEC.items():
            a = nom + self.strength * (lo - nom)
            b = nom + self.strength * (hi - nom)
            v = rng.uniform(min(a, b), max(a, b))
            if name in BOUNDS:
                v = float(np.clip(v, *BOUNDS[name]))
            p[name] = v
        return p

    def simulate(self, p: dict, T: int, rng) -> np.ndarray:
        c, N = self.c, len(self.c.sigma_d)
        E = p["equicorr"] * np.ones((N, N)) + (1 - p["equicorr"]) * np.eye(N)
        C = (1 - p["corr_blend"]) * c.corr + p["corr_blend"] * E
        try:
            L = np.linalg.cholesky(C + 1e-8 * np.eye(N))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(C)
            L = V @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))

        pr = p["p_stay"]
        ps = max(0.85, pr - 0.03)           # stress regimes are shorter-lived
        P = np.array([
            [pr, (1 - pr) * 0.8, (1 - pr) * 0.2],
            [(1 - pr) * 0.5, pr, (1 - pr) * 0.5],
            [(1 - ps) * 0.2, (1 - ps) * 0.8, ps],
        ])
        cum = np.cumsum(P, axis=1)
        vol_mult = np.array([CALM_VOL, 1.0, p["stress_vol"]])
        drift_extra = np.array([CALM_DRIFT, 0.0, p["stress_drift"]])

        mu = (p["drift_keep"] * c.mu_ann + p["drift_shift"]
              + rng.normal(0.0, p["asset_drift_sd"], N))
        sig = c.sigma_d * p["vol_scale"]
        df = p["tail_df"]
        Z = rng.standard_t(df, size=(T, N)) * np.sqrt((df - 2.0) / df) @ L.T
        u = rng.random(T)

        lr = np.zeros((T, N))
        s = rng.choice(3, p=START_PROBS)
        for t in range(T):
            s = int(np.searchsorted(cum[s], u[t]))
            s = min(s, 2)
            lr[t] = (mu + drift_extra[s]) / config.TRADING_DAYS + sig * vol_mult[s] * Z[t]
            if t > 0:
                lr[t] += p["phi"] * lr[t - 1]
        return np.expm1(lr)

    def sample_episode(self, rng) -> Episode:
        p = self.sample_params(rng)
        T = config.WARMUP + config.EPISODE_LEN + 1
        return Episode(self.simulate(p, T, rng), config.WARMUP, p["cost"])
