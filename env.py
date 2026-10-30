"""
Portfolio environment + the shared one-day portfolio dynamics.

Action: N+1 numbers -> softmax -> long-only weights over N stocks + cash.
Timing: at the close of day t the agent sees features up to day t, picks
weights, and earns day t+1's return. Costs = cost_rate * one-way turnover of
the stock positions. Reward = REWARD_SCALE * (log(1+net) - (gamma/2)*net^2).
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from features import compute_features, n_features


@dataclass
class Episode:
    R: np.ndarray        # (T, N) daily simple returns
    t0: int              # first decision day
    cost_rate: float


def softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def portfolio_step(w_target, w_drift, r_assets, cost_rate, rf_daily):
    """One day of trading. Returns (net_return, new_drifted_weights, turnover, cost)."""
    n = len(r_assets)
    turnover = float(np.abs(w_target[:n] - w_drift[:n]).sum())
    gross = float(np.dot(w_target[:n], r_assets) + w_target[n] * rf_daily)
    cost = cost_rate * turnover
    denom = 1.0 + gross
    new_w = np.empty_like(w_target)
    new_w[:n] = w_target[:n] * (1.0 + r_assets) / denom
    new_w[n] = w_target[n] * (1.0 + rf_daily) / denom
    new_w /= new_w.sum()
    return gross - cost, new_w, turnover, cost


class PortfolioEnv(gym.Env):
    def __init__(self, source, n_assets: int):
        super().__init__()
        self.source, self.n = source, n_assets
        self.rf_d = (1 + config.RF_ANNUAL) ** (1 / config.TRADING_DAYS) - 1
        obs_dim = n_features(n_assets) + n_assets + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-config.ACTION_CLIP, config.ACTION_CLIP,
                                       (n_assets + 1,), np.float32)

    def _obs(self):
        return np.concatenate([self.F[self.t], self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ep = self.source.sample(self.np_random)
        self.R, self.t, self.cost_rate = ep.R, ep.t0, ep.cost_rate
        self.t_last = len(ep.R) - 1
        self.F = compute_features(ep.R)
        self.w = np.append(np.full(self.n, 1.0 / self.n), 0.0)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -config.ACTION_CLIP, config.ACTION_CLIP)
        net, self.w, turnover, _ = portfolio_step(
            softmax(a), self.w, self.R[self.t + 1], self.cost_rate, self.rf_d)
        reward = config.REWARD_SCALE * (np.log1p(net) - 0.5 * config.RISK_AVERSION * net ** 2)
        self.t += 1
        return self._obs(), float(reward), self.t >= self.t_last, False, {"net": net, "turnover": turnover}
