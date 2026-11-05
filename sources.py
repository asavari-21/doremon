import numpy as np

import config
from env import Episode
from simulator import DomainRandomizer, calibrate

EP_LEN = config.WARMUP + config.EPISODE_LEN + 1


class HistoricalSource:
    """Random 1-year windows cut from the real TRAINING data."""
    def __init__(self, R_train):
        self.R = R_train

    def sample(self, rng):
        s = int(rng.integers(0, len(self.R) - EP_LEN + 1))
        return Episode(self.R[s:s + EP_LEN], config.WARMUP, config.COST_RATE)


class SimSource:
    """Simulated markets from the (optionally domain-randomized) simulator."""
    def __init__(self, randomizer: DomainRandomizer):
        self.rand = randomizer

    def sample(self, rng):
        return self.rand.sample_episode(rng)


class MixedSource:
    def __init__(self, sources, probs=None):
        self.sources, self.probs = sources, probs

    def sample(self, rng):
        i = int(rng.choice(len(self.sources), p=self.probs))
        return self.sources[i].sample(rng)


class FixedSource:
    """One fixed path (used for evaluation)."""
    def __init__(self, R, t_start):
        self.ep = Episode(R, t_start, config.COST_RATE)

    def sample(self, rng):
        return self.ep


def build_source(mode: str, R_train, strength: float = 1.0):
    if mode == "hist":
        return HistoricalSource(R_train)
    calib = calibrate(R_train)
    if mode == "sim":      # calibrated simulator, NO randomization (ablation)
        return SimSource(DomainRandomizer(calib, 0.0))
    if mode == "dr":       # calibrated simulator WITH domain randomization
        return SimSource(DomainRandomizer(calib, strength))
    if mode == "mixed":    # half real windows, half randomized simulation
        return MixedSource([HistoricalSource(R_train),
                            SimSource(DomainRandomizer(calib, strength))])
    raise ValueError(f"unknown mode {mode!r} (use hist, sim, dr, mixed)")


def strategy_name(mode: str, strength: float) -> str:
    return f"ppo_{mode}_s{strength:g}" if mode in ("dr", "mixed") else f"ppo_{mode}"
