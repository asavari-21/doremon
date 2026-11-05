import numpy as np

import config


def n_features(n_assets: int) -> int:
    return 4 * n_assets + 2


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """out[t] = sum(x[t-w+1 .. t]) for t >= w-1, else 0."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.zeros_like(x)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def compute_features(R: np.ndarray) -> np.ndarray:
    lr = np.log1p(R)
    w = config.WARMUP
    s1, s2 = _rolling_sum(lr, w), _rolling_sum(lr ** 2, w)
    sig63 = np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 1e-10))
    s1b, s2b = _rolling_sum(lr, 21), _rolling_sum(lr ** 2, 21)
    sig21 = np.sqrt(np.maximum(s2b / 21 - (s1b / 21) ** 2, 1e-10))
    z1 = lr / sig63
    z5 = _rolling_sum(lr, 5) / (sig63 * np.sqrt(5))
    z21 = s1b / (sig63 * np.sqrt(21))
    vr = np.log(sig21 / sig63)
    F = np.concatenate(
        [z1, z5, z21, vr, z21.mean(1, keepdims=True), vr.mean(1, keepdims=True)], axis=1
    )
    F = np.clip(F, -5.0, 5.0)
    F[: w - 1] = 0.0
    return F.astype(np.float32)
