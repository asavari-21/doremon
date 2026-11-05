import numpy as np
from scipy.optimize import minimize

REBAL = 21  # monthly


def _full(w_stocks):
    return np.append(w_stocks, 0.0)


def buy_hold_equal(R, t_start):
    return lambda t, w: None


def equal_weight_monthly(R, t_start):
    N = R.shape[1]
    return lambda t, w: _full(np.full(N, 1.0 / N)) if (t - t_start) % REBAL == 0 else None


def inverse_vol_monthly(R, t_start):
    lr = np.log1p(R)
    def f(t, w):
        if (t - t_start) % REBAL:
            return None
        v = lr[max(0, t - 62): t + 1].std(0) + 1e-8
        iv = 1.0 / v
        return _full(iv / iv.sum())
    return f


def min_variance_monthly(R, t_start):
    lr = np.log1p(R)
    N = R.shape[1]
    def f(t, w):
        if (t - t_start) % REBAL:
            return None
        X = lr[max(0, t - 251): t + 1]
        S = np.cov(X.T)
        S = 0.7 * S + 0.3 * np.diag(np.diag(S))        # shrink toward diagonal
        S = S * (N / np.trace(S))   # rescale: at raw daily scale (~1e-4) SLSQP stops
                                    # immediately and silently returns equal weights
        x0 = np.full(N, 1.0 / N)
        res = minimize(lambda x: x @ S @ x, x0, method="SLSQP", bounds=[(0, 1)] * N,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}])
        x = res.x if res.success else x0
        return _full(np.clip(x, 0, None) / np.clip(x, 0, None).sum())
    return f


BASELINES = {
    "buy_hold_equal": buy_hold_equal,
    "equal_weight_monthly": equal_weight_monthly,
    "inverse_vol_monthly": inverse_vol_monthly,
    "min_variance_monthly": min_variance_monthly,
}
