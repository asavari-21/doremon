import numpy as np

import config
from env import portfolio_step


def run_backtest(R, t_start, t_end, decide, cost_rate=None):
    """Trade from day t_start to t_end. At each day t, decide(t, w_drift) returns
    target weights (N+1, last = cash) or None to hold. Earns R[t+1]. Everyone
    starts equal-weight in stocks (no cost for the initial allocation)."""
    cost_rate = config.COST_RATE if cost_rate is None else cost_rate
    rf_d = (1 + config.RF_ANNUAL) ** (1 / config.TRADING_DAYS) - 1
    N = R.shape[1]
    w = np.append(np.full(N, 1.0 / N), 0.0)
    rets, turns = [], []
    for t in range(t_start, t_end):
        target = decide(t, w)
        if target is None:
            target = w
        net, w, turnover, _ = portfolio_step(np.asarray(target, float), w, R[t + 1], cost_rate, rf_d)
        rets.append(net)
        turns.append(turnover)
    return np.array(rets), np.array(turns)
