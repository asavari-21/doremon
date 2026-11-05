import numpy as np
import pandas as pd

import config

D = config.TRADING_DAYS


def max_drawdown(r: np.ndarray) -> float:
    """Largest peak-to-trough fall of the wealth curve, as a fraction (0.25 = -25%)."""
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.max(1.0 - wealth / peak))


def summarize(r, dates=None, turnover=None, rf_annual=None) -> dict:
    r = np.asarray(r, dtype=float)
    rf_annual = config.RF_ANNUAL if rf_annual is None else rf_annual
    n = len(r)
    wealth = float(np.prod(1.0 + r))
    cagr = wealth ** (D / n) - 1.0
    vol = float(r.std(ddof=1) * np.sqrt(D))
    rf_d = (1 + rf_annual) ** (1 / D) - 1
    ex = r - rf_d
    sharpe = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(D))
    down = np.minimum(ex, 0.0)
    sortino = float(ex.mean() / (np.sqrt((down ** 2).mean()) + 1e-12) * np.sqrt(D))
    mdd = max_drawdown(r)
    w21 = pd.Series(r).add(1).rolling(21).apply(np.prod, raw=True).dropna() - 1.0
    q = np.quantile(r, 0.05)
    out = dict(
        CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino, MaxDD=mdd,
        Calmar=cagr / mdd if mdd > 1e-9 else np.nan,
        Worst21d=float(w21.min()) if len(w21) else np.nan,
        CVaR5=float(r[r <= q].mean()),
        Turnover_ann=float(np.mean(turnover) * D) if turnover is not None else np.nan,
        FinalWealth=wealth,
    )
    if dates is not None:
        for name, (a, b) in config.STRESS_WINDOWS.items():
            m = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
            if m.sum() > 5:
                out[f"{name}_ret"] = float(np.prod(1.0 + r[m]) - 1.0)
                out[f"{name}_mdd"] = max_drawdown(r[m])
    return out
