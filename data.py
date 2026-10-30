"""
Data loading. Real data comes from yfinance (needs internet) and is cached to
data/prices.csv. `--synthetic` swaps in fake prices so you can test the whole
pipeline offline -- synthetic results mean NOTHING about real markets.

    python data.py            # download + cache real data, print a summary
"""
import os

import numpy as np
import pandas as pd

import config


def load_prices(force_download: bool = False):
    """Returns (asset_prices DataFrame, benchmark_prices Series)."""
    if os.path.exists(config.DATA_CACHE) and not force_download:
        df = pd.read_csv(config.DATA_CACHE, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        raw = yf.download(config.TICKERS + [config.BENCHMARK], start=config.START_DATE,
                          end=config.END_DATE, auto_adjust=True, progress=False)["Close"]
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned no data. Check your internet / tickers.")
        df = raw
        os.makedirs(os.path.dirname(config.DATA_CACHE) or ".", exist_ok=True)
        df.to_csv(config.DATA_CACHE)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:       # yfinance sometimes returns tz-aware dates
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    missing = [t for t in config.TICKERS if t not in df.columns]
    if missing:
        raise RuntimeError(f"No data for tickers: {missing}")
    assets = df[config.TICKERS].ffill(limit=5).dropna(how="any")
    bench = df[config.BENCHMARK].reindex(assets.index).ffill() if config.BENCHMARK in df.columns else None
    return assets, bench


def make_synthetic_prices(n_days: int = 3300, seed: int = 0):
    """Fake prices (GARCH-style market factor + stock noise + a 2020-style crash),
    generated differently from the DR simulator so tests aren't circular."""
    rng = np.random.default_rng(seed)
    N = len(config.TICKERS)
    dates = pd.bdate_range("2012-01-02", periods=n_days)
    h, mkt = 1e-4, np.zeros(n_days)
    for t in range(n_days):
        mkt[t] = np.sqrt(h) * rng.standard_normal()
        h = 2e-6 + 0.08 * mkt[t] ** 2 + 0.90 * h
    crash = (dates >= "2020-02-20") & (dates < "2020-04-01")
    mkt[crash] += -0.012
    beta = rng.uniform(0.6, 1.4, N)
    alpha = rng.uniform(0.0, 0.0006, N)
    r = alpha + np.outer(mkt, beta) + 0.008 * rng.standard_normal((n_days, N))
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=config.TICKERS)
    bench = pd.Series(100 * np.cumprod(1 + (alpha.mean() + mkt)), index=dates)
    return prices, bench


def load_data(synthetic: bool = False, force_download: bool = False) -> dict:
    prices, bench = make_synthetic_prices() if synthetic else load_prices(force_download)
    R = prices.pct_change().dropna()
    bench_R = bench.pct_change().reindex(R.index).fillna(0.0) if bench is not None else None
    dates = R.index
    return dict(
        dates=dates, R=R.values.astype(np.float64), bench_R=bench_R,
        i_val=int(dates.searchsorted(pd.Timestamp(config.VAL_START))),
        i_test=int(dates.searchsorted(pd.Timestamp(config.TEST_START))),
    )


if __name__ == "__main__":
    d = load_data(force_download=True)
    print(f"{d['R'].shape[1]} stocks, {len(d['dates'])} days "
          f"({d['dates'][0].date()} to {d['dates'][-1].date()})")
    print(f"train: {d['i_val']} days | val: {d['i_test'] - d['i_val']} days | "
          f"test: {len(d['dates']) - d['i_test']} days")
