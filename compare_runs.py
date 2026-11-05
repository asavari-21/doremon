import argparse

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="results/metrics_by_run.csv")
ap.add_argument("--ref", default="ppo_hist")
ap.add_argument("--period", default="test", choices=["val", "test"])
a = ap.parse_args()

METRICS = ["CAGR", "Sharpe", "MaxDD", "Worst21d", "CVaR5", "Turnover_ann",
           "covid_crash_ret", "covid_crash_mdd"]
m = pd.read_csv(a.file)
m = m[(m.period == a.period) & m.strategy.str.startswith("ppo")]
if a.ref not in set(m.strategy):
    raise SystemExit(f"reference {a.ref!r} not found. Available: {sorted(set(m.strategy))}")

ref = m[m.strategy == a.ref]
rows = []
for name, g in m.groupby("strategy"):
    if name == a.ref:
        continue
    for k in [k for k in METRICS if k in m.columns]:
        x, y = g[k].dropna().values, ref[k].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        rows.append(dict(strategy=name, metric=k, n=f"{len(x)} vs {len(y)}",
                         mean=x.mean(), ref_mean=y.mean(), diff=x.mean() - y.mean(),
                         welch_p=stats.ttest_ind(x, y, equal_var=False).pvalue,
                         mwu_p=stats.mannwhitneyu(x, y).pvalue))
out = pd.DataFrame(rows)
if out.empty:
    raise SystemExit("not enough seeds (need at least 3 per strategy).")

# Holm correction over all Welch p-values shown
order = np.argsort(out.welch_p.values)
m_ = len(out); holm = np.empty(m_)
running = 0.0
for rank, idx in enumerate(order):
    running = max(running, min(1.0, out.welch_p.values[idx] * (m_ - rank)))
    holm[idx] = running
out["holm_p"] = holm
out["survives_correction"] = np.where(out.holm_p < 0.05, "YES", "")

pd.set_option("display.width", 220)
print(f"\n{a.period.upper()} period, compared with {a.ref} "
      f"(diff = strategy minus {a.ref}; for MaxDD / Turnover / covid_crash_mdd lower is better)\n")
print(out.round(4).to_string(index=False))
print("\nRule of thumb: p < 0.05 AFTER correction (last column) is real evidence; "
      "an uncorrected p between 0.05 and 0.2 is only a hint.")
