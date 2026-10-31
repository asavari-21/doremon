"""
End-to-end experiment: train PPO agents under different training regimes,
evaluate them and simple baselines on the SAME held-out real data, and write
a results table + wealth plot.

    python run_experiment.py --quick --synthetic                     # offline smoke test
    python run_experiment.py --modes hist dr --seeds 0 1 2           # real data
    python run_experiment.py --modes hist sim dr --seeds 0 1 2 3 4   # fuller ablation

Outputs (results/): metrics_by_run.csv, summary.csv, wealth_test.png
"""
import argparse
import os

import numpy as np
import pandas as pd

import config
from backtest import run_backtest
from baselines import BASELINES
from data import load_data
from env import policy_to_target
from features import compute_features
from metrics import summarize
from sources import strategy_name
from train import model_path, train_one


def rl_decide(model, F):
    def decide(t, w):
        obs = np.concatenate([F[t], w]).astype(np.float32)
        a, _ = model.predict(obs, deterministic=True)
        return policy_to_target(a, w)          # same smoothing rule the agent trained under
    return decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["hist", "dr"], choices=["hist", "sim", "dr", "mixed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=config.TIMESTEPS)
    ap.add_argument("--dr-strength", type=float, default=1.0)
    ap.add_argument("--dr-strengths", nargs="+", type=float, default=None,
                    help="several DR strengths in one run, e.g. 0.5 1 2 (overrides --dr-strength)")
    ap.add_argument("--synthetic", action="store_true", help="fake data, offline test only")
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--retrain", action="store_true", help="ignore saved models")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.quick:
        a.timesteps, a.seeds = min(a.timesteps, 20_000), a.seeds[:1]
    os.makedirs(a.out, exist_ok=True)

    d = load_data(synthetic=a.synthetic)
    R, dates, i_val, i_test = d["R"], d["dates"], d["i_val"], d["i_test"]
    T = len(R)
    F = compute_features(R)
    periods = {"val": (i_val, i_test - 1), "test": (i_test, T - 1)}
    print(f"{R.shape[1]} stocks | train {i_val}d | val {i_test - i_val}d | test {T - i_test}d"
          f"{'  [SYNTHETIC DATA]' if a.synthetic else ''}")

    rows, curves = [], {}

    def record(name, seed, period, rets, turns, t_start):
        dts = dates[t_start + 1: t_start + 1 + len(rets)]
        rows.append(dict(strategy=name, seed=seed, period=period, **summarize(rets, dts, turns)))
        curves[(name, seed, period)] = (dts, np.cumprod(1.0 + rets))

    # --- baselines (no training) ---
    for period, (ts, te) in periods.items():
        for name, make in BASELINES.items():
            r, tu = run_backtest(R, ts, te, make(R, ts))
            record(name, 0, period, r, tu, ts)
        if d["bench_R"] is not None:
            br = d["bench_R"].values[ts + 1: te + 1]
            record("nifty50_buy_hold", 0, period, br, None, ts)

    # --- RL agents ---
    from stable_baselines3 import PPO
    strengths = a.dr_strengths or [a.dr_strength]
    combos = []                       # (mode, strength); hist and sim have no strength
    for mode in a.modes:
        for st in (strengths if mode in ("dr", "mixed") else strengths[:1]):
            combos.append((mode, st))
    for mode, st in combos:
        for seed in a.seeds:
            path = model_path(mode, st, seed)
            if a.retrain or not os.path.exists(path):
                print(f"training {strategy_name(mode, st)} seed={seed} "
                      f"({a.timesteps:,} steps) ...", flush=True)
                train_one(mode, seed, a.timesteps, st, d)
            model = PPO.load(path, device="cpu")
            for period, (ts, te) in periods.items():
                r, tu = run_backtest(R, ts, te, rl_decide(model, F))
                record(strategy_name(mode, st), seed, period, r, tu, ts)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "metrics_by_run.csv"), index=False)

    num = [c for c in res.columns if c not in ("strategy", "seed", "period")]
    g = res.groupby(["period", "strategy"])[num]
    summary = g.mean().join(g.std().add_suffix("_sd"))
    summary["n_seeds"] = res.groupby(["period", "strategy"]).size()
    summary.to_csv(os.path.join(a.out, "summary.csv"))

    show = ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Worst21d", "Turnover_ann"]
    show += [c for c in num if c.endswith("_ret") or c.endswith("_mdd")]
    pd.set_option("display.width", 200)
    for period in ("val", "test"):
        print(f"\n=== {period.upper()} (mean over seeds; baselines have 1 run) ===")
        print(summary.loc[period][show + ["n_seeds"]].round(3).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in res["strategy"].unique():
            cs = [curves[k] for k in curves if k[0] == name and k[2] == "test"]
            dts = cs[0][0]
            w = np.mean([c[1] for c in cs], axis=0)
            ax.plot(dts, w, label=name, lw=2 if name.startswith("ppo") else 1)
        ax.set_title("Test-period wealth (start = 1.0, after costs)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "wealth_test.png"), dpi=120)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"\nSaved to {a.out}/: metrics_by_run.csv, summary.csv, wealth_test.png")


if __name__ == "__main__":
    main()