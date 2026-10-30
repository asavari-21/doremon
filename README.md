# PPO + Domain Randomization for Portfolio Optimization

A PPO agent allocates across 10 NSE stocks + cash. It is trained in one of
several ways, then every strategy is tested on the same held-out real data.

**The question this answers (and what it does not):** *Does training on
randomized simulated markets (domain randomization, DR) make a portfolio
policy more robust on real data it has never seen, compared with training on
real history alone?* It does **not** claim to predict prices. DR can make a
policy robust; it cannot create a signal that isn't in the data.

## Setup (once)
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it, in this order
```bash
# 1. Known-answer checks (no internet, ~2 min). Must say ALL CHECKS PASSED.
python sanity_check.py

# 2. Download + cache the real NSE data (needs internet) and look at the split
python data.py

# 3. Offline smoke test of the whole pipeline on FAKE data (~15 s). Results are meaningless.
python run_experiment.py --synthetic --quick

# 4. The real experiment (~10-20 min): history-only vs domain-randomized
python run_experiment.py --modes hist dr --seeds 0 1 2

# 5. Fuller ablation + more seeds (~1 hr): adds "sim" (simulator with NO randomization)
python run_experiment.py --modes hist sim dr --seeds 0 1 2 3 4
```
Outputs go to `results/`: `summary.csv` (main table), `metrics_by_run.csv`
(every run), `wealth_test.png`. Trained models are cached in `models/`
(use `--retrain` to redo them). The same table is printed to the terminal.

Train a single agent: `python train.py --mode dr --seed 0` (modes: `hist`, `sim`, `dr`, `mixed`).

## What the strategies are
| name | what it is |
|---|---|
| `ppo_hist` | PPO trained on random 1-year windows of real **training** data |
| `ppo_sim` | PPO trained on a simulator calibrated to training data, **no** randomization |
| `ppo_dr_s1` | same simulator **with** domain randomization (`--dr-strength 1`) |
| `ppo_mixed_s1` | half real windows, half randomized simulation |
| baselines | `buy_hold_equal`, `equal_weight_monthly`, `inverse_vol_monthly`, `min_variance_monthly`, `nifty50_buy_hold` |

`ppo_sim` vs `ppo_dr` isolates the effect of the **randomization itself**,
separate from the effect of simply using a simulator.

## Data split (edit in `config.py`)
Train: 2012 to end-2017. Validation: 2018-2019. Test: 2020 onward. The test
period contains the March-2020 crash, which the training years do not. Train
only on train; use validation to choose settings; **look at test once, at the end.**

## How to read the results
Look at the **TEST** table, comparing `ppo_dr` to `ppo_hist` **and** to the baselines:
- `Sharpe`, `CAGR` - return per unit of risk / growth after costs
- `MaxDD`, `Worst21d`, `covid_crash_ret`, `covid_crash_mdd` - the robustness
  metrics this idea is really about (lower drawdown / smaller crash loss is better)
- `Turnover_ann` - 20 means trading the whole book 20x a year; costs are charged

DR is a **win** only if `ppo_dr` beats `ppo_hist` on the robustness metrics
**consistently across seeds** and is not just beaten by `min_variance_monthly`
or `equal_weight_monthly`. With 3 seeds treat differences as hints, not proof;
use 5+ seeds before claiming anything. An honest "DR did not help" is a valid result.

## Things you can change (all in `config.py`)
- `TICKERS`, `START_DATE`, `VAL_START`, `TEST_START`
- `COST_RATE` (10 bps), `RF_ANNUAL` (5%), `RISK_AVERSION` (5) - **my assumptions**
- `DR_SPEC` - which market parameters are randomized and over what range
  (this is the heart of the idea); `--dr-strength 0.5 / 1 / 2` narrows or widens all ranges at once
- `TIMESTEPS` (300k), `PPO_KW`, `NET_ARCH`

## Known limits (read these)
- **The yfinance download path was not tested by me** (my sandbox has no
  internet). Everything else was tested on synthetic data. If step 2 fails,
  it is almost certainly a ticker/network issue - fix `TICKERS` or retry.
- The simulator is stylised (3 regimes, fat tails, AR(1), random drifts and
  correlations). DR helps only if reality falls inside the randomized range.
- Synthetic-data results are meaningless; they only prove the code runs.
- A bug found while building this: the `min_variance` baseline silently
  returned equal weights (optimizer tolerance). It is fixed and has a
  regression check; keep `sanity_check.py` in your workflow if you edit things.
- This is research code, not financial advice.

## Files
`config.py` settings | `data.py` download/cache/synthetic data | `features.py` no-look-ahead features |
`simulator.py` domain-randomized market | `env.py` Gym environment + portfolio dynamics |
`sources.py` where training episodes come from | `train.py` PPO training |
`backtest.py` shared backtester | `baselines.py` | `metrics.py` | `run_experiment.py` | `sanity_check.py`
