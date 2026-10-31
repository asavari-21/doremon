"""
Central configuration. Change settings here, not in the other files.

Everything marked ASSUMPTION is a choice I made because it wasn't specified.
Change it if it doesn't match what you want.
"""

# ---------------------------------------------------------------------------
# Data (ASSUMPTION: same NSE large-cap universe as your FLSPI project)
# ---------------------------------------------------------------------------
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
]
BENCHMARK = "^NSEI"          # NIFTY 50, used only as a buy-and-hold comparison
START_DATE = "2012-01-01"
END_DATE = None              # None = up to today
DATA_CACHE = "data/prices.csv"

# Chronological split (ASSUMPTION). The agent trains ONLY on data before
# VAL_START. VAL is for sanity checks / choosing settings. TEST is touched
# for the final report only -- do not tune on it. With these dates the test
# period includes the March-2020 crash, which the training period does not.
VAL_START = "2018-01-01"
TEST_START = "2020-01-01"

# Windows (inside the test period) reported separately as stress tests.
STRESS_WINDOWS = {"covid_crash": ("2020-02-15", "2020-04-30")}

# ---------------------------------------------------------------------------
# Trading environment
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
EPISODE_LEN = 252            # one training episode = one simulated year
WARMUP = 63                  # days of history needed to compute features
COST_RATE = 0.001            # ASSUMPTION: 10 bps per unit of one-way turnover
RF_ANNUAL = 0.05             # ASSUMPTION: cash earns 5% a year
RISK_AVERSION = 5.0          # reward = log growth - (gamma/2) * return^2
REWARD_SCALE = 100.0
ACTION_CLIP = 3.0            # actions are clipped, then softmax -> weights

# Anti-churn settings. Without them the agent trades its whole book ~100x a year
# and costs eat all of its gains. Both apply in training AND in evaluation.
POLICY_SMOOTHING = 0.15      # each day the agent moves only this fraction of the way
                             # toward the weights it asks for. 1.0 = old behaviour (no smoothing).
TURNOVER_PENALTY = 0.01      # extra reward penalty per unit of turnover (a regulariser on top of
                             # the real COST_RATE). 0.0 = old behaviour.

# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
TIMESTEPS = 300_000
N_ENVS = 8
NET_ARCH = [64, 64]
PPO_KW = dict(
    n_steps=256, batch_size=512, n_epochs=10, learning_rate=3e-4,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
)

# ---------------------------------------------------------------------------
# Domain randomization.  name: (nominal, low, high)
# Each episode, every parameter is drawn uniformly from
#   [nominal + s*(low-nominal), nominal + s*(high-nominal)]
# where s is --dr-strength (0 = no randomization, 1 = these ranges,
# 2 = twice as wide). The simulator is calibrated to TRAINING data only.
# ---------------------------------------------------------------------------
DR_SPEC = {
    "vol_scale":      (1.00, 0.70, 1.80),    # x all volatilities
    "drift_shift":    (0.00, -0.10, 0.15),   # added to every asset's annual drift
    "drift_keep":     (1.00, 0.00, 1.00),    # x each asset's historical drift
    "asset_drift_sd": (0.00, 0.00, 0.12),    # per-asset annual drift noise (sd)
    "corr_blend":     (0.00, 0.00, 0.60),    # weight on an equicorrelation matrix
    "equicorr":       (0.50, 0.30, 0.90),    # that matrix's correlation level
    "stress_vol":     (2.00, 1.50, 3.50),    # vol multiplier in the stress regime
    "stress_drift":   (-0.40, -0.80, -0.10), # extra annual drift in stress regime
    "p_stay":         (0.985, 0.95, 0.995),  # probability a regime persists a day
    "tail_df":        (8.0, 3.5, 12.0),      # Student-t degrees of freedom
    "phi":            (0.00, -0.08, 0.08),   # AR(1): momentum (+) / reversal (-)
    "cost":           (COST_RATE, 0.0005, 0.003),
}

SEED = 0