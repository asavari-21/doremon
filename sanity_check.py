"""
Known-answer checks. Run this FIRST; it needs no internet.

    python sanity_check.py          # all checks (the RL one takes a few minutes)
    python sanity_check.py --fast   # skip the PPO known-answer test
"""
import sys

import numpy as np

import config
from backtest import run_backtest
from baselines import buy_hold_equal
from env import Episode, PortfolioEnv, policy_to_target, softmax
from features import compute_features
from metrics import max_drawdown
from simulator import DomainRandomizer, calibrate

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


rng = np.random.default_rng(0)

# 1. max drawdown on a known path: +10%, -20%, +5%  ->  1.1, 0.88, 0.924  -> DD = 20%
check("max drawdown on known path", abs(max_drawdown(np.array([0.10, -0.20, 0.05])) - 0.20) < 1e-9)

# 2. features use no future data
R = rng.normal(0.0005, 0.01, (400, 5))
F1 = compute_features(R)
R2 = R.copy(); R2[301:] = rng.normal(0, 0.05, (99, 5))
F2 = compute_features(R2)
check("features at t=300 ignore data after t=300", np.allclose(F1[:301], F2[:301]))

# 3. buy-and-hold equal weight, zero cost == average of the stocks' own wealth
r, _ = run_backtest(R, 100, 399, buy_hold_equal(R, 100), cost_rate=0.0)
expect = np.mean(np.prod(1 + R[101:400], axis=0))
check("buy&hold backtest matches hand calculation", abs(np.prod(1 + r) - expect) < 1e-9)

# 3b. min-variance baseline must actually optimise (a past bug made it equal-weight)
from baselines import equal_weight_monthly, min_variance_monthly
Rm = rng.normal(0.0004, 0.012, (600, 6)) * np.array([0.6, 1, 1.4, 0.8, 1.8, 1.1])
wmv = min_variance_monthly(Rm, 300)(300, None)
check("min-variance baseline differs from equal weight and favours the low-vol stock",
      not np.allclose(wmv, equal_weight_monthly(Rm, 300)(300, None)) and wmv[0] == wmv[:-1].max())

# 3c. smoothing: weights stay valid, move only part of the way, and k=1 gives the old behaviour
wd = np.array([0.5, 0.3, 0.2, 0.0]); act = np.array([3.0, -3.0, 0.0, 0.0])
t_ = policy_to_target(act, wd)
check("smoothed target is a valid weight vector", abs(t_.sum() - 1) < 1e-9 and (t_ >= 0).all())
check("smoothing moves only part of the way",
      np.abs(t_ - wd).sum() < np.abs(softmax(act) - wd).sum())
_k = config.POLICY_SMOOTHING; config.POLICY_SMOOTHING = 1.0
check("smoothing = 1.0 reproduces plain softmax", np.allclose(policy_to_target(act, wd), softmax(act)))
config.POLICY_SMOOTHING = _k

# 4. simulator: nominal (strength 0) is deterministic in params; strength 1 varies
calib = calibrate(rng.normal(0.0005, 0.012, (1500, 6)))
p0 = [DomainRandomizer(calib, 0.0).sample_params(rng) for _ in range(5)]
p1 = [DomainRandomizer(calib, 1.0).sample_params(rng) for _ in range(5)]
check("strength 0 -> identical parameters", all(p == p0[0] for p in p0))
check("strength 1 -> parameters vary", len({round(p["vol_scale"], 6) for p in p1}) > 1)
sim = DomainRandomizer(calib, 0.0)
long_path = sim.simulate(sim.sample_params(rng), 20000, rng)
ratio = long_path.std(0).mean() / calib.sigma_d.mean()
check("nominal simulated vol is near calibrated vol", 0.7 < ratio < 1.8, f"(ratio {ratio:.2f})")

# 5. environment mechanics
class _S:
    def sample(self, rng_): return Episode(rng.normal(0, 0.01, (400, 4)), config.WARMUP, 0.001)
env = PortfolioEnv(_S(), 4)
obs, _ = env.reset(seed=0)
done, steps, tot_w_ok = False, 0, True
while not done:
    obs, rew, done, _, info = env.step(env.action_space.sample())
    tot_w_ok &= abs(env.w.sum() - 1) < 1e-9
    steps += 1
check("episode runs to the end, weights always sum to 1", tot_w_ok and steps == 400 - config.WARMUP - 1, f"({steps} steps)")

# 6. known-answer RL test: one stock has a big edge, PPO should find it
if "--fast" not in sys.argv:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    class _Edge:
        def sample(self, rng_):
            T = config.WARMUP + config.EPISODE_LEN + 1
            R_ = rng_.normal(0.0, 0.01, (T, 4))
            R_[:, 0] += 0.002           # stock 0: +0.2%/day edge, same risk
            return Episode(R_, config.WARMUP, 0.001)

    venv = make_vec_env(lambda: PortfolioEnv(_Edge(), 4), n_envs=4, seed=0)
    model = PPO("MlpPolicy", venv, seed=0, n_steps=252, batch_size=252, verbose=0)
    print("training known-answer PPO (200k steps, ~1-3 min) ...")
    model.learn(200_000)
    e = PortfolioEnv(_Edge(), 4)
    ws = []
    for s in range(5):
        o, _ = e.reset(seed=100 + s)
        d_ = False
        while not d_:
            a, _ = model.predict(o, deterministic=True)
            o, _, d_, _, _ = e.step(a)
            ws.append(e.w[0])
    check("PPO puts most weight on the stock with the edge", np.mean(ws) > 0.5,
          f"(avg weight {np.mean(ws):.2f}, equal-weight would be 0.20)")

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED -- fix before trusting any results")
sys.exit(0 if ok_all else 1)