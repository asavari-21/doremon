import argparse
import os

# PPO with a small MLP is as fast or faster on CPU. Hide any GPU so PyTorch does not
# print the "NVIDIA driver too old" warning. Set CUDA_VISIBLE_DEVICES yourself to override.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import config
from data import load_data
from env import PortfolioEnv
from sources import build_source, strategy_name


def model_path(mode, strength, seed, out_dir="models"):
    return os.path.join(out_dir, f"{strategy_name(mode, strength)}_seed{seed}.zip")


def train_one(mode, seed, timesteps, strength, data, out_dir="models", n_envs=None, verbose=0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    n_envs = n_envs or config.N_ENVS
    R_train = data["R"][: data["i_val"]]
    N = R_train.shape[1]
    source = build_source(mode, R_train, strength)
    set_random_seed(seed)
    venv = make_vec_env(lambda: PortfolioEnv(source, N), n_envs=n_envs, seed=seed)
    ppo_kw = dict(config.PPO_KW)
    ppo_kw["n_steps"] = max(16, config.PPO_KW["n_steps"] * config.N_ENVS // n_envs)
    ppo_kw["batch_size"] = min(config.PPO_KW["batch_size"], ppo_kw["n_steps"] * n_envs)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=verbose, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=config.NET_ARCH, vf=config.NET_ARCH)),
                **ppo_kw)
    model.learn(total_timesteps=timesteps)
    os.makedirs(out_dir, exist_ok=True)
    path = model_path(mode, strength, seed, out_dir)
    model.save(path)
    venv.close()
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dr", choices=["hist", "sim", "dr", "mixed"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--timesteps", type=int, default=config.TIMESTEPS)
    ap.add_argument("--dr-strength", type=float, default=1.0)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    d = load_data(synthetic=a.synthetic)
    print("saved:", train_one(a.mode, a.seed, a.timesteps, a.dr_strength, d, verbose=1))