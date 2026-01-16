import os
import csv
from datetime import datetime

import numpy as np
import torch
import random

from reach_robotic_hand.envs.reach_env import ReachEnvWrapper


def make_run_dir(base_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_random(
    steps_per_epoch=4096,
    epochs=400,
    seed=0,
    results_dir="results/random",
    env_id="FetchReachDense-v4",
    obs_norm=True,
    max_episode_steps=100,
    success_threshold=0.05,
    terminate_on_success=True,
):
    """
    Random baseline:
      - samples actions from env.action_space (native bounds)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ReachEnvWrapper(
        env_id=env_id,
        obs_norm=obs_norm,
        max_episode_steps=max_episode_steps,
        success_threshold=success_threshold,
        terminate_on_success=terminate_on_success,
    )

    run_dir = make_run_dir(results_dir)
    csv_path = os.path.join(run_dir, "metrics.csv")

    fieldnames = [
        "epoch",
        "env_steps",
        "avg_ep_return",
        "avg_ep_len",
        "success_rate",
        "avg_final_goal_dist",
        "avg_min_goal_dist",
        "policy_loss",
        "value_loss",
        "entropy",
        "loss",
    ]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    obs, _ = env.reset()
    env_steps = 0

    for epoch in range(epochs):
        ep_returns, ep_lens = [], []
        ep_success, ep_final_dist, ep_min_dist = [], [], []

        ep_ret, ep_len = 0.0, 0
        ep_succ = 0.0
        ep_min_gd = np.inf
        ep_final_gd = np.nan

        for _ in range(steps_per_epoch):
            env_steps += 1

            act = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_len += 1

            if info.get("is_success", 0.0) == 1.0:
                ep_succ = 1.0
            if "goal_dist" in info:
                gd = float(info["goal_dist"])
                ep_min_gd = min(ep_min_gd, gd)
                ep_final_gd = gd

            if done:
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                ep_success.append(ep_succ)
                ep_final_dist.append(ep_final_gd)
                ep_min_dist.append(ep_min_gd)

                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
                ep_succ = 0.0
                ep_min_gd = np.inf
                ep_final_gd = np.nan

        row = {
            "epoch": epoch + 1,
            "env_steps": env_steps,
            "avg_ep_return": float(np.mean(ep_returns)) if ep_returns else np.nan,
            "avg_ep_len": float(np.mean(ep_lens)) if ep_lens else np.nan,
            "success_rate": float(np.mean(ep_success)) if ep_success else np.nan,
            "avg_final_goal_dist": float(np.nanmean(ep_final_dist)) if ep_final_dist else np.nan,
            "avg_min_goal_dist": float(np.nanmean(ep_min_dist)) if ep_min_dist else np.nan,
            "policy_loss": np.nan,
            "value_loss": np.nan,
            "entropy": np.nan,
            "loss": np.nan,
        }

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(
            f"[RANDOM] Epoch {epoch+1:4d}/{epochs} | "
            f"Steps {env_steps:8d} | "
            f"AvgRet {row['avg_ep_return']:+.3f} | "
            f"Succ {row['success_rate']:.3f} | "
            f"FinalDist {row['avg_final_goal_dist']:.4f} | "
            f"MinDist {row['avg_min_goal_dist']:.4f}"
        )

    print(f"Done. Random baseline saved to: {csv_path}")


if __name__ == "__main__":
    run_random(
        steps_per_epoch=4096,
        epochs=400,
        seed=0,
        results_dir="results/random",
        env_id="FetchReachDense-v4",
        obs_norm=True,
        max_episode_steps=100,
        success_threshold=0.05,
        terminate_on_success=True,
    )
