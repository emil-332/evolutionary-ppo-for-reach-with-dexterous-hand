import os
import csv
from datetime import datetime

import torch
import numpy as np
import random

from reach_robotic_hand.algos.ppo.actor_critic import ActorCritic
from reach_robotic_hand.algos.ppo.buffer import RolloutBuffer
from reach_robotic_hand.algos.ppo.ppo import PPO
from reach_robotic_hand.envs.reach_env import ReachEnvWrapper


def make_run_dir(base_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_training(
    steps_per_epoch=4096,
    epochs=400,
    seed=0,
    device="cpu",
    results_dir="results/baseline",
):
    env = ReachEnvWrapper(
        env_id="FetchReachDense-v4",
        obs_norm=True,
        obs_norm_warmup_steps=50_000,
        max_episode_steps=100,
        success_threshold=0.05,
        terminate_on_success=True,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ac = ActorCritic(obs_dim, act_dim).to(device)
    agent = PPO(ac, device=device, lr=3e-5, target_kl=0.01, value_coef=1.0, vf_clip=0.2)

    buffer = RolloutBuffer(
        buffer_size=steps_per_epoch,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
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
        buffer.ptr = 0

        ep_returns, ep_lens = [], []
        ep_success, ep_final_dist, ep_min_dist = [], [], []

        ep_ret, ep_len = 0.0, 0
        ep_succ = 0.0
        ep_min_gd = np.inf
        ep_final_gd = np.nan

        for _ in range(steps_per_epoch):
            env_steps += 1

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            act, logp, val = ac.act(obs_t)
            act = act.squeeze(0)
            logp = logp.squeeze(0)
            val = val.squeeze(0)

            next_obs, reward, terminated, truncated, info = env.step(act.detach().cpu().numpy())
            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_len += 1

            if info.get("is_success", 0.0) == 1.0:
                ep_succ = 1.0
            if "goal_dist" in info:
                gd = float(info["goal_dist"])
                ep_min_gd = min(ep_min_gd, gd)
                ep_final_gd = gd

            buffer.store(
                obs=obs_t.squeeze(0).detach().cpu(),
                act=act.detach().cpu(),
                logp=logp.detach().cpu(),
                val=val.detach().cpu(),
                rew=float(reward),
                done=float(done),
            )

            obs = next_obs

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

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_val = float(ac.value(obs_t).item())

        buffer.finish_path(last_value=last_val)

        stats = agent.update(buffer.get())

        row = {
            "epoch": epoch + 1,
            "env_steps": env_steps,
            "avg_ep_return": float(np.mean(ep_returns)) if ep_returns else np.nan,
            "avg_ep_len": float(np.mean(ep_lens)) if ep_lens else np.nan,
            "success_rate": float(np.mean(ep_success)) if ep_success else np.nan,
            "avg_final_goal_dist": float(np.nanmean(ep_final_dist)) if ep_final_dist else np.nan,
            "avg_min_goal_dist": float(np.nanmean(ep_min_dist)) if ep_min_dist else np.nan,
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "loss": stats["loss"],
        }

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(
            f"Epoch {epoch+1:4d}/{epochs} | "
            f"Steps {env_steps:8d} | "
            f"AvgRet {row['avg_ep_return']:+.3f} | "
            f"Succ {row['success_rate']:.3f} | "
            f"FinalDist {row['avg_final_goal_dist']:.4f} | "
            f"MinDist {row['avg_min_goal_dist']:.4f} | "
        )

    print(f"Done. Metrics saved to: {csv_path}")


if __name__ == "__main__":
    run_training()
