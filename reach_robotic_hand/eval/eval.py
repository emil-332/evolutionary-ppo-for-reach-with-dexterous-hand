from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import random


@torch.no_grad()
def evaluate_policy(
    env,
    actor_critic,
    device: str = "cpu",
    n_episodes: int = 30,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate the current policy for n_episodes.

    deterministic:
      If True: uses mean action if the policy distribution exposes .normal.mean / mu-like access.
      If not available, falls back to sampling.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    ep_returns: List[float] = []
    ep_lens: List[int] = []
    ep_success: List[float] = []
    ep_min_dist: List[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=None if seed is None else (seed + 10_000 + ep))
        done = False

        ep_ret = 0.0
        ep_len = 0
        succ = 0.0
        min_gd = float("inf")

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            dist = actor_critic.policy(obs_t)

            if deterministic:
                a = None
                if hasattr(dist, "normal") and hasattr(dist.normal, "mean"):
                    u = dist.normal.mean
                    a = torch.tanh(u)
                if a is None:
                    a, _ = dist.sample()
            else:
                a, _ = dist.sample()

            action = a.squeeze(0).detach().cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_len += 1

            if info.get("is_success", 0.0) == 1.0:
                succ = 1.0
            if "goal_dist" in info:
                gd = float(info["goal_dist"])
                if gd < min_gd:
                    min_gd = gd

        ep_returns.append(ep_ret)
        ep_lens.append(ep_len)
        ep_success.append(succ)
        ep_min_dist.append(min_gd if np.isfinite(min_gd) else np.nan)

    return {
        "eval_avg_ep_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "eval_avg_ep_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "eval_success_rate": float(np.mean(ep_success)) if ep_success else float("nan"),
        "eval_avg_min_goal_dist": float(np.nanmean(ep_min_dist)) if ep_min_dist else float("nan"),
    }
