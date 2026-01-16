from __future__ import annotations

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import random

from reach_robotic_hand.algos.ppo.actor_critic import ActorCritic
from reach_robotic_hand.algos.ppo.buffer import RolloutBuffer
from reach_robotic_hand.algos.ppo.ppo import PPO
from reach_robotic_hand.eval.eval import evaluate_policy
from reach_robotic_hand.algos.eppo.eppo_utils import Fitness, copy_params, mutate_offspring, select_elites, make_offspring_from_elites

from reach_robotic_hand.envs.reach_env import ReachEnvWrapper

POP_SIZE = 8
N_ELITES = 4

GENERATIONS = 10
TRAIN_EPOCHS_PER_GEN = 5

STEPS_PER_EPOCH = 4096  
EVAL_EPISODES = 10

SIGMA_W = 0.01
SIGMA_LOGSTD = 0.05
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5

DETERMINISTIC_EVAL = True
OBS_NORM_WARMUP_STEPS = 50_000  
FREEZE_RMS_DURING_EPPO = True


def make_run_dir(base_dir: str, prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_metrics_row(csv_path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    first = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if first:
            w.writeheader()
        w.writerow(row)


def train_one_individual(
    env: ReachEnvWrapper,
    ac: ActorCritic,
    device: str,
    steps_per_epoch: int,
    train_epochs: int,
    ppo_kwargs: Dict[str, Any],
    out_dir: str,
    start_env_steps: int = 0,
) -> Tuple[Dict[str, float], int]:
    """
    Train one individual for `train_epochs` PPO epochs and write metrics.csv in out_dir
    Returns (final_train_stats, total_env_steps_consumed)
    """
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "metrics.csv")

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

    agent = PPO(ac, device=device, **ppo_kwargs)

    buffer = RolloutBuffer(
        buffer_size=steps_per_epoch,
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        device=device,
    )

    obs, _ = env.reset()
    env_steps = int(start_env_steps)

    for epoch in range(train_epochs):
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
            "policy_loss": stats.get("policy_loss", np.nan),
            "value_loss": stats.get("value_loss", np.nan),
            "entropy": stats.get("entropy", np.nan),
            "loss": stats.get("loss", np.nan),
        }
        write_metrics_row(csv_path, fieldnames, row)

    final_train = {
        "train_success_rate": float(row["success_rate"]) if row.get("success_rate") is not None else float("nan"),
        "train_avg_min_goal_dist": float(row["avg_min_goal_dist"]) if row.get("avg_min_goal_dist") is not None else float("nan"),
        "train_avg_ep_return": float(row["avg_ep_return"]) if row.get("avg_ep_return") is not None else float("nan"),
    }
    return final_train, (env_steps - start_env_steps)


def run_eppo(
    results_dir: str = "results/eppo",
    run_name: Optional[str] = None,
    device: str = "cpu",
    seed: int = 0,
    env_id: str = "FetchReachDense-v4",
    success_threshold: float = 0.05,
    max_episode_steps: int = 100,
):
    """
    EPPO main loop.

    Output structure:
      results/eppo/<run_name>/gen_<g>/ind_<i>/metrics.csv
      results/eppo/<run_name>/summary.csv
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if run_name is None:
        run_name = Path(make_run_dir(results_dir, prefix="run")).name

    root = os.path.join(results_dir, run_name)
    ensure_dir(root)

    env_tmp = ReachEnvWrapper(
        env_id=env_id,
        obs_norm=True,
        obs_norm_warmup_steps=OBS_NORM_WARMUP_STEPS,
        max_episode_steps=max_episode_steps,
        success_threshold=success_threshold,
        terminate_on_success=True,
    )

    shared_rms = env_tmp.get_shared_rms()  # optional helper; else use env_tmp._rms
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]


    env_train = ReachEnvWrapper(
        env_id=env_id,
        obs_norm=True,
        obs_norm_warmup_steps=OBS_NORM_WARMUP_STEPS,
        max_episode_steps=max_episode_steps,
        success_threshold=success_threshold,
        terminate_on_success=True,
        shared_rms=shared_rms,
    )

    env_eval = ReachEnvWrapper(
        env_id=env_id,
        obs_norm=True,
        obs_norm_warmup_steps=OBS_NORM_WARMUP_STEPS,
        max_episode_steps=max_episode_steps,
        success_threshold=success_threshold,
        terminate_on_success=True,
        shared_rms=shared_rms,
    )

    env_eval.set_obs_norm_update(False)
    env_train.set_obs_norm_update(True)

    obs_dim = env_train.observation_space.shape[0]
    act_dim = env_train.action_space.shape[0]

    ppo_kwargs = dict(
        lr=3e-5,
        target_kl=0.01,
        value_coef=1.0,
        vf_clip=0.2,
        clip_ratio=0.2,
        train_iters=10,
        minibatch_size=256,
        entropy_coef=0.0,
        max_grad_norm=0.5,
    )

    # Initialize population with identical weights (generation 0)
    population: List[ActorCritic] = []
    base = ActorCritic(obs_dim, act_dim).to(device)
    for _ in range(POP_SIZE):
        m = ActorCritic(obs_dim, act_dim).to(device)
        copy_params(m, base)
        population.append(m)

    summary_csv = os.path.join(root, "summary.csv")
    summary_fields = [
        "gen",
        "ind",
        "env_steps_ind",
        "env_steps",
        "train_success_rate",
        "train_avg_min_goal_dist",
        "train_avg_ep_return",
        "eval_success_rate",
        "eval_avg_min_goal_dist",
        "eval_avg_ep_return",
        "is_elite",
    ]

    if FREEZE_RMS_DURING_EPPO:
        env_train.set_obs_norm_update(True)  # allow warmup updates at the beginning

    env_steps = 0

    for g in range(GENERATIONS):
        gen_dir = os.path.join(root, f"gen_{g}")
        ensure_dir(gen_dir)

        fitness_list: List[Fitness] = []
        train_stats_list: List[Dict[str, float]] = []
        eval_stats_list: List[Dict[str, float]] = []
        env_steps_after_ind: List[int] = []

        # freeze once warmup is passed
        if FREEZE_RMS_DURING_EPPO:
            env_train.maybe_freeze_obs_norm()

        for i in range(POP_SIZE):
            ind_dir = os.path.join(gen_dir, f"ind_{i}")
            ensure_dir(ind_dir)

            # train
            train_stats, steps_used = train_one_individual(
                env=env_train,
                ac=population[i],
                device=device,
                steps_per_epoch=STEPS_PER_EPOCH,
                train_epochs=TRAIN_EPOCHS_PER_GEN,
                ppo_kwargs=ppo_kwargs,
                out_dir=ind_dir,
                start_env_steps=0,
            )

            env_steps += int(steps_used)
            env_steps_after_ind.append(env_steps)

            # evaluate
            eval_stats = evaluate_policy(
                env=env_eval,
                actor_critic=population[i],
                device=device,
                n_episodes=EVAL_EPISODES,
                deterministic=DETERMINISTIC_EVAL,
                seed=seed + 1_000_000 * g + 10_000 * i,
            )

            # Build fitness from eval metrics
            fit = Fitness(
                success_rate=eval_stats["eval_success_rate"],
                avg_min_goal_dist=eval_stats["eval_avg_min_goal_dist"],
                avg_ep_return=eval_stats["eval_avg_ep_return"],
            )

            fitness_list.append(fit)
            train_stats_list.append(train_stats)
            eval_stats_list.append(eval_stats)

        elites = select_elites(fitness_list, N_ELITES)
        elite_set = set(elites)

        for i in range(POP_SIZE):
            row = {
                "gen": g,
                "ind": i,
                "env_steps_ind": int(steps_used),
                "env_steps": env_steps_after_ind[i],
                "train_success_rate": train_stats_list[i]["train_success_rate"],
                "train_avg_min_goal_dist": train_stats_list[i]["train_avg_min_goal_dist"],
                "train_avg_ep_return": train_stats_list[i]["train_avg_ep_return"],
                "eval_success_rate": eval_stats_list[i]["eval_success_rate"],
                "eval_avg_min_goal_dist": eval_stats_list[i]["eval_avg_min_goal_dist"],
                "eval_avg_ep_return": eval_stats_list[i]["eval_avg_ep_return"],
                "is_elite": 1 if i in elite_set else 0,
            }
            write_metrics_row(summary_csv, summary_fields, row)

       
        next_population: List[ActorCritic] = []

        # Keeping elites
        for ei in elites:
            m = ActorCritic(obs_dim, act_dim).to(device)
            copy_params(m, population[ei])
            next_population.append(m)

        # Offspring
        parents = make_offspring_from_elites(elites, POP_SIZE)
        for parent_idx in parents:
            m = ActorCritic(obs_dim, act_dim).to(device)
            copy_params(m, population[parent_idx])
            mutate_offspring(
                m,
                sigma_w=SIGMA_W,
                sigma_logstd=SIGMA_LOGSTD,
                log_std_min=LOG_STD_MIN,
                log_std_max=LOG_STD_MAX,
            )
            next_population.append(m)

        assert len(next_population) == POP_SIZE
        population = next_population

        best = max(range(POP_SIZE), key=lambda i: fitness_list[i].key())
        print(
            f"[Gen {g:03d}] "
            f"Best eval_succ={eval_stats_list[best]['eval_success_rate']:.3f} | "
            f"eval_min_dist={eval_stats_list[best]['eval_avg_min_goal_dist']:.4f} | "
            f"eval_ret={eval_stats_list[best]['eval_avg_ep_return']:+.3f} | "
            f"elites={elites}"
        )

    print(f"EPPO done. Results in: {root}")


if __name__ == "__main__":
    run_eppo(
        results_dir="results/eppo",
        device="cpu",
        seed=0,
        env_id="FetchReachDense-v4",
        success_threshold=0.05,
        max_episode_steps=100,
    )
