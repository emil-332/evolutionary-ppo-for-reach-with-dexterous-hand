from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import random

@dataclass
class Fitness:
    """
    Lexicographic fitness:
      1) success_rate
      2) avg_min_goal_dist
      3) avg_ep_return 
    """
    success_rate: float
    avg_min_goal_dist: float
    avg_ep_return: float

    def key(self) -> Tuple[float, float, float]:
        # descending by success, descending by negative distance, descending by return:
        return (self.success_rate, -self.avg_min_goal_dist, self.avg_ep_return)


@torch.no_grad()
def copy_params(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Copy parameters and buffers from src -> dst."""
    dst.load_state_dict(src.state_dict(), strict=True)


@torch.no_grad()
def mutate_offspring(
    model: torch.nn.Module,
    sigma_w: float = 0.01,
    sigma_logstd: float = 0.05,
    log_std_name: str = "log_std",
    log_std_min: float = -3.0,
    log_std_max: float = -0.5,
) -> None:
    """
    Apply Gaussian noise to parameters (offspring only).
    - Parameters named 'log_std' get sigma_logstd and are clamped.
    - All other floating parameters get sigma_w.
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.is_floating_point(p.data):
            continue

        std = sigma_logstd if name.endswith("log_std") or name == "log_std" else sigma_w
        noise = torch.randn_like(p.data) * std
        p.data.add_(noise)

        if log_std_name in name:
            p.data.clamp_(log_std_min, log_std_max)

def select_elites(
    fitness_list: List[Fitness],
    k: int,
) -> List[int]:
    """
    Return indices of top-k elites by lexicographic fitness.
    """
    assert k <= len(fitness_list)
    order = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i].key(), reverse=True)
    return order[:k]


def make_offspring_from_elites(
    elite_indices: List[int],
    pop_size: int,
) -> List[int]:
    """
    Returns a parent index for each offspring slot (size pop_size - len(elites)).
    Offspring parents are sampled uniformly from elites.
    """
    n_elites = len(elite_indices)
    n_offspring = pop_size - n_elites
    parents = [random.choice(elite_indices) for _ in range(n_offspring)]
    return parents
