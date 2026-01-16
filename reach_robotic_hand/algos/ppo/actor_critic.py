import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SquashedNormal:
    """
    Tanh-squashed diagonal Gaussian with log-prob correction.

    a = tanh(u),  u ~ N(mu, std)
    log pi(a) = log N(u) - sum log(1 - a^2)
    """

    def __init__(self, mu: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.normal = Normal(mu, std)
        self.eps = eps

    def sample(self):
        u = self.normal.rsample()
        a = torch.tanh(u)
        logp = self.log_prob(a, pre_tanh=u)
        return a, logp

    def log_prob(self, a: torch.Tensor, pre_tanh: torch.Tensor = None) -> torch.Tensor:
        if pre_tanh is None:
            pre_tanh = atanh(a, eps=self.eps)
        logp_u = self.normal.log_prob(pre_tanh).sum(dim=-1)
        correction = torch.log(1.0 - a.pow(2) + self.eps).sum(dim=-1)
        return logp_u - correction

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy().sum(dim=-1)


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic
    Policy: tanh-squashed diagonal Gaussian (see: https://deepwiki.com/hnsqdtt/GRALP/3.2-ppopolicy#action-distribution-with-tanh-squashing)
    clamps log_std into [log_std_min, log_std_max] to prevent excessive exploration
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        log_std_init: float = -1.0,
        log_std_min: float = -3.0,
        log_std_max: float = -0.5,
    ):
        super().__init__()

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h

        self.shared_net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * float(log_std_init))
        self.v_head = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared_net(obs)

    def policy(self, obs: torch.Tensor) -> SquashedNormal:
        feat = self.forward(obs)
        mu = self.mu_head(feat)

        # clamp for exploration control
        log_std = torch.clamp(self.log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)

        return SquashedNormal(mu, std)

    def act(self, obs: torch.Tensor):
        dist = self.policy(obs)
        action, logp = dist.sample()
        value = self.value(obs)
        return action, logp, value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.forward(obs)
        return self.v_head(feat).squeeze(-1)
