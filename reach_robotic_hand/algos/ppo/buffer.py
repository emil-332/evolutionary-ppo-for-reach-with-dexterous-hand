import torch


class RolloutBuffer:
    """
    Stores trajectories for PPO.
    Computes discounted returns and advantages via GAE
    """

    def __init__(self, buffer_size, obs_dim, act_dim, gamma=0.99, lam=0.95, device="cpu"):
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)
        self.ptr = 0
        self.max_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def store(self, obs, act, logp, val, rew, done):
        assert self.ptr < self.max_size, "Buffer overflow!"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.log_probs[self.ptr] = logp
        self.values[self.ptr] = val
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_value=0.0):
        rewards = self.rewards[:self.ptr]
        values = self.values[:self.ptr]
        dones = self.dones[:self.ptr]

        adv = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            adv[t] = last_gae

        self.advantages[:self.ptr] = adv
        self.returns[:self.ptr] = adv + values

    def get(self):
        assert self.ptr == self.max_size, "Buffer not full!"

        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages = adv

        return {
            "obs": self.obs.to(self.device),
            "actions": self.actions.to(self.device),
            "log_probs": self.log_probs.to(self.device),
            "returns": self.returns.to(self.device),
            "advantages": self.advantages.to(self.device),
            "values": self.values.to(self.device),
        }
