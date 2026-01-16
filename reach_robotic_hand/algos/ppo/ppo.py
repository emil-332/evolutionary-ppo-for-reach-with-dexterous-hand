import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    """
    Minimal PPO agent with KL early stopping.

    - value function clipping (PPO-style) to stabilize critic
    """

    def __init__(
        self,
        actor_critic,
        clip_ratio=0.2,
        lr=3e-5,
        train_iters=10,
        minibatch_size=256,
        value_coef=1.0,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        device="cpu",
        target_kl=0.01,
        vf_clip=0.2,
    ):
        self.ac = actor_critic.to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.target_kl = target_kl
        self.vf_clip = float(vf_clip)

    def update(self, data):
        obs = data["obs"]
        actions = data["actions"]
        old_logp = data["log_probs"]
        returns = data["returns"]
        advantages = data["advantages"]
        old_values = data["values"]

        total_steps = obs.shape[0]

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
            "n_minibatches": 0,
            "early_stop": 0,
        }

        for _ in range(self.train_iters):
            idxs = torch.randperm(total_steps, device=obs.device)

            kl_sum = 0.0
            kl_count = 0

            for start in range(0, total_steps, self.minibatch_size):
                mb_idx = idxs[start : start + self.minibatch_size]

                mb_obs = obs[mb_idx]
                mb_act = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_old_v = old_values[mb_idx]

                # policy
                dist = self.ac.policy(mb_obs)
                logp = dist.log_prob(mb_act)
                ratio = torch.exp(logp - mb_old_logp)

                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()

                # value 
                v_pred = self.ac.value(mb_obs)
                v_pred_clipped = mb_old_v + torch.clamp(v_pred - mb_old_v, -self.vf_clip, self.vf_clip)

                vf_loss1 = (mb_ret - v_pred).pow(2)
                vf_loss2 = (mb_ret - v_pred_clipped).pow(2)
                value_loss = torch.max(vf_loss1, vf_loss2).mean()

                # entropy
                entropy = dist.entropy().mean()

                # total loss 
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # KL for early stopping
                with torch.no_grad():
                    approx_kl = (mb_old_logp - logp).mean()
                    kl_sum += float(approx_kl.abs().item())
                    kl_count += 1

                    stats["policy_loss"] += float(policy_loss.item())
                    stats["value_loss"] += float(value_loss.item())
                    stats["entropy"] += float(entropy.item())
                    stats["loss"] += float(loss.item())
                    stats["n_minibatches"] += 1

            mean_kl = kl_sum / max(kl_count, 1)
            if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                stats["early_stop"] = 1
                break

        n = max(stats["n_minibatches"], 1)
        for k in ["policy_loss", "value_loss", "entropy", "loss"]:
            stats[k] /= n

        return stats
