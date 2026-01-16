import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional

class RunningMeanStd:
    """
    Numerically stable running mean / variance estimator (Welford).
    Per-sample updates.
    """

    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x
        batch_var = np.zeros_like(x, dtype=np.float64)
        batch_count = 1.0
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)

        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ReachEnvWrapper(gym.Wrapper):
    """
    Wrapper for Gymnasium-Robotics goal-conditioned envs (FetchReachDense-v4 etc.)

    Provides:
      - deterministic dict->flat observation
      - optional observation normalization (RunningMeanStd)
          * warmup: update rms for first obs_norm_warmup_steps observations
          * freeze: stop updating after warmup (or when manually disabled)
      - info["goal_dist"], info["is_success"]
      - reward shaping
      - optional terminate-on-success

    Notes on obs_norm:
      - This wrapper normalizes the FLATTENED observation vector.
      - RMS update is controlled by:
          (a) self._obs_norm_update_enabled AND
          (b) warmup budget not exhausted
    """

    def __init__(
        self,
        env_id: str = "FetchReachDense-v4",
        obs_norm: bool = False,
        obs_norm_warmup_steps: int = 0,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        success_threshold: float = 0.05,
        terminate_on_success: bool = True,
        shared_rms: Optional[RunningMeanStd] = None,
    ):
        make_kwargs = {}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps

        env = gym.make(env_id, **make_kwargs)
        super().__init__(env)

        self.env_id = env_id
        self.obs_norm = bool(obs_norm)
        self.reward_scale = float(reward_scale)
        self.success_threshold = float(success_threshold)
        self.terminate_on_success = bool(terminate_on_success)

        self.obs_norm_warmup_steps = int(obs_norm_warmup_steps) if obs_norm_warmup_steps is not None else 0
        self._obs_seen = 0
        self._obs_norm_update_enabled = True

        # Build flattened observation space deterministically
        example_obs, _ = self.env.reset()
        flat = self.flatten_obs(example_obs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32
        )

        # Action space
        self.action_space = self.env.action_space
        self._act_low = self.action_space.low.astype(np.float32)
        self._act_high = self.action_space.high.astype(np.float32)

        # Mapping tanh-actions (-1,1) to [low,high] if needed
        self._act_mid = (self._act_high + self._act_low) / 2.0
        self._act_half_range = (self._act_high - self._act_low) / 2.0

        # RMS (either per-env or shared)
        if self.obs_norm:
            if shared_rms is not None:
                self._rms = shared_rms
            else:
                self._rms = RunningMeanStd(shape=flat.shape)
        else:
            self._rms = None

    def set_obs_norm_update(self, enabled: bool) -> None:
        """Enable/disable RMS updates (normalization still applied using frozen stats)."""
        self._obs_norm_update_enabled = bool(enabled)

    def maybe_freeze_obs_norm(self) -> None:
        """
        Freeze RMS updates once warmup steps are reached.
        If obs_norm_warmup_steps <= 0, this does nothing.
        """
        if self._rms is None:
            return
        if self.obs_norm_warmup_steps <= 0:
            return
        if self._obs_seen >= self.obs_norm_warmup_steps:
            self._obs_norm_update_enabled = False

    def flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """Deterministic dict flattening (sorted keys)."""
        if isinstance(obs, dict):
            pieces = []
            for key in sorted(obs.keys()):
                value = obs[key]
                if isinstance(value, dict):
                    pieces.append(self.flatten_obs(value))
                else:
                    pieces.append(np.asarray(value, dtype=np.float32).ravel())
            return np.concatenate(pieces, axis=0).astype(np.float32)
        return np.asarray(obs, dtype=np.float32).ravel()

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._rms is None:
            return obs.astype(np.float32)

        self._obs_seen += 1

        if self.obs_norm_warmup_steps > 0 and self._obs_seen >= self.obs_norm_warmup_steps:
            self._obs_norm_update_enabled = False

        if self._obs_norm_update_enabled:
            self._rms.update(obs)

        std = np.sqrt(self._rms.var) + 1e-8
        return ((obs - self._rms.mean) / std).astype(np.float32)

    def _compute_goal_metrics(self, obs_dict: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obs_dict, dict):
            return info

        if "achieved_goal" in obs_dict and "desired_goal" in obs_dict:
            ag = np.asarray(obs_dict["achieved_goal"], dtype=np.float32).ravel()
            dg = np.asarray(obs_dict["desired_goal"], dtype=np.float32).ravel()
            goal_dist = float(np.linalg.norm(ag - dg, ord=2))
            is_success = float(goal_dist < self.success_threshold)

            info["goal_dist"] = goal_dist
            info["is_success"] = is_success

        return info


    ### GYM API

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        info = dict(info) if isinstance(info, dict) else {}
        info = self._compute_goal_metrics(obs_dict, info)

        flat = self.flatten_obs(obs_dict)
        flat = self._normalize_obs(flat)
        return flat, info

    def _map_action_to_env_bounds(self, action: np.ndarray) -> np.ndarray:
        if np.allclose(self._act_low, -1.0) and np.allclose(self._act_high, 1.0):
            return action
        return self._act_mid + action * self._act_half_range

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = self._map_action_to_env_bounds(action)
        action = np.clip(action, self._act_low, self._act_high)

        obs_dict, _, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        info = self._compute_goal_metrics(obs_dict, info)

        if self.terminate_on_success and info.get("is_success", 0.0) == 1.0:
            terminated = True

        flat = self.flatten_obs(obs_dict)
        flat = self._normalize_obs(flat)

        goal_dist = float(info.get("goal_dist", 0.0))
        reward = -np.sqrt(goal_dist + 1e-6)
        if info.get("is_success", 0.0) == 1.0:
            reward += 1.0
        reward *= self.reward_scale

        return flat, float(reward), bool(terminated), bool(truncated), info
    
    def get_shared_rms(self) -> Optional[RunningMeanStd]:
        return self._rms

