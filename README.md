# Reach Robotic Hand - PPO and Evolutionary PPO (EPPO) on FetchReachDense-v4

This repository contains a small, reproducible reinforcement learning project for goal-reaching with the Gymnasium Robotics environment FetchReachDense-v4. It includes:

- A PPO algorithm with a few stabilizing adaptations
- An Evolutionary PPO (EPPO) variant that combines PPO updates with population-based selection and mutation
- A Random policy baseline for sanity checking

---

## 1. Environment and Task

Agents are trained on a goal-conditioned reaching task. For full environment description see https://robotics.farama.org/envs/shadow_dexterous_hand/reach/. 

- Environment: `FetchReachDense-v4` (Gymnasium Robotics).
- Success: Achieved goal is within a configurable `success_threshold` of the desired goal.
- Custom Wrapper: `ReachEnvWrapper` in `reach_robotic_hand/envs/reach_env.py`:
  - Flattens dict observations into a deterministic 1D vector.
  - Optionally normalizes observations via a Running Mean/Std (RMS) estimator.
  - Adds `info["goal_dist"]` and `info["is_success"]`.
  - Supports `terminate_on_success` for shorter episodes when success is reached.
  - Applies a shaped dense reward derived from goal distance.

---

## 2. Repository Structure
```text
reach_robotic_hand/
├─ algos/
│  ├─ ppo/
│  │  ├─ __init__.py
│  │  ├─ actor_critic.py
│  │  ├─ buffer.py
│  │  ├─ ppo.py
│  │  └─ train_ppo.py
│  ├─ eppo/
│  │  ├─ __init__.py
│  │  ├─ eppo_utils.py
│  │  └─ train_eppo.py
│  └─ random/
│     ├─ __init__.py
│     └─ train_random.py
├─ envs/
│  ├─ __init__.py
│  └─ reach_env.py
├─ eval/
│  ├─ __init__.py
│  └─ eval.py
├─ utils/
│  ├─ __init__.py
│  └─ plot.py
└─ results/
   ├─ baseline/
   ├─ random/
   ├─ eppo/
   └─ plots/
```
---

## 3. PPO Baseline

### 3.1 Core PPO components
- Policy/value network: `ActorCritic` (`algos/ppo/actor_critic.py`)
- Rollout storage + GAE: `RolloutBuffer` (`algos/ppo/buffer.py`)
- PPO update rule: `PPO` (`algos/ppo/ppo.py`)
- Training script: `train_ppo.py`

### 3.2 PPO adaptations made in this project
This PPO algorithm includes a few modifications to improve stability on tight-reaching thresholds:

1. Tanh-squashed Gaussian policy
   - Actions are produced via a Gaussian in pre-tanh space and then squashed using `tanh`.
   - Log-probabilities include the standard squashing correction term.
   - This ensures actions remain bounded and avoids mismatch with the environment action space.

2. Clamped exploration via `log_std` bounds
   - The policy maintains a learned diagonal log standard deviation parameter (`log_std`).
   - To prevent pathological exploration collapse/explosion, `log_std` is clamped into `[log_std_min, log_std_max]` each forward pass.

3. KL early stopping
   - An approximate KL is computed internally for early stopping only.
   - The KL diagnostic is not logged to CSV by default.

---

## 4. Random Baseline

The random baseline is a non-learning reference:

- Samples actions directly from `env.action_space.sample()`.
- Uses the same wrapper, reward shaping, termination behavior, and CSV schema.
- Loss/entropy columns are written as `NaN` so plotting logic remains consistent.

This baseline is useful to confirm that success is non-trivial and that learning algorithms provide a clear improvement.

---

## 5. EPPO: Evolutionary PPO

EPPO is a simple hybrid algorithm combining:

- Gradient-based improvement (PPO) within each individual, and
- Evolutionary selection/mutation across a population.

### 5.1 High-level idea
Each generation:
1. Maintain a population of `POP_SIZE` actor–critic models
2. Train each individual for a fixed amount of on-policy PPO interaction
3. Evaluate each individual in a separate evaluation environment
4. Select the top `N_ELITES` individuals
5. Create offspring by copying elite weights and applying Gaussian mutation to offspring only.
6. Repeat.

### 5.2 What changed vs PPO
Compared to standard single-agent PPO training:

1. Population
   - Instead of a single model, EPPO maintains `POP_SIZE` models.

2. Selection mechanism
   - Individuals are ranked by a lexicographic fitness:
     1) higher success rate  
     2) lower average minimum distance to goal  
     3) higher average return  
   - Implemented in `algos/eppo/eppo_utils.py`.

3. Mutation
   - Offspring are created by:
     - copying a randomly chosen elite, then
     - adding Gaussian noise to parameters (`sigma_w`) and to `log_std` separately (`sigma_logstd`).
   - `log_std` is clamped after mutation into `[LOG_STD_MIN, LOG_STD_MAX]`.

4. Optimizer reset
   - Offspring do not inherit Adam optimizer state.
   - Each individual’s PPO optimizer state is re-initialized when training begins (no optimizer carryover).

5. Train vs Eval split
   - EPPO computes both:
     - training metrics (from rollouts used to update PPO), and
     - evaluation metrics (from rollouts used only to assess fitness).
   - the hold-out of rollouts is used for unbiased evaluation of the current policy.

---

## 6. Running Experiments

### 6.1 PPO
  ```bash
  python -m reach_robotic_hand.algos.ppo.train_ppo
```

### 6.2 Random baseline
  ```bash
  python -m reach_robotic_hand.algos.ppo.train_random
```
### 6.3 EPPO
  ```bash 
  python -m reach_robotic_hand.algos.ppo.train_eppo
```
## 7. Results

TODO 




