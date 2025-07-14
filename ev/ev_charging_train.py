import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from ev_charging import EVChargingEnv  # Update if the env is in a different file

# === Config ===
SEED = 42
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 5000
STEP_MINUTES = 5
MODEL_PATH = "ppo_ev_final"
NORM_PATH = "vec_normalize.pkl"
CHECKPOINT_PATH = "./logs/checkpoints/"
BEST_MODEL_PATH = "./logs/best_model/"
TENSORBOARD_LOG = "./tensorboard_logs/ppo_ev/"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

# === Environment Setup ===
def make_env():
    def _init():
        env = EVChargingEnv()
        return Monitor(env)
    return _init

def make_vec_env(training=True):
    env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    vec_env.training = training
    vec_env.norm_reward = training
    return vec_env

# Check if environment is valid
check_env(EVChargingEnv(), warn=True)

# === Custom Evaluation Callback with Reward Logging ===
class RewardLoggingEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_rewards = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            self.eval_rewards.append(self.last_mean_reward)
        return result

# === Create Environments ===
train_env = make_vec_env(training=True)
eval_env = make_vec_env(training=False)

# === Callbacks ===
eval_callback = RewardLoggingEvalCallback(
    eval_env=eval_env,
    eval_freq=EVAL_FREQ,
    best_model_save_path=BEST_MODEL_PATH,
    log_path="./logs/ppo_ev",
    deterministic=True,
    render=False,
    callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=100.0, verbose=1),
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=CHECKPOINT_PATH,
    name_prefix="ppo_ev"
)

callback = CallbackList([eval_callback, checkpoint_callback])

# === PPO Policy Configuration ===
policy_kwargs = dict(
    net_arch=[256, 256, 128],
    activation_fn=torch.nn.ReLU
)

# === Model Initialization ===
print("Initializing PPO model...")
model = PPO(
    "MlpPolicy",
    train_env,
    seed=SEED,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=lambda f: f * 1.5e-4,
    n_steps=4096,
    batch_size=128,
    n_epochs=15,
    gamma=0.98,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.02,
    vf_coef=2.0,
    max_grad_norm=0.5,
    tensorboard_log=TENSORBOARD_LOG,
)

# === Training ===
print(f"\n=== Training for {TOTAL_TIMESTEPS:,} timesteps ===")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# === Save final model and normalization stats ===
model.save(MODEL_PATH)
train_env.save(NORM_PATH)
print("\nTraining complete. Model and VecNormalize stats saved.")

# === Evaluation ===
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"\nFinal Evaluation: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")

# === Plot evaluation reward trend ===
plt.figure(figsize=(10, 6))
timesteps = np.arange(len(eval_callback.eval_rewards)) * EVAL_FREQ
plt.plot(timesteps, eval_callback.eval_rewards, label="Evaluation Reward")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Evaluation Reward Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("eval_rewards.png")
plt.show()
