import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from ev_charging import EVChargingEnv  # assumes your environment is defined in ev_charging.py

# === Configuration ===
SEED = 42
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5000
STEP_MINUTES = 5
STEP_HOURS = STEP_MINUTES / 60

# === Paths ===
MODEL_PATH = "ppo_ev_charging_model.zip"
NORM_PATH = "vec_normalize.pkl"
CHECKPOINT_PATH = "./logs/checkpoints/"
BEST_MODEL_PATH = "./logs/best_model/"
TENSORBOARD_LOG = "./ppo_ev_tensorboard/"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

# === Environment Wrappers ===
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

# === Custom Eval Callback ===
class RewardLoggingEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_rewards = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            self.eval_rewards.append(self.last_mean_reward)
        return result

# === Initialize Environments ===
train_env = make_vec_env(training=True)
eval_env = make_vec_env(training=False)

# === Callbacks ===
eval_callback = RewardLoggingEvalCallback(
    eval_env=eval_env,
    eval_freq=EVAL_FREQ,
    best_model_save_path=BEST_MODEL_PATH,
    verbose=1
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

# === Load or Initialize Model ===
if os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH):
    print("Loading saved model and normalization stats...")
    train_env = VecNormalize.load(NORM_PATH, DummyVecEnv([make_env()]))
    train_env.training = True
    train_env.norm_reward = True

    model = PPO.load(MODEL_PATH, env=train_env)

    eval_env = VecNormalize.load(NORM_PATH, DummyVecEnv([make_env()]))
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.reset()
else:
    print("Initializing new PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
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
        learning_rate=lambda f: f * 1.5e-4,
    )

# === Train Model ===
print(f"=== Starting training for {TOTAL_TIMESTEPS:,} timesteps ===")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# === Save Model and Normalization ===
model.save(MODEL_PATH)
train_env.save(NORM_PATH)
print("\nTraining complete. Model and normalization saved.")

# === Plot Rewards ===
plt.figure(figsize=(10, 6))
timesteps = np.arange(len(eval_callback.eval_rewards)) * EVAL_FREQ
plt.plot(timesteps, eval_callback.eval_rewards, label="Eval Mean Reward")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Evaluation Reward Progress")
plt.grid(True)
plt.tight_layout()
plt.savefig("eval_rewards.png")
plt.show()
