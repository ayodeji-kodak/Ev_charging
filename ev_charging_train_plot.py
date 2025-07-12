import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from ev_charging import EVChargingEnv

# === Constants ===
SEED = 42
MODEL_PATH = "ppo_ev_charging_model.zip"
NORM_PATH = "vec_normalize.pkl"

STEP_MINUTES = 5
STEP_HOURS = STEP_MINUTES / 60

# === Helpers ===
def make_env():
    def _init():
        return EVChargingEnv()
    return _init

def make_vec_normalized_env():
    env = DummyVecEnv([make_env()])
    return VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# === Set up environments ===
train_env = make_vec_normalized_env()
eval_env = make_vec_normalized_env()

# === Callbacks ===
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    eval_freq=5000,
    best_model_save_path="./logs/best_model/",
    verbose=1
)

# === Model Training or Loading ===
policy_kwargs = dict(net_arch=[256, 256])

if os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH):
    print("Loading model and normalization stats...")
    train_env = VecNormalize.load(NORM_PATH, DummyVecEnv([make_env()]))
    train_env.training = True
    train_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=train_env)
else:
    print("Training new model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_ev_tensorboard/"
    )

total_timesteps = 100_000
print(f"\nTraining for {total_timesteps} timesteps...\n")
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# === Save model and env ===
model.save(MODEL_PATH)
train_env.save(NORM_PATH)
print("\nModel and normalization stats saved.\n")

# === Load eval env and model ===
eval_env = VecNormalize.load(NORM_PATH, DummyVecEnv([make_env()]))
eval_env.training = False
eval_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=eval_env)

# === Start Evaluation ===
normalized_state = eval_env.reset()
state = normalized_state
raw_state = eval_env.unnormalize_obs(normalized_state)

# === Extract base env and parameters ===
base_env = eval_env.venv.envs[0]
while hasattr(base_env, "env"):
    base_env = base_env.env

battery_capacity = base_env.battery_capacity_kWh
P_available = base_env.P_available
max_charge_power = base_env.max_charge_power
max_heat_power = base_env.max_heat_power
max_cool_power = base_env.max_cool_power

# === Extract observation ===
obs_values = raw_state.flatten()
obs_labels = ["Tbatt", "tarrival", "tdeparture", "SOC", "x_target", "P_available", "trem"]
obs_dict = dict(zip(obs_labels, obs_values))

# === Consistent trem calculation ===
tarrival = obs_dict["tarrival"]
tdeparture = obs_dict["tdeparture"]
initial_trem = (tdeparture - tarrival) % 24  # Wrap-around handling
trem = initial_trem

# === Print Evaluation Info ===
print("\n=== Starting Evaluation ===\n")
print("Observation:")
for k, v in obs_dict.items():
    print(f"  {k}: {v:.3f}")

print("\nEnvironment parameters:")
print(f"  Battery capacity (kWh): {battery_capacity}")
print(f"  Available power (kW): {P_available:.3f}")
print(f"  Max charge power (kW): {max_charge_power}")
print(f"  Max heat power (kW): {max_heat_power}")
print(f"  Max cool power (kW): {max_cool_power}")
print(f"  Initial time remaining (trem): {initial_trem:.2f} hrs")
print("==========================\n")

# === Evaluation Loop ===
done = False
step_count = 0
max_steps = int(24 * 60 / STEP_MINUTES)

soc_history, temp_history, time_history, trem_history = [], [], [], []

print("Evaluating policy...\n")

while not done and step_count < max_steps:
    if trem <= 0:
        print(f"Time remaining reached zero at step {step_count}.")
        break

    action, _ = model.predict(state, deterministic=True)

    charge = np.clip(action[0][0], 0, max_charge_power)
    heat = np.clip(action[0][1], 0, max_heat_power)
    cool = np.clip(action[0][2], 0, max_cool_power)

    # Mutual exclusion of heat/cool
    if heat > cool:
        cool = 0.0
    else:
        heat = 0.0

    clipped_action = np.array([charge, heat, cool], dtype=np.float32)

    result = eval_env.step([clipped_action])

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated[0] or truncated[0]
        info = info[0]
        reward = reward[0]
    else:
        next_state, reward, done_array, info = result
        done = done_array[0]
        info = info[0]
        reward = reward[0]

    state = next_state
    trem = max(0.0, trem - STEP_HOURS)

    soc = info.get("soc", np.nan)
    temp = info.get("tbatt", np.nan)

    print(f"Step {step_count:03d} | SOC: {soc:.3f} | Temp: {temp:.2f} | trem: {trem:.3f} hrs | Reward: {reward:.3f} | Action: {clipped_action}")

    soc_history.append(soc)
    temp_history.append(temp)
    time_history.append(step_count * STEP_MINUTES)
    trem_history.append(trem)

    step_count += 1

# === Results Summary ===
final_soc = soc_history[-1]
target_soc = base_env.x_target
percent_error = 100 * abs(final_soc - target_soc) / target_soc

print("\n===== Final Results =====")
print(f"Final SOC: {final_soc:.3f}")
print(f"Target SOC: {target_soc:.3f}")
print(f"Percent Error: {percent_error:.2f}% {'✅' if percent_error <= 0.1 else '❌'}")
print("==========================\n")

# === Plotting ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(time_history, soc_history, label="SOC")
plt.axhline(target_soc, color="r", linestyle="--", label="Target SOC")
plt.xlabel("Time (minutes)")
plt.ylabel("SOC")
plt.title("SOC Over Time")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(time_history, temp_history, label="Battery Temp")
plt.axhline(25, color="r", linestyle="--", label="Ideal Temp")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Battery Temperature")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(time_history, trem_history, label="Time Remaining")
plt.xlabel("Time (minutes)")
plt.ylabel("trem (hrs)")
plt.title("Time Remaining")
plt.legend()

plt.tight_layout()
plt.show()
