import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ev_charging import EVChargingEnv

# === Constants ===
STEP_MINUTES = 5
STEP_HOURS = STEP_MINUTES / 60
MODEL_PATH = "ppo_ev_charging_model.zip"
NORM_PATH = "vec_normalize.pkl"

# === Helpers ===
def make_env(seed=None):
    def _init():
        env = EVChargingEnv()
        env.seed(seed)
        return env
    return _init


# === Set seed and reset properly ===
import random
seed = random.randint(0, 10000)
print(f"Using random seed: {seed}")

# === Load eval env and model ===
eval_env = VecNormalize.load(NORM_PATH, DummyVecEnv([make_env(seed)]))
eval_env.training = False
eval_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=eval_env)

# === Start Evaluation ===


# Seed base env manually
base_env = eval_env.venv.envs[0]
while hasattr(base_env, "env"):
    base_env = base_env.env
base_env.seed(seed)

# Reset using the seed in VecNormalize (indirectly)
obs_raw, _ = base_env.reset(seed=seed) # Pass seed here to reset RNG properly
# Normalize observation using VecNormalize
obs = eval_env.normalize_obs(obs_raw)
state = obs

raw_state = eval_env.unnormalize_obs(obs)

# === Extract base env and parameters ===
base_env = eval_env.venv.envs[0]
while hasattr(base_env, "env"):
    base_env = base_env.env

battery_capacity = base_env.battery_capacity_kWh
P_available = base_env.P_available
max_charge_power = base_env.max_charge_power
max_heat_power = base_env.max_heat_power
max_cool_power = base_env.max_cool_power
internal_resistance = base_env.internal_resistance  # corrected attribute for internal resistance

# Manually set estimated battery thermal capacity (kJ/°C)
thermal_capacity = 5000  # Example value, adjust as needed

# === Extract observation ===
obs_values = raw_state.flatten()
obs_labels = ["Tbatt", "tarrival", "tdeparture", "SOC", "x_target", "P_available", "trem"]
obs_dict = dict(zip(obs_labels, obs_values))

tarrival = obs_dict["tarrival"]
tdeparture = obs_dict["tdeparture"]
initial_trem = (tdeparture - tarrival) % 24
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

    # === Predict Action ===
    action, _ = model.predict(state, deterministic=True)
    action = np.array(action).flatten()

    raw_charge = action[0]
    raw_heat = action[1]
    raw_cool = action[2]

    current_temp = eval_env.get_attr("Tbatt")[0]

    charge = np.clip(raw_charge, 0, max_charge_power)
    heat = np.clip(raw_heat, 0, max_heat_power)
    cool = np.clip(raw_cool, 0, max_cool_power)

    if heat > cool:
        cool = 0.0
    else:
        heat = 0.0

    if current_temp < 22:
        charge = 0.0
        cool = 0.0
    elif current_temp > 28:
        charge = 0.0
        heat = 0.0

    total_requested_power = charge + heat + cool
    if total_requested_power > P_available:
        scale = P_available / total_requested_power
        charge *= scale
        heat *= scale
        cool *= scale

    clipped_action = np.array([[charge, heat, cool]], dtype=np.float32)

    # Fixed unpacking here for gym < 0.26
    next_state, reward, done, info = eval_env.step(clipped_action)

    # Unpack from lists (VecEnv wraps outputs in lists/arrays)
    done = done[0] if isinstance(done, (list, np.ndarray)) else done
    reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
    info = info[0] if isinstance(info, (list, np.ndarray)) else info

    state = next_state

    trem = max(0.0, trem - STEP_HOURS)
    soc = info.get("soc", np.nan)
    temp = info.get("tbatt", np.nan)

    # Calculate heat generated from internal resistance: I^2 * R
    # Assuming current I = charge power / battery voltage (assuming 400V nominal)
    battery_voltage = 400  # Volts, assumed nominal
    current = charge * 1000 / battery_voltage  # Convert kW to W before dividing
    heat_internal_resistance = (current ** 2) * internal_resistance  # Watts

    # Convert heat power (W) to energy for this timestep (Joules)
    # Energy = Power * time (seconds)
    step_seconds = STEP_HOURS * 3600
    energy_joules = heat_internal_resistance * step_seconds  # J

    # Convert energy (J) to kJ
    energy_kj = energy_joules / 1000

    # Calculate temperature increase (°C) = energy (kJ) / thermal capacity (kJ/°C)
    temp_increase_C = energy_kj / thermal_capacity

    print(
        f"Step {step_count:03d} | SOC: {soc:.3f} | Batt Temp: {temp:.2f} | trem: {trem:.3f} hrs | "
        f"Reward: {reward:.3f} | Action: [{charge:.2f}, {heat:.2f}, {cool:.2f}] | "
        f"Heat from internal resistance: {heat_internal_resistance/1000:.4f} kW | "
        f"Estimated Temp Increase: {temp_increase_C:.4f} °C"
    )

    soc_history.append(soc)
    temp_history.append(temp)
    time_history.append(step_count * STEP_MINUTES)
    trem_history.append(trem)

    step_count += 1

# === Final Results ===
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