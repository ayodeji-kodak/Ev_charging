import gymnasium as gym
import numpy as np
from pprint import pprint
import os

from sustaingym.envs.evcharging.ev_env_soc import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# === Load the trained model ===
model_path = "./soc_evcharging_logs/soc_evcharging_final.zip"  # adjust if different
model = PPO.load(model_path)
print(f"\n✅ Loaded trained model from: {model_path}")

# === Recreate the same environment used during training ===
trace_gen = RealTraceGenerator(
    site='caltech',
    date_period=('2019-05-01', '2019-08-31'),
    sequential=True,
    use_unclaimed=False,
    requested_energy_cap=100,
    seed=42
)

env = EVChargingEnv(
    data_generator=trace_gen,
    moer_forecast_steps=36,
    project_action_in_env=True,
    verbose=1
)

check_env(env, warn=True)

# === Reset the environment ===
obs, info = env.reset()
print("\n=== Environment Reset ===\n")

# === Run the evaluation ===
for timestep in range(288):
    # Predict action using the trained model
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Only print if there's non-zero demand
    if 'demands' in obs and np.any(obs['demands'] > 0):
        print(f"\n\n=== TIMESTEP {timestep} WITH DEMAND {'='*40}")
        
        print("\n--- DEMAND INFORMATION ---")
        demand_stations = np.where(obs['demands'] > 0)[0]
        num_demands = len(demand_stations)
        print(f"Number of stations with demand: {num_demands}")
        print(f"Stations with demand: {demand_stations}")
        print(f"Demand values: {obs['demands'][demand_stations]}")

        print("\n--- NON-ZERO VALUES ONLY ---")
        arrays_to_check = ['est_departures', 'demands', 'arrival_soc', 'target_soc', 'current_soc']
        for array_name in arrays_to_check:
            if array_name in obs:
                array = obs[array_name]
                non_zero_indices = np.where(array != 0)[0] if array_name != 'est_departures' else np.where(array != -288)[0]
                if len(non_zero_indices) > 0:
                    print(f"\n{array_name}:")
                    for idx in non_zero_indices:
                        print(f"  Station {idx}: {array[idx]}")
                else:
                    print(f"\n{array_name}: All values are zero")

        print("\n--- FULL OBSERVATION ---")
        for key, value in obs.items():
            print(f"\n{key}:")
            print(f"Shape: {np.shape(value)}")
            print(f"Dtype: {np.array(value).dtype}")
            print(f"Number of elements: {len(value)}")
            print("Values:")
            pprint(value)

        print("\n--- COMPLETE INFO DICTIONARY ---")
        for key, value in info.items():
            print(f"\n{key}:")
            if isinstance(value, list):
                print(f"Type: list (length {len(value)})")
                print("All items:")
                for i, item in enumerate(value):
                    if hasattr(item, '__dict__'):
                        print(f"[{i}]:")
                        pprint(vars(item))
                    else:
                        print(f"[{i}]: {item}")
            elif isinstance(value, dict):
                print("Type: dict")
                print(f"Number of items: {len(value)}")
                print("All items:")
                pprint(value)
            else:
                print(f"Value: {value}")

        if "active_evs" in info and info["active_evs"]:
            print("\n--- DETAILED ACTIVE EV INFO ---")
            print(f"Number of active EVs: {len(info['active_evs'])}")
            for i, ev in enumerate(info["active_evs"]):
                if ev.station_id in [env.cn.station_ids[i] for i in demand_stations]:
                    print(f"\nEV {i+1} FULL DETAILS:")
                    ev_dict = vars(ev)
                    print(f"Number of attributes: {len(ev_dict)}")
                    for attr, val in ev_dict.items():
                        print(f"{attr:>20}: {val}")

        if hasattr(env, 'moer') and env.moer is not None:
            print("\n--- COMPLETE MOER DATA ---")
            print(f"Shape: {env.moer.shape}")
            print(f"Number of values: {env.moer.size}")
            print("Current timestep MOER values:")
            pprint(env.moer[timestep])

        print(f"\nREWARD: {reward}")
        print(f"TERMINATED: {terminated}")
    
    if terminated:
        print("\n=== EPISODE TERMINATED EARLY ===")
        break

print("\n=== SIMULATION COMPLETE ===")
env.close()
