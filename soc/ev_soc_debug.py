import gymnasium as gym
import numpy as np
import sys
from pprint import pprint

from sustaingym.envs.evcharging.ev_env_soc import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import GMMsTraceGenerator
from stable_baselines3.common.env_checker import check_env

# --- Setup trace generator ---
trace_gen = GMMsTraceGenerator(
    site='caltech',
    date_period=('2019-05-01', '2019-08-31'),
    n_components=40,
    requested_energy_cap=100,
    seed=42
)

# --- Initialize environment ---
env = EVChargingEnv(
    data_generator=trace_gen,
    moer_forecast_steps=36,
    project_action_in_env=True,
    verbose=1
)

# --- Optional: Check Gym API validity ---
check_env(env, warn=True)

# --- Reset the environment ---
obs, info = env.reset()
print("\n=== Environment Reset ===\n")

# Define a zero action (no charging) for demonstration
zero_action = np.zeros(env.num_stations)

# Run simulation for all 288 timesteps
for timestep in range(288):
    # Step the environment with zero action
    obs, reward, terminated, truncated, info = env.step(zero_action)
    
    # Only print if there's non-zero demand
    if 'demands' in obs and np.any(obs['demands'] > 0):
        print(f"\n\n=== TIMESTEP {timestep} WITH DEMAND {'='*40}")
        
        # Print demand information first
        print("\n--- DEMAND INFORMATION ---")
        demand_stations = np.where(obs['demands'] > 0)[0]
        num_demands = len(demand_stations)
        print(f"Number of stations with demand: {num_demands}")
        print(f"Stations with demand: {demand_stations}")
        print(f"Demand values: {obs['demands'][demand_stations]}")
        
        # NEW: Print only non-zero values for specified arrays
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
        
        # Print Observation with full details (original output)
        print("\n--- FULL OBSERVATION ---")
        for key, value in obs.items():
            print(f"\n{key}:")
            print(f"Shape: {np.shape(value)}")
            print(f"Dtype: {np.array(value).dtype}")
            print(f"Number of elements: {len(value)}")
            print("Values:")
            pprint(value)
        
        # Rest of your existing code remains the same...
        # Print Complete Info Dictionary
        print("\n--- COMPLETE INFO DICTIONARY ---")
        for key, value in info.items():
            print(f"\n{key}:")
            if isinstance(value, list):
                print(f"Type: list (length {len(value)})")
                print("All items:")
                for i, item in enumerate(value):
                    if hasattr(item, '__dict__'):  # If it's an object
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
        
        # Print detailed EV information for active stations
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
        
        # Print MOER data if available
        if hasattr(env, 'moer') and env.moer is not None:
            print("\n--- COMPLETE MOER DATA ---")
            print(f"Shape: {env.moer.shape}")
            print(f"Number of values: {env.moer.size}")
            print("Current timestep MOER values:")
            pprint(env.moer[timestep])
        
        # Print reward and termination status
        print(f"\nREWARD: {reward}")
        print(f"TERMINATED: {terminated}")
    
    if terminated:
        print("\n=== EPISODE TERMINATED EARLY ===")
        break

print("\n=== SIMULATION COMPLETE ===")
env.close()
