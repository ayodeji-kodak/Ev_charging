import gymnasium as gym
import numpy as np
from pprint import pprint
import os

from sustaingym.envs.evcharging.ev_env_soc import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def get_theoretical_max_energy(timestep_minutes=5):
    """Return the absolute maximum energy any charger can provide in kWh for one timestep"""
    return (EVChargingEnv.ACTION_SCALE_FACTOR * EVChargingEnv.VOLTAGE / 1000) * (timestep_minutes / 60)

def get_total_available_energy(cn, timestep_minutes=5):
    """
    Calculate the total available energy from the grid considering all constraints.
    Returns total energy in kWh for one timestep.
    """
    phase_factor = np.exp(1j * np.deg2rad(cn._phase_angles))
    A_tilde = cn.constraint_matrix * phase_factor[None, :]
    
    # Using cvxpy to match how the environment does it
    import cvxpy as cp
    
    # Variables representing the current at each station (normalized 0-1)
    action = cp.Variable(len(cn._phase_angles))
    
    # Objective is to maximize total current
    objective = cp.Maximize(cp.sum(action))
    
    # Constraints
    constraints = [
        action >= 0,
        action <= 1,
        cp.abs(A_tilde @ action) * EVChargingEnv.ACTION_SCALE_FACTOR <= cn.magnitudes
    ]
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status != 'optimal':
        raise ValueError("Could not find optimal solution for total available power")
    
    # Convert total current to total energy (kWh)
    total_current = np.sum(action.value) * EVChargingEnv.ACTION_SCALE_FACTOR
    total_power = total_current * EVChargingEnv.VOLTAGE / 1000  # kW
    total_energy = total_power * (timestep_minutes / 60)  # kWh
    
    return total_energy

# === Load the trained model ===
model_path = "./soc_evcharging_logs/soc_evcharging_final.zip"
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

# Assuming 5-minute timesteps (as is common in EV charging simulations)
timestep_minutes = 5

# === Run the evaluation ===
for timestep in range(288):
    # Calculate energy values (kWh per timestep)
    charger_max = get_theoretical_max_energy(timestep_minutes)
    total_available = get_total_available_energy(env.cn, timestep_minutes)
    
    # Predict action using the trained model
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Calculate delivered energy (kWh) per charger
    energy_delivered_per_charger = action * charger_max  # action is [0-1], charger_max is in kWh
    
    # Print energy delivered information
    if 'demands' in obs and np.any(obs['demands'] > 0):
        print(f"\n\n=== TIMESTEP {timestep} WITH DEMAND {'='*40}")
        
        # Print energy information
        print(f"\n--- ENERGY INFORMATION (kWh) ---")
        print(f"Maximum possible per charger per timestep: {charger_max:.3f} kWh")
        print(f"Total available from grid per timestep: {total_available:.3f} kWh")
        
        # Print detailed energy allocation per charger
        print("\n--- ENERGY ALLOCATION PER CHARGER (kWh) ---")
        print(f"\n{'='*40}")
        demand_stations = np.where(obs['demands'] > 0)[0]
        
        # Create a table of energy allocation
        print(f"{'Station':<10}{'Action':<10}{'Energy':<10}{'Remaining':<12}{'Demand':<10}")
        print("-" * 52)
        for station_idx in demand_stations:
            energy = energy_delivered_per_charger[station_idx]
            remaining = obs['demands'][station_idx] - energy
            print(
                f"{station_idx:<10}"
                f"{action[station_idx]:<10.3f}"
                f"{energy:<10.3f}"
                f"{remaining:<12.3f}"
                f"{obs['demands'][station_idx]:<10.3f}"
            )
        
        # Calculate and print utilization statistics
        total_allocated = np.sum(energy_delivered_per_charger[demand_stations])
        avg_utilization = np.mean(energy_delivered_per_charger[demand_stations] / charger_max)
        print(f"\nTotal allocated to EVs: {total_allocated:.3f} kWh")
        print(f"Grid utilization: {total_allocated/total_available:.1%}")
        print(f"Average charger utilization: {avg_utilization:.1%}")
        
        # Rest of your diagnostic print statements...
        print("\n--- DEMAND INFORMATION (kWh) ---")
        print(f"Number of stations with demand: {len(demand_stations)}")
        print(f"Stations with demand: {demand_stations}")

        print("\n--- NON-ZERO VALUES ONLY ---")
        arrays_to_check = ['est_departures', 'demands', 'arrival_soc', 'target_soc', 'current_soc']
        for array_name in arrays_to_check:
            if array_name in obs:
                array = obs[array_name]
                non_zero_indices = np.where(array != 0)[0] if array_name != 'est_departures' else np.where(array != -288)[0]
                if len(non_zero_indices) > 0:
                    print(f"\n{array_name}:")
                    for idx in non_zero_indices:
                        print(f"  Station {idx}: {array[idx]}" + (" kWh" if array_name == 'demands' else ""))

        if "active_evs" in info and info["active_evs"]:
            print("\n--- DETAILED ACTIVE EV INFO ---")
            for i, ev in enumerate(info["active_evs"]):
                if ev.station_id in [env.cn.station_ids[i] for i in demand_stations]:
                    print(f"\nEV at Station {ev.station_id}:")
                    print(f"  Requested energy: {ev.requested_energy:.3f} kWh")
                    print(f"  Delivered energy: {ev.delivered_energy:.3f} kWh")
                    print(f"  Arrival SOC: {ev.arrival_soc:.1%}")
                    print(f"  Current SOC: {ev.current_soc:.1%}")
                    print(f"  Target SOC: {ev.target_soc:.1%}")

        print(f"\nREWARD: {reward}")
    
    if terminated:
        print("\n=== EPISODE TERMINATED EARLY ===")
        break

print("\n=== SIMULATION COMPLETE ===")
env.close()
