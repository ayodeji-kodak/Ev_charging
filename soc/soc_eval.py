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
total_rewards = 0

# Store previous cumulative reward breakdown
prev_reward_breakdown = {
    'profit': 0.0,
    'carbon_cost': 0.0,
    'excess_charge': 0.0
}


# === Run the evaluation ===
for timestep in range(288):
    # Calculate energy values (kWh per timestep)
    charger_max = get_theoretical_max_energy(timestep_minutes)
    total_available = get_total_available_energy(env.cn, timestep_minutes)
    
    # Predict action using the trained model
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    reward_breakdown = info['reward_breakdown']
    total_rewards += reward

    # Compute per-step reward components
    current_reward_breakdown = info['reward_breakdown']
    step_profit = current_reward_breakdown['profit'] - prev_reward_breakdown['profit']
    step_carbon_cost = current_reward_breakdown['carbon_cost'] - prev_reward_breakdown['carbon_cost']
    step_excess_charge = current_reward_breakdown['excess_charge'] - prev_reward_breakdown['excess_charge']

    # === max possible reward per step ===
    num_active_evs = len(env._simulator.get_active_evs()) # Number of EVs charging this step
    max_profit_step = num_active_evs * 32 * env.PROFIT_FACTOR
    min_carbon_cost_step = num_active_evs * 32 * env.CARBON_COST_FACTOR * env.moer[env.t, 0]
    max_reward_step = max_profit_step - min_carbon_cost_step

    # Calculate delivered energy (kWh) per charger
    energy_delivered_per_charger = action * charger_max  # action is [0-1], charger_max is in kWh
    
    # Print energy delivered information
    if 'demands' in obs and np.any(obs['demands'] > 0):
        print(f"\n\n=== TIMESTEP {timestep} WITH DEMAND {'='*40}")
        
        # Print energy information
        print(f"\n--- ENERGY INFORMATION (kWh) ---")
        print(f"Maximum possible per charger per timestep: {charger_max:.3f} kWh")
        print(f"Total available from grid per timestep: {total_available:.3f} kWh")
        
        # Print detailed energy allocation per charger with SOC information
        print("\n--- CHARGER STATUS TABLE ---")
        demand_stations = np.where(obs['demands'] > 0)[0]
        
        # Create header for the table
        header = [
            "Station", "Action", "Energy (kWh)", "Remaining (kWh)", 
            "Demand (kWh)", "Est Depart", "Arrival SOC", 
            "Current SOC", "Target SOC"
        ]
        
        # Format the header
        print(
            f"{header[0]:<8} {header[1]:<7} {header[2]:<12} {header[3]:<12} "
            f"{header[4]:<12} {header[5]:<10} {header[6]:<12} "
            f"{header[7]:<12} {header[8]:<12}"
        )
        print("-" * 100)
        
        # Populate the table rows
        for station_idx in demand_stations:
            energy = energy_delivered_per_charger[station_idx]
            remaining = obs['demands'][station_idx] - energy
            est_depart = obs['est_departures'][station_idx]
            
            
            print(
                f"{station_idx:<8} "
                f"{action[station_idx]:<7.3f} "
                f"{energy:<12.3f} "
                f"{remaining:<12.3f} "
                f"{obs['demands'][station_idx]:<12.3f} "
                f"{obs['est_departures'][station_idx]:<10}"
                f"{obs['arrival_soc'][station_idx]:<12.1%} "
                f"{obs['current_soc'][station_idx]:<12.1%} "
                f"{obs['target_soc'][station_idx]:<12.1%}"
            )
        
        # Calculate and print utilization statistics
        total_allocated = np.sum(energy_delivered_per_charger[demand_stations])
        avg_utilization = np.mean(energy_delivered_per_charger[demand_stations] / charger_max)
        print(f"\nTotal allocated to EVs: {total_allocated:.3f} kWh")
        print(f"Grid utilization: {total_allocated/total_available:.1%}")
        print(f"Average charger utilization: {avg_utilization:.1%}")
        
        # Print reward information
        print(f"\nREWARD: {reward}")

        # Print per-step reward breakdown
        print(f"\nTimestep {env.t}:")
        print(f"  Step reward: {reward:.2f} (Max possible: {max_reward_step:.2f})")
        print(f"  Step profit: ${step_profit:.2f}")
        print(f"  Step carbon cost: ${step_carbon_cost:.2f}")
        print(f"  Step violation cost: ${step_excess_charge:.2f}")

        # Update previous reward breakdown for next step
        prev_reward_breakdown = current_reward_breakdown.copy()
    
    if terminated:
        print("\n=== EPISODE TERMINATED EARLY ===")
        print("Episode finished")
        print(f"Total reward: {total_rewards}")
        print(f"Profit: ${info['reward_breakdown']['profit']:.2f}")
        print(f"Carbon cost: ${info['reward_breakdown']['carbon_cost']:.2f}")
        print(f"Grid violation cost: ${info['reward_breakdown']['excess_charge']:.2f}")
        print(f"Maximum possible profit: ${info['max_profit']:.2f}")
        break

print("\n=== SIMULATION COMPLETE ===")
env.close()
