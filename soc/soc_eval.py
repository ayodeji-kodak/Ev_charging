import gymnasium as gym
import numpy as np
from pprint import pprint
import os

from sustaingym.envs.evcharging.ev_env_soc import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def get_theoretical_max_energy(timestep_minutes=5):
    """Absolute max energy any charger can provide in kWh for one timestep."""
    return (EVChargingEnv.ACTION_SCALE_FACTOR * EVChargingEnv.VOLTAGE / 1000) * (timestep_minutes / 60)

def get_total_available_energy(cn, timestep_minutes=5):
    """Total available energy from grid considering constraints (kWh per timestep)."""
    phase_factor = np.exp(1j * np.deg2rad(cn._phase_angles))
    A_tilde = cn.constraint_matrix * phase_factor[None, :]
    import cvxpy as cp
    x = cp.Variable(len(cn._phase_angles))
    prob = cp.Problem(
        cp.Maximize(cp.sum(x)),
        [
            x >= 0,
            x <= 1,
            cp.abs(A_tilde @ x) * EVChargingEnv.ACTION_SCALE_FACTOR <= cn.magnitudes
        ]
    )
    prob.solve()
    if prob.status != 'optimal':
        raise ValueError("Could not find optimal solution for total available power")
    total_current = np.sum(x.value) * EVChargingEnv.ACTION_SCALE_FACTOR
    total_power = total_current * EVChargingEnv.VOLTAGE / 1000  # kW
    return total_power * (timestep_minutes / 60)  # kWh

def amps_after_rounding(proj_norm, env):
    """
    Reproduce env._to_schedule rounding (without re-solving) to show the pilot amps.
    proj_norm: normalized [0,1] projected action per station
    returns np.array of amps after rounding/discretization.
    """
    amps = proj_norm * EVChargingEnv.ACTION_SCALE_FACTOR
    rounded = np.zeros_like(amps)
    for i in range(env.num_stations):
        if env.cn.min_pilot_signals[i] == 6:
            # {0} U {6,7,8,...,32}, below 6 -> 0, else integer-rounded
            rounded[i] = np.round(amps[i]) if amps[i] >= 6 else 0
        else:
            # {0,8,16,24,32}
            rounded[i] = np.round(amps[i] / 8) * 8
    return rounded

# === Load the trained model ===
model_path = "./soc_evcharging_logs/soc_evcharging_final.zip"
model = PPO.load(model_path)
print(f"\n✅ Loaded trained model from: {model_path}")

# === Recreate the same environment ===
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
    project_action_in_env=True,   # important so the projector is initialized
    verbose=1
)

check_env(env, warn=True)

# === Reset the environment ===
obs, info = env.reset()
print("\n=== Environment Reset ===\n")

timestep_minutes = 5

# === Run the evaluation ===
for timestep in range(288):
    charger_max = get_theoretical_max_energy(timestep_minutes)
    total_available = get_total_available_energy(env.cn, timestep_minutes)

    # Agent proposes an action from current obs
    agent_action, _ = model.predict(obs, deterministic=True)
    agent_action = np.asarray(agent_action).reshape(-1)

    # --- KEY: get the env's projected action for this exact timestep BEFORE stepping ---
    # This uses the same demands/env state the env will use inside step().
    projected_action = env._project_action(agent_action.copy()) if env.project_action_in_env else agent_action.copy()

    # For reporting energy delivered this step, use the projected action (pre-rounding)
    energy_delivered_per_charger = projected_action * charger_max

    # For transparency, also compute pilot amps after rounding/discretization like the env does
    pilot_amps = amps_after_rounding(projected_action, env)

    # Now actually step the env with the agent's raw action (env will project again internally)
    obs, reward, terminated, truncated, info = env.step(agent_action)

    # Get per-timestep reward breakdown
    current_reward = info['current_reward_breakdown']
    
    # Print reward breakdown for every timestep (even if no demand)
    print(f"\n--- TIMESTEP {timestep} REWARD BREAKDOWN ---")
    print(f"Total Reward: {current_reward['total']:.4f}")
    print(f"Profit: {current_reward['profit']:.4f}")
    print(f"Carbon Cost: {-current_reward['carbon_cost']:.4f}")
    print(f"Excess Charge Penalty: {-current_reward['excess_charge']:.4f}")
    if 'follow_projection' in current_reward:
        print(f"Projection Following: {current_reward['follow_projection']:.4f}")

    # Print details only when there is demand at any station
    if 'demands' in obs and np.any(obs['demands'] > 0):
        demand_stations = np.where(obs['demands'] > 0)[0]
        print(f"\n\n=== TIMESTEP {timestep} WITH DEMAND {'='*40}")

        # Energy info
        print(f"\n--- ENERGY INFORMATION (kWh) ---")
        print(f"Maximum possible per charger per timestep: {charger_max:.3f} kWh")
        print(f"Total available from grid per timestep: {total_available:.3f} kWh")

        # Table header
        headers = [
            "Idx", "StationID", "Agent", "Proj", "Δ(Proj-Agent)",
            "Pilot (A)", "Energy kWh", "Remaining kWh",
            "Demand kWh", "EstDepart", "Arr SOC", "Cur SOC", "Tgt SOC"
        ]
        print(
            f"{headers[0]:<4} {headers[1]:<12} {headers[2]:<7} {headers[3]:<7} {headers[4]:<13} "
            f"{headers[5]:<9} {headers[6]:<10} {headers[7]:<12} "
            f"{headers[8]:<10} {headers[9]:<9} {headers[10]:<8} {headers[11]:<8} {headers[12]:<7}"
        )
        print("-" * 132)

        for i in demand_stations:
            station_id = env.cn.station_ids[i]
            energy = energy_delivered_per_charger[i]
            remaining = obs['demands'][i] - energy
            delta = projected_action[i] - agent_action[i]
            print(
                f"{i:<4d} "
                f"{station_id:<12} "
                f"{agent_action[i]:<7.3f} "
                f"{projected_action[i]:<7.3f} "
                f"{delta:<13.3f} "
                f"{int(pilot_amps[i]):<9d} "
                f"{energy:<10.3f} "
                f"{remaining:<12.3f} "
                f"{obs['demands'][i]:<10.3f} "
                f"{int(obs['est_departures'][i]):<9d} "
                f"{obs['arrival_soc'][i]:<8.1%} "
                f"{obs['current_soc'][i]:<8.1%} "
                f"{obs['target_soc'][i]:<7.1%}"
            )

        total_allocated = float(np.sum(energy_delivered_per_charger[demand_stations]))
        avg_utilization = float(np.mean(projected_action[demand_stations])) if len(demand_stations) > 0 else 0.0
        grid_util = (total_allocated / total_available) if total_available > 0 else 0.0

        print(f"\nTotal allocated to EVs: {total_allocated:.3f} kWh")
        print(f"Grid utilization: {grid_util:.1%}")
        print(f"Average charger utilization (proj): {avg_utilization:.1%}")
        print(f"\nREWARD: {reward}")

    if terminated:
        print("\n=== EPISODE TERMINATED EARLY ===")
        print(f"Maximum possible profit: ${info['max_profit']:.2f}")
        break

print("\n=== SIMULATION COMPLETE ===")
env.close()
