import numpy as np
from sustaingym.envs.evcharging import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator, GMMsTraceGenerator

def main():
    # Configuration parameters
    site = 'caltech'  # or 'jpl'
    date_period = ('2019-05-01', '2019-08-31')  # must be a tuple of two strings
    moer_forecast_steps = 36
    project_action_in_env = True
    verbose = 2  # set to 2 for maximum debug output
    
    # Choose between real traces or GMM-generated traces
    use_real_traces = True
    
    try:
        # Create the appropriate trace generator
        if use_real_traces:
            print("Using RealTraceGenerator...")
            data_generator = RealTraceGenerator(
                site=site,
                date_period=date_period,
                sequential=True,
                use_unclaimed=False,
                requested_energy_cap=100,
                seed=42
            )
        else:
            print("Using GMMsTraceGenerator...")
            data_generator = GMMsTraceGenerator(
                site=site,
                date_period=date_period,
                n_components=30,
                requested_energy_cap=100,
                seed=42
            )
        
        # Create the environment
        env = EVChargingEnv(
            data_generator=data_generator,
            moer_forecast_steps=moer_forecast_steps,
            project_action_in_env=project_action_in_env,
            verbose=verbose
        )
        
        print("\nEnvironment created successfully!")
        print(f"Number of stations: {env.num_stations}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Reset the environment to start a new episode
        print("\nResetting environment...")
        obs, info = env.reset()

        print("\nEV Arrival Times:")
        for ev in info.get('evs', []):
            print(f"EV {ev.session_id}: arrives at {ev.arrival} periods")
                
        print("\nInitial observation:")
        for key, value in obs.items():
            print(f"{key}: shape {value.shape}, min {value.min()}, max {value.max()}")
        
        print("\nEnvironment info keys:", info.keys())  # Debug: show available keys
        
        # Safely access info dictionary with .get() to handle missing keys
        print("\nEnvironment info:")
        print(f"Number of EVs: {info.get('num_evs', 'Key not available')}")
        print(f"Average plugin time: {info.get('avg_plugin_time', 'Key not available')}")
        print(f"Max possible profit: ${info.get('max_profit', 'Key not available')}")
        print(f"Reward breakdown: {info.get('reward_breakdown', 'Key not available')}")
        
        # Run a few steps with random actions to test the environment
        num_test_steps = 288
        print(f"\nRunning {num_test_steps} test steps with random actions...")
        
        for step in range(num_test_steps):
            # Generate random action
            action = env.action_space.sample()
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if there are active EVs
            active_evs = np.sum(obs['demands'] > 0)
            if active_evs > 0:
                print(f"\nStep {step + 1}:")
                print(f"Active EVs: {active_evs}/{env.num_stations}")
                print(f"Action: min {action.min()}, max {action.max()}, mean {action.mean()}")
                print(f"Reward: {reward:.2f}")
                print(f"Terminated: {terminated}")
                
                # Print some observation details
                print(f"Current MOER: {obs['prev_moer'][0]:.4f}")
                print(f"MOER forecast: {obs['forecasted_moer'][:5]}...")
                
                # Print step info (use .get() to handle missing keys)
                print("Step info:")
                print(f"Active EVs: {len(info.get('active_evs', []))}")
                print(f"Reward breakdown: {info.get('reward_breakdown', {})}")
            
            if terminated:
                print("Episode terminated early!")
                break
        
        # Print final reward breakdown if available
        if 'reward_breakdown' in info:
            print("\nFinal reward breakdown:")
            for key, value in info['reward_breakdown'].items():
                print(f"{key}: ${value:.2f}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        # Close the environment
        if 'env' in locals():
            env.close()
            print("\nEnvironment closed successfully!")

if __name__ == "__main__":
    main()