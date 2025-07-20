import numpy as np
from sustaingym.envs.evcharging import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator, GMMsTraceGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", 
    message="Invalid schedule provided at iteration",
    category=UserWarning,
    module="acnportal.*")
warnings.filterwarnings("ignore", 
    message="Simulating \d+ events using Real trace generator",
    category=UserWarning)

def create_env():
    """Create and return the EV charging environment"""
    site = 'caltech'
    date_period = ('2019-05-01', '2019-08-31')
    moer_forecast_steps = 36
    project_action_in_env = True
    
    data_generator = RealTraceGenerator(
        site=site,
        date_period=date_period,
        sequential=True,
        use_unclaimed=False,
        requested_energy_cap=100,
        seed=42
    )
    
    env = EVChargingEnv(
        data_generator=data_generator,
        moer_forecast_steps=moer_forecast_steps,
        project_action_in_env=project_action_in_env,
        verbose=0
    )
    return env

def train_model():
    """Train the PPO model and save it"""
    log_dir = "./ppo_evcharging_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = create_env()
    env = Monitor(env, log_dir)
    
    print("\nEnvironment created successfully!")
    print(f"Number of stations: {env.num_stations}")
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=log_dir
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="ppo_evcharging"
    )
    
    print("\nTraining PPO for 100,000 timesteps...")
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )
    
    model.save(os.path.join(log_dir, "ppo_evcharging_final"))
    print(f"\nModel saved to {log_dir}")
    env.close()

if __name__ == "__main__":
    train_model()
