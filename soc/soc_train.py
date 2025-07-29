import numpy as np
from sustaingym.envs.evcharging.ev_env_soc import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", 
    message="Invalid schedule provided at iteration",
    category=UserWarning,
    module="acnportal.*")
warnings.filterwarnings("ignore", 
    message="Simulating \d+ events using Real trace generator",
    category=UserWarning)

class EnhancedPlottingCallback(BaseCallback):
    """Enhanced callback with SOC tracking and additional metrics"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.carbon_costs = []
        self.profits = []
        self.actions = []
        self.active_evs = []
        self.moer_values = []
        self.steps = 0
        # New SOC tracking variables
        self.arrival_socs = []
        self.target_socs = []
        self.current_socs = []
        self.soc_progress = []  # (current_soc - arrival_soc)/(target_soc - arrival_soc)
        
    def _on_step(self) -> bool:
        # Collect metrics from info dictionary
        info = self.locals.get('infos')[0]
        
        # Collect reward components if available
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            self.rewards.append(self.locals['rewards'][0])
            self.carbon_costs.append(breakdown.get('carbon_cost', 0))
            self.profits.append(breakdown.get('profit', 0))
            
        # Collect action data
        actions = self.locals.get('actions')
        if actions is not None:
            self.actions.append(actions[0].mean())  # Store mean charging rate
            
        # Collect environment state from observation
        obs = self.locals.get('new_observations') or self.locals.get('observations')
        if obs is not None and isinstance(obs, dict):
            # Track active EVs
            active = np.sum(obs['demands'] > 0) if 'demands' in obs else 0
            self.active_evs.append(active)
            
            # Get MOER value if available
            if 'prev_moer' in obs:
                self.moer_values.append(obs['prev_moer'][0])
            
            # Track SOC metrics
            if all(key in obs for key in ['arrival_soc', 'target_soc', 'current_soc']):
                active_mask = obs['demands'] > 0 if 'demands' in obs else np.ones_like(obs['arrival_soc'], dtype=bool)
                
                # Only calculate for active EVs
                if np.any(active_mask):
                    arrival = np.mean(obs['arrival_soc'][active_mask])
                    target = np.mean(obs['target_soc'][active_mask])
                    current = np.mean(obs['current_soc'][active_mask])
                    
                    self.arrival_socs.append(arrival)
                    self.target_socs.append(target)
                    self.current_socs.append(current)
                    
                    # Calculate SOC progress (0 at arrival, 1 at target)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        progress = np.nan_to_num((current - arrival) / (target - arrival), nan=0.0, posinf=0.0, neginf=0.0)
                    self.soc_progress.append(progress)
        
        self.steps += 1
        return True
    
    def plot_metrics(self):
        """Plot all collected metrics including SOC data"""
        plt.figure(figsize=(18, 15))
        
        # Create steps array based on collected data
        min_length = min(len(self.rewards), len(self.carbon_costs), len(self.profits))
        steps = np.arange(min_length)
 
        # Plot 1: Reward Components
        if len(self.rewards) > 0:
            plt.subplot(3, 3, 1)
            plt.plot(steps, self.rewards[:min_length], label='Total Reward')
            plt.plot(steps, self.profits[:min_length], label='Profit')
            plt.plot(steps, self.carbon_costs[:min_length], label='Carbon Cost')
            plt.title('Reward Components Over Time')
            plt.xlabel('Steps')
            plt.ylabel('Value ($)')
            plt.legend()
            plt.grid(True)
        
        # Plot 2: Cumulative Profits vs Carbon Costs
        if len(self.profits) > 0 and len(self.carbon_costs) > 0:
            plt.subplot(3, 3, 2)
            plt.plot(steps, np.cumsum(self.profits[:min_length]), label='Cumulative Profit')
            plt.plot(steps, np.cumsum(self.carbon_costs[:min_length]), label='Cumulative Carbon Cost')
            plt.title('Cumulative Profit vs Carbon Cost')
            plt.xlabel('Steps')
            plt.ylabel('Cumulative Value ($)')
            plt.legend()
            plt.grid(True)
        
        # Plot 3: Charging Behavior
        if len(self.actions) > 0:
            plt.subplot(3, 3, 3)
            plt.plot(np.arange(len(self.actions)), self.actions, label='Mean Charging Rate')
            plt.title('Charging Behavior')
            plt.xlabel('Steps')
            plt.ylabel('Charging Rate')
            plt.ylim(0, 1)
            plt.grid(True)
        
        # Plot 4: Profit vs Carbon Cost Scatter
        if len(self.profits) > 0 and len(self.carbon_costs) > 0:
            plt.subplot(3, 3, 4)
            plt.scatter(self.profits[:min_length], self.carbon_costs[:min_length], alpha=0.5)
            plt.title('Profit vs Carbon Cost Tradeoff')
            plt.xlabel('Profit ($)')
            plt.ylabel('Carbon Cost ($)')
            plt.grid(True)

        # Plot 5: State of Charge Progress
        if len(self.soc_progress) > 0:
            plt.subplot(3, 3, 5)
            plt.plot(np.arange(len(self.soc_progress)), self.soc_progress, label='SOC Progress')
            plt.title('Average SOC Progress (0=arrival, 1=target)')
            plt.xlabel('Steps')
            plt.ylabel('Normalized SOC Progress')
            plt.ylim(0, 1)
            plt.grid(True)

        # Plot 6: SOC Values Over Time
        if len(self.current_socs) > 0:
            plt.subplot(3, 3, 6)
            plt.plot(np.arange(len(self.arrival_socs)), self.arrival_socs, label='Arrival SOC')
            plt.plot(np.arange(len(self.current_socs)), self.current_socs, label='Current SOC')
            plt.plot(np.arange(len(self.target_socs)), self.target_socs, label='Target SOC')
            plt.title('SOC Values Over Time')
            plt.xlabel('Steps')
            plt.ylabel('State of Charge')
            plt.legend()
            plt.grid(True)

        # Plot 7: Active EVs vs SOC Progress
        if len(self.active_evs) > 0 and len(self.soc_progress) > 0:
            plt.subplot(3, 3, 7)
            plt.plot(np.arange(len(self.active_evs)), self.active_evs, label='Active EVs')
            plt.plot(np.arange(len(self.soc_progress)), np.array(self.soc_progress)*max(self.active_evs), 
                    label='SOC Progress (scaled)')
            plt.title('Active EVs vs SOC Progress')
            plt.xlabel('Steps')
            plt.ylabel('Count/Progress')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics_with_soc.png')
        plt.close()
        
        # Save data to CSV with SOC metrics
        data_dict = {
            'step': np.arange(min_length),
            'reward': self.rewards[:min_length],
            'profit': self.profits[:min_length],
            'carbon_cost': self.carbon_costs[:min_length],
        }
        
        if len(self.actions) > 0:
            data_dict['mean_charging_rate'] = self.actions[:min_length]
        if len(self.active_evs) > 0:
            data_dict['active_evs'] = self.active_evs[:min_length]
        if len(self.moer_values) > 0:
            data_dict['moer'] = self.moer_values[:min_length]
        if len(self.soc_progress) > 0:
            data_dict.update({
                'arrival_soc': self.arrival_socs[:min_length],
                'target_soc': self.target_socs[:min_length],
                'current_soc': self.current_socs[:min_length],
                'soc_progress': self.soc_progress[:min_length]
            })
            
        data = pd.DataFrame(data_dict)
        data.to_csv('training_metrics_with_soc.csv', index=False)

def create_env():
    """Create and return the EV charging environment with SOC tracking"""
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
    """Train the PPO model and save it with enhanced SOC tracking"""
    log_dir = "./soc_evcharging_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = create_env()
    env = Monitor(env, log_dir)
    
    print("\nEnvironment created successfully!")
    print(f"Number of stations: {env.num_stations}")
    
    # Create our enhanced callback
    plotting_callback = EnhancedPlottingCallback()
    
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
    
    print("\nTraining PPO for 250,000 timesteps...")
    model.learn(
        total_timesteps=250000,
        callback=plotting_callback,
        tb_log_name="ppo_run"
    )
    
    # Generate enhanced plots after training
    plotting_callback.plot_metrics()
    print("Training metrics plots with SOC data saved to training_metrics_with_soc.png")
    
    model.save(os.path.join(log_dir, "soc_evcharging_final"))
    print(f"\nModel saved to {log_dir}")
    env.close()

if __name__ == "__main__":
    train_model()
