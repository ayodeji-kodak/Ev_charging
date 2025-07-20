import numpy as np
from sustaingym.envs.evcharging import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator, GMMsTraceGenerator
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

class PlottingCallback(BaseCallback):
    """Custom callback for collecting training metrics and plotting"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.carbon_costs = []
        self.profits = []
        self.actions = []
        self.active_evs = []
        self.moer_values = []
        self.steps = 0
        
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
        if obs is not None and isinstance(obs, dict) and 'demands' in obs:
            active = np.sum(obs['demands'] > 0)
            self.active_evs.append(active)
            
            # Get MOER value if available
            if 'prev_moer' in obs:
                self.moer_values.append(obs['prev_moer'][0])
        
        self.steps += 1
        return True
    
    def plot_metrics(self):
        """Plot all collected metrics"""
        plt.figure(figsize=(15, 12))
        
        # Create steps array based on collected data
        min_length = min(len(self.rewards), len(self.carbon_costs), len(self.profits))
        steps = np.arange(min_length)
 
        # Plot 1: Only plot if we have reward data
        if len(self.rewards) > 0:
            plt.subplot(3, 2, 1)
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
            plt.subplot(3, 2, 2)
            plt.plot(steps, np.cumsum(self.profits[:min_length]), label='Cumulative Profit')
            plt.plot(steps, np.cumsum(self.carbon_costs[:min_length]), label='Cumulative Carbon Cost')
            plt.title('Cumulative Profit vs Carbon Cost')
            plt.xlabel('Steps')
            plt.ylabel('Cumulative Value ($)')
            plt.legend()
            plt.grid(True)
        
        # Plot 3: Charging Behavior
        if len(self.actions) > 0:
            plt.subplot(3, 2, 3)
            plt.plot(np.arange(len(self.actions)), self.actions, label='Mean Charging Rate')
            plt.title('Charging Behavior')
            plt.xlabel('Steps')
            plt.ylabel('Charging Rate')
            plt.ylim(0, 1)
            plt.grid(True)
        
        # Plot 4: Active EVs (only if we have data)
        '''
        if len(self.active_evs) > 0:
            plt.subplot(3, 2, 4)
            plt.plot(np.arange(len(self.active_evs)), self.active_evs, label='Active EVs')
            plt.title('Active EVs Over Time')
            plt.xlabel('Steps')
            plt.ylabel('Number of Active EVs')
            plt.grid(True)
        
        # Plot 5: MOER Values
        if len(self.moer_values) > 0:
            plt.subplot(3, 2, 5)
            plt.plot(np.arange(len(self.moer_values)), self.moer_values, label='MOER', color='red')
            plt.title('Grid Carbon Intensity (MOER)')
            plt.xlabel('Steps')
            plt.ylabel('MOER Value')
            plt.grid(True)
        '''
        # Plot 6: Profit vs Carbon Cost Scatter
        if len(self.profits) > 0 and len(self.carbon_costs) > 0:
            plt.subplot(3, 2, 6)
            plt.scatter(self.profits[:min_length], self.carbon_costs[:min_length], alpha=0.5)
            plt.title('Profit vs Carbon Cost Tradeoff')
            plt.xlabel('Profit ($)')
            plt.ylabel('Carbon Cost ($)')
            plt.grid(True)

        # Plot 7: Moving average of total reward (convergence)
        if len(self.rewards) > 100:
            plt.figure(figsize=(10, 4))
            moving_avg = pd.Series(self.rewards).rolling(window=100).mean()
            plt.plot(self.rewards, label='Reward')
            plt.plot(moving_avg, label='100-Step Moving Avg', linewidth=2)
            plt.title('Reward Convergence Over Training')
            plt.xlabel('Steps')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('training_reward_convergence.png')
            plt.close()

        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
        # Save data to CSV for further analysis
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
            
        data = pd.DataFrame(data_dict)
        data.to_csv('training_metrics.csv', index=False)

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
    
    # Create our custom callback
    plotting_callback = PlottingCallback()
    
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
    
    print("\nTraining PPO for 100,000 timesteps...")
    model.learn(
        total_timesteps=100000,
        callback=plotting_callback,
        tb_log_name="ppo_run"
    )
    
    # Generate plots after training
    plotting_callback.plot_metrics()
    print("Training metrics plots saved to training_metrics.png")
    
    model.save(os.path.join(log_dir, "ppo_evcharging_final"))
    print(f"\nModel saved to {log_dir}")
    env.close()

if __name__ == "__main__":
    train_model()
