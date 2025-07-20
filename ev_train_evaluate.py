import numpy as np
import matplotlib.pyplot as plt
from sustaingym.envs.evcharging import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import RealTraceGenerator
from stable_baselines3 import PPO
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

def load_env():
    """Load and return the EV charging environment"""
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

def plot_rewards(steps, rewards, max_rewards, profit_ratios):
    """Plot the reward progression"""
    plt.figure(figsize=(12, 6))
    
    # Reward comparison plot
    plt.subplot(1, 2, 1)
    plt.plot(steps, rewards, label='Achieved Reward', color='blue')
    plt.plot(steps, max_rewards, label='Max Possible Reward', color='green', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Comparison')
    plt.legend()
    plt.grid(True)
    
    # Profit ratio plot
    plt.subplot(1, 2, 2)
    plt.plot(steps, profit_ratios, label='Profit Ratio', color='red')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    plt.title('Achieved/Max Profit Ratio')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('ppo_evcharging_logs', 'reward_analysis.png'))
    plt.close()

def evaluate_model():
    """Evaluate the trained model"""
    log_dir = "./ppo_evcharging_logs/"
    model_path = os.path.join(log_dir, "ppo_evcharging_final.zip")
    os.makedirs(log_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    
    env = load_env()
    model = PPO.load(model_path)
    
    print("\n=== Running Evaluation ===")
    obs, info = env.reset()
    num_test_steps = 288
    total_reward = 0
    max_possible_profit = info['max_profit']

    # Data storage for plotting
    steps = []
    rewards = []
    max_rewards = []
    profit_ratios = []
    current_profits = []

    print("\nInitial Session Info:")
    print(f"- Max possible profit: ${max_possible_profit:.2f}")
    print(f"- Reward breakdown keys: {list(info['reward_breakdown'].keys())}")
    
    for step in range(num_test_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Store data for plotting
        current_profit = info['reward_breakdown']['profit']
        profit_ratio = current_profit / max_possible_profit
        
        steps.append(step)
        rewards.append(total_reward)
        max_rewards.append(max_possible_profit)
        profit_ratios.append(profit_ratio)
        current_profits.append(current_profit)
        
        active_evs = np.sum(obs['demands'] > 0)
        if active_evs > 0 or step % 50 == 0:
            print(f"\nStep {step + 1}:")
            print(f"Active EVs: {active_evs}/{env.num_stations}")
            print(f"Action stats: min={action.min():.3f}, max={action.max():.3f}")
            print(f"Step reward: {reward:.2f} | Cumulative: {total_reward:.2f}")
            print(f"Profit: ${current_profit:.2f} ({profit_ratio:.1%} of max)")
            print(f"MOER: Current={obs['prev_moer'][0]:.4f}")
        
        if terminated:
            print("\nEpisode terminated early!")
            break
    
    print("\n=== Final Results ===")
    print(f"Total reward: {total_reward:.2f}")
    if 'reward_breakdown' in info:
        print("Final reward breakdown:")
        for k, v in info['reward_breakdown'].items():
            print(f"- {k}: {v:.2f}")
    
    # Generate plots
    plot_rewards(steps, rewards, max_rewards, profit_ratios)
    print("\nSaved reward analysis plots to ppo_evcharging_logs/reward_analysis.png")
    
    env.close()

if __name__ == "__main__":
    evaluate_model()
