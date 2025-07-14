import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EVChargingEnv(gym.Env):
    def __init__(self):
        super(EVChargingEnv, self).__init__()

        # Constants
        self.max_charge_power = 22.0  # kW (max charging power)
        self.max_heat_power = 3.0     # kW max heating
        self.max_cool_power = 3.0     # kW max cooling
        self.mu = 0.95                # charging efficiency
        self.battery_capacity_kWh = 60.0
        self.nominal_voltage = 400.0  # volts
        self.internal_resistance = 0.01  # Ohms

        # Observation space: Tbatt, tav, tdep, SOC, x_target, P_available, trem
        self.observation_space = spaces.Box(
            low=np.array([-40, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([60, 24, 24, 1, 1, 30, 24], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: charge_power, heat_power, cool_power
        self.action_space = spaces.Box(
            low=np.array([0., 0., 0.], dtype=np.float32),
            high=np.array([self.max_charge_power, self.max_heat_power, self.max_cool_power], dtype=np.float32),
            dtype=np.float32
        )

        # Added: max episode steps
        self.max_steps = 288  # 24 hours at 5-minute steps
        self.current_step = 0

        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)  # reinitialize self.np_random with the new seed
        else:
            self.seed(np.random.randint(0, 10000))  # ensure new seed every time


        self.Tbatt = self.np_random.uniform(-35, 55)  # battery temp °C
        self.tav = self.np_random.uniform(0, 24)     # arrival time hr
        self.tdep = (self.tav + self.np_random.uniform(3, 8)) % 24  # departure time hr
        self.current_time = self.tav

        self.x = self.np_random.uniform(0.2, 0.6)    # initial SOC
        self.x_target = self.np_random.uniform(0.8, 1.0)
        self.P_available = self.np_random.uniform(5, self.max_charge_power)

        self._update_trem()
        self._update_state()

        # Reset step counter
        self.current_step = 0

        self.history = {"soc": [self.x], "temp": [self.Tbatt], "time": [self.current_time]}

        return self.state, {}

    def _update_trem(self):
        if self.tdep >= self.current_time:
            self.trem = self.tdep - self.current_time
        else:
            self.trem = (24 - self.current_time) + self.tdep


    def _update_state(self):
        self.state = np.array([
            self.Tbatt, self.tav, self.tdep, self.x,
            self.x_target, self.P_available, self.trem
        ], dtype=np.float32)

    def step(self, action):
        # Clip actions
        charge_power, heat_power, cool_power = np.clip(action, self.action_space.low, self.action_space.high)

        # Prevent heating and cooling simultaneously
        if heat_power > 0 and cool_power > 0:
            # Prioritize heating if cold (<20), else cooling
            if self.Tbatt < 20:
                cool_power = 0
            else:
                heat_power = 0

        # Prevent charging and cooling simultaneously if battery hot
        if charge_power > 0 and cool_power > 0:
            if self.Tbatt > 30:
                charge_power = 0  # prioritize cooling to protect battery
            else:
                cool_power = 0   # prioritize charging

        # If SOC reached target, stop charging
        if self.x >= self.x_target:
            charge_power = 0
        else:
            # Slow down charging if we're too early to reach full SOC
            charging_buffer = 0.167  # 10 minutes
            timestep_hr = 5 / 60
            required_energy_kWh = max(0, (self.x_target - self.x) * self.battery_capacity_kWh)
            estimated_time_hr = required_energy_kWh / (charge_power * self.mu) if charge_power > 0 else float('inf')

            # If charging would finish too early, reduce charge power
            if estimated_time_hr + charging_buffer < self.trem:
                max_charge_duration = self.trem - charging_buffer
                adjusted_power = required_energy_kWh / max_charge_duration / self.mu
                charge_power = min(charge_power, adjusted_power, self.max_charge_power)

        # Ensure total power does not exceed available
        total_power = charge_power + heat_power + cool_power
        if total_power > self.P_available:
            scale = self.P_available / total_power
            charge_power *= scale
            heat_power *= scale
            cool_power *= scale
            overdraw_penalty = (total_power - self.P_available) * 10  # harsher penalty
        else:
            overdraw_penalty = 0

        timestep_hr = 5 / 60  # 5 minute timestep

        # CCCV Charging: constant current until 80% SOC, then exponential taper
        if self.x < 0.8:
            effective_charge_power = charge_power
        else:
            # Exponential tapering beyond 0.8 SOC (constant voltage)
            taper_factor = np.exp(-10 * (self.x - 0.8))  # sharper taper
            effective_charge_power = charge_power * taper_factor

        # Update SOC based on effective charging power
        delta_x = (effective_charge_power * self.mu * timestep_hr) / self.battery_capacity_kWh
        prev_soc = self.x
        self.x = np.clip(self.x + delta_x, 0, 1)

        # Temperature dynamics
        ambient_temp = 25.0
        thermal_resistance = 0.1  # K/kW
        thermal_capacity = 0.12   # kWh/K

        # Resistive heat from charging current
        current = (effective_charge_power * 1000) / self.nominal_voltage if effective_charge_power > 0 else 0
        heat_from_charge = (self.internal_resistance * current ** 2) / 1000  # kW heat

        # Net heat power: heating minus cooling plus resistive heat
        net_heat_power = heat_power - cool_power + heat_from_charge

        # Heat exchange with ambient using Newton's law of cooling
        temp_diff = self.Tbatt - ambient_temp
        #heat_loss = temp_diff / thermal_resistance
        heat_loss = 0
        total_heat_flow = net_heat_power - heat_loss

        dT = (total_heat_flow * timestep_hr) / thermal_capacity
        self.Tbatt = np.clip(self.Tbatt + dT, -10, 60)

        # Update time and remaining time
        self.current_time = (self.current_time + timestep_hr) % 24
        self._update_trem()

        # Reward components
        soc_delta = self.x - prev_soc
        soc_reward = soc_delta * 50  # emphasize SOC increase

        # Temperature penalty: Quadratic penalty outside 20-30°C, reward within range
        if 20 <= self.Tbatt <= 30:
            temp_reward = 2.0  # positive reward for staying within target range
            temp_penalty = 0
        else:
            temp_reward = 0
            temp_penalty = -((self.Tbatt - 25) ** 2) * 0.2  # stronger penalty for temp deviation

        # Encourage using power efficiently (charging + thermal)
        power_used = charge_power + heat_power + cool_power
        power_usage_penalty = -0.05 * (self.P_available - power_used) if self.x < self.x_target else 0

        # Encourage using at least 80% of available power
        power_utilization_ratio = power_used / self.P_available if self.P_available > 0 else 0

        if power_utilization_ratio >= 0.8:
            power_utilization_bonus = (power_utilization_ratio - 0.8) * 5  # scaled bonus for exceeding 80%
        else:
            power_utilization_bonus = (power_utilization_ratio - 0.8) * 10  # penalty for falling short

        # Encourage charging and heating coordination
        coordinated_bonus = min(1.0, abs(self.Tbatt - 25) / 5.0)

        # Reward per kWh used for SOC gain
        energy_used_kWh = (charge_power + heat_power + cool_power) * timestep_hr
        efficiency_reward = 0
        if soc_delta > 0 and energy_used_kWh > 0:
            efficiency_reward = (soc_delta * self.battery_capacity_kWh) / energy_used_kWh * 2.0  # reward SOC gain per energy

        # Bonus for completion of SOC target on time
        completion_bonus = 25 if self.x >= self.x_target and self.trem > 0 else 0

        reward = soc_reward + temp_reward + temp_penalty + power_usage_penalty + completion_bonus - overdraw_penalty + coordinated_bonus + power_utilization_bonus

        early_finish_penalty = 0
        MIN_REMAINING_TIME = 0.1667  # hours

        # Check if the target SOC is reached too early
        if self.x >= self.x_target and self.trem > MIN_REMAINING_TIME:
            early_finish_penalty = (self.trem - MIN_REMAINING_TIME) * 1
            reward -= early_finish_penalty

        # Update state and history
        self._update_state()
        self.history["soc"].append(self.x)
        self.history["temp"].append(self.Tbatt)
        self.history["time"].append(self.current_time)

        info = {
            "soc": self.x,
            "tbatt": self.Tbatt,
            "trem": self.trem,
            "charge_power": charge_power,
            "heat_power": heat_power,
            "cool_power": cool_power,
            "reward_components": {
                "soc_reward": soc_reward,
                "temp_reward": temp_reward,
                "temp_penalty": temp_penalty,
                "power_usage_penalty": power_usage_penalty,
                "completion_bonus": completion_bonus,
                "early_finish_penalty": early_finish_penalty if self.x >= self.x_target else 0,
                "overdraw_penalty": overdraw_penalty,
                "coordinated_bonus": coordinated_bonus,
                "power_utilization_bonus": power_utilization_bonus,
            }
        }

        self.current_step += 1

        reward = float(reward)

        done = self.trem <= 0 or self.current_step >= self.max_steps

        return self.state, reward, done, False, info


    def render(self, mode='human'):
        print(f"\nTime: {self.current_time:.2f} hr | Remaining time: {self.trem:.2f} hr")
        print(f"SOC: {self.x:.3f} | Tbatt: {self.Tbatt:.2f}°C | Available Power: {self.P_available:.2f} kW")

    def close(self):
        pass