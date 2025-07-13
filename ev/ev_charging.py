import gymnasium as gym
from gymnasium import spaces
import numpy as np

STEP_HOURS = 5 / 60  # 5 minutes in hours

class EVChargingEnv(gym.Env):
    def __init__(self, debug=False):
        super(EVChargingEnv, self).__init__()

        self.debug = debug  # Added debug flag

        # Constants
        self.max_charge_power = 22.0  # kW (max charging power)
        self.max_heat_power = 3.0     # kW max heating
        self.max_cool_power = 3.0     # kW max cooling
        self.mu = 0.95                # charging efficiency (nominal)
        self.battery_capacity_kWh = 60.0
        self.nominal_voltage = 400.0  # volts
        self.internal_resistance_25C = 0.01  # Ohms at 25°C
        self.alpha = 0.0039  # Temperature coefficient of resistance (per °C)

        # Observation space: Tbatt, tav, tdep, SOC, x_target, P_available, trem
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([60, 24, 24, 1, 1, 30, 24], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: charge_power, heat_power, cool_power
        self.action_space = spaces.Box(
            low=np.array([0., 0., 0.], dtype=np.float32),
            high=np.array([self.max_charge_power, self.max_heat_power, self.max_cool_power], dtype=np.float32),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def temp_dependent_resistance(self, T):
        return self.internal_resistance_25C * (1 + self.alpha * (T - 25))
    
    def temp_dependent_efficiency(self, T):
        nominal_eff = self.mu
        efficiency = nominal_eff - 0.005 * (T - 25)**2
        return np.clip(efficiency, 0.8, nominal_eff)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.Tbatt = self.np_random.uniform(10, 40)
        self.tav = self.np_random.uniform(0, 24)
        self.tdep = (self.tav + self.np_random.uniform(3, 8)) % 24
        self.current_time = self.tav

        self.x = self.np_random.uniform(0.2, 0.6)
        self.x_target = self.np_random.uniform(0.8, 1.0)
        self.P_available = self.np_random.uniform(5, self.max_charge_power)

        self._update_trem()
        self._update_state()

        self.current_step = 0               # <-- Add step counter
        self.max_steps = int(24 / STEP_HOURS)  # 24 hours / 5 min steps = 288

        self.history = {"soc": [self.x], "temp": [self.Tbatt], "time": [self.current_time]}

        return self.state, {}

    def _update_trem(self):
        self.trem = (self.tdep - self.current_time) % 24

    def _update_state(self):
        self.state = np.array([
            self.Tbatt, self.tav, self.tdep, self.x,
            self.x_target, self.P_available, self.trem
        ], dtype=np.float32)

    def step(self, action):
        charge_power, heat_power, cool_power = np.clip(action, self.action_space.low, self.action_space.high)

        if heat_power > 0 and cool_power > 0:
            if heat_power >= cool_power:
                cool_power = 0
            else:
                heat_power = 0

        if self.x >= self.x_target:
            charge_power = 0
        else:
            charging_buffer = 0.167
            timestep_hr = STEP_HOURS
            required_energy_kWh = max(0, (self.x_target - self.x) * self.battery_capacity_kWh)
            estimated_time_hr = required_energy_kWh / (charge_power * self.mu) if charge_power > 0 else float('inf')

            if estimated_time_hr + charging_buffer < self.trem:
                max_charge_duration = self.trem - charging_buffer
                adjusted_power = required_energy_kWh / max_charge_duration / self.mu
                charge_power = min(charge_power, adjusted_power, self.max_charge_power)

        total_power = charge_power + heat_power + cool_power
        if total_power > self.P_available:
            scale = self.P_available / total_power
            charge_power *= scale
            heat_power *= scale
            cool_power *= scale
            overdraw_penalty = (total_power - self.P_available) * 10
        else:
            overdraw_penalty = 0

        timestep_hr = STEP_HOURS

        self.current_time = (self.current_time + timestep_hr) % 24
        self._update_trem()

        self.current_step += 1
        done = self.current_step >= self.max_steps  

        internal_resistance = self.temp_dependent_resistance(self.Tbatt)

        V_nominal = self.nominal_voltage
        R_internal = internal_resistance
        V_max = V_nominal + 20.0
        V_ocv = V_nominal + (V_max - V_nominal) * self.x

        if self.x < 0.8:
            I = charge_power * 1000 / V_nominal
            V_batt = V_ocv + I * R_internal
            effective_charge_power = I * V_batt / 1000
        else:
            V_batt = V_max
            I_cc = charge_power * 1000 / V_nominal
            taper_factor = np.exp(-10 * (self.x - 0.8))
            I = I_cc * taper_factor
            effective_charge_power = I * V_batt / 1000

        temp_efficiency = self.temp_dependent_efficiency(self.Tbatt)

        delta_x = (effective_charge_power * temp_efficiency * timestep_hr) / self.battery_capacity_kWh
        prev_soc = self.x
        self.x = np.clip(self.x + delta_x, 0, 1)

        ambient_temp = 25.0
        thermal_resistance = 0.1
        thermal_capacity = 10.0

        heat_from_charge = (R_internal * I ** 2) / 1000 if I > 0 else 0.0

        net_heat_power = heat_power - cool_power + heat_from_charge

        temp_diff = self.Tbatt - ambient_temp
        heat_loss = temp_diff / thermal_resistance
        total_heat_flow = net_heat_power - heat_loss
        dT = (total_heat_flow * timestep_hr) / thermal_capacity
        self.Tbatt = np.clip(self.Tbatt + dT, 0, 60)

        soc_delta = self.x - prev_soc
        soc_reward = soc_delta * 200

        zero_charge_penalty = 0
        if charge_power == 0 and self.x < self.x_target:
            zero_charge_penalty = -50

        if self.Tbatt < 20:
            temp_reward = 0
            temp_penalty = -0.1 * (20 - self.Tbatt)
        elif self.Tbatt <= 35:
            temp_reward = 5.0
            temp_penalty = 0
        else:
            temp_reward = 0
            temp_penalty = -0.5 * (self.Tbatt - 35) ** 2

        thermal_action_reward = 0
        if self.Tbatt < 20 and heat_power > 0:
            thermal_action_reward = 2.0
        elif self.Tbatt > 35 and cool_power > 0:
            thermal_action_reward = 2.0

        power_used = charge_power + heat_power + cool_power
        power_usage_penalty = -0.2 * (self.P_available - power_used) if self.x < self.x_target else 0
        completion_bonus = 200 if self.x >= self.x_target and self.trem > 0 else 0

        reward = (
            soc_reward + temp_reward + temp_penalty +
            thermal_action_reward + power_usage_penalty +
            completion_bonus - overdraw_penalty +
            zero_charge_penalty
        )
        reward = float(reward)

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
                "overdraw_penalty": overdraw_penalty
            }
        }

        # Conditional debug print
        if self.debug:
            print(f"Step {len(self.history['soc']) - 1:03d} | SOC: {self.x:.3f} | Temp: {self.Tbatt:.2f} | trem: {self.trem:.3f} hrs "
                f"| Reward: {reward:.3f} | Action: [{charge_power:.2f}, {heat_power:.2f}, {cool_power:.2f}] "
                f"| Heat from internal resistance: {heat_from_charge:.4f} kW | Estimated Temp Increase: {dT:.4f} °C")

        return self.state, reward, done, False, info

    def render(self, mode='human'):
        print(f"\nTime: {self.current_time:.2f} hr | Remaining time: {self.trem:.2f} hr")
        print(f"SOC: {self.x:.3f} | Tbatt: {self.Tbatt:.2f}°C | Available Power: {self.P_available:.2f} kW")

    def close(self):
        pass
