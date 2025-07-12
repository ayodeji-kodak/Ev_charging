import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EVChargingEnv(gym.Env):
    """
    EV Charging Environment.

    Observation space (state):
        Tbatt: Battery temperature at arrival (°C)
        tav: Time of arrival (hour in 24h format, 0–24)
        tdep: Time of departure (hour in 24h format, 0–24)
        SOC: State of Charge [0–1]
        x_target: Target SOC [0–1]
        P_available: Available charging power (kW)
        trem: Remaining time to charge (hours) = (tdep - current_time), updated each step

    Action space:
        [charging_power (0–22 kW), heat_power (0–3 kW), cool_power (0–3 kW)]
    """

    def __init__(self):
        super(EVChargingEnv, self).__init__()

        self.max_charge_power = 22.0  # kW
        self.max_heat_power = 3.0     # kW
        self.max_cool_power = 3.0     # kW
        self.mu = 0.95                # Charging efficiency
        self.battery_capacity_kWh = 60.0
        self.nominal_voltage = 400.0
        self.internal_resistance = 0.01  # Ohms

        # Observation: [Tbatt, tav, tdep, SOC, x_target, P_available, trem]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([60, 24, 24, 1, 1, 30, 24], dtype=np.float32),
            dtype=np.float32
        )

        # Action: [charge_power, heat_power, cool_power]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.max_charge_power, self.max_heat_power, self.max_cool_power], dtype=np.float32),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.Tbatt = self.np_random.uniform(10, 40)
        self.tav = self.np_random.uniform(0, 24)
        self.tdep = self.tav + self.np_random.uniform(3, 8)  # Departure 3–8 hours after arrival
        self.tdep = self.tdep % 24  # Wrap within 24h
        self.current_time = self.tav

        self.x = self.np_random.uniform(0.2, 0.6)  # Initial SOC
        self.x_target = self.np_random.uniform(0.8, 1.0)
        self.P_available = self.np_random.uniform(5, self.max_charge_power)

        self._update_trem()
        self._update_state()

        self.history = {
            "soc": [self.x],
            "temp": [self.Tbatt],
            "time": [self.current_time]
        }

        return self.state, {}

    def _update_trem(self):
        """Update remaining time based on current time and tdep, considering wrap-around."""
        delta = (self.tdep - self.current_time) % 24
        self.trem = delta

    def _update_state(self):
        self.state = np.array([
            self.Tbatt, self.tav, self.tdep, self.x,
            self.x_target, self.P_available, self.trem
        ], dtype=np.float32)

    def step(self, action):
        charge_power, heat_power, cool_power = action
        charge_power = np.clip(charge_power, 0, self.max_charge_power)
        heat_power = np.clip(heat_power, 0, self.max_heat_power)
        cool_power = np.clip(cool_power, 0, self.max_cool_power)

        total_power = charge_power + heat_power + cool_power

        # Handle overdraw penalty
        if total_power > self.P_available:
            scale = self.P_available / total_power
            charge_power *= scale
            heat_power *= scale
            cool_power *= scale
            overdraw_penalty = (total_power - self.P_available) * 5
        else:
            overdraw_penalty = 0

        timestep_hr = 1 / 60  # 1-minute timestep

        # CCCV tapering after 80% SOC
        if self.x >= 0.8:
            taper_factor = np.exp(-5 * (self.x - 0.8))
            charge_power *= taper_factor

        delta_x = (charge_power * self.mu * timestep_hr) / self.battery_capacity_kWh
        self.x = np.clip(self.x + delta_x, 0, 1)

        # Temperature update
        current = (charge_power * 1000) / self.nominal_voltage if charge_power > 0 else 0
        heat_from_charge = self.internal_resistance * current ** 2 / 1000  # kW

        ambient_temp = 25.0
        thermal_resistance = 0.1  # K/kW
        thermal_capacity = 10.0   # kWh/K

        net_heat_power = heat_power - cool_power + heat_from_charge
        heat_loss = (self.Tbatt - ambient_temp) / thermal_resistance
        dT = (net_heat_power - heat_loss) * timestep_hr / thermal_capacity
        self.Tbatt = np.clip(self.Tbatt + dT, 0, 60)

        # Advance time
        self.current_time = (self.current_time + timestep_hr) % 24
        self._update_trem()

        done = self.trem <= 0 or self.x >= self.x_target

        # Reward
        soc_reward = -abs(self.x_target - self.x) * 10
        temp_penalty = -5 if (self.Tbatt < 0 or self.Tbatt > 45) else -abs(25 - self.Tbatt) * 0.1
        power_penalty = -total_power * 0.01
        reward = soc_reward + temp_penalty + power_penalty - overdraw_penalty

        # Update state and history
        self._update_state()
        self.history["soc"].append(self.x)
        self.history["temp"].append(self.Tbatt)
        self.history["time"].append(self.current_time)

        info = {
            "soc": self.x,
            "tbatt": self.Tbatt,
            "trem": self.trem
        }

        return self.state, reward, done, False, info

    def render(self, mode='human'):
        print(f"\nTime: {self.current_time:.2f} hr | Remaining time: {self.trem:.2f} hr")
        print(f"SOC: {self.x:.3f} | Tbatt: {self.Tbatt:.2f}°C | Available Power: {self.P_available:.2f} kW")

    def close(self):
        pass
