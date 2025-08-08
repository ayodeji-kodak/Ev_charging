"""
The module implements the EVChargingEnv class with ACN-Sim battery model.

run with 
python -m sustaingym.envs.evcharging.ev_env_soc
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import acnportal.acnsim as acns
from acnportal.acnsim.models.battery import Linear2StageBattery
import cvxpy as cp
from gymnasium import Env, spaces
import numpy as np

from .event_generation import AbstractTraceGenerator
from .utils import MINS_IN_DAY, site_str_to_site
from sustaingym.envs.utils import solve_mosek


class EVChargingEnv(Env):
    """EVCharging class with ACN-Sim battery model.

    This classes simulates the charging schedule of electric vehicles (or EVs)
    connected to an EV charging network using ACN-Sim's battery model. Each episode 
    is a 24-hour day of charging, and the simulation can be done using real data 
    from ACN-Data or a Gaussian mixture model (GMM) fitted on the data.
    
    The environment now uses ACN-Sim's Linear2StageBattery model for more realistic
    battery charging behavior.

    This environment's API is known to be compatible with Gymnasium v0.28, v0.29.

    In what follows:

    - ``n`` = number of stations in the EV charging network
    - ``k`` = number of steps for the MOER CO2 forecast

    Actions:

    .. code:: none

        Type: Box(n)
        Action                              Shape   Min     Max
        normalized pilot signal             n       0       1

    Observations:

    .. code:: none

        Type: Dict(Box(1), Box(n), Box(n), Box(1), Box(k))
                                            Shape   Min     Max
        Timestep (fraction of day)          1       0       1
        Estimated departures (timesteps)    n       -288    288
        Demands (kWh)                       n       0       Max Allowed Energy Request
        Previous MOER value                 1       0       1
        Forecasted MOER (kg CO2 / kWh)      k       0       1
        Battery SOC                         n       0       1
    """
    TIMESTEP_DURATION = 5  # in minutes
    ACTION_SCALE_FACTOR = 32  # Max charging rate in A for garage EVSEs

    # Reward calculation factors
    VOLTAGE = 208  # in volts (V), default value from ACN-Sim
    MARGINAL_REVENUE_PER_KWH = 0.15  # revenue in $ / kWh
    OPERATING_MARGIN = 0.20  # profit / revenue as a %
    MARGINAL_PROFIT_PER_KWH = MARGINAL_REVENUE_PER_KWH * OPERATING_MARGIN  # $ / kWh
    CO2_COST_PER_METRIC_TON = 30.85  # carbon cost in $ / 1000 kg CO2
    A_MINS_TO_KWH = (1 / 60) * (VOLTAGE / 1000)  # (kWh / A * mins)
    VIOLATION_WEIGHT = 0.8  # cost in $ / kWh of violation

    A_PERS_TO_KWH = A_MINS_TO_KWH * TIMESTEP_DURATION  # (kWh / A * periods)
    PROFIT_FACTOR = A_PERS_TO_KWH * MARGINAL_PROFIT_PER_KWH  # $ / (A * period)
    VIOLATION_FACTOR = A_PERS_TO_KWH * VIOLATION_WEIGHT  # $ / (A * period)
    CARBON_COST_FACTOR = A_PERS_TO_KWH * (CO2_COST_PER_METRIC_TON / 1000)  # ($ * kV * hr) / (kg CO2 * period)

    def __init__(self, data_generator: AbstractTraceGenerator,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 verbose: int = 0):
        assert 1 <= moer_forecast_steps <= 36

        # Set arguments
        self.data_generator = data_generator
        self.max_timestep = MINS_IN_DAY // self.TIMESTEP_DURATION
        self.moer_forecast_steps = moer_forecast_steps
        self.project_action_in_env = project_action_in_env
        self.verbose = verbose
        
        if self.verbose < 2:
            warnings.filterwarnings('ignore')

        # Set up infrastructure info
        self.cn = site_str_to_site(self.data_generator.site)
        self.num_stations = len(self.cn.station_ids)
        self._evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}
        
        # Initialize battery models for each station (will be populated when EVs arrive)
        self._batteries = [None] * self.num_stations

        # Initialize observation arrays
        self._est_departures = np.zeros(self.num_stations, dtype=np.float32)
        self._demands = np.zeros(self.num_stations, dtype=np.float32)
        self._prev_moer = np.zeros(1, dtype=np.float32)
        self._forecasted_moer = np.zeros(self.moer_forecast_steps, dtype=np.float32)
        self._timestep_obs = np.zeros(1, dtype=np.float32)
        self._soc = np.zeros(self.num_stations, dtype=np.float32)  # Current SOC for each station

        self.observation_space = spaces.Dict({
            'timestep':        spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'est_departures':  spaces.Box(-288, 288, shape=(self.num_stations,), dtype=np.float32),
            'demands':         spaces.Box(0, self.data_generator.requested_energy_cap,
                                        shape=(self.num_stations,), dtype=np.float32),
            'prev_moer':       spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'forecasted_moer': spaces.Box(0, 1, shape=(self.moer_forecast_steps,), dtype=np.float32),
            'soc':             spaces.Box(0, 1, shape=(self.num_stations,), dtype=np.float32),
        })

        self._obs = {
            'timestep': self._timestep_obs,
            'est_departures': self._est_departures,
            'demands': self._demands,
            'prev_moer': self._prev_moer,
            'forecasted_moer': self._forecasted_moer,
            'soc': self._soc,
        }

        # Track cumulative components of reward signal
        self._reward_breakdown = {
            'profit': 0.0,
            'carbon_cost': 0.0,
            'excess_charge': 0.0,
        }

        # Initialize variables for gym resetting
        self.t = 0
        self._simulator: acns.Simulator = None

        # Define action space for the pilot signals
        self.action_space = spaces.Box(
            low=0, high=1.0, shape=(self.num_stations,), dtype=np.float32)

        # Define reward range
        self.reward_range = (-np.inf, self.PROFIT_FACTOR * 32 * self.num_stations)

        # Set up action projection
        if self.project_action_in_env:
            self._init_action_projection()

    def _init_action_projection(self) -> None:
        """Initializes optimization problem, parameters, and variables."""
        # Projected action to be sent as actual pilot signal, normalized to [0, 1]
        self._projected_action = cp.Variable(self.num_stations, nonneg=True)

        # Parameters to be set when stepping through environment
        self._agent_action = cp.Parameter(self.num_stations, nonneg=True)
        self._demands_cvx = cp.Parameter(self.num_stations, nonneg=True)

        # Action cannot exceed maximum pilot signal or total demand of vehicle
        max_action = cp.minimum(
            1., self._demands_cvx / self.A_PERS_TO_KWH / self.ACTION_SCALE_FACTOR)

        objective = cp.Minimize(cp.norm(self._projected_action - self._agent_action, p=2))
        constraints = [
            self._projected_action <= max_action,
            magnitude_constraint(self._projected_action, self.cn)
        ]
        self.prob = cp.Problem(objective, constraints)

        assert self.prob.is_dpp() and self.prob.is_dcp()

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """Projects action to satisfy charging network constraints."""
        filtered_demands = np.where(self._est_departures > 0, self._demands, 0)

        self._projected_action.value = action  # initialize value for faster convergence
        self._agent_action.value = action
        self._demands_cvx.value = filtered_demands
        solve_mosek(self.prob, self.verbose)
        action = self._projected_action.value

        action = np.where(self._est_departures > 0, action, 0)

        return action

    def __repr__(self) -> str:
        """Returns the string representation of charging gym."""
        return (f'EVChargingGym (action projection = {self.project_action_in_env}, '
                f'moer forecast steps = {self.moer_forecast_steps}) '
                f'using {self.data_generator.__repr__()}')

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Steps the environment with battery model updates."""
        self.t += 1

        # Convert action to schedule
        schedule = self._to_schedule(action)
        
        # Update battery states before stepping simulator
        for station_id, pilot in schedule.items():
            station_idx = self._evse_name_to_idx[station_id]
            if self._batteries[station_idx] is not None:
                # Charge the battery
                self._batteries[station_idx].charge(
                    pilot[0],  # Current in A
                    self.VOLTAGE,
                    self.TIMESTEP_DURATION
                )
        
        # Step the simulator
        done = self._simulator.step(schedule)
        self._simulator._resolve = False  # work-around to keep iterating

        # Get updated observation
        observation = self._get_observation()
        reward = self._get_reward(schedule)
        info = self._get_info()

        return observation, reward, done, False, info


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed)
        self.data_generator.set_seed(seed)

        if options is not None and 'verbose' in options:
            self.verbose = options['verbose']

        # Initialize network, events, MOER data, simulator, interface, and timestep
        self.cn = site_str_to_site(self.data_generator.site)
        events, self._evs, num_plugs = self.data_generator.get_event_queue()
        self._max_profit = self._calculate_max_profit()
        self.moer = self.data_generator.get_moer()
        
        # Reset all batteries
        self._batteries = [None] * self.num_stations
        
        self._simulator = acns.Simulator(
            network=self.cn, scheduler=None, events=events,
            start=self.data_generator.day,
            period=self.TIMESTEP_DURATION, verbose=False)
        self._interface = acns.Interface(self._simulator)
        self.t = 0

        # Restart information tracking for reward component
        for reward_component in self._reward_breakdown:
            self._reward_breakdown[reward_component] = 0.0

        if self.verbose >= 1:
            print(f'Simulating {num_plugs} events using {self.data_generator}')

        return self._get_observation(), self._get_info()

    def _to_schedule(self, action: np.ndarray) -> dict[str, list[float]]:
        """Returns EVSE pilot signals given a numpy action."""
        # Set action to 0 for any EVs that have departed
        for station_idx in range(self.num_stations):
            if self._est_departures[station_idx] <= 0:
                action[station_idx] = 0

        if self.project_action_in_env:
            action = self._project_action(action)

        action *= self.ACTION_SCALE_FACTOR  # convert to (A), in [0, 32]
        pilot_signals = {}
        for i in range(self.num_stations):
            station_id = self.cn.station_ids[i]
            if self.cn.min_pilot_signals[i] == 6:
                pilot_signals[station_id] = [np.round(action[i]) if action[i] >= 6 else 0]
            else:
                pilot_signals[station_id] = [np.round(action[i] / 8) * 8]
        return pilot_signals

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Returns observations for the current state of simulation."""
        self._est_departures.fill(0)
        self._demands.fill(0)
        self._soc.fill(0)

        for session_info in self._interface.active_sessions():
            station_idx = self._evse_name_to_idx[session_info.station_id]
            
            # Get EV object
            ev = next((ev for ev in self._simulator.get_active_evs() 
                     if ev.station_id == session_info.station_id), None)
            
            if ev:
                # Initialize battery model if not already done
                if self._batteries[station_idx] is None:
                    capacity = ev.requested_energy
                    max_power = 32 * self.VOLTAGE / 1000  # Max charging power in kW (32A * 208V)
                    self._batteries[station_idx] = Linear2StageBattery(
                        capacity=capacity,
                        init_charge=0,  # Start with empty battery
                        max_power=max_power,
                        transition_soc=0.8  # Standard transition point
                    )
                
                # Update SOC based on charging progress
                remaining_energy = ev.requested_energy * ev.percent_remaining
                delivered_energy = ev.requested_energy - remaining_energy
                
                # Calculate current SOC (0-1)
                if ev.requested_energy > 0:
                    self._soc[station_idx] = delivered_energy / ev.requested_energy
                else:
                    self._soc[station_idx] = 0.0
                
                # Update demands
                self._demands[station_idx] = remaining_energy

            self._est_departures[station_idx] = session_info.estimated_departure - self.t

        self._prev_moer[0] = self.moer[self.t, 0]
        self._forecasted_moer[:] = self.moer[self.t, 1:self.moer_forecast_steps + 1]
        self._timestep_obs[0] = self.t / self.max_timestep

        return self._obs


    def _get_info(self, all: bool = False) -> dict[str, Any]:
        """Returns info. See `step()`."""
        info = {
            'max_profit': self._max_profit,
            'reward_breakdown': self._reward_breakdown
        }
        if all:
            info.update({
                'num_evs': len(self._evs),
                'avg_plugin_time': self._calculate_avg_plugin_time(),
                'evs': self._evs,
                'active_evs': self._simulator.get_active_evs(),
                'moer': self.moer,
                'pilot_signals': self._simulator.pilot_signals_as_df(),
                'batteries': self._batteries
            })
        return info

    def _calculate_avg_plugin_time(self) -> float:
        """Calculate average plug-in times for evs in periods."""
        return np.mean([ev.departure - ev.arrival for ev in self._evs])

    def _calculate_max_profit(self) -> float:
        """Calculate max profits without regards to network constraints."""
        requested_energy = np.array([ev.requested_energy for ev in self._evs])
        duration_in_periods = np.array([ev.departure - ev.arrival for ev in self._evs])
        max_kwh_in_duration = duration_in_periods * self.ACTION_SCALE_FACTOR * self.A_PERS_TO_KWH
        max_kwh_to_provide = np.minimum(requested_energy, max_kwh_in_duration)
        max_profit = np.sum(max_kwh_to_provide * self.MARGINAL_PROFIT_PER_KWH)
        return max_profit

    def _get_reward(self, schedule: Mapping[str, Sequence[float]]) -> float:
        """Returns total reward for scheduler performance on current timestep."""
        # profit calculation (Amp * period) -> ($)
        total_charging_rate = np.sum(self._simulator.charging_rates[:, self.t-1])  # in (A)
        profit = self.PROFIT_FACTOR * total_charging_rate

        # Network constraints - amount of charge over maximum allowed rates ($)
        schedule = np.array([x[0] for x in schedule.values()])  # convert to numpy
        current_sum = np.abs(self._simulator.network.constraint_current(schedule))
        excess_current = np.sum(np.maximum(0, current_sum - self._simulator.network.magnitudes))
        excess_charge = excess_current * self.VIOLATION_FACTOR

        # Carbon cost (Amp * period * kg CO2 / kWh) -> ($)
        carbon_cost = self.CARBON_COST_FACTOR * total_charging_rate * self.moer[self.t, 0]

        total_reward = profit - carbon_cost - excess_charge

        # Update reward information-tracking
        self._reward_breakdown['profit'] += profit
        self._reward_breakdown['carbon_cost'] += carbon_cost
        self._reward_breakdown['excess_charge'] += excess_charge

        return total_reward

    def close(self) -> None:
        """Close the environment. Delete internal variables."""
        del self._simulator, self.cn


def magnitude_constraint(action: cp.Variable, cn: acns.ChargingNetwork
                         ) -> cp.Constraint:
    """Creates constraint requiring that aggregate magnitude (A) must be less
    than observation magnitude (A)."""
    phase_factor = np.exp(1j * np.deg2rad(cn._phase_angles))  # shape [num_stations]
    A_tilde = cn.constraint_matrix * phase_factor[None, :]  # shape [num_constraints, num_stations]

    # convert to A
    agg_magnitude = cp.abs(A_tilde @ action) * EVChargingEnv.ACTION_SCALE_FACTOR

    if len(action.shape) == 1:
        # agg_magnitude has shape [num_constraints]
        return agg_magnitude <= cn.magnitudes
    elif len(action.shape) == 2:
        # agg_magnitude has shape [num_constraints, T]
        return agg_magnitude <= cn.magnitudes[:, None]
    else:
        raise ValueError(
            'Action should have shape [num_stations] or [num_stations, T], '
            f'but received shape {action.shape} instead.')
