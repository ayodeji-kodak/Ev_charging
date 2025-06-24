# EV Laxity-Based Heating and Charging Scheduler (Multi-EV)

from typing import List
import math

class EV:
    def __init__(self, id, SoC, T_batt, T_ambient, time_left, target_SoC, battery_capacity):
        self.id = id
        self.SoC = SoC  # State of Charge [0.0 - 1.0]
        self.T_batt = T_batt  # Battery temperature in Celsius
        self.T_ambient = T_ambient  # Ambient temperature in Celsius
        self.time_left = time_left  # Time left until departure (minutes)
        self.target_SoC = target_SoC
        self.battery_capacity = battery_capacity  # in kWh
        self.laxity = None

# System Parameters
P_station_max = 100  # kW total power available at the station
DELTA_T = 1  # Time step in minutes

# Helper Functions
def estimate_max_charging_rate(T_batt):
    if T_batt < 10:
        return 10  # kW
    elif T_batt > 20:
        return 50
    return 20 + 3 * (T_batt - 10)

def f_temp(T_batt):
    return max(0.5, min(1.0, (T_batt - 0) / 20))

def decide_heating_power(T_batt):
    return 3 if T_batt < 15 else 0

def heating_model(P_heat, T_batt, T_ambient):
    # Simplified heating dynamics
    heat_transfer_coeff = 0.05
    temp_gain = (P_heat * 0.2) - heat_transfer_coeff * (T_batt - T_ambient)
    return temp_gain

def log(id, laxity, P_heat, P_charge, SoC, T_batt):
    print(f"EV{id}: Laxity={laxity:.2f}, P_heat={P_heat:.2f}kW, P_charge={P_charge:.2f}kW, SoC={SoC:.2f}, T_batt={T_batt:.2f}")

# Scheduler Loop
def laxity_scheduler(fleet: List[EV]):
    while any(ev.SoC < ev.target_SoC for ev in fleet):

        # 1. Compute laxity
        for ev in fleet:
            energy_needed = (ev.target_SoC - ev.SoC) * ev.battery_capacity
            avg_charging_rate = estimate_max_charging_rate(ev.T_batt)
            time_to_full_charge = energy_needed / avg_charging_rate * 60  # to minutes
            ev.laxity = ev.time_left - time_to_full_charge

        # 2. Sort fleet by laxity
        fleet_sorted = sorted(fleet, key=lambda x: x.laxity)
        remaining_power = P_station_max

        for ev in fleet_sorted:
            if ev.SoC >= ev.target_SoC:
                continue

            P_heat = decide_heating_power(ev.T_batt)
            max_charge = estimate_max_charging_rate(ev.T_batt)
            P_charge = max(0, min(remaining_power - P_heat, max_charge))

            # Scale down if needed
            total_power = P_heat + P_charge
            if total_power > remaining_power:
                scale = remaining_power / total_power
                P_heat *= scale
                P_charge *= scale

            # 4. Update EV states
            delta_temp = heating_model(P_heat, ev.T_batt, ev.T_ambient)
            ev.T_batt += delta_temp
            charging_eff = f_temp(ev.T_batt)
            ev.SoC += (P_charge * charging_eff * DELTA_T) / (60 * ev.battery_capacity)
            ev.SoC = min(ev.SoC, ev.target_SoC)
            ev.time_left -= DELTA_T

            remaining_power -= (P_heat + P_charge)

            log(ev.id, ev.laxity, P_heat, P_charge, ev.SoC, ev.T_batt)

        print("-- End of Time Step --\n")

# Sample Fleet
ev_fleet = [
    EV(id=1, SoC=0.4, T_batt=5, T_ambient=-10, time_left=120, target_SoC=0.9, battery_capacity=60),
    EV(id=2, SoC=0.2, T_batt=10, T_ambient=-5, time_left=90, target_SoC=0.8, battery_capacity=70),
    EV(id=3, SoC=0.6, T_batt=15, T_ambient=0, time_left=60, target_SoC=0.95, battery_capacity=50),
]

# Run Scheduler
laxity_scheduler(ev_fleet)
