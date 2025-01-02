from mesa import Agent
import numpy as np
import pandas as pd
from datetime import time
# from abm_model import Dishwasher, Toilet, Garden, Sink, Shower, WashingMachine, Pool, DynamicPricing

rng = np.random.default_rng(1234)

# path = '/home/cade/Documents/Research/ABM/dynamic_pricing/py_files+data/'
# data = pd.read_csv(path+'dataset_80.csv')   
# df = pd.DataFrame(data)

class Device:
    def __init__(self, device_name, weekly_sched, daily_sched, flow_rate, duration):
        self.device_name = device_name
        # self.n_weekly = n_weekly
        self.weekly_sched = weekly_sched
        # self.n_daily = n_daily
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate        
        self.duration = duration

    def update_flow(self, new_flow):
        self.flow_rate = new_flow

    def update_duration(self, perc):
        self.duration *= perc

    def update_daily_schedule(self, new_sched):
        self.daily_sched = new_sched

    def update_weekly_schedule(self, new_sched):
        self.weekly_sched = new_sched
        
    def volume_used(self):
        self.volume = np.round((self.duration * self.flow_rate), 2)
        return self.volume
    
# class Shower(Device, Schedule):
#     def __init__(self, weekly_sched, daily_sched, tou_rate, flow_rate, duration, person):
#         Device.__init__(self, "shower", tou_weight, flow_rate, duration)
#         Schedule.__init__(self, "shower", weekly_sched, daily_sched)
#         self.person = person

        
class Shower(Device):
    def __init__(self, weekly_sched, daily_sched, flow_rate, duration, person):
        super().__init__("shower", weekly_sched, daily_sched, flow_rate, duration)

        self.weekly_sched = weekly_sched
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration
        self.person = person

class Toilet(Device):
    def __init__(self, daily_sched, flow_rate, duration, person):
        super().__init__("toilet", [1,1,1,1,1,1,1], daily_sched, flow_rate, duration)
        
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration
        self.person = person

class Sink(Device):
    def __init__(self, daily_sched, flow_rate, duration, person):
        super().__init__("sink", [1,1,1,1,1,1,1], daily_sched, flow_rate, duration)
        
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration
        self.person = person

class Dishwasher(Device):
    def __init__(self, weekly_sched, daily_sched, flow_rate, duration):
        super().__init__("dishwasher", weekly_sched, daily_sched, flow_rate, duration)

        self.weekly_sched = weekly_sched
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration
        
class WashingMachine(Device):
    def __init__(self, weekly_sched, daily_sched, flow_rate, duration):
        super().__init__("washing machine", weekly_sched, daily_sched, flow_rate, duration)

        self.weekly_sched = weekly_sched
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration

class Garden(Device):
    def __init__(self, weekly_sched, daily_sched, flow_rate, duration):
        super().__init__("garden", weekly_sched, daily_sched, flow_rate, duration)
        
        self.weekly_sched = weekly_sched
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration 
        

class Pool(Device):
    def __init__(self, weekly_sched, daily_sched, flow_rate, duration):
        super().__init__("pool", weekly_sched, daily_sched, flow_rate, duration)
        
        self.weekly_sched = weekly_sched
        self.daily_sched = daily_sched
        self.flow_rate = flow_rate
        self.duration = duration 

class WaterUser(Agent):
    """An agent that uses water. """
    def __init__(self, model):
        super().__init__(model)
        self.time = 0
        
        # [-1.0, 1.0]
        self.elasticity = 0.0
        # [5,000, 150,000]
        self.income = 0
        # [1, 7]
        self.inhabitants = np.random.choice(np.arange(1,8), p=[0.246, 0.261, 0.193, 0.179, 0.076, 0.030, 0.015])
        # "low income", "medium income", "high income"
        self.income_level = None
        # "environmentalist", "committed", "techno-solutionist", "client"
        self.social_values = None
        # self.res_density = np.random.choice(['low density', 'medium density', 'high density'], p=[0.12, 0.02, 0.86])
        self.res_density = rng.choice(['low density', 'high density'], p=[0.13, 0.87])
        # "owner", "renter"
        self.home_tenure = None
        # "low info", "medium info", "high info"
        self.info_level = None
        # True, False
        self.has_garden = False
        # True, False
        self.has_pool = False
        # True, False
        self.pool_on = False
        # "DEM", "DM", "DE", "SMD", "DLN"
        self.cluster = None
        self.neighbors = None
        self.garden_rec = None

        self.tou_weights = {
            "toilet": "standard",
            "sink": "standard",
            "shower": "standard",
            "washing machine": "standard",
            "dishwasher": "standard",
            "garden": "standard",
        }

        self.flows = {
            "toilet": "standard",
            "sink": "standard",
            "shower": "standard",
            "washing machine": "standard",
            "dishwasher": "standard",
            "garden": "standard"
        }

        self.durations = {
            "toilet": "standard",
            "sink": "standard",
            "shower": "standard",
            "washing machine": "standard",
            "dishwasher": "standard",
            "garden": "standard"
        }

        self.toilet_consumption = 0
        self.shower_consumption = 0
        self.sink_consumption = 0
        self.washing_machine_consumption = 0
        self.dishwasher_consumption = 0 
        self.garden_consumption = 0
        self.pool_consumption = 0
        
        self.shower_schedule = list()         
        self.toilet_schedule = list()
        self.sink_schedule = list()
        self.dishwasher_schedule = None
        self.washing_machine_schedule = None
        self.garden_schedule = None
        self.pool_schedule = None

        self.water_use_schedule = {
            "shower": self.shower_schedule,
            "toilet": self.toilet_schedule,
            "sink": self.sink_schedule,
            "diswasher": self.dishwasher_schedule,
            "washing machine": self.washing_machine_schedule,
            "garden": self.garden_schedule,
            "pool": self.pool_schedule
        }
        
        # in Netlogo "volume-consumed" is list with length(devices). each value is equal to use-duration * standard-flow ( * consions-flow if agent has adapted consious flow rates)
        # in Python "{device}_schedule" is a list of length(self.inhabitants) of Device objects (see abm_model.py), or a single Device object
        # volume_consumed is returned from a Device object function
        
        self.weekly_consumption = 0
        self.previous_weekly_consumption = 0 
        self.bimonthly_consumption = 0
        self.monthly_consumption = 0
        self.annual_bill = 0
        self.daily_consumption = 0
        self.cum_consumption = 0 

        self.peak_consumption = 0
        self.flat_consumption = 0
        self.valley_consumption = 0 
        
        self.bimonthly_peak_consumption = 0
        self.bimonthly_valley_consumption = 0
        self.bimonthly_flat_consumption = 0 
        
        self.device_changes = list()
        self.practice_changes = list()
        self.schedule_changes = list()

        # self.change_schedule = False
        # self.change_device = False
        # self.change_practice = False

        self.device_count = 0
        self.practice_count = 0
        self.schedule_count = 0

        self.count_weeks = 0
        self.to_change_schedule = False
        self.to_change_device = False
        self.to_change_behavior = False
        self.to_change = False
        
    def set_shower_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["shower"]
        weights = weights.xs(weights_type, level=0)["shower"].to_list()
        for i in np.arange(1, self.inhabitants+1):
            if self.model.season == "winter":
                n_weekly = 4 + rng.integers(4)
            elif self.model.season in ["spring", "autumn"]:
                n_weekly = 5 + rng.integers(3)
            else:
                n_weekly = 6 + rng.integers(2)
            weekly_sched = [True] * 7
            if n_weekly != 7:
                for x in rng.integers(7, size=7-n_weekly):
                    weekly_sched[x] = False
            daily_sched = {}
            for day in np.arange(7):
                if weekly_sched[day]:
                    tim = rng.choice(24, p=weights)
                    daily_sched[day] = [time(tim)]
                else:
                    daily_sched[day] = []
            if initial:
                flows_type = self.flows["shower"]
                flow_rate = flow[flows_type]["shower"]
                duration = rng.triangular(2,5,20)
                self.shower_schedule.append(Shower(
                                                   weekly_sched, 
                                                   daily_sched, 
                                                   flow_rate, 
                                                   duration, 
                                                   i
                                               )
                                        )
            else:
                self.shower_schedule[i].update_daily_sched(daily_sched)
            
    def set_toilet_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["toilet"]
        weights = weights.xs(weights_type)["toilet"].to_list()
        for i in np.arange(1, self.inhabitants+1):
            daily_sched = {}
            for day in np.arange(7):
                daily_sched[day] = []
                n_daily = rng.triangular(4,5,8)
                for n in np.arange(n_daily+1):
                    tim = rng.choice(24, p=weights)
                    daily_sched[day].append(time(tim))
            if initial:
                duration = 1
                flows_type = self.flows["toilet"]
                flow_rate = flow[flows_type]["toilet"]
                self.toilet_schedule.append(Toilet(
                                                   daily_sched, 
                                                   flow_rate, 
                                                   duration, 
                                                   i
                                               )
                                        )
            else:
                self.toilet_schedule[i].update_daily_sched(daily_sched)

    def set_sink_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["sink"]
        weights = weights.xs(weights_type)["sink"].to_list()
        for i in np.arange(1, self.inhabitants+1):
            n_daily = rng.triangular(4,5,8)
            daily_sched = {}
            for day in np.arange(7):
                daily_sched[day] = []
                for n in np.arange(n_daily+1):
                    tim = rng.choice(24, p=weights)
                    daily_sched[day].append(time(tim))
            if initial:
                duration = rng.triangular(0.1, 0.2, 2.0)
                flows_type = self.flows["sink"]
                flow_rate = flow[flows_type]["sink"]
                self.sink_schedule.append(Sink(daily_sched, 
                                                 flow_rate, 
                                                 duration, 
                                                 i
                                               )
                                        )
            else:
                self.sink_schedule[i].update_daily_sched(daily_sched)

    def set_dishwasher_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["dishwasher"]
        weights = weights.xs(weights_type)["dishwasher"].to_list()
        n_weekly = self.inhabitants + rng.integers(low=-2, high=2)
        weekly_sched = [True] * 7
        if n_weekly != 7:
            for x in rng.integers(7, size=7-n_weekly):
                weekly_sched[x] = False
        daily_sched = {}
        for day in np.arange(7):
            if weekly_sched[day]:
                tim = rng.choice(24, p=weights)
                daily_sched[day] = [time(tim)]
            else:
                daily_sched[day] = []
        if initial:
            duration = 1
            flows_type = self.flows["dishwasher"]
            flow_rate = flow[flows_type]["dishwasher"]
            self.dishwasher_schedule = Dishwasher( 
                                               weekly_sched, 
                                               daily_sched, 
                                               flow_rate, 
                                               duration 
                                           )
        else:
            self.dishwasher_schedule.update_daily_schedule(daily_sched)
            self.dishwasher_schedule.update_weekly_schedule(weekly_sched)
        
    def set_washing_machine_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["washing_machine"]
        weights = weights.xs(weights_type)["washing_machine"].to_list()
        n_weekly = min([2 * self.inhabitants + rng.integers(2), 7])
        weekly_sched = [True] * 7
        if n_weekly != 7:
            for x in rng.integers(7, size=7-n_weekly):
                weekly_sched[x] = False
        daily_sched = {}
        for day in np.arange(7):
            if weekly_sched[day]:
                tim = rng.choice(24, p=weights)
                daily_sched[day] = [time(tim)]
            else:
                daily_sched[day] = []
        if initial:
            duration = 1
            flows_type = self.flows["washing_machine"]
            flow_rate = flow[flows_type]["washing_machine"]
            self.washing_machine_schedule = WashingMachine( 
                                               weekly_sched, 
                                               daily_sched, 
                                               flow_rate, 
                                               duration 
                                           )                               
        else:
            self.washing_machine_schedule.update_daily_schedule(daily_sched)
            self.washing_machine_schedule.update_weekly_schedule(weekly_sched)
        
    def set_garden_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["garden"]
        weights = weights.xs(weights_type)["garden"].to_list()
        n_weekly = 1 + rng.integers(4) 
        weekly_sched = [False] * 7 
        for x in rng.integers(7, size=n_weekly):
            weekly_sched[x] = True
        daily_sched = {}
        for day in np.arange(7):
            if weekly_sched[day]:
                tim = rng.choice(24, p=weights)
                daily_sched[day] = [time(tim)]
            else:
                daily_sched[day] = []
        if initial:
            duration = rng.triangular(10,20,60)
            flows_type = self.flows["garden"]
            flow_rate = flow[flows_type]["garden"]
            self.garden_schedule = Garden(weekly_sched, 
                                          daily_sched, 
                                          flow_rate, 
                                          duration
                                      )
        else:
            self.garden_schedule.update_weekly_schedule(weekly_sched)
            self.garden_schedule.update_daily_schedule(daily_sched)

    def set_pool_schedule(self, weights, flow, initial=False):
        weights_type = self.tou_weights["pool"]
        weights = weights.xs(weights_type)["pool"].to_list()
        n_weekly = rng.integers(2) 
        weekly_sched = [False] * 7 
        for x in rng.integers(7, size=n_weekly):
            weekly_sched[x] = True
        daily_sched = {}
        for day in np.arange(7):
            if weekly_sched[day]:
                tim = rng.choice(24, p=weights)
                daily_sched[day] = [time(tim)]
            else:
                daily_sched[day] = []
        if initial:
            flows_type = self.flows["pool"]
            flow_rate = flow[flows_type]["pool"]
            duration = rng.triangular(10,20,60)
            self.pool_schedule = Pool(
                                      weekly_sched, 
                                      daily_sched, 
                                      flow_rate, 
                                      duration
                                  )        
        else:
            self.pool_schedule.update_daily_schedule(daily_sched)
            self.pool_schedule.update_weekly_schedule(weekly_sched)
        
    def set_schedules(self, weights, flows):
        self.set_dishwasher_schedule(weights, flows)
        self.set_washing_machine_schedule(weights, flows)
        self.set_garden_schedule(weights, flows)
        self.set_pool_schedule(weights, flows)
        self.set_shower_schedule(weights, flows)
        self.set_sink_schedule(weights, flows)
        self.set_toilet_schedule(weights, flows)
        
    def check_schedule(self):
        current_day = self.model.datetime.weekday()
        current_time = self.model.datetime.time()
        for i in self.inhabitants:
            if current_time in self.toilet_schedule[i].daily_sched[current_day]:
                volume = self.shower_schedule[i].volume_used()
                self.shower_consumption += volume
                self.use_water(volume, current_time)
            if current_time in self.sink_schedule[i].daily_sched[current_day]:
                volume = self.sink_schedule[i].volume_used()
                self.sink_consumption += volume
                self.use_water(volume, current_time)
            if self.shower_schedule[i].weekly_sched[current_day]:
                if current_time in self.shower_schedule[i].daily_sched[current_day]:
                    volume = self.shower_schedule[i].volume_used()
                    self.shower_consumption += volume
                    self.use_water(volume, current_time)
        if self.dishwasher_schedule.weekly_sched[current_day]:
            if current_time in self.dishwasher_schedule.daily_sched[current_day]:
                volume = self.dishwasher_schedule.volume_used()
                self.dishwasher_consumption += volume
                self.use_water(volume, current_time)
        if self.washing_machine_schedule.weekly_sched[current_day]:
            if current_time in self.washing_machine_schedule.daily_sched[current_day]:
                volume = self.washing_machine_schedule.volume_used()
                self.washing_machine_consumption += volume
                self.use_water(volume, current_time)
        if self.garden_schedule.weekly_sched[current_day]:
            if current_time in self.garden_schedule.daily_sched[current_day]:
                volume = self.garden_schedule.volume_used()
                self.garden_consumption += volume
                self.use_water(volume, current_time)
        if self.pool_schedule.weekly_sched[current_day]:
            if current_time in self.pool_schedule.daily_sched[current_day]:
                volume = self.pool_schedule.volume_used()
                self.pool_consumption += volume
                self.use_water(volume, current_time)
            else:
                pass
        else:
            pass
        
    def use_water(self, volume, time):
        self.weekly_consumption += volume
        self.bimonthly_consumption += volume
        self.monthly_consumption += volume
        self.daily_consumption += volume
        self.cum_consumption += volume
        if self.model.tariff != "Standard IBR":
            for times in self.model.peak_times:
                if time >= times[0] and time <= times[1]:
                    self.peak_consumption += volume
                    self.bimonthly_peak_consumption += volume
                else:
                    pass
            for times in self.model.flat_times:
                if time >= times[0] and time <= times[1]:
                    self.flat_consumption += volume
                    self.bimonthly_flat_consumption += volume
                else:
                    pass
            for times in self.model.valley_times:
                if time >= times[0] and time <= times[1]:
                    self.valley_consumption += volume
                    self.bimonthly_valley_consumption += volume

    def reset_tou_consumption(self):
        self.peak_consumption = 0
        self.valley_consumption = 0
        self.flat_consumption = 0

    def reset_bi_consumption(self):
        self.bimonhtly_peak_consumption = 0
        self.bimonthly_valley_consumption = 0
        self.bimonthly_flat_consumption = 0
        self.bimonthly_consumption = 0

    def reset_weekly_consumption(self):
        self.weekly_consumption = 0

    def bimonthly_bill(self):
        if self.model.tariff == "Standard IBR":
            p = self.model.fixed_fee
            q = np.round(self.bimonthly_consumption / 748, 2)
            if self.model.tier1v < q <= self.model.tier2v:
                p += (q-self.model.tier1v) * self.model.tier1p
            elif q > self.model.tier2v:
                p = p + self.model.tier1p + (q-self.model.tier2v) * self.model.tier2p 
            else:
                print("Error calculating bill")
            self.bill = np.round(p * 1.03, 2)
        elif self.model.tariff == "Dynamic Pricing 1":
            valley_fraction = np.round(self.bimonthly_valley_consumption / 748, 2)
            flat_fraction = np.round(self.bimonthly_flat_consumption / 748, 2)
            peak_fracton = np.round(self.bimonthly_peak_consumption / 748, 2)
            bill = self.model.fixed_fee + valley_fraction * self.model.valley_rate + flat_fraction * self.model.flat_rate + peak_fraction * self.model.peak_rate
            self.bill = np.round(bill * 1.03, 2)
        self.annual_bill += self.bill

    def calculate_projected_bill(self, mult):
        projected_valley = self.valley_consumption * mult
        projected_peak = self.peak_consumption * mult
        projected_flat = self.flat_consumption * mult
        projected_bill = (projected_valley * self.model.valley_rate + projected_flat * self.model.flat_rate + projected_peak * self.model.peak_rate) * 1.03
        return projected_bill
        
    def check_change(self):
        if self.model.datetime not in self.model.billing_dates:
            if self.info_level == "high info":
                mult = 8
            elif self.info_level == "medium info":
                mult = 2
            elif self.info_level == "low info":
                mult = 1
            else:
                mult = 1
            bill = self.calculate_projected_bill(mult)
        else:
            bill = self.bill
        if self.social_values == "client" or self.social_values == "techno-solutionist":
            if bill / self.income > self.model.thresholds[self.income_level] and self.device_count + self.practice_count < 8 and self.schedule_count < 4:
                change = rng.integers(1, 4)
                if change == 1:
                    self.to_change = True
                elif change == 2:
                    self.to_change_schedule = True
                else:
                    self.to_change = True
                    self.to_change_schedule = True
            else:
                self.to_change = False
                self.to_change_schedule = False
                
        elif self.social_values == "committed" or self.social_values == "environmentalist":
            neigbors_average = self.model.agents.select(lambda a: a.unique_id in self.neighbors).agg("weekly_consumption", np.mean) 
            if self.weekly_consumption < neighbors_average and self.device_count + self.practice_count < 6:
            # if self.weekly_consumption < self.model.average_weekly_consumption:
                self.to_change = True
                self.to_change_schedule = True
            else:
                self.to_change = False
                self.to_change_schedule = False

    def get_neighbors(self):
        try:
            self.neighbors = list(self.model.social_network.nodes[self.res_density]["graph"][self.unique_id])
        except Exception as e:
            print("An error occurred: ", e)
            print("Make sure model.social_network is defined.")
        
    def apply_change(self):
        if self.social_values == "client" and self.home_tenure == "owner":
            if self.device_count < 5 and self.practice_count < 4:
                p = rng.choice([self.change_practice(), self.change_device()], p=[0.35, 0.65])
                p()
            elif self.device_count == 5 and self.practice_count < 4:
                self.change_practice()
            elif self.practice_count == 4 and self.device_count < 5:
                self.change_device()
            else:
                pass
            # self.change_schedule()
        elif self.social_values == "client" and self.home_tenure == "renter":
            self.change_practice()

        elif self.social_values == "techno-solutionist" and self.home_tenure == "owner":
            self.change_device()
            # self.check_neighbors()
            # self.change_schedule()
            
    def apply_change_schedule(self):
        if self.social_values == "client":
            self.change_schedule()
        elif self.social_values == "techno-solutionist":
            self.check_social()
            self.change_schedule()
        
        
        

    def check_social(self):
        social = len(self.neighbors)
        mornings = []
        evenings = []
        for i in self.neighbors:
            if 2 <= np.mean(i.garden_schedule.daily_sched) < 9:
                mornings.append(True)
            elif np.mean(i.garden_schedule.daily_sched) > 20 or np.mean(i.garden_schedule.daily_sched) < 2:
                evenings.append(True)           
        if len(mornings) / social > 0.60:
            self.tou_weights["garden"] = "morning"
            # TODO: use set_garden_schedule
            self.garden_ref = "morning"
        if len(evenings) / social > 0.60:
            self.tou_weights["garden"] = "evening"
            # TODO: use set_garden_schedule
            self.garden_ref = "evening"
            
    def change_device(self):
        devices = []
        weights = []
        for dev, flo in self.device_flows.items():
            if flo == "standard":
                devices.append(dev)
                weights.append(self.model.device_costs[dev])
        device = rng.choice(devices, p=weights)
        self.flows[device] = "conscious"        
        schedule = self.water_use_schedule[device]
        flow = self.model.flows[self.flows[device]][device]
        if isinstance(device, dict):
            schedule.update_flow(flow)
        elif isinstance(device, list):
            for i in schedule:
                i.update_flow(flow)
        self.device_count += 1
                
    def change_practice(self):
        devices = []
        for dev, duration in self.durations.items():
            if duration == "standard":
                devices.apend(dev)
        device = rng.choice(devices)
        schedule = self.water_use_schedule[device]
        self.durations[device] = "conscious"
        # device = rng.choice(self.shower_schedule, self.dishwasher_schedule, self.washing_machine_schedule, self.toilet_schedule, self.sink_schedule, self.garden_schedule)
        if isinstance(schedule, dict):
            percent_decrease = rng.uniform(low=0.6, high=1.0)
            schedule.update_duration(percent_decrease)
        elif isinstance(schedule, list):
            for i in schedule:
                percent_decrease = rng.uniform(low=0.6, high=1.0)
                i.update_duration(percent_decrease)

    def change_schedule(self):
        devices = []
        for dev, time in self.tou_weights.items():
            if time == "standard":
                devices.append(dev)
        device = rng.choice(devices)
        schedule = self.water_use_schedule[device]
        self.tou_weights[device] = "conscious"
        
                  
    def adapt_to_price(self):
        """checks price of water and adjusts use if the user is able to
            adapt to price increases"""
        amount = self.use_water()
        if self.water_system.price == self.water_system.price1:
            delta_p = self.water_system.price1 - self.water_system.price
        elif self.water_system.price == self.water_system.price2:
            delta_p = self.water_system.price2 - self.water_system.price
        elif self.water_system.price == self.water_system.price3:
            delta_p = self.water_system.price3 - self.water_system.price
        delta_q = self.e*delta_p
        self.water_use = (1+delta_q)*amount
        self.tot_water_use += (1+delta_q)*amount
        self.water_system.water_use_cap += (1+delta_q)*amount
        return self.water_use, self.tot_water_use, self.water_system.water_use_cap

    def step(self):
        # each user agent uses water
        self.use_water()
        # turn water on/off and increment water usage
        self.adapt_to_price()
        # change price of water
        self.change_price()
        # change time at each step
        self.change_time()


        





