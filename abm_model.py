from mesa import Model
from abm_agents import WaterUser
from mesa.datacollection import DataCollector
# from mesa.time import RandomActivation 
import numpy as np
import networkx as nx
from datetime import datetime, timedelta, time
import schedule
import os
import pandas as pd
rng = np.random.default_rng(1234)

# Datacollector functions
def get_water_price(model):
    """return price of water"""
    water_price = model.water_system.price
    return water_price

def get_water_usage(model):
    """return the total water usage of all water users"""
    water_usage = [a.usage for a in model.schedule.agents]
    return np.sum(water_usage)

def get_num_adapt_users(model):
    """return the number of water users that can adapt to price increases"""
    adapt_users = [a for a in model.schedule.agents if
                    a.adapt == 1]
    return len(adapt_users)

def get_num_nonadapt_users(model):
    """return the number of water users that cannot adapt to price increaes"""
    nonadapt_users = [a for a in model.schedule.agents if
                        a.adapt == 0]
    return len(nonadapt_users)

def get_total_water_use(model):
    """return total water use of all water users"""
    total_water_use = [a.water_use for a in model.schedule.agents]
    #return the sum of all water use
    return np.sum(total_water_use)

def get_users_on(model):
    """return the number of water users who are currently using water"""
    users_on = [a for a in model.schedule.agents if a.water_on == 1]
    return np.sum(users_on)

# class Device:
#     def __init__(self, device_name, tou_weight, flow_rate, duration):
#         self.device_name = device_name
#         self.tou_weight = tou_weight
#         self.flow_rate = flow_rate
#         self.duration = duration

#         def update_flow(self, new_flow):
#             self.flow_rate = new_flow

#         def update_duration(self, perc):
#             self.duration *= perc

#         def volume_used(self):
#             self.volume = np.round((self.duration * self.flow_rate), 2)
#             return self.volume
        
# class Schedule:
#     def __init__(self, device_name, weekly_sched, daily_sched):
#         self.device_name = device_name
#         self.weekly_sched = weekly_sched
#         self.daily_sched = daily_sched

#     def update_daily_schedule(self, new_sched):
#         self.daily_sched = new_sched

#     def update_weekly_schedule(self, new_sched):
#         self.weekly_sche = new_sched
        
class DynamicPricing(Model):
    """init parameters "init_people", "capacity" are all UserSettableParameters"""
    def __init__(self, 
                 N=1800,
                 days = 365,
                 id=0,
                 seed=1234,
                 **kwargs
             ):
        super().__init__(seed=seed)
        init_start = datetime.now()
        self.days = days
        self.id = id
        self.num_agents = N
        self.timestep = 0
        self.endstep = self.days * 24
        self.datetime = datetime(2018, 1, 1)
        self.season = "winter"
        self.enddate = self.datetime + timedelta(days=self.days)
        # self.base_demand_pattern = pd.read_csv("files/dailyhouseholdconsumption.csv")
        self.social_network = nx.Graph()
        self.neighbors_n = {
            "low density": 4,
            "medium density": 6,
            "high density": 8
        }

        self.average_weekly_consumption = 0 
        
        if 'thresholds' in kwargs:
            self.thresholds = kwargs['thresholds']
        else:
            self.thresholds = {
                "low income": 0.008 ,
                "medium income": 0.004 ,
                "high income": 0.0025
            }

        self.billing_dates = [
            datetime(2018, 3, 1),
            datetime(2018, 5, 1),
            datetime(2018, 7, 1),
            datetime(2018, 9, 1),
            datetime(2018, 11, 1)
        ]
    
        self.flows = {
            "standard":{
                "toilet": 1.6,
                "sink": 1.2,
                "shower": 2.5,
                "dishwasher": 8.6,
                "washing machine": 40,
                "garden": 12.0,
                "pool": 12.0
                },
            "conscious": {
                "toilet": 1.28,
                "sink": 0.95,
                "shower": 2.15,
                "dishwasher": 5.8,
                "washing machine": 24,
                "garden": 6.0,
                "pool": 6.0
                }
            }
        
        self.device_costs = {
            "toilet": 0.022, 
            "sink": 0.743,
            "shower": 0.111,
            "washing machine": 0.005,
            "dishwasher": 0.008,
            "garden": 0.111
        }

        if 'tariff-type' in kwargs:
            self.tariff = kwargs['tariff-type']
        else:
            self.tariff = "Standard IBR"
            self.tier0p = 0.0
            self.tier1p = 2.89
            self.tier2p = 3.50
            self.tier0v = 0.0
            self.tier1v = 3.00
            self.tier2v = 4.00
            self.fixed_fee = 18.40
        if self.tariff == "Dynamic Pricing 1":
            self.fixed_fee = 18.40
            self.peak_rate = 1.25
            self.flat_rate = 1.00
            self.valley_rate = 0.50
            if 'times' in kwargs:
                times = kwargs['times']
                self.peak_times = times['peak times']
                self.flat_times = times['flat times']
                self.valley_times = times['valley times']
            else:
                self.peak_times = [[time(hour=6), time(hour=10)], [time(hour=17), time(hour=20)]]
                self.flat_times = [[time(hour=5), time(hour=6)], [time(hour=10), time(hour=17)], [time(hour=22)]]
                self.valley_times = [[time(hour=0), time(hour=5)], [time(hour=22), time(hour=23)]]

        if 'weights' in kwargs:
            weights_file = kwargs['weights']
            if not os.path.exists(weights_file):
                raise ValueError(f"File {weights_file} does not exist.")
            if not weights_file.endswith('.csv'):
                raise ValueError(f'File {weights_file} is not a .csv file. Please pass a .csv file in kwargs "weights".')
        else:
            weights_file = ('input/tou_weights.csv')
        weights_data = pd.read_csv(weights_file, header=0, skiprows=[1], index_col=[0,1])
        self.weights = pd.DataFrame(weights_data)
        # self.standard_weights = weights.xs('standard')
        # self.conscious_weights = weights.xs('conscious')
        
            # self.standard_weights = {
                #                    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23 
                # "toilet":          [ 5,  5,  5,  5,  5,  5, 10, 10, 15, 10, 10,  5,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10, 10,  5],
                # "sink":            [ 0,  0,  0,  0,  0,  0, 10, 15, 15, 15, 10, 10, 10,  5,  5,  5, 10, 15, 20, 15,  5,  5,  5,  0],
                # "shower":          [ 0,  0,  0,  0,  5, 10, 20, 20, 15, 10, 10, 10,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10,  5,  5],
                # "dishwasher":      [ 0,  0,  0,  0,  0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                # "washing machine": [ 0,  0,  0,  0,  0,  0,  5, 10, 15, 20, 20, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10,  5,  5,  0],
                # "garden":          [ 0,  0,  0,  0,  0, 10, 15, 15, 15, 10,  0,  0,  0,  0,  0,  0,  0, 10, 10, 10, 10, 10, 10,  5],
            # }
            # self.conscious_weights= {
                # "toilet":          [],
                # "sink":            [],
                # "shower":          [ 0,  0,  0,  0,  5, 10, 20, 20, 15, 10, 10, 10,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10,  5,  5],
                # "dishwasher":      [ 5,  5,  5,  5,  5,  5,  5,  0,  0,  0,  0,  0, 15, 10, 10, 10, 15,  0,  0,  0, 20, 15, 15, 15],
                # "washing machine": [ 5,  5,  5,  5,  5,  5, 10,  5,  5,  5,  5,  5, 20, 15, 10, 10, 10, 10,  5,  5,  5, 25, 10, 10],
                # "garden":          [10, 10, 10, 10, 10, 10,  5,  5,  5,  5,  0,  0,  0,  0,  0,  0,  0,  0,  5,  5, 10, 10, 15, 20],
            # }
        
           
        
        # self.datacollector = DataCollector(model_reporters={
        #                                     "Water Use": get_total_water_use, 
        #                                     "Water Users": get_total_water_users,
        #                                     "Users with water on": get_users_on
        #                                     }
        #                                 )

        # create water users for the model 
        for i in range(self.num_agents):
            p = WaterUser(self)
            # assign gardens and pools based on residential density
            if p.res_density == "low density":
                p.has_garden = rng.choice([True, False], p=[0.6, 0.4])       
                p.has_pool = rng.choice([True, False], p=[0.12, 0.88])
            # assign income by inhabitants and income level based on income + inhabitants
            if p.inhabitants == 1:
                p.income = rng.gamma(1.84, 0.000036)
                if p.income < 55000:
                    p.income_level = "low income"
                else:
                    p.income_level = "medium income"
            elif p.inhabitants == 2:
                p.income = rng.gamma(3.92, 0.000048)
                if p.income < 62000:
                    p.income_leve = "low income"
                else:
                    p.income_level = "medium income"
            elif p.inhabitants == 3:
                p.income = rng.gamma(7.34, 0.000073)
                if p.income < 70000:
                    p.income_level = "low income"
                else:
                    p.income_level = "medium income"
            elif p.inhabitants == 4:
                p.income = rng.gamma(3.51, 0.000032)
                if p.income < 77500:
                    p.income_level = "low income"
                else:
                    p.income_level = "medium income"
            elif p.inhabitants == 5:
                p.income = rng.gamma(12.0, 0.00010)
                if p.income < 83700:
                    p.income_level = "low income"
                else:
                    p.income_level = "medium income"
            elif p.inhabitants == 6:
                p.income = rng.gamma(12.8, 0.000096)
                if p.income < 89900:
                    p.income_level = "low income"
                else:
                    p.income_level = "medium income"
            elif p.inhabitants == 7:
                p.income = rng.gamma(16.7, 0.00014)
                if p.income < 96100:
                    p.income_level = "low income"
                else:
                    p.income_level = "medium income"
            if p.income > 155000:
                p.income_level = "high income"
            else:
                pass
            # p.income = p.income / p.inhabitants / 12

            # assign social values and home tenure based on income-level
            if p.income_level == "low income":
                p.social_values = rng.choice(["client", "techno-optimist", "committed", "environmentalist"], p=[0.245, 0.392, 0.118, 0.245])
                p.home_tenure = rng.choice(["owner", "renter"], p=[0.714, 0.286])
            elif p.income_level == "medium income":
                p.social_values = rng.choice(["client", "techno-optimist", "committed", "environmentalist"], p=[0.079, 0.334, 0.269, 0.318])
                p.home_tenure = rng.choice(["owner", "renter"], p=[0.714, 0.286])
            elif p.income_level == "high income":
                p.social_values = rng.choice(["client", "techno-optimist", "committed", "environmentalist"], p=[0.239, 0.329, 0.180, 0.252])
            else:
                pass

            #assign info-level by income
            if p.income < 25000:
                p.info_level = rng.choice(["low info", "medium info", "high info"], p=[0.35, 0.40, 0.25])
            elif p.income >= 25000 and p.income < 75000:
                p.info_level = rng.choice(["low info", "medium info", "high info"], p=[0.15, 0.40, 0.45])
            elif income >= 75000:
                p.info_level = rng.choice(["low info", "medium info", "high info"], p=[0.05, 0.50, 0.45])

            # assign cluster by income and inhabitants
            if (p.income_level == "high income" or p.income_level == "medium income") and p.inhabitants > 3:
                p.cluster = rng.choice(["DEM", "DM", "SMD", "DLN", "DE"], p=[0.25, 0.25, 0.20, 0.15, 0.15])
            elif (p.income == "high income" or p.income == "medium income") and p.inhabitants <= 3:
                p.cluster = rng.choice(["DEM", "DM", "SMD", "DLN", "DE"], p=[0.15, 0.20, 0.20, 0.20, 0.25])
            if p.income_level == "low income" and p.inhabitants > 3:
                p.cluster = rng.choice(["DEM", "DM", "SMD", "DLN", "DE"], p=[0.25, 0.20, 0.20, 0.20, 0.15])
            elif p.income_level == "low income" and p.inhabitants <= 3:
                p.cluster = rng.choice(["DEM", "DM", "SMD", "DLN", "DE"], p=[0.15, 0.20, 0.20, 0.20, 0.25])
            else:
                pass

            # assign elasticity based on cluster
            if p.cluster == "DEM":
                p.elasticity = -0.249
            elif p.cluster == "DM":
                p.elasticity = -0.330
            elif p.cluster == "DE":
                p.elasticity = -0.379
            elif p.cluster == "SMD":
                p.elasticity = -0.392
            elif p.cluster == "DLN":
                p.elasticity = -0.302
            else:
                pass

            # for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                # p.shower_schedule[day] =             
        # def define_social_network(self):
        network_dict = self.agents.groupby("res_density").count()
        for key, value in network_dict.items():
            new_labels = self.agents.select(lambda a: a.res_density == key).get('unique_id')
            nw = nx.watts_strogatz_graph(n=value, k=self.neighbors_n[key], p=0.2, seed=seed)
            mapping = dict(zip(nw, new_labels))
            nw = nx.relabel_nodes(nw, mapping)
            self.social_network.add_node(key, graph=nw)             
        for u in self.social_network.nodes:
            for v in self.social_network.nodes:
                if u != v:
                    self.social_network.add_edge(u, v)
                        
        self.agents.do("get_neighbors")
        self.agents.do("set_shower_schedule", self.weights, self.flows, initial=True)
        self.agents.do("set_toilet_schedule", self.weights, self.flows, initial=True)
        self.agents.do("set_sink_schedule", self.weights, self.flows, initial=True)
        self.agents.do("set_dishwasher_schedule", self.weights, self.flows, initial=True)
        self.agents.do("set_washingmachine_schedule", self.weights, self.flows, initial=True)
        self.agents.do("set_garden_schedule", self.weights, self.flows, initial=True)
        self.agents.do("set_pool_schedule", self.weights, self.flows, initial=True)
        self.running = True
        self.datacollector.collect(self)
    

    def change_time(self):
        self.timestep += 1
        self.datetime += timedelta(hours=1)
        if self.datetime == datetime(2018, 3, 1, 0, 0, 0):
            self.season = "spring"
        elif self.datetime == datetime(2018, 6, 1, 0, 0, 0):
            self.season = "summer"
        elif self.datetime == datetime(2018, 9, 1, 0, 0, 0):
            self.season = "autumn"
        elif self.datetime == datetime(2018, 12, 1, 0, 0, 0):
            self.season = "winter"
        else:
            pass

    def check_time(self):
        # daily schedule
        # if self.datetime.hour == 0:
        #     self.agents.do("set_shower_schedule", self.weights, self.flows)
        #     self.agents.do("set_toilet_schedule", self.weights, self.flows)
        #     self.agents.do("set_sink_schedule", self.weights, self.flows)
        # weekly schedule
        if self.datetime.weekday == 0:
            if self.tariff == "Dynamic Pricing 1":
                high_agents = self.agents.select(lambda agent: agent.info_level == "high info")
                high_agents.shuffle().do("check_change")
                high_agents.select(lambda a: a.to_change).shuffle().do("apply_change")
                high_agents.select(lambda a: a.to_change_schedule).shuffle().do("apply_change_schedule")
                high_agents.shuffle().do("apply_change")
                high_agents.shuffle().do("reset_tou_consumption")
            self.agents.shuffle_do("reset_weekly_consumption")
            self.agents.shuffle_do("set_schedules", self.weights, self.flows)
            # self.agents.do("set_dishwasher_schedule", self.weights, self.flows)
            # self.agents.do("set_washing_machine_schedule", self.weights, self.flows)
            # self.agents.do("set_garden_schedule", self.weights, self.flows)
            # self.agents.do("set_shower_schedule", self.weights, self.flows)
            # self.agents.do("set_toilet_schedule", self.weights, self.flows)
            # self.agents.do("set_sink_schedule", self.weights, self.flows)           self.agents.do("set_pool_schedule", self.weights, self.flows)        # monthly schedule
        # monthly schedule    
        if self.datetime.day == 1 and self.tariff == "Dynamic Pricing 1":
            # if self.tariff == "Dynamic Pricing 1":
            med_agents = self.agents.select(lambda agent: agent.info_level == "medium info")
            med_agents.shuffle().do("check_change")
            # med_agents.select(lambda a: a.to_change or a.to_change_schedule).shuffle().do("apply_change")
            med_agents.select(lambda a: a.to_change).shuffle().do("apply_change")
            med_agents.select(lambda a: a.to_change_schedule).shuffle().do("apply_change_schedule")
            med_agents.shuffle_do("set_schedules", self.weights, self.flows)
            med_agents.shuffle().do("reset_tou_consumption")
            # self.agents.do("set_dishwasher_schedule", self.weights, self.flows)
            # self.agents.do("set_washing_machine_schedule", self.weights, self.flows)
            # self.agents.do("set_garden_schedule", self.weights, self.flows)
            # self.agents.do("set_pool_schedule", self.weights, self.flows)        
        # bimonthly schedule
        if self.datetime in self.billing_dates:
            self.agents.suffle().do("bimonthly_bill")
            if self.tariff == "Dynamic Pricing 1":
                low_agents = self.agents.select(lambda agent: agent.info_level == "low info")
                low_agents.shuffle().do("check_change")
                # low_agents.select(lambda a: a.to_change or a.to_change_schedule).shuffle().do("apply_change")
                low_agents.select(lambda a: a.to_change_schedule).shuffle().do("apply_change_schedule")
                low_agents.select(lambda a: a.to_change).shuffle().do("apply_change")
                low_agents.shuffle().do("reset_tou_consumption")
            self.agents.do("reset_bi_consumption")
            self.agents.shuffle_do("reset_weekly_consumption")
            self.agents.do("set_dishwasher_schedule", self.weights, self.flows)
            self.agents.do("set_washing_machine_schedule", self.weights, self.flows)
            self.agents.do("set_garden_schedule", self.weights, self.flows)
            self.agents.do("set_pool_schedule", self.weights, self.flows)        # monthly schedule
            
    def step(self):
        # tell all the agents in the model to run their step function 
        # use the AgentSet do method
        self.agents.do("check_schedule")
        self.change_time()
        self.check_time()

        # self.schedule.step was deprecated in Mesa 3.0
        # self.schedule.step()
        # collect data 
        # self.datacollector.collect(self) 
    

