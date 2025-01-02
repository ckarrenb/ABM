from mesa import Agent

class WaterSystem(Agent):
    def __init__(self, unique_id, model):
        # initialize the parent class with required parameters
        super().__init__(unique_id, model)
        # for tracking current water price
        self.price = 0
    
    def change_price(self):
        """change the price of water, price change generator;  change price"""
        self.change_gen = self.random.randint(0,100)
        if self.change_gen > 75:
            if self.price == 0:
                self.price += 1
            else:
                self.price -= 1
        else:
            self.price = self.price
        return self.price


class WaterUser(Agent):
    """An agent that uses water. """
    def __init__(self, unique_id, model, water_system):
        super().__init__(unique_id, model)
        self.water_system = water_system
        # randomly assign adaptability to dynamic pricing
        self.adapt = self.random.randint(0,1)
        # total amount of water used 
        self.water_use = 0
        # is water user currently using water
        self.water_on = 1
        self.time = 0
        self.daily_use = self.random.uniform(15,50)

    def change_time(self):
        """change time counter at each step"""
        self.time += 1
        return self.time

    def use_water(self):
        """use water with diurnal pattern"""
        if self.time > 0 and self.time < 120:
            n = 0.085
        elif self.time > 120 and self.time < 240:
            n = 0.034
        elif self.time > 240 and self.time < 360:
            n = 0.026
        elif self.time > 360 and self.time < 480:
            n = 0.021
        elif self.time > 480 and self.time < 600:
            n = 0.051
        elif self.time > 600 and self.time < 720:
            n = 0.136
        elif self.time > 720 and self.time < 840:
            n = 0.128
        elif self.time > 840 and self.time < 960:
            n = 0.11
        elif self.time > 960 and self.time < 1080:
            n = 0.094
        elif self.time > 1080 and self.time < 1200:
            n = 0.085
        elif self.time > 1200 and self.time < 1320:
            n = 0.119
        else:
            n = 0.0111
        return n

    def adapt_to_price(self):
        """checks price of water and adjusts users use if the user is able to
            adapt to price increases"""
        n = self.use_water()
        if  self.adapt == 1 and self.water_system.price == 1:
            self.water_use = n*self.daily_use
            self.water_on = 1
        else:
            self.water_use = n*self.daily_use
            self.water_on = 1
            
    def step(self):
        # each user agent uses water
        self.use_water()
        # turn water on/off and increment water usage
        self.adapt_to_price()
        # change price of water
        self.water_system.change_price()
        # change time at each step
        self.change_time()
        





