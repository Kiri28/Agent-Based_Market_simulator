from agents.base_agent import BaseAgent
import pandas as pd
import numpy as np
import random as rand

# mean reversion trader logic
# Надо исправить время наличия точек входа!!!
class MeanReversionTrader(BaseAgent):

    def __init__(self, lob_book, data, external_id, alpha, k , v_mr):
        self.LOB_book = lob_book
        self.data = data
        self.agent_id = "MRT"+str(external_id)
        # minimal difference between orders
        self.minimal_diff = 0.01
        self.alpha = alpha
        self.k = k
        self.v = v_mr
        self.initial_spread = 0.05
        
        
    def get_statistics(self):
        self.ema = pd.DataFrame(self.data).ewm(alpha = self.alpha, adjust=False).mean().values[-1][0]

    def profit_calculation(self):
        # id our agent exists in profit table...
        if self.agent_id in self.LOB_book.curr_portfolio:
            # append his immediate profit
            return self.LOB_book.curr_portfolio[self.agent_id]*(self.data[-1]-self.data[-2])
        else:
            # else append zero immediate profit
            return 0

    def trading_logic(self):
        pass
    
    def trading_step(self):
        MeanReversionTrader.get_statistics(self)
        #if p_t - ema_t >= k..
        #adding = 0.25
        adding = 0.6
        if self.data[-1] - self.ema + adding >= self.k*np.std(self.data):
            if  len(self.LOB_book.LOB_book_ask) > 0:
                pricing = self.LOB_book.get_price('ask_head')
            else:
                pricing = self.data[-1] + self.initial_spread/2
            #print("MRT in game!")
            self.LOB_book.add_ask(self.agent_id, self.v, pricing + self.minimal_diff)
        
        #else if ema_t - p_t >= k...
        elif self.ema - self.data[-1] + adding >= self.k*np.std(self.data):
            if  len(self.LOB_book.LOB_book_bid) > 0:
                pricing = self.LOB_book.get_price('bid_head')
            else:
                pricing = self.data[-1] + self.initial_spread/2
            #print("MRT in game!")
            self.LOB_book.add_bid(self.agent_id, self.v, pricing - self.minimal_diff)
            
