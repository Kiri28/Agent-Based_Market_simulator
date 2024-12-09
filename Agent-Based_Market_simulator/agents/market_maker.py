from agents.base_agent import BaseAgent
import numpy as np
import pandas as pd
import random as rand

# market maker logic class
class MarketMaker(BaseAgent):

    def __init__(self, lob_book, data, external_id, v_min, v_max, v, w):
        self.LOB_book = lob_book
        self.data = data
        self.orders_list = []
        self.agent_id = "MM"+str(external_id)
        # minimal difference between orders
        self.minimal_diff = 0.01
        #external params
        self.rolling_window = w
        self.initial_spread = 0.05
        self.v_min = v_min
        self.shares_limit = v_max

    def get_statistics(self):
        # get forecast by w period rolling-mean estimate
        res = pd.DataFrame(self.data).rolling(self.rolling_window).mean()
        trend = np.transpose(res.values)[0,-1] - np.transpose(res.values)[0,-2]
        # positive values says that trand increasing
        # negative - decreasing
        #print(trend)
        return trend
    
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
        #cancel any existing orders from ask_book
        for order in self.LOB_book.LOB_book_ask:
            if order[2] == self.agent_id:
                self.LOB_book.delete_order(order[0])
                
        #cancel any existing orders from bid_book
        for order in self.LOB_book.LOB_book_bid:
            if order[2] == self.agent_id:
                self.LOB_book.delete_order(order[0])
        
        # узнаём лучшие цены на выставление лимитных ордеров
        if  len(self.LOB_book.LOB_book_ask) > 0:
            ask_pricing = self.LOB_book.get_price('ask_head')
        else:
            ask_pricing = self.data[-1] + self.initial_spread/2
            #print(ask_pricing, self.data[-1], self.initial_spread/2)
            
        if  len(self.LOB_book.LOB_book_bid) > 0:
            bid_pricing = self.LOB_book.get_price('bid_head')
        else:
            bid_pricing = self.data[-1] - self.initial_spread/2
            
        #print(self.data)
        #strict conditions
        # if predicts next order is buy, then...
        if MarketMaker.get_statistics(self) > 0:
            #Submit sell at best price with volume...
            sell_submit = rand.randint(1,self.shares_limit)
            self.LOB_book.add_ask(self.agent_id, sell_submit, ask_pricing)
            self.LOB_book.add_bid(self.agent_id, self.v_min, bid_pricing)
            
        elif MarketMaker.get_statistics(self) < 0:
            # Submit buy at best price with volume...
            buy_submit = rand.randint(1,self.shares_limit)
            self.LOB_book.add_bid(self.agent_id, buy_submit, bid_pricing)
            self.LOB_book.add_ask(self.agent_id, self.v_min, ask_pricing)
        
        else:
            pass
