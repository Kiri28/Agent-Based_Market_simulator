from agents.base_agent import BaseAgent
import numpy as np
import pandas as pd
import random as rand

#Liquidity consumer logic
# calling only in start of the day
class LiquidConsumer(BaseAgent):

    def __init__(self, lob_book, data, external_id, deal_type, 
        order_volume, h_min, h_max):
        self.LOB_book = lob_book
        self.data = data
        self.orders_list = []
        self.agent_id = "LC"+str(external_id)
        # minimal difference between orders
        self.minimal_diff = 0.01
        self.deal_type = deal_type
        self.order_volume = order_volume
        #external param
        self.shares_limit = 100000

    def get_statistics(self):
        # get num of shares at best price
        if self.deal_type == 'buy':
            cout = k = 0
            min_val = self.LOB_book.LOB_book_ask[0][-1]
            #print(self.LOB_book.LOB_book_ask, min_val, k)
            while self.LOB_book.LOB_book_ask[k][-1] == min_val:
                cout += self.LOB_book.LOB_book_ask[k][-2]
                k+=1
                if k >= len(self.LOB_book.LOB_book_ask):
                    break

        else:
            cout = k = 0
            min_val = self.LOB_book.LOB_book_bid[0][-1]
            while self.LOB_book.LOB_book_bid[k][-1] == min_val:
                cout += self.LOB_book.LOB_book_bid[k][-2]
                k+=1
                if k >= len(self.LOB_book.LOB_book_bid):
                    break
                
        return cout
    
    def profit_calculation(self):
        # id our agent exists in profit table...
        if self.agent_id in self.LOB_book.curr_portfolio:
            # append his immediate profit
            return self.LOB_book.curr_portfolio[self.agent_id]*(self.data[-1]-self.data[-2])
        else:
            # else append zero immediate profit
            return 0
    
    def trading_logic(self, shares_limit):
        pass
    
    
    def trading_step(self):
        #if rand() < 0.5 then...
        if self.deal_type == 'buy' and len(self.LOB_book.LOB_book_ask) > 0:
            #print(len(self.LOB_book.LOB_book_ask))
            F_t = LiquidConsumer.get_statistics(self)
            if F_t > 0:
                if self.order_volume <= F_t:
                    self.LOB_book.make_purchase(self.agent_id, self.order_volume)
                    return self.order_volume
                else:
                    self.LOB_book.make_purchase(self.agent_id, F_t)
                    return F_t
                    
        elif self.deal_type == 'sell' and len(self.LOB_book.LOB_book_bid) > 0:
            F_t = LiquidConsumer.get_statistics(self)
            #print(F_t)
            if F_t > 0:
                if self.order_volume <= F_t:
                    self.LOB_book.make_sell(self.agent_id, self.order_volume)
                    return self.order_volume
                else:
                    self.LOB_book.make_sell(self.agent_id, F_t)
                    return F_t
        
        return 0
