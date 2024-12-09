from agents.base_agent import BaseAgent
import numpy as np
import pandas as pd
import random as rand


# momentum trader logic
class MomentumTrader(BaseAgent):

    def __init__(self, lob_book, data, external_id, k, n_memory, wealth):
        self.LOB_book = lob_book
        self.k = k
        self.wealth = 10  # wealth
        self.n_memory = n_memory
        self.data = data
        self.agent_id = "MT"+str(external_id)
        # minimal difference between orders
        self.minimal_diff = 0.01

    def get_statistics(self):
        roc = 0
        if len(self.data) > 1+self.n_memory:
            if self.data[-1 - self.n_memory] != 0:
                roc = (self.data[-1] - self.data[-1 - self.n_memory]) / self.data[-1 - self.n_memory]
        return roc

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

    def trading_step(self, wealth_val):
        roc_val = MomentumTrader.get_statistics(self)
        wealth = max(self.wealth + wealth_val, 0)
        # обновим состояние системы
        self.LOB_book.update_total()
        # if roc>=k
        if roc_val >= self.k:
            # print("I am here!")
            # Submit buy market order with v...
            v_t = max(1, int(abs(roc_val)*wealth))
            if v_t > 0 and self.LOB_book.total_ask >= v_t:
                # проверяем не пуста ли LOB_ask и если нет, то делаем 
                self.LOB_book.make_purchase(self.agent_id, v_t)
        elif roc_val <= -self.k:
            v_t = max(1, int(abs(roc_val)*wealth))
            if v_t > 0 and self.LOB_book.total_bid >= v_t:
                # проверяем не пуста ли LOB_bid и если нет, то делаем покупку
                self.LOB_book.make_sell(self.agent_id, v_t)
