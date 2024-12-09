from agents.base_agent import BaseAgent
import pandas as pd
import numpy as np
import random as rand

#!!!
# Разобраться какие ставить параметры для среднего и дисперсии в величинах трейдеров.
#!!!
class NoiseTrader(BaseAgent):

    def __init__(self, lob_book, data, external_id, x_min, betacoeff, 
        lmbda_m, lmbda_l, lmbda_c, mu_mo, omeg_mo, mu_lo, omeg_lo):

        self.LOB_book = lob_book
        self.data = data
        self.agent_id = "NT"+str(external_id)
        # minimal difference between orders
        self.minimal_diff = 0.01
        self.x_min = x_min
        self.betacoeff = betacoeff
        # external params
        self.mu_mo = mu_mo
        self.gamma_mo = omeg_mo
        self.mu_lo = mu_lo
        self.gamma_lo = omeg_lo
        self.initial_spread = 0.05

        self.lmbda_m = lmbda_m
        self.lmbda_l = lmbda_l
        self.lmbda_c = lmbda_c
        
    def get_statistics(self, mu, gamma):
        #Здесь неправильно! Надо доделать этот момент!
        # заменить 0 на единицу! так делать нельзя, но мы сделаем!
        self.v_t = max(1, np.exp(mu + gamma*rand.random()))
        self.p_offspr = self.x_min*(1-min(0.99, round(rand.random(), 2)))**(-1/(self.betacoeff-1))

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
        # set long or short position
        NoiseTrader.get_statistics(self, 1, 1)
        position = rand.choice(['long', 'short'])
        trading_type = rand.choices(['market', 'limit', 'cancel'], weights = [self.lmbda_m, self.lmbda_l, self.lmbda_c])[0]
        
        if trading_type == 'market':
            NoiseTrader.get_statistics(self, self.mu_mo, self.gamma_mo)
            if position == 'long' and len(self.LOB_book.LOB_book_ask) > 0:
                self.LOB_book.make_purchase(self.agent_id, int(self.v_t))
            elif position == 'short' and len(self.LOB_book.LOB_book_bid) > 0:
                self.LOB_book.make_sell(self.agent_id, int(self.v_t))
            
        elif trading_type == 'limit':
            NoiseTrader.get_statistics(self, self.mu_lo, self.gamma_lo)
            limit_type = rand.choices(['cross', 'inspr', 'spr', 'offspr'], weights = [0.003, 0.098, 0.173, 0.726])[0]
            
            # узнаём лучшие цены на выставление лимитных ордеров
            if  len(self.LOB_book.LOB_book_ask) > 0:
                ask_pricing = self.LOB_book.get_price('ask_head')
            else:
                ask_pricing = self.data[-1] + self.initial_spread/2
            
            if  len(self.LOB_book.LOB_book_bid) > 0:
                bid_pricing = self.LOB_book.get_price('bid_head')
            else:
                bid_pricing = self.data[-1] - self.initial_spread/2
                
            #print(ask_pricing, self.data[-1], self.initial_spread/2)
                    
            if limit_type == 'cross':
                self.LOB_book.add_bid(self.agent_id, int(self.v_t), ask_pricing) if position == 'long' else self.LOB_book.add_ask(self.agent_id, int(self.v_t), bid_pricing)
            
            elif limit_type == 'inspr':
                p_inspr = rand.uniform(bid_pricing, ask_pricing)
                self.LOB_book.add_bid(self.agent_id, int(self.v_t), p_inspr) if position == 'long' else self.LOB_book.add_ask(self.agent_id, int(self.v_t), p_inspr)
            
            elif limit_type == 'spr':

                self.LOB_book.add_bid(self.agent_id, int(self.v_t), bid_pricing) if position == 'long' else self.LOB_book.add_ask(self.agent_id, int(self.v_t), ask_pricing)
             
            elif limit_type == 'offspr':
                off_ask = ask_pricing + self.p_offspr
                off_bid = bid_pricing - self.p_offspr
                self.LOB_book.add_bid(self.agent_id, int(self.v_t), off_bid) if position == 'long' else self.LOB_book.add_ask(self.agent_id, int(self.v_t), off_ask)
                
        elif trading_type == 'cancel':
            # выбираем список ордеров с id нужного нам агента
            if len(self.LOB_book.LOB_book_bid) > 0 or len(self.LOB_book.LOB_book_ask) > 0:
                change_arr_long = list(filter(lambda t: t[2]==self.agent_id, self.LOB_book.LOB_book_bid))
                change_arr_short = list(filter(lambda t: t[2]==self.agent_id, self.LOB_book.LOB_book_ask))
                change_arr = change_arr_long + change_arr_short
                # Сортируем и удаляем самый старый!
                ss = change_arr
                change_arr.sort(key=lambda point: point[1])
                if len(change_arr) > 0:
                    self.LOB_book.delete_order(change_arr[0][0])
            
