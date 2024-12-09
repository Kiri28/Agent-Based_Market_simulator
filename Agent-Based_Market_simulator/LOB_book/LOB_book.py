import math
import numpy as np
import pandas as pd
import datetime
import random as rand
from itertools import product
import itertools
import time
import re


class LOB:
    
    def __init__(self):
        #need to reallocate!!!

        #data of price changing and volume shifts
        self.trading_data = []
        self.volume_data = []

        #bid's orders array
        # has such signature as:
        # Buy_orders:[..[order_id, time, agent, bid_size, bid_price]..]
        self.LOB_book_bid = []

        #ask's orders array
        # has such signature as:
        # Sell_orders:[..[order_id, time, agent, bid_size, bid_price]..]
        self.LOB_book_ask = []

        # zero-order id allocation
        self.num_order_id = -1
        # book of agents_and_their_orders
        self.agents_book = {}

        #making the dictionary for deals.
        #It helps us to integrate money in model.
        self.curr_portfolio = {}
        
        
    # !!! changes in last version-1
    # add bid limit order
    def add_bid(self, agent_id, bid_size, bid_price):

        #bid_price1 = round(bid_price, 2)
        bid_price1 = bid_price
        
        self.num_order_id += 1
        #compile each order's id
        order_id = 'B'+'0'*(7-len(str(self.num_order_id))) + str(self.num_order_id)
        self.LOB_book_bid.append([order_id, 
                                  datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S.%f"), 
                                  agent_id, bid_size, bid_price1])
        
        self.LOB_book_bid.sort(key=lambda point: (-point[-1], point[1]))
        LOB.update_total(self)
        #проверка не было ли дороже
        return LOB.get_checking(self, agent_id)
        #return order_id
        

    # !!! changes in last version-1
    # add ask limit order
    def add_ask(self, agent_id, ask_size, ask_price):

        #ask_price1 = round(ask_price, 2)
        ask_price1 = ask_price

        self.num_order_id += 1
        #compile each order's id
        order_id = 'A'+'0'*(7-len(str(self.num_order_id))) + str(self.num_order_id)
        self.LOB_book_ask.append([order_id, 
                                  datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S.%f"), 
                                  agent_id, ask_size, ask_price1])
        
        self.LOB_book_ask.sort(key=lambda point: (point[-1], point[1]))
        LOB.update_total(self)
        #проверка не было ли спроса дешевле
        return LOB.get_checking(self, agent_id)
        #return order_id
    
    
    def get_checking(self, agent_id):
        #если всё впорядке, то ничего не делаем
        if len(self.LOB_book_bid)>0 and len(self.LOB_book_ask)>0:
            #если спрос больше предложения, то осуществляем транзакцию.
            if self.LOB_book_bid[0][-1] >= self.LOB_book_ask[0][-1]:

                # if bid order was later than ask...
                # make deal by ask price
                if self.LOB_book_bid[0][1] >= self.LOB_book_ask[0][1]:
                    min_val = min(self.LOB_book_bid[0][-2], self.total_ask)
                    self.LOB_book_bid[0][-2] -= min_val
                    # если мы исчерпали наше желание, то удаляем ордер
                    if self.LOB_book_bid[0][-2] == 0:
                        self.LOB_book_bid.pop(0)
                    # !!!!!!
                    return LOB.make_purchase(self, agent_id, min_val)

                elif self.LOB_book_bid[0][1] < self.LOB_book_ask[0][1]:
                    min_val = min(self.LOB_book_ask[0][-2], self.total_bid)
                    self.LOB_book_ask[0][-2] -= min_val
                    # если мы исчерпали наше желание, то удаляем ордер
                    if self.LOB_book_ask[0][-2] == 0:
                        self.LOB_book_ask.pop(0)
                    # !!!!!!
                    return LOB.make_sell(self, agent_id, min_val)
                
        return {"status":"none"}
                
                
        
    #delete order by order_id
    def delete_order(self, order_id):
        if type(order_id)==str:
            if order_id[0]=='A':
                try:
                    self.LOB_book_ask.remove(list(filter(lambda x: x[0] == order_id, self.LOB_book_ask))[0])
                except:
                    # it should be better to put smth message here...
                    pass

            elif order_id[0]=='B':
                try:
                    self.LOB_book_bid.remove(list(filter(lambda x: x[0] == order_id, self.LOB_book_bid))[0])
                except:
                    # it should be better to put smth message here...
                    pass
        else: 
            try:
                self.LOB_book_bid.remove(list(filter(lambda x: x[0] == order_id, self.LOB_book_bid))[0])
                self.LOB_book_ask.remove(list(filter(lambda x: x[0] == order_id, self.LOB_book_bid))[0])
            except:
                # it should be better to put smth message here...
                pass
        LOB.update_total(self)
    
    # get current sell and buy pricing
    def get_price(self, ask):
        if ask=='sell':
            return self.LOB_book_bid[0][-1]
        elif ask=='buy':
            return self.LOB_book_ask[0][-1]
        elif ask=="bid_head":
            if len(self.LOB_book_bid) > 0:
                return self.LOB_book_bid[0][-1]
            else:
                return self.trading_data[-1]
        elif ask=="ask_head":
            if len(self.LOB_book_ask) > 0:
                return self.LOB_book_ask[0][-1]
            else:
                return self.trading_data[-1]
        
    # get data from limit order books
    def get_data(self, type1='both'):
        if type1=='ask':
            return self.LOB_book_ask
        elif type1=='bid':
            return self.LOB_book_bid
        elif type1=='both':
            return [self.LOB_book_bid, self.LOB_book_ask]
    
    # get current trading data
    def get_trading_data(self):
        new = self.trading_data
        new1 = self.volume_data
        self.trading_data = []
        self.volume_data = []
        return [new, new1]
        
    # update total information
    # need to call if you want to get current trading data..
    # ..after some completed deals
    def update_total(self):
        self.total_bid = sum([self.LOB_book_bid[k][-2] 
                              for k in range(len(self.LOB_book_bid))])
        self.total_ask = sum([self.LOB_book_ask[k][-2] 
                              for k in range(len(self.LOB_book_ask))])
        
        
    # !!! changes in last version 
        
    # purchasing process on market
    # can using for forward orders making
    def make_purchase(self, agent_id, size):
        # mean sell/buy value
        mkk = min(size, self.total_ask)
        #dirty hack
        #if mkk == 0:
        #    mkk = 1

        curr_size = size
        res_price = 0
        result_dict = {'total_sum':0}
        if len(self.LOB_book_ask)>0:
            result_dict[self.LOB_book_ask[0][-1]] = 0

        while(curr_size > 0) and (self.total_ask > 0):
            self.LOB_book_ask[0][-2] -= 1
            curr_size -= 1
            result_dict['total_sum'] += self.LOB_book_ask[0][-1]
            result_dict[self.LOB_book_ask[0][-1]] += 1

            #################
            # change the num of stocks in portfolio
            # если сделки купли/продажи на текущем объёме уже есть
            # то добавляем
            # else write -1
            if self.LOB_book_ask[0][2] in self.curr_portfolio: 
                # пишем что он продал
                self.curr_portfolio[self.LOB_book_ask[0][2]] -= 1
            else:
                # иначе пишем что продал одну цб
                self.curr_portfolio[self.LOB_book_ask[0][2]] = -1
                
            if agent_id in self.curr_portfolio: 
                # пишем что он продал
                self.curr_portfolio[agent_id] += 1
            else:
                # иначе пишем что продал одну цб
                self.curr_portfolio[agent_id] = 1
                
            ########### 

            if self.LOB_book_ask[0][-2] == 0:
                self.LOB_book_ask.pop(0)
                if len(self.LOB_book_ask) > 0:
                    if (curr_size > 0) and (self.LOB_book_ask[0][-1] not in result_dict):
                        result_dict[self.LOB_book_ask[0][-1]] = 0
            #update data about total bid/ask
            LOB.update_total(self)

        # new round rule
        #result_dict['mean_sum'] = round(result_dict['total_sum']/mkk, 2)
        result_dict['mean_sum'] = result_dict['total_sum']/mkk
        
        self.trading_data.append(result_dict['mean_sum'])
        self.volume_data.append(mkk)
        result_dict['total_sum']
            
        # if completed deal has a rest of order, 
        # this order adds into bid book by actual price
        if (self.total_ask <= 0) and (curr_size > 0):
            result_dict['new_order'] = LOB.add_bid(self, agent_id, curr_size, result_dict['mean_sum'])
        
        LOB.update_total(self)
        #print(self.curr_portfolio)
        return result_dict
    
    # !!! changes in last version 

    #selling process on market
    # can using for forward orders making
    def make_sell(self, agent_id, size):
        
        # mean sell/buy value
        mkk = min(size, self.total_bid)
        curr_size = size
        res_price = 0
        result_dict = {'total_sum':0, 'mean_sum':0}
        if len(self.LOB_book_bid)>0:
            result_dict[self.LOB_book_bid[0][-1]]=0

        while(curr_size > 0) and (self.total_bid > 0):
            self.LOB_book_bid[0][-2] -= 1
            curr_size -= 1
            result_dict['total_sum'] += self.LOB_book_bid[0][-1]
            result_dict[self.LOB_book_bid[0][-1]] += 1

            #################
            # change the num of stocks in portfolio
            # если сделки купли/продажи на текущем объёме уже есть
            # то добавляем
            # else write -1
            if self.LOB_book_bid[0][2] in self.curr_portfolio: 
                # пишем что он продал
                self.curr_portfolio[self.LOB_book_bid[0][2]] += 1
            else:
                # иначе пишем что продал одну цб
                self.curr_portfolio[self.LOB_book_bid[0][2]] = 1
                
            if agent_id in self.curr_portfolio: 
                # пишем что он продал
                self.curr_portfolio[agent_id] -= 1
            else:
                # иначе пишем что продал одну цб
                self.curr_portfolio[agent_id] = -1
                
            ########### 

            if self.LOB_book_bid[0][-2] == 0:
                self.LOB_book_bid.pop(0)
                if len(self.LOB_book_bid) > 0:
                    if (curr_size > 0) and (self.LOB_book_bid[0][-1] not in result_dict):
                        result_dict[self.LOB_book_bid[0][-1]] = 0
            #update data about total bid/ask
            LOB.update_total(self)

        # new round rule
        #print(size, self.total_bid, self.LOB_book_bid)
        #result_dict['mean_sum'] = round(result_dict['total_sum']/mkk, 2)
        result_dict['mean_sum'] = result_dict['total_sum']/mkk
        
        self.trading_data.append(result_dict['mean_sum'])
        self.volume_data.append(mkk)
        result_dict['total_sum']
            
        # if completed deal has a rest of order, 
        # this order adds into ask book by actual price
        if (self.total_bid <= 0) and (curr_size > 0):
            result_dict['new_order'] = LOB.add_ask(self, agent_id, curr_size, result_dict['mean_sum'])
        
        LOB.update_total(self)
        return result_dict
            
