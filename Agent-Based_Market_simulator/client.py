import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as ss
import random as rand
from tqdm import tqdm
from itertools import product
import itertools
import time
import re
import csv
from tqdm import tqdm

from agents.liquidity_consumer import LiquidConsumer
from agents.market_maker import MarketMaker
from agents.mean_reversion_trader import MeanReversionTrader
from agents.momentum_trader import MomentumTrader
from agents.noise_trader import NoiseTrader
# DDPG
from agents.ddpg_agent.agent import DDPG_agent

from LOB_book.LOB_book import LOB
from config.parameters_list import parameters_list

from other_functions import MACD, RSI, OBV, preprocessing_foo


class SimulateMarket:
    def __init__(self):
        self.lob_book = LOB()
        self.params = parameters_list()

        self.LC_shares_limit = self.params.lc_h_max
        self.data = self.params.init_data
        self.volume = self.params.init_volume

        # params for aggregation
        self.close_price = self.params.close_price
        self.min_price = self.params.min_price
        self.max_price = self.params.max_price
        self.spread = self.params.spread
        self.volume_data = self.params.volume_data
        self.volume_inb = self.params.volume_inb
        self.macd = self.params.macd
        self.macdsignal = self.params.macdsignal
        self.macdhist = self.params.macdhist
        self.data_rsi = self.params.data_rsi
        self.data_obv = self.params.data_obv

        # aggregate value
        self.aggregate_value = self.params.aggregate_value

        # bid_price and ask price added
        self.bid_price = self.params.bid_price
        self.ask_price = self.params.ask_price
        self.bid_volume = self.params.bid_volume
        self.ask_volume = self.params.ask_volume
        self.initial_spread = 0.05

        # DDPG initialization
        # HERE!!!
        self.ddpg_agent = DDPG_agent(1)


        # делаем таблицу где будем хранить 
        # профит каждого агента
        self.agents_profit_table = {"MM"+str(1): [0 for _ in range(len(self.data))], 
                                    "LC"+str(1): [0 for _ in range(len(self.data))],
                                    "MT"+str(1): [0 for _ in range(len(self.data))],
                                    "MRT"+str(1): [0 for _ in range(len(self.data))], 
                                    "NT"+str(1): [0 for _ in range(len(self.data))], 
                                    "DDPG"+str(1): [0 for _ in range(len(self.data))]}


    def write(self, filename, 
        aggregate_filename = "data/comparing_rows/params_215_ddpg_aggr.csv"):
        with open(filename, mode = 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', 
                quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # write parameters in parameters list
            columns = ['Id', 'price', 'volume', 'bid_price', 'ask_price', 
            'bid_volume', 'ask_volume'] + list(self.agents_profit_table.keys())

            writer.writerow(columns)
            for k in range(len(self.data)):
                writer.writerow([k, self.data[k], self.volume[k], 
                    self.bid_price[k], self.ask_price[k], self.bid_volume[k], 
                    self.ask_volume[k]] + [
                    self.agents_profit_table[a][k] for a in list(self.agents_profit_table.keys())])

        # save aggregate row
        with open(aggregate_filename, mode = 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',',
             quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # write parameters in parameters list
            columns = ['Id', 'close_price', 'min_price', 'max_price', 
            'spread', 'volume_data', 'volume_inb', 'macd', 'macdsignal',
             'macdhist', 'data_rsi', 'data_obv']

            writer.writerow(columns)
            for k in range(len(self.close_price)):
                writer.writerow([k, self.close_price[k], self.min_price[k], 
                self.max_price[k], self.spread[k], self.volume_data[k], 
                self.volume_inb[k], self.macd[k], self.macdsignal[k], 
                self.macdhist[k], self.data_rsi[k], self.data_obv[k]])

        # save parameters_list in separate file
        with open(filename[:-4]+"_params.csv", mode = 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',',
             quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['param_type', 'param_value'])
            writer.writerow(['ticks_per_day', self.params.ticks_per_day])
            writer.writerow(['num_of_days', self.params.num_of_days])
            writer.writerow(['tick_size', self.params.tick_size])
            writer.writerow(['delta_mm', self.params.delta_mm])
            writer.writerow(['delta_lc', self.params.delta_lc])
            writer.writerow(['delta_mr', self.params.delta_mr])
            writer.writerow(['delta_mt', self.params.delta_mt])
            writer.writerow(['delta_nt', self.params.delta_nt])
            writer.writerow(['mm_v_min', self.params.mm_v_min])
            writer.writerow(['mm_v_max', self.params.mm_v_max])
            writer.writerow(['mm_v', self.params.mm_v])
            writer.writerow(['mm_w', self.params.mm_w])
            writer.writerow(['lc_h_min', self.params.lc_h_min])
            writer.writerow(['lc_h_max', self.params.lc_h_max])
            writer.writerow(['mrt_v_mr', self.params.mrt_v_mr])
            writer.writerow(['mrt_alpha', self.params.mrt_alpha])
            writer.writerow(['mt_nr', self.params.mt_nr])
            writer.writerow(['mt_k', self.params.mt_k])
            writer.writerow(['b_or_s_prob', self.params.b_or_s_prob])
            writer.writerow(['lmbda_m', self.params.lmbda_m])
            writer.writerow(['lmbda_l', self.params.lmbda_l])
            writer.writerow(['lmbda_c', self.params.lmbda_c])
            writer.writerow(['mu_mo', self.params.mu_mo])
            writer.writerow(['omeg_mo', self.params.omeg_mo])
            writer.writerow(['mu_lo', self.params.mu_lo])
            writer.writerow(['omeg_lo', self.params.omeg_lo])
            writer.writerow(['xmin_offfspr', self.params.xmin_offfspr])
            writer.writerow(['beta_offspr', self.params.beta_offspr])
        
    def make_aggregation(self, aggregate_value = 30):
	    data_cls = np.mean(self.data[-aggregate_value:])
	    data_min = np.min(self.data[-aggregate_value:])
	    data_max = np.max(self.data[-aggregate_value:])
	    
	    data_vol = np.sum(self.volume[-aggregate_value:])


	    data_spread = np.mean(np.array(self.ask_price[-aggregate_value:]) - np.array(self.bid_price[-aggregate_value:]))
	    data_volume_inb = (np.sum(self.ask_volume[-aggregate_value:]) - 
	                   np.sum(self.bid_volume[-aggregate_value:]))/(np.sum(self.ask_volume[-aggregate_value:]) + 
	                                                                np.sum(self.bid_volume[-aggregate_value:]))
	    
	    self.close_price.append(data_cls)
	    self.min_price.append(data_min)
	    self.max_price.append(data_max)
	    self.spread.append(data_spread)
	    self.volume_data.append(data_vol)
	    self.volume_inb.append(data_volume_inb)
	    
	    try: 
	        ku = list(MACD(self.close_price, 12, 26, 9)[0])[-1]
	    except: 
	        ku = 0

	    try: 
	        km = list(MACD(self.close_price, 12, 26, 9)[1])[-1]
	    except: 
	        km = 0

	    try: 
	        kl = list(MACD(self.close_price, 12, 26, 9)[2])[-1]
	    except: 
	        kl = 0

	    self.macd.append(ku)
	    self.macdsignal.append(km)
	    self.macdhist.append(kl)
	        
	    self.data_rsi.append(preprocessing_foo(RSI,
         self.close_price, min(10, aggregate_value)))
	    self.data_obv.append(preprocessing_foo(OBV,
         self.close_price, self.volume_data))


    def start(self):
        for vv in tqdm(range(self.params.num_of_days)):
            for tt in range(self.params.ticks_per_day):
                #!!!!!!!!
                # Market maker logic
                if rand.random()<self.params.delta_mm:
                    m=MarketMaker(self.lob_book, self.data, 1, self.params.mm_v_min, 
                        self.params.mm_v_max, self.params.mm_v, self.params.mm_w)
                    m.trading_step()

                # Liquidity consumer logic
                # будем для каждого агента типа Liquid consumer хранить массив значений
                # которые будут показывать что он выбрал в начале дня
                # и на какую сумму
                if (tt == 0) == True:
                    LC_dict = {}
                # если у нас начало дня...
                if (tt == 0) == True:
                    deal_type = rand.choice(["buy","sell"])
                    deal_vol = rand.randint(1, self.LC_shares_limit)
                    LC_dict[1] = [deal_type, deal_vol]
                    if rand.random()<self.params.delta_lc:
                        L = LiquidConsumer(self.lob_book, self.data, 1, deal_type, deal_vol, 
                            h_min = self.params.lc_h_min, h_max = self.params.lc_h_max)
                        new_ord = L.trading_step()
                        LC_dict[1][1] -= new_ord

                else:
                    if LC_dict[1][1] > 0 and rand.random()<self.params.delta_lc:
                        L=LiquidConsumer(self.lob_book, self.data, 1, LC_dict[1][0], LC_dict[1][1], 
                            h_min = self.params.lc_h_min, h_max = self.params.lc_h_max)
                        new_ord = L.trading_step()
                        LC_dict[1][1] -= new_ord


                # Momentum trader logic
                if rand.random()<self.params.delta_mt:
                    MT=MomentumTrader(self.lob_book, self.data, 1, 
                        k = self.params.mt_k, n_memory = self.params.mt_nr, wealth = 200)
                    MT.trading_step(np.sum(self.agents_profit_table["MT" + str(1)]))

                # Mean reversion trader logic
                if rand.random()<self.params.delta_mr:
                    MRT=MeanReversionTrader(self.lob_book, self.data, 1, 
                        alpha = self.params.mrt_alpha, k = 5, v_mr = self.params.mrt_v_mr)
                    MRT.trading_step()

                # Noise trader logic
                # if rand.random()<self.params.delta_nt:
                if rand.random()<self.params.delta_nt:
                    NT=NoiseTrader(self.lob_book, self.data, 1,
                     x_min = self.params.xmin_offfspr, betacoeff = self.params.beta_offspr, 
                        lmbda_m = self.params.lmbda_m, lmbda_l = self.params.lmbda_l,
                         lmbda_c = self.params.lmbda_c, 
                        mu_mo = self.params.mu_mo, omeg_mo = self.params.omeg_mo, 
                        mu_lo = self.params.omeg_lo, omeg_lo = self.params.omeg_lo)

                    NT.trading_step()

                # DDPG agent logic
                if len(self.close_price) > 0:
                    self.ddpg_agent.run_analytics(np.array([self.close_price[-1], self.min_price[-1], 
                    self.max_price[-1], self.spread[-1], self.volume_data[-1], 
                    self.volume_inb[-1], self.macd[-1], self.macdsignal[-1], 
                    self.macdhist[-1], self.data_rsi[-1], self.data_obv[-1]]))

                    if rand.random() < self.params.delta_ddpg:
                        self.ddpg_agent.trading_step(self.lob_book)

                # Self-learning trader logic
                # Make precomputing
                if ((int(vv)+1)*tt)%self.aggregate_value == 0:
                	SimulateMarket.make_aggregation(self)

                # update exchange statement
                arr_for_add = self.lob_book.get_trading_data()
                if len(arr_for_add[0])>0:
                    self.data.append(np.mean(arr_for_add[0]))
                    self.volume.append(np.mean(arr_for_add[1]))
                else:
                    self.data.append(self.data[-1])
                    self.volume.append(0)

                # узнаём лучшие цены на выставление лимитных ордеров
                if  len(self.lob_book.LOB_book_ask) > 0:
                    ask_pricing = self.lob_book.get_price('ask_head')
                else:
                    ask_pricing = self.data[-1] + self.initial_spread/2
                    # print(ask_pricing, self.data[-1], self.initial_spread/2)

                if  len(self.lob_book.LOB_book_bid) > 0:
                    bid_pricing = self.lob_book.get_price('bid_head')
                else:
                    bid_pricing = self.data[-1] - self.initial_spread/2

                self.bid_price.append(bid_pricing)
                self.ask_price.append(ask_pricing)

                # get num of shares at best price
                cout_ask = k_ask = 0
                if len(self.lob_book.LOB_book_ask) > 0:
                    min_val_ask = self.lob_book.LOB_book_ask[0][-1]
                    # print(self.LOB_book.LOB_book_ask, min_val, k)
                    while self.lob_book.LOB_book_ask[k_ask][-1] == min_val_ask:
                        cout_ask += self.lob_book.LOB_book_ask[k_ask][-2]
                        k_ask+=1
                        if k_ask >= len(self.lob_book.LOB_book_ask):
                            break

                cout_bid = k_bid = 0
                if len(self.lob_book.LOB_book_bid) > 0:
                    min_val_bid = self.lob_book.LOB_book_bid[0][-1]
                    while self.lob_book.LOB_book_bid[k_bid][-1] == min_val_bid:
                        cout_bid += self.lob_book.LOB_book_bid[k_bid][-2]
                        k_bid+=1
                        if k_bid >= len(self.lob_book.LOB_book_bid):
                            break

                # update profit tables for all agents
                m = MarketMaker(self.lob_book, self.data, 1, self.params.mm_v_min, 
                        self.params.mm_v_max, self.params.mm_v, self.params.mm_w)
                self.agents_profit_table["MM" + str(1)].append(m.profit_calculation())

                L=LiquidConsumer(self.lob_book, self.data, 1, deal_type, deal_vol, 
                            h_min = self.params.lc_h_min, h_max = self.params.lc_h_max)
                self.agents_profit_table["LC" + str(1)].append(L.profit_calculation())

                MT=MomentumTrader(self.lob_book, self.data, 1, 
                        k = self.params.mt_k, n_memory = self.params.mt_nr, wealth = 200)
                self.agents_profit_table["MT" + str(1)].append(MT.profit_calculation())

                MRT=MeanReversionTrader(self.lob_book, self.data, 1, 
                        alpha = self.params.mrt_alpha, k = 5, v_mr = self.params.mrt_v_mr)
                self.agents_profit_table["MRT" + str(1)].append(MRT.profit_calculation())

                NT=NoiseTrader(self.lob_book, self.data, 1,
                 x_min = self.params.xmin_offfspr, betacoeff = self.params.beta_offspr, 
                        lmbda_m = self.params.lmbda_m, lmbda_l = self.params.lmbda_l,
                         lmbda_c = self.params.lmbda_c, 
                        mu_mo = self.params.mu_mo, omeg_mo = self.params.omeg_mo, 
                        mu_lo = self.params.omeg_lo, omeg_lo = self.params.omeg_lo)
                self.agents_profit_table["NT" + str(1)].append(NT.profit_calculation())

                self.agents_profit_table["DDPG" + str(1)].append(self.ddpg_agent.profit_calculation(self.lob_book, self.data))

                self.bid_volume.append(cout_bid)
                self.ask_volume.append(cout_ask)
         



if __name__ == '__main__':
    market = SimulateMarket()
    print('Compiled successfully. Model building data...')
    market.start()
    print('Model was builder successfully')
    market.write("data/" + parameters_list().save_folder)
    print('Data saved by way:', "data/" + parameters_list().save_folder)
