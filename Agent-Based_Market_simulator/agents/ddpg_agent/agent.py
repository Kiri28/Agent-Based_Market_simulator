import time
import copy
import random
import numpy as np
import pandas as pd
from collections import deque

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.autograd import Variable

from agents.ddpg_agent.ddpg import DDPG
#from ddpg import DDPG
#from networks import Actor, Critic
from agents.ddpg_agent.replay_buffer import ExpirienceReplay
#from agents.ddpg_agent.data_distributor import data_distributor

#class DDPG_agent(BaseAgent):
class DDPG_agent:
    
    def __init__(self, external_id):
        self.agent_id = "DDPG_agent"+str(external_id)
        self.ddpg = DDPG(12, 1, tau = 0.001, gamma = 0.99)
        self.step = 0
        self.start_train = 10000
        #cls = data_distributor()
        self.done = True
        buffer_size = 500
        self.buffer = ExpirienceReplay(buffer_size)
        self.actor_loss_sum = 0
        self.critic_loss_sum = 0
        self.loss_ctn = 0
        self.test_every=1000
    
        self.log_ts = []
        self.log_mean = []
        self.log_std = []
        self.reward_arr = []

        #rng = tqdm(range(timesteps))

    def profit_calculation(self, LOB_book, data):
        # id our agent exists in profit table...
        if self.agent_id in LOB_book.curr_portfolio:
            # append his immediate profit
            return LOB_book.curr_portfolio[self.agent_id]*(data[-1]-data[-2])
        else:
            # else append zero immediate profit
            return 0
    
    def run_analytics(self, last_data):

        if self.done:
            self.done = False
            self.state = np.append(last_data, np.array([0]))
        action = self.ddpg.get_action(self.state)
        #print(self.state)
        # observe state s and select action a
        action = np.round(np.clip(action + 0.1 * np.random.randn(len(action)), -1.0, 1.0))
        next_state = np.append(last_data, self.state[-1]+action)
        reward = (next_state[0] - self.state[0])*next_state[-1]
        done = False
        self.buffer.add((self.state, action, next_state, reward, done))
        self.state = next_state

        if self.step > self.start_train:
            # randomly sample a batch of transitions
            batch_size = 256
            batch = self.buffer.sample(batch_size)
            critic_loss, actor_loss = self.ddpg.update(batch)
            self.actor_loss_sum += actor_loss
            self.critic_loss_sum += critic_loss
            self.loss_ctn += 1

            if self.step % self.test_every == 0: #or self.step == timesteps - 1:
                self.log_ts.append(self.step)
                #mean, std = test(ddpg, test_count)
                #self.log_mean.append(mean)
                #self.log_std.append(std)
                self.actor_loss_sum = 0
                self.critic_loss_sum = 0
                self.loss_ctn = 0

        #reward_data
        self.reward_arr.append(reward)
    
        self.step += 1
        return 1

    def trading_step(self, LOB_book):
        action = self.ddpg.get_action(self.state)
        act_value = np.round(np.clip(action + 0.2 * np.random.randn(len(action)), -1.0, 1.0))[0][0][0]
        #print(act_value)
        if act_value > 0 and LOB_book.total_ask > 0:
            LOB_book.make_purchase(self.agent_id, 1)
        elif act_value < 0 and LOB_book.total_bid > 0:
            LOB_book.make_sell(self.agent_id, 1)
        #print(action)
