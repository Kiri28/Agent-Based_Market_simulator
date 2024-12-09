import copy
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.autograd import Variable

#from networks import Actor, Critic
from agents.ddpg_agent.networks import Actor, Critic

device = torch.device("cpu") # Model trains faster in GPU
#device = torch.device("cuda")

class DDPG:
    def __init__(self, state_size, action_size, tau=0.001, gamma=0.99):
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor.to(device)
        self.target_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.001)
        
    def get_action(self, state):
        action = self.actor(torch.from_numpy(state)).detach().numpy()
        #action = np.array(self.actor(torch.from_numpy(state))).squeeze(0)
        action = np.clip(action, -1., 1.)
        
        return action
    
    def compute_actor_loss(self, state):

        # сначала выдаём предсказание состояния, а затем...
        actor_loss = -self.critic(state, self.actor(state))
        actor_loss = actor_loss.mean()

        return actor_loss
    
    def compute_critic_loss(self, state, action, next_state, reward, done):
        target_1 = self.target_actor(next_state)
        #y_value = reward + self.gamma*self.target_critic(next_state, target_1)
        y_value = reward + self.gamma*torch.squeeze(self.target_critic(next_state, target_1))
        critic_value = self.critic(state, action)

        #print(y_value.size())
        #Q_value_estimation = self.critic(y_value, critic_value)
        
        loss_foo = nn.MSELoss()
        #123123123
        #print(critic_value.size(), y_value.size())
        value_loss = loss_foo(torch.squeeze(critic_value), y_value)
        loss = value_loss
        return loss
    
    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def update(self, batch):
        state, action, next_state, reward, done = batch
        state = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
        done = torch.tensor(done, device=device)
        action = torch.tensor(action, device=device, dtype=torch.float32)
        
        actor_loss = self.compute_actor_loss(state)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Actor soft update
        self.soft_update(self.target_actor, self.actor)

        critic_loss = self.compute_critic_loss(state, action, next_state, reward, done)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
            
        # Critic soft update
        self.soft_update(self.target_critic, self.critic)

        return critic_loss.item(), actor_loss.item()