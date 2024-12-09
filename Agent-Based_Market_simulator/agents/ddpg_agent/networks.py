import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm.notebook import tqdm


# Actor network
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        #print(state_size)
        hidden_layer_1 = 200
        hidden_layer_2 = 100
        self.gru = nn.GRU(state_size, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, action_size)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        if len(state.shape) == 1:
            state = torch.reshape(state, (1, 1, state.shape[-1]))
        else:
            state = torch.reshape(state, (1, state.shape[-2], state.shape[-1]))
        #print(state.shape)
        state = state.float()
        #print(state)
        net, h = self.gru(state)
        #net = self.softmax(net)
        net = self.fc3(net)
        net = self.tanh(net)
        return net

# Critic network
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        #hidden_layer_1 = 200
        hidden_layer_2 = 100
        self.gru = nn.GRU(state_size, hidden_layer_2)
        self.fc2 = nn.Linear(hidden_layer_2 + action_size, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, 1)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state, action):
        if len(state.shape) == 1:
            state = torch.reshape(state, (1, 1, state.shape[-1]))
        else:
            state = torch.reshape(state, (1, state.shape[-2], state.shape[-1]))
        if len(action.shape) > 3:
            action = torch.reshape(action, (1, action.shape[0], action.shape[-1]))
        state = state.float()
        net, h = self.gru(state)
        # It may be important!!!
        #net = self.softmax(net)
        #print(net, net.shape, action, action.shape)
        
        #It is neccesary to check this point!
        net = self.fc2(torch.cat([net, action], 2))
        #net = self.relu(net)
        #net = self.softmax(net)
        net = self.fc3(net)
        return net