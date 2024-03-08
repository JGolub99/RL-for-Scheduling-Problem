import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

import Rlclasses as RL
import helpfunctions as help

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = state.type(T.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=5000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=8, fc2_dims=16)
        self.state_memory = np.zeros((self.mem_size, input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action


    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        if self.iter_cntr % 1000 == 0:
            self.epsilon = self.epsilon*0.99
            print("Epsilon: ",self.epsilon)
        #self.epsilon = self.epsilon - self.eps_dec \
        #    if self.epsilon > self.eps_min else self.eps_min



# Instantiate the environment:
myObstacles = [(7,3),(7,4),(7,5),
               (8,3),(8,4),(8,5),
               (9,3),(9,4),(9,5),
               (0,0),(0,1),(0,2),(0,4),(0,5),(0,6),
               (1,0),
               (2,0),
               (3,0),
               (5,0),
               (6,0),
               (7,0),
               (9,0),
               (10,0),
               (12,0),
               (13,0),
               (15,0)]
station1 = RL.Station(1,1,(0,3))
station2 = RL.Station(1,1,(4,0))
station3 = RL.Station(1,1,(8,0))
station4 = RL.Station(1,1,(11,0))
station5 = RL.Station(1,1,(14,0))
myStations = [station1,station2,station3,station4,station5]
myAgent = RL.Agent(0,2,(14,4))
deposit = (14,5)
myFactory = RL.Factory(7,16,myObstacles,myStations,myAgent,deposit)


observation = myFactory.getState()
actions = myFactory.possibleActions
numActions = len(actions)
observationList = help.flatten_tuple(observation)
numStates = len(observationList)

GAMMA = 0.95
ALPHA = 0.003
EPSILON = 1.0
BATCHSIZE = 2000

myAgent = Agent(GAMMA,EPSILON,ALPHA,numStates,BATCHSIZE,numActions,50000)

while myAgent.mem_cntr < myAgent.mem_size:
    myFactory.reset()
    done = False
    while not done:
        action = myFactory.randomAction()
        currentState = myFactory.getState()
        nextState, reward, done = myFactory.step(action)
        myAgent.store_transition(help.flatten_tuple(currentState),help.get_index(action,myFactory.possibleActions),reward,help.flatten_tuple(nextState),done)
print("Memory initilised")

scores = []
epsHistory = []
numEpisodes = 500

while myAgent.epsilon > 0.1 :
    epsHistory.append(myAgent.epsilon)
    done = False
    myFactory.reset()

    score = 0

    while not done:
        state = myFactory.getState() # (position tuple, agent load, factory load 1, factor load 2)
        #print(state)
        action = myAgent.choose_action(help.flatten_tuple(state))
        actionString = myFactory.possibleActions[action]
        nextState, reward, done = myFactory.step(actionString)
        score += reward
        myAgent.store_transition(help.flatten_tuple(state),action,reward,help.flatten_tuple(nextState),done)
        myAgent.learn()
        #if score % 100 == 0:
        #    print(score)
    print(score)

    scores.append(score)

done = False
myFactory.reset()
myAgent.epsilon = 0
score = 0
while not done:
    state = myFactory.getState()
    action = myAgent.choose_action(help.flatten_tuple(state))
    actionString = myFactory.possibleActions[action]
    nextState, reward, done = myFactory.step(actionString)
    score += reward
    #myAgent.store_transition(help.flatten_tuple(state),action,reward,help.flatten_tuple(nextState),done)    
    print(actionString)
print("Final score: ",score)
plt.plot(scores)
plt.show()