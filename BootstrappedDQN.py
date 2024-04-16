import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import math
import utils
import random

import Rlclasses2 as RL
import helpfunctions as help

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, width, height, fc1_dims, fc2_dims,
                 n_actions):
        super(DuelingDeepQNetwork, self).__init__()

        self.width = width
        self.height = height
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(2,8,3)
        self.conv2 = nn.Conv2d(8,16,3)
        self.fc1 = nn.Linear(16*(height-4)*(width-4), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.values = nn.Linear(fc2_dims,1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = state.type(T.float32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,16*(self.height-4)*(self.width-4))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        V = self.values(x)

        return actions, V

class SharedNet(nn.Module):
    def __init__(self, width, height, fc1_dims):
        super(SharedNet, self).__init__()
        self.width = width
        self.height = height
        self.fc1_dims = fc1_dims

        self.conv1 = nn.Conv2d(2,8,3)
        self.conv2 = nn.Conv2d(8,16,3)
        self.fc1 = nn.Linear(16*(height-4)*(width-4), self.fc1_dims)


    def forward(self, x):
        x = x.type(T.float32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,16*(self.height-4)*(self.width-4))
        x = F.relu(self.fc1(x))
        return x

class HeadNet(nn.Module):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(HeadNet, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.values = nn.Linear(self.fc2_dims,1)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        V = self.values(x)
        return actions, T.add(V, (actions-actions.mean(dim=1, keepdim=True)))

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, lr, width, height, fc1_dims, fc2_dims,
                 n_actions):
        super(EnsembleNet, self).__init__()

        self.shared_net = SharedNet(width, height, fc1_dims)

        self.net_list = nn.ModuleList([HeadNet(n_actions,fc1_dims,fc2_dims) for k in range(n_ensemble)])
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _core(self, x):
        return self.shared_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        if k is not None:
            return self.net_list[k](self.shared_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads

class Agent:
    def __init__(self, gamma, epsilon, lr, width,height, batch_size, n_actions, myBeta, numHeads, bProb,
                 max_mem_size=5000, eps_end=0.1, eps_dec=0.99, reduce_eps = 1000, annealBias=True):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.reduce_eps = reduce_eps
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 10
        self.actionCounts = T.ones(n_actions)
        self.beta = myBeta
        self.numHeads = numHeads
        self.bProb = bProb

        if annealBias:
            self.final_p = 1.0
        else:
            self.final_p = self.beta

        self.beta_schedule = utils.LinearSchedule(math.log(self.eps_min)*self.reduce_eps/math.log(self.eps_dec), initial_p=self.beta, final_p=self.final_p)

        self.Q_eval = EnsembleNet(self.numHeads,lr, n_actions=n_actions,
                                   width=width,height=height,
                                   fc1_dims=8, fc2_dims=16)
        
        self.Q_target = EnsembleNet(self.numHeads,lr, n_actions=n_actions,
                                   width=width,height=height,
                                   fc1_dims=8, fc2_dims=16)

        #self.state_memory = np.zeros((self.mem_size, input_dims),
        #                             dtype=np.float32)
        #self.new_state_memory = np.zeros((self.mem_size, input_dims),
        #                                 dtype=np.float32)
        #self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        #self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        #self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.memory = help.PrioritizedReplayBufferEnsemble(size=self.mem_size,alpha=0.6,numHeads=self.numHeads, bProb=self.bProb)

    def store_transition(self, state, action, reward, state_, terminal):
        #index = self.mem_cntr % self.mem_size
        #self.state_memory[index] = state
        #self.new_state_memory[index] = state_
        #self.reward_memory[index] = reward
        #self.action_memory[index] = action
        #self.terminal_memory[index] = terminal
        self.memory.add(state,action,reward,state_,terminal)

        self.mem_cntr += 1

    def choose_action(self, observation,k):
        state = T.tensor(observation).to(self.Q_eval.device)
        if k == None:
            actions = self.Q_eval.forward(state,k)
            action_list = []
            for head in actions:
                option = T.argmax(head[0]).item()
                action_list.append(option)
            action = max(set(action_list), key=action_list.count)
        elif np.random.random() > self.epsilon:
            actions, _ = self.Q_eval.forward(state,k)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def replace_target_network(self):
        if self.iter_cntr % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())


    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        #state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        #new_state_batch = T.tensor(
        #        self.new_state_memory[batch]).to(self.Q_eval.device)
        #action_batch = self.action_memory[batch]
        #reward_batch = T.tensor(
        #        self.reward_memory[batch]).to(self.Q_eval.device)
        #terminal_batch = T.tensor(
        #        self.terminal_memory[batch]).to(self.Q_eval.device)
        
        samples = self.memory.sample(self.batch_size,self.beta)
        state_batch = T.tensor(samples[0]).to(self.Q_eval.device)
        new_state_batch = T.tensor(samples[3]).to(self.Q_eval.device)
        action_batch = samples[1]
        reward_batch = T.tensor(samples[2]).to(self.Q_eval.device)
        terminal_batch = T.tensor(samples[4]).to(self.Q_eval.device)
        mask = T.tensor(samples[5]).to(self.Q_eval.device)
        weights = samples[6]
        batch_idxes = samples[7]


        weights = np.sqrt(weights)
        weights = T.FloatTensor(weights).to(self.Q_eval.device)

        #q = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        #print("Initialised q values: ", q)

        q_pred_list = self.Q_eval.forward(new_state_batch,None) #[(action,q),...]
        q_eval_list = self.Q_eval.forward(state_batch,None)
        q_target_list = self.Q_target.forward(new_state_batch,None)

        cnt_losses = []
        for k in range(self.numHeads):
            q = T.zeros(self.batch_size)
            total_used = T.sum(mask[:,k])

            if total_used > 0.0:
                q_target = q_target_list[k][1]
                q_pred = q_pred_list[k][1]
                q_eval = q_eval_list[k][1]
                

                q_eval = T.gather(q_eval,1,T.tensor(action_batch,dtype=T.int64).view(-1,1))

                maxA = T.argmax(q_pred, dim=1).to(self.Q_eval.device)
                maxA = maxA.view(-1,1)

                q_target = T.gather(q_target,1,maxA)


                q[T.logical_not(terminal_batch)] = reward_batch[T.logical_not(terminal_batch)] + self.gamma*q_target[T.logical_not(terminal_batch)].squeeze()
                q[terminal_batch] = reward_batch[terminal_batch].float()

                TD_error = q - q_eval.squeeze()
                weighted_TD_errors = T.mul(TD_error, weights).to(self.Q_eval.device)
                zero_tensor = T.zeros(weighted_TD_errors.shape).to(self.Q_eval.device)
        
                partial_loss = self.Q_eval.loss(weighted_TD_errors, zero_tensor).to(self.Q_eval.device)

                full_loss = mask[:,k]*partial_loss
                cnt_losses.append(T.sum(full_loss/total_used))

        loss = sum(cnt_losses)/self.numHeads
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        if self.iter_cntr % self.reduce_eps == 0:
            self.epsilon = self.epsilon*self.eps_dec
            self.beta = self.beta_schedule.value(self.iter_cntr)
            print("Epsilon: ",self.epsilon)
            #print('Beta: ', self.beta)

        td_errors = TD_error.detach().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(batch_idxes, new_priorities)
        #self.epsilon = self.epsilon - self.eps_dec \
        #    if self.epsilon > self.eps_min else self.eps_min


'''
# Instantiate the environment:
myObstacles = [(7,3),(7,4),(7,5),
               (8,3),(8,4),(8,5),
               (9,3),(9,4),(9,5),
               (0,0),(0,1),(0,2),(0,4),(0,5),(0,6),(0,7),
               (1,0),(1,7),
               (2,0),(2,7),
               (3,0),(3,7),
               (4,7),
               (5,0),(5,7),
               (6,0),(6,7),
               (7,0),(7,7),
               (8,7),
               (9,0),(9,7),
               (10,0),(10,7),
               (11,7),
               (12,0),(12,7),
               (13,0),(13,7),
               (14,7),
               (15,0),(15,7),
               (16,0),(16,1),(16,2),(16,3),(16,4),(16,5),(16,6),(16,7)]

block1 = [(1,4),(1,5),(1,6),
          (2,4),(2,5),(2,6),
          (3,4),(3,5),(3,6),
          (4,4),(4,5),(4,6),
          (5,4),(5,5),(5,6)]

block2 = [(11,3),(11,4),(11,5),
          (12,3),(12,4),(12,5)]

block3 = [(15,1),(15,2),(15,3),(15,4),(15,5),(15,6)]

myObstacles += block1
myObstacles += block2
myObstacles += block3

station1 = RL.Station(1,1,(0,3))
station2 = RL.Station(1,1,(4,0))
station3 = RL.Station(1,1,(8,0))
station4 = RL.Station(1,1,(11,0))
station5 = RL.Station(1,1,(14,0))
myStations = [station1,station2,station3,station4,station5]
myAgent = RL.Agent(0,2,(14,4))
deposit = (14,5)
myFactory = RL.Factory(8,17,myObstacles,myStations,myAgent,deposit)
print(myFactory.grid)
'''

# Instantiate the environment:
myObstacles = [(3,2),
               (0,0),(0,1),(0,2),(0,3),(0,4),
               (1,0),(1,4),
               (2,0),(2,4),
               (3,0),(3,4),
               (4,0),(4,4),
               (5,0),(5,1),(5,2),(5,3),(5,4)]
station1 = RL.Station(2,3,(1,2))
station2 = RL.Station(1,1,(1,3))
myStations = [station1,station2]
myAgent = RL.Agent(1,2,(4,1))
deposit = (4,3)
myFactory = RL.Factory(5,6,myObstacles,myStations,myAgent,deposit)

actions = myFactory.possibleActions
numActions = len(actions)

GAMMA = 0.95
ALPHA = 0.003
EPSILON = 1.0
BATCHSIZE = 30

INITIAL_BETA = 0.99
RANDOM_RESET = False

NUM_HEADS = 5
BERNOULLI_PROB = 0.5

myAgent = Agent(GAMMA,EPSILON,ALPHA,5,6,BATCHSIZE,numActions,INITIAL_BETA, NUM_HEADS,BERNOULLI_PROB,5000,reduce_eps=500,annealBias=False)

while myAgent.mem_cntr < myAgent.mem_size:
    done = False
    while not done:
        action = myFactory.randomAction()
        currentState = myFactory.getState()
        nextState, reward, done = myFactory.step(action)
        if done:
            print("Terminated")
        myAgent.store_transition(currentState,help.get_index(action,myFactory.possibleActions),reward,nextState,done)
    if RANDOM_RESET:
        myFactory.reset2()
    else:
        myFactory.reset()
print("Memory initilised")

scores = []
epsHistory = []
numEpisodes = 500
maxEpisodeLength = 10000


counter = 0
while myAgent.epsilon > myAgent.eps_min:
    epsHistory.append(myAgent.epsilon)
    done = False
    if RANDOM_RESET:
        myFactory.reset2()
    else:
        myFactory.reset()

    score = 0

    k = random.randint(0,NUM_HEADS-1)

    counter = 0 
    while not done:
        state = myFactory.getState()
        action = myAgent.choose_action(state,k)
        actionString = myFactory.possibleActions[action]
        nextState, reward, done = myFactory.step(actionString)
        score += reward
        myAgent.store_transition(state,action,reward,nextState,done)
        myAgent.learn()
        #if score % 5000 == 0:
            #print(score)
        counter +=1
        if counter > maxEpisodeLength:
            break
    print(score)

    scores.append(score)

done = False
if RANDOM_RESET:
    myFactory.reset2()
else:
    myFactory.reset()
print(myFactory.getState())
myAgent.epsilon = 0
score = 0
while not done:
    state = myFactory.getState()
    action = myAgent.choose_action(help.flatten_tuple(state),None)
    actionString = myFactory.possibleActions[action]
    nextState, reward, done = myFactory.step(actionString)
    score += reward
    #myAgent.store_transition(help.flatten_tuple(state),action,reward,help.flatten_tuple(nextState),done)    
    print(actionString)
print("Final score: ",score)
plt.plot(scores)
plt.show()
