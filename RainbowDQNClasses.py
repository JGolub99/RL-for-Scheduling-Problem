import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import utils

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
        self.conv1 = nn.Conv2d(5,8,3)
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

class Agent:
    def __init__(self, gamma, epsilon, lr, width,height, batch_size, n_actions, myBeta,
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

        if annealBias:
            self.final_p = 1.0
        else:
            self.final_p = self.beta

        self.beta_schedule = utils.LinearSchedule(math.log(self.eps_min)*self.reduce_eps/math.log(self.eps_dec), initial_p=self.beta, final_p=self.final_p)

        self.Q_eval = DuelingDeepQNetwork(lr, n_actions=n_actions,
                                   width=width,height=height,
                                   fc1_dims=8, fc2_dims=16)
        
        self.Q_target = DuelingDeepQNetwork(lr, n_actions=n_actions,
                                   width=width,height=height,
                                   fc1_dims=8, fc2_dims=16)

        #self.state_memory = np.zeros((self.mem_size, input_dims),
        #                             dtype=np.float32)
        #self.new_state_memory = np.zeros((self.mem_size, input_dims),
        #                                 dtype=np.float32)
        #self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        #self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        #self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.memory = help.PrioritizedReplayBuffer(size=self.mem_size,alpha=0.6)

    def store_transition(self, state, action, reward, state_, terminal):
        #index = self.mem_cntr % self.mem_size
        #self.state_memory[index] = state
        #self.new_state_memory[index] = state_
        #self.reward_memory[index] = reward
        #self.action_memory[index] = action
        #self.terminal_memory[index] = terminal
        self.memory.add(state,action,reward,state_,terminal)

        self.mem_cntr += 1

    def choose_action(self, observation, k = None):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation).to(self.Q_eval.device)
            actions, values = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def replace_target_network(self):
        if self.iter_cntr % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def choose_ucb_action(self,observation):     

        state = T.tensor(observation).to(self.Q_eval.device)
        Q = self.Q_eval.forward(state)



        added = T.sqrt(T.div(math.log(self.iter_cntr+0.001),self.actionCounts))

        actions = Q + 3.0*added
        action = T.argmax(actions).item()
        self.actionCounts[action]+=1

        return action

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
        weights = samples[5]
        batch_idxes = samples[6]


        weights = np.sqrt(weights)
        weights = T.FloatTensor(weights).to(self.Q_eval.device)

        #q = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q = T.zeros(self.batch_size)
        #print("Initialised q values: ", q)


        a_pred, v_pred = self.Q_eval.forward(new_state_batch)
        a_eval, v_eval = self.Q_eval.forward(state_batch)

        q_pred = T.add(v_pred, (a_pred-a_pred.mean(dim=1, keepdim=True)))
        q_eval = T.add(v_eval, (a_eval-a_eval.mean(dim=1, keepdim=True)))

        #q_pred = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)
        #q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
        q_eval = T.gather(q_eval,1,T.tensor(action_batch,dtype=T.int64).view(-1,1))
        #print("Qpred: ", q_pred)
        #print("Qeval: ", q_eval)
        #print(q_eval.size())

        #q_next = self.Q_next.forward(new_state_batch)
        
        maxA = T.argmax(q_pred, dim=1).to(self.Q_eval.device)
        maxA = maxA.view(-1,1)
        #print("Best actions: ", maxA)

        a_target, v_target = self.Q_target.forward(new_state_batch)
        q_target = T.add(v_target, (a_target-a_target.mean(dim=1, keepdim=True)))
        #print("Ungathers Qtarget: ", q_target)
        q_target = T.gather(q_target,1,maxA)
        #print("Gathered target: ", q_target)
        #print(q_target[T.logical_not(terminal_batch)].squeeze().size())
        #print(q[T.logical_not(terminal_batch)])

        #print("Rewards:", reward_batch)

        q[T.logical_not(terminal_batch)] = reward_batch[T.logical_not(terminal_batch)] + self.gamma*q_target[T.logical_not(terminal_batch)].squeeze()
        q[terminal_batch] = reward_batch[terminal_batch].float()
        #print("q_values: ", q)
        #q_target[maxA] = reward_batch + self.gamma*T.max(q_next, dim=1)
        TD_error = q - q_eval.squeeze()
        weighted_TD_errors = T.mul(TD_error, weights).to(self.Q_eval.device)
        zero_tensor = T.zeros(weighted_TD_errors.shape).to(self.Q_eval.device)
        
        loss = self.Q_eval.loss(weighted_TD_errors, zero_tensor).to(self.Q_eval.device)
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