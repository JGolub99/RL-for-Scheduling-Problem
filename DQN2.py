'''
In this file we are going to try to perform DQN again, but using a different representation for the agent.
'''

import Rlclasses as RL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import helpfunctions as help
import random
from matplotlib import pyplot as plt

class QNetwork(nn.Module):
    def __init__(self,alpha,input_space,output_space):
        super(QNetwork,self).__init__()
        
        hidden_space1 = 8  # Nothing special with 8, feel free to change
        hidden_space2 = 16  # Nothing special with 16, feel free to change

        self.fc1 = nn.Linear(input_space,hidden_space1)
        self.fc2 = nn.Linear(hidden_space1,hidden_space2)
        self.fc3 = nn.Linear(hidden_space2,output_space)

        self.optimizer = optim.AdamW(self.parameters(), lr = alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        x = torch.Tensor(x).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

def learn(agent,qNetwork,qNext,factory,batchsize,gamma, replace):
    qNetwork.optimizer.zero_grad()
    
    if replace:
        qNext.load_state_dict(qNetwork.state_dict())

    # Randomly select the batch of data

    if agent.memCntr < batchsize:
        miniBatch = agent.memory
    else:
        miniBatch = random.sample(agent.memory,batchsize)
        #print(memStart)
    #print("Batch: ", len(miniBatch))
    memory = []
    nextMemory = []
    QTarget = []
    actions = []
    for item in miniBatch:
        memory.append(help.flatten_tuple(item[0]))
        nextMemory.append(help.flatten_tuple(item[3]))
        actions.append(help.get_index(item[1],factory.possibleActions))
        if factory.isTerminalState(item[3]):
            QTarget.append(item[2])
        else:
            QNext = qNext.forward(help.flatten_tuple(item[3])).to(qNetwork.device)
            QTarget.append(item[2] + gamma*torch.max(QNext))
    memory = np.array(memory)
    nextMemory = np.array(nextMemory)
    #print("Memory size: ", np.size(memory))
    #print("Actions: ", actions)
    QPredFull = qNetwork.forward(memory).to(qNetwork.device)
    QPred = torch.Tensor(QPredFull[list(range(0,len(memory))),actions])
    QTarget = torch.Tensor(QTarget)
    #print("QPredFull: ", QPredFull)
    #print("QPred: ", QPred)
    #print("QTarget: ", QTarget)
    loss = qNetwork.loss(QTarget,QPred).to(qNetwork.device)
    #print("Loss: ", loss)
    loss.backward()
    qNetwork.optimizer.step()
    #print(torch.Tensor.size(QPred))

# Instantiate the environment:
myObstacles = [(2,1)]
station1 = RL.Station(2,3,(0,1))
station2 = RL.Station(1,1,(0,2))
myStations = [station1,station2]
myAgent = RL.Agent(1,2,(3,0))
deposit = (3,2)
myFactory = RL.Factory(3,4,myObstacles,myStations,myAgent,deposit)


'''
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
print(myFactory.grid)
'''
observation = myFactory.getState()
actions = myFactory.possibleActions
numActions = len(actions)
observationList = help.flatten_tuple(observation)
numStates = len(observationList)

ALPHA = 0.01
EPSILON = 1
GAMMA = 1.0
numEpisodes = 100
totalReward = np.zeros(numEpisodes)
#QEval = QNetwork(ALPHA,numStates,numActions)
policyNet = QNetwork(ALPHA,numStates,numActions)
targetNet = QNetwork(ALPHA,numStates,numActions)
targetNet.load_state_dict(policyNet.state_dict())
#currentState = help.flatten_tuple(myFactory.getState())
#actions = QNext.forward(currentState)
#actionIndex = torch.argmax(actions).item()
#action = myFactory.possibleActions[actionIndex]
#newState, reward, doneFlag = myFactory.step(action)
#myAgent.storeTransition(currentState, action, reward, newState)
#myAgent.storeTransition(currentState, action, reward, newState)
#myAgent.storeTransition(currentState, action, reward, newState)

#learn(myAgent,QNext,myFactory,10,GAMMA)
#print(myAgent.memory)
#print(myFactory.randomAction())

swapCnt = 0
for i in range(numEpisodes):
    swapCnt += 1
    episodeReward = 0
    doneFlag = False
    myFactory.reset()
    while not doneFlag:
        randNum = random.random()
        currentState = myFactory.getState()
        currentStateList = help.flatten_tuple(currentState)
        if randNum < EPSILON:
            action = myFactory.randomAction()
        else:
            actionsTensor = policyNet.forward(currentStateList)
            actionIndex = torch.argmax(actionsTensor).item()
            action = myFactory.possibleActions[actionIndex]

        newState, reward, doneFlag = myFactory.step(action)
        newStateList = help.flatten_tuple(newState)
        myAgent.storeTransition(currentStateList, action, reward, newStateList)
        episodeReward+=reward
        if swapCnt%1000==0:
            learn(myAgent,policyNet,targetNet,myFactory,100,GAMMA,True)
        else:
            learn(myAgent,policyNet,targetNet,myFactory,100,GAMMA,False)
        if episodeReward % 1000 == 0:
            print(episodeReward)
    print(episodeReward)
    totalReward[i] = episodeReward
    if i%10 == 0:
        print("Epsilon:", EPSILON)
        #if episodeReward>-30:
        #    print("Actions: ", actionList)
        EPSILON = EPSILON*0.9

plt.plot(totalReward)
plt.show()

#print(myAgent.memory)

        




#print(myNetwork.forward(observationList))