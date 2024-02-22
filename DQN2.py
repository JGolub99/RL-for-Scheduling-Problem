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

        self.optimizer = optim.RMSprop(self.parameters(), lr = alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        x = torch.Tensor(x).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

def learn(agent,qNetwork,factory,batchsize,gamma):
    qNetwork.optimizer.zero_grad()
    
    # Randomly select the batch of data

    if agent.memCntr + batchsize < agent.maxMem:
        memStart = int(np.random.choice(range(agent.memCntr)))
    else:
        memStart = int(np.random.choice(range(agent.maxMem-batchsize-1)))
        #print(memStart)
    miniBatch = agent.memory[memStart:memStart+batchsize]
    #print("Batch: ", len(miniBatch))
    memory = []
    QTarget = []
    actions = []
    for item in miniBatch:
        memory.append(help.flatten_tuple(item[0]))
        actions.append(help.get_index(item[1],factory.possibleActions))
        if factory.isTerminalState(item[3]):
            QTarget.append(item[2])
        else:
            QNext = qNetwork.forward(help.flatten_tuple(item[3])).to(qNetwork.device)
            QTarget.append(item[2] + gamma*torch.max(QNext))
    memory = np.array(memory)
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

observation = myFactory.getState()
actions = myFactory.possibleActions
numActions = len(actions)
observationList = help.flatten_tuple(observation)
numStates = len(observationList)

ALPHA = 0.03
EPSILON = 1
GAMMA = 1.0
numEpisodes = 1000
totalReward = np.zeros(numEpisodes)
#QEval = QNetwork(ALPHA,numStates,numActions)
QNext = QNetwork(ALPHA,numStates,numActions)
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


for i in range(numEpisodes):
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
            actionsTensor = QNext.forward(currentStateList)
            actionIndex = torch.argmax(actionsTensor).item()
            action = myFactory.possibleActions[actionIndex]

        newState, reward, doneFlag = myFactory.step(action)
        newStateList = help.flatten_tuple(newState)
        myAgent.storeTransition(currentStateList, action, reward, newStateList)
        episodeReward+=reward
        learn(myAgent,QNext,myFactory,30,GAMMA)
    print(episodeReward)
    totalReward[i] = episodeReward
    if i%10 == 0:
        print("Epsilon:", EPSILON)
        #if episodeReward>-30:
        #    print("Actions: ", actionList)
        EPSILON = EPSILON*0.99

plt.plot(totalReward)
plt.show()

#print(myAgent.memory)

        




#print(myNetwork.forward(observationList))