import Rlclasses as RL
import numpy as np
import helpfunctions as help
from matplotlib import pyplot as plt
import random

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

# Set the hyperparameters:
ALPHA = 0.1
GAMMA = 1
EPSILON = 1

# Create the Q table:
Q1 = {}
Q2 = {}

for state in myFactory.stateSpacePlus:
    for action in myFactory.possibleActions:
        Q1[state,action] = 0
        Q2[state,action] = 0

numberEpisodes = 1000
totalReward = np.zeros(numberEpisodes)

for i in range(numberEpisodes):
    episodeReward = 0
    doneFlag = False
    myFactory.reset()
    #actionList = []
    while not doneFlag:
        randNum = random.random()
        if randNum < EPSILON:
            action = myFactory.randomAction()
        else:
            action = help.bestActionDouble(myFactory,Q1,Q2)
        #print(action)
        #actionList.append(action)
        oldState = myFactory.getState()
        newState, reward, doneFlag = myFactory.step(action)
        episodeReward += reward

        randQ = random.random()
        if randQ < 0.5:

            updatedAction = help.bestAction(myFactory,Q1)
            #Update Q values:
            Q1[oldState,action] = Q1[oldState,action] + ALPHA*(reward + GAMMA*Q2[newState,updatedAction] - Q1[oldState,action])
        else:
            updatedAction = help.bestAction(myFactory,Q2)
            #Update Q values:
            Q2[oldState,action] = Q2[oldState,action] + ALPHA*(reward + GAMMA*Q1[newState,updatedAction] - Q2[oldState,action])  

    print("Episode ", i)
    print("Reward: ", episodeReward)
    if i%10 == 0:
        print("Epsilon:", EPSILON)
        #if episodeReward>-30:
        #    print("Actions: ", actionList)
        EPSILON = EPSILON*0.99
    totalReward[i] = episodeReward

episodeReward = 0
doneFlag = False
myFactory.reset()
actionList = []
while not doneFlag:
    action = help.bestActionDouble(myFactory,Q1,Q2)
    #print(action)
    actionList.append(action)
    oldState = myFactory.getState()
    newState, reward, doneFlag = myFactory.step(action)
    episodeReward += reward
    randQ = random.random()
    if randQ < 0.5:

        updatedAction = help.bestAction(myFactory,Q1)
        #Update Q values:
        Q1[oldState,action] = Q1[oldState,action] + ALPHA*(reward + GAMMA*Q2[newState,updatedAction] - Q1[oldState,action])
    else:
        updatedAction = help.bestAction(myFactory,Q2)
        #Update Q values:
        Q2[oldState,action] = Q2[oldState,action] + ALPHA*(reward + GAMMA*Q1[newState,updatedAction] - Q2[oldState,action]) 

print("Reward: ", episodeReward)
print(actionList)

plt.plot(totalReward)
plt.show()