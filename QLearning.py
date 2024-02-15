import Rlclasses as RL
import numpy as np
import helpfunctions as help
from matplotlib import pyplot as plt
import random

# Instantiate the environment:
myObstacles = [(2,1)]
station1 = RL.Station(2,3,(0,1))
station2 = RL.Station(1,1,(0,2))
myStations = [station1,station2]
myAgent = RL.Agent(1,2,(3,0))
deposit = (3,2)
myFactory = RL.Factory(3,4,myObstacles,myStations,myAgent,deposit)

# Set the hyperparameters:
ALPHA = 0.1
GAMMA = 1
EPSILON = 1

# Create the Q table:
Q = {}

for state in myFactory.stateSpacePlus:
    for action in myFactory.possibleActions:
        Q[state,action] = 0

numberEpisodes = 5000
totalReward = np.zeros(numberEpisodes)

for i in range(numberEpisodes):
    episodeReward = 0
    doneFlag = False
    myFactory.reset()
    actionList = []
    while not doneFlag:
        randNum = random.random()
        if randNum < EPSILON:
            action = myFactory.randomAction()
        else:
            action = help.bestAction(myFactory,Q)
        #print(action)
        actionList.append(action)
        oldState = myFactory.getState()
        newState, reward, doneFlag = myFactory.step(action)
        episodeReward += reward
        updatedAction = help.bestAction(myFactory,Q)

        #Update Q values:
        Q[oldState,action] = Q[oldState,action] + ALPHA*(reward + GAMMA*Q[newState,updatedAction])
    if i%100 == 0:
        print("Episode ", i)
        print("Epsilon:", EPSILON)
        print("Reward: ", episodeReward)
        if episodeReward>-30:
            print("Actions: ", actionList)
        EPSILON = EPSILON*0.95
    totalReward[i] = episodeReward

plt.plot(totalReward)
plt.show()