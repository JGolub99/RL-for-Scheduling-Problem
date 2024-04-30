from BoostrappedDQNClasses import Agent as BootAgent
from RainbowDQNClasses import Agent as RainbowAgent
from tests import performTestOne, performTestTwo, performTestThree
import Rlclasses2 as RL

import movingObstacleSchedule as mo
import helpfunctions as help

from matplotlib import pyplot as plt

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
agent = RL.Agent(1,2,(4,1))
deposit = (4,3)
myFactory = RL.Factory(5,6,myObstacles,myStations,agent,deposit)

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
agent = RL.Agent(1,2,(4,1))
deposit = (4,3)
myFactory = RL.Factory(5,6,myObstacles,myStations,agent,deposit)

'''
In this section of code we will instatiate the test environments:
'''

myFactory = RL.Factory(5,6,myObstacles,myStations,agent,deposit)
myFactory2 = RL.Factory(5,6,myObstacles,myStations,RL.Agent(1,2,(1,1)),deposit)
myFactory3 = RL.Factory(5,6,myObstacles,[RL.Station(3,3,(1,2)),RL.Station(1,1,(1,3))],RL.Agent(2,2,(4,1)),deposit)

myObstacles2 = [(3,1),
               (0,0),(0,1),(0,2),(0,3),(0,4),
               (1,0),(1,4),
               (2,0),(2,4),
               (3,0),(3,4),
               (4,0),(4,4),
               (5,0),(5,1),(5,2),(5,3),(5,4)]

myFactory4 = RL.Factory(5,6,myObstacles2,myStations,agent,deposit)
myFactory5 = RL.Factory(5,6,myObstacles,[RL.Station(1,1,(1,1)),RL.Station(2,3,(1,2)),RL.Station(1,1,(1,3))],agent,deposit)

factories = [myFactory,myFactory2,myFactory3,myFactory4,myFactory5]
optimals = [-15,-14,-18,-13,-19]

schedule1 = mo.ObstacleSchedule([{1:(2,3),2:"Left",3:"Left",4:"Up",5:"Remove"},
                                 {1:(4,2),5:"Left",6:"Remove"}])
schedule2 = mo.ObstacleSchedule([{1:(2,3),2:"Left",3:"Left",4:"Up",5:"Remove"}])
schedules = [schedule1,schedule2]

'''
End of test environments
'''

actions = myFactory.possibleActions
numActions = len(actions)

GAMMA = 0.95
ALPHA = 0.003
EPSILON = 1.0
BATCHSIZE = 30

INITIAL_BETA = 0.99
RANDOM_RESET = True

myAgent = RainbowAgent(GAMMA,EPSILON,ALPHA,5,6,BATCHSIZE,numActions,INITIAL_BETA,5000,reduce_eps=1000,annealBias=False)

while myAgent.mem_cntr < myAgent.mem_size:
    done = False
    while not done:
        action = myFactory.randomAction()
        currentState = myFactory.getState()
        nextState, reward, done, _ = myFactory.step(action)
        if done:
            print("Terminated")
        myAgent.store_transition(currentState,help.get_index(action,myFactory.possibleActions),reward,nextState,done)
    if RANDOM_RESET:
        myFactory.reset2()
    else:
        myFactory.reset()
print("Memory initilised")

scores = []
testScores = []
testScores2 = []
testScores3 = []
epsHistory = []
episodeNumber = 0

while myAgent.epsilon > myAgent.eps_min :
    episodeNumber += 1
    epsHistory.append(myAgent.epsilon)
    done = False
    if RANDOM_RESET:
        myFactory.reset2()
    else:
        myFactory.reset()

    score = 0

    while not done:
        state = myFactory.getState()
        #print(state)
        action = myAgent.choose_action(state)
        actionString = myFactory.possibleActions[action]
        nextState, reward, done, _ = myFactory.step(actionString)
        score += reward
        myAgent.store_transition(state,action,reward,nextState,done)
        myAgent.learn()
        #if score % 5000 == 0:
            #print(score)
    print("Episode {}: ".format(episodeNumber),score)

    scores.append(score)

    if (episodeNumber%300) == 0:
        temp_list = []
        temp_list2 = []
        temp_list3 = []
        for _ in range(5):
            finalScore = performTestOne(myAgent,factories,optimals,50000)
            temp_list.append(finalScore)
            finalScore2 = performTestTwo(myAgent,myFactory,[[1.0,0.0,0.0],[0.6,0.4,0.0],[0.6,0.3,0.1]],5,50000)
            temp_list2.append(finalScore2)
            finalScore3 = performTestThree(myAgent,myFactory,schedules,50000)
            temp_list3.append(finalScore3)
        averagedFinalScores = help.elementwise_average(temp_list)
        averagedFinalScores2 = help.elementwise_average(temp_list2)
        averagedFinalScores3 = help.elementwise_average(temp_list3)
        testScores.append(averagedFinalScores)
        testScores2.append(averagedFinalScores2)
        testScores3.append(averagedFinalScores3)
        print(averagedFinalScores)
        print(averagedFinalScores2)
        print(averagedFinalScores3)


'''
done = False
if RANDOM_RESET:
    myFactory.reset2()
else:
    myFactory.reset()
print(myFactory.getState())
'''
for x in testScores:
    print(x)

performances = help.transpose(testScores)
plt.figure()
for index,performance in enumerate(performances):
    plt.plot(performance,label="Testcase {}".format(index))


plt.xlabel("Training episodes / 500")
plt.ylabel("Reward")
plt.title("Rewards for different initial states throughout the agen trianing process")
plt.legend()
plt.show()

performances2 = help.transpose(testScores2)
plt.figure()
for index,performance in enumerate(performances2):
    plt.plot(performance,label="Testcase {}".format(index))


plt.xlabel("Training episodes / 500")
plt.ylabel("Deviation from optimal policy return")
plt.title("Sensitivity to varying levels of disturbance")
plt.legend()
plt.show()

def averageFilter(myList,windowSize):
    listLength = len(myList)
    averagedList = []
    j=0
    for i in range(windowSize,listLength):
        partialSum = sum(myList[j:i])
        partialAverage = partialSum/windowSize
        averagedList.append(partialAverage)
        j+=1
    return averagedList

averaged_scores = averageFilter(scores,30)

plt.figure()
plt.plot(averaged_scores)
plt.title("Training performance")
plt.xlabel("Training episodes")
plt.ylabel("Reward")
plt.show()