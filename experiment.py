from BoostrappedDQNClasses import Agent as BootAgent
from RainbowDQNClasses import Agent as RainbowAgent
from tests import performTestOne, performTestTwo, performTestThree
import Rlclasses2 as RL

import movingObstacleSchedule as mo
import helpfunctions as help

from matplotlib import pyplot as plt
import random
import sys

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

probability1 = [1.0,0.0,0.0]
probability2 = [0.8,0.0,0.2]
probability3 = [0.6,0.0,0.4]

probabilities = [probability1,probability2,probability3]

schedule2 = mo.ObstacleSchedule([{1:(2,3),2:"Left",3:"Left",4:"Up",5:"Remove"},
                                 {1:(4,2),5:"Left",6:"Remove"}])
schedule1 = mo.ObstacleSchedule([{1:(2,3),2:"Left",3:"Left",4:"Up",5:"Remove"}])

schedule3 = mo.ObstacleSchedule([{1:(2,3), 2:"Random", 3:"Random", 4:"Random", 5:"Random",
                                6:"Random", 7:"Random", 8:"Random", 9:"Random", 10:"Remove"}])
schedules = [schedule1,schedule2,schedule3]

'''
End of test environments
'''

actions = myFactory.possibleActions
numActions = len(actions)

GAMMA = 0.95
ALPHA = 0.003
EPSILON = 1.0
EPS_REDUCTION_RATE = 1000   #1000
BATCHSIZE = 30

MAX_MEMORY_SIZE = 5000

INITIAL_BETA = 0.99
RANDOM_RESET = True

NUM_HEADS = 10
BERNOULLI_PROB = 0.25

TEST_THRESH = 10000

alg_type = input("Give agent learning type (Rainbow or Bootstrapped): ")

if alg_type == "Rainbow":
    myAgent = RainbowAgent(GAMMA,EPSILON,ALPHA,5,6,BATCHSIZE,numActions,INITIAL_BETA,MAX_MEMORY_SIZE,eps_end=0.2,reduce_eps=EPS_REDUCTION_RATE,annealBias=False)
elif alg_type == "Bootstrapped":
    myAgent = BootAgent(GAMMA,EPSILON,ALPHA,5,6,BATCHSIZE,numActions,INITIAL_BETA,NUM_HEADS,BERNOULLI_PROB,MAX_MEMORY_SIZE,eps_end=0.2,reduce_eps=EPS_REDUCTION_RATE,annealBias=False)
else:
    sys.exit("Invalid learning agent")

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
testScores2_1 = []
testScores2_2 = []
testScores2_3 = []
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

    k = random.randint(0,NUM_HEADS-1)

    while not done:
        state = myFactory.getState()
        #print(state)
        action = myAgent.choose_action(state,k)
        actionString = myFactory.possibleActions[action]
        nextState, reward, done, _ = myFactory.step(actionString)
        score += reward
        myAgent.store_transition(state,action,reward,nextState,done)
        myAgent.learn()
        #if score % 5000 == 0:
            #print(score)
    print("Episode {}: ".format(episodeNumber),score)

    scores.append(score)

    if (episodeNumber%500) == 0:
        temp_list = []
        temp_list2_1 = []
        temp_list2_2 = []
        temp_list2_3 = []
        temp_list3 = []
        for _ in range(5):
            finalScore = performTestOne(myAgent,factories,optimals,TEST_THRESH)
            temp_list.append(finalScore)
            finalScore2 = performTestTwo(myAgent,myFactory,probabilities,1,TEST_THRESH)
            temp_list2_1.append(finalScore2)
            finalScore2 = performTestTwo(myAgent,myFactory,probabilities,3,TEST_THRESH)
            temp_list2_2.append(finalScore2)
            finalScore2 = performTestTwo(myAgent,myFactory,probabilities,5,TEST_THRESH)
            temp_list2_3.append(finalScore2)
            finalScore3 = performTestThree(myAgent,myFactory,schedules,TEST_THRESH)
            temp_list3.append(finalScore3)
        averagedFinalScores = help.elementwise_average(temp_list)
        averagedFinalScores2_1 = help.elementwise_average(temp_list2_1)
        averagedFinalScores2_2 = help.elementwise_average(temp_list2_2)
        averagedFinalScores2_3 = help.elementwise_average(temp_list2_3)
        averagedFinalScores3 = help.elementwise_average(temp_list3)
        testScores.append(averagedFinalScores)
        testScores2_1.append(averagedFinalScores2_1)
        testScores2_2.append(averagedFinalScores2_2)
        testScores2_3.append(averagedFinalScores2_3)

        testScores3.append(averagedFinalScores3)
        print(averagedFinalScores)
        print(averagedFinalScores2_1)
        print(averagedFinalScores2_2)
        print(averagedFinalScores2_3)
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
plt.ylabel("Empirical regret")
plt.title("Empirical regret from initial states throughout the agent trianing process")
plt.legend()
plt.show()

performances2_1 = help.transpose(testScores2_1)
performances2_2 = help.transpose(testScores2_2)
performances2_3 = help.transpose(testScores2_3)
plt.figure()
for index,performance in enumerate(performances2_1):
    plt.subplot(1, 3, 1)
    plt.plot(performance,label="Testcase {}".format(index))
    plt.ylabel("Deviation from optimal policy return")
    plt.title("High frequency disturbance")

for index,performance in enumerate(performances2_2):
    plt.subplot(1, 3, 2)
    plt.plot(performance,label="Testcase {}".format(index))
    plt.xlabel("Training episodes / 500")
    plt.title("Medium frequency disturbance")

for index,performance in enumerate(performances2_3):
    plt.subplot(1, 3, 3)
    plt.plot(performance,label="Testcase {}".format(index))
    plt.title("Low frequency disturbance")



plt.suptitle("Sensitivity to varying levels of disturbance")
plt.legend()
plt.show()

performances3 = help.transpose(testScores3)
plt.figure()
for index,performance in enumerate(performances3):
    plt.plot(performance,label="Testcase {}".format(index))


plt.xlabel("Training episodes / 500")
plt.ylabel("Number of crashes with moving obstacles")
plt.title("Obstacle avoidance performance")
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