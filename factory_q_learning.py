'''
This script will simulate a single agent that collects and deposits items from a 
static environment factory.
We shall allow the agent to hold a maximum number of items

We will represent the world as a discrete numpy grid where:
    - 0 represents floorspace that the agent can move to
    - 1 represents the agent
    - 2 represents the deposit
    - 3 represents an impassable obstacle
    - higher numbers represent stations, whose objects will be accessed via a dictionary

An object position on the factory floor is represented by the tuple (x,y).    

'''

import numpy as np

class Station:

    def __init__(self, load, maximum, position, mounted=False, id=None):
        self.load = load
        self.maximum = maximum
        self.mounted = mounted
        self.position = position
        self.id = id
    
    def getLoad(self):
        return self.load
    
    def getMaximum(self):
        return self.maximum
    
    def addLoad(self):
        self.load += 1

    def takeLoad(self):
        self.load -= 1
    

class Agent:

    def __init__(self,load,maximum,position):
        self.load = load
        self.maximum = maximum
        self.position = position
    
    def getLoad(self):
        return self.load
    
    def getMaximum(self):
        return self.maximum
    
    def addLoad(self):
        self.load += 1

    def takeLoad(self):
        self.load -= 1


class Factory:

    def __init__(self, height, width, listOfObstacles, listOfStations, agent, deposit):
        self.height = height
        self.width = width
        self.grid = np.zeros((width,height))
        self.addAgent(agent)
        self.addDeposit(deposit)
        self.addObstacles(listOfObstacles)
        self.addStations(listOfStations)
        print(self.grid)
    

    def addAgent(self,agent):
        positionX, positionY = agent.position
        self.grid[positionX][positionY] = 1

    def addDeposit(self,deposit):
        positionX, positionY = deposit
        self.grid[positionX][positionY] = 2
    
    def addObstacles(self,obstacles):
        for obstacle in obstacles:
            positionX, positionY = obstacle
            self.grid[positionX][positionY] = 3

    def addStations(self,stations):
        index = 4
        for station in stations:
            positionX, positionY = station.position
            self.grid[positionX][positionY] = index
            index += 1


myObstacles = [(2,1)]
station1 = Station(2,5,(0,1))
station2 = Station(1,3,(0,2))
myStations = [station1,station2]
myAgent = Agent(0,2,(3,0))
deposit = (3,2)
myFactory = Factory(3,4,myObstacles,myStations,myAgent,deposit)
