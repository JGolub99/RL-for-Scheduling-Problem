import helpfunctions as help
import random
import numpy as np
import copy

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

    def __init__(self,load,maximum,position,name="Agent"):
        self.load = load
        self.maximum = maximum
        self.position = position
        self.name = name
    
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
        self.stateSpace = [] # Does not include terminal states
        self.stateSpacePlus = [] # Does include terminal 

        self.initialAgent = agent
        self.initialStations = listOfStations

        self.deposit = deposit
        self.obstacles = listOfObstacles

        self.reset()
        self.createStateSpace(agent)
        self.possibleActions = ["Up", "Right", "Down", "Left", "Load", "Unload"]
    

    def addAgent(self):
        positionX, positionY = self.agent.position
        self.grid[positionX][positionY] = 1
    
    def removeAgent(self):
        positionX, positionY = self.agent.position
        self.grid[positionX][positionY] = 0

    def addDeposit(self):
        positionX, positionY = self.deposit
        self.grid[positionX][positionY] = 2
    
    def addObstacles(self):
        for obstacle in self.obstacles:
            positionX, positionY = obstacle
            self.grid[positionX][positionY] = 3

    def addStations(self):
        index = 4
        for station in self.stations:
            station.id = index
            positionX, positionY = station.position
            self.grid[positionX][positionY] = index
            index += 1

    def createStateSpace(self, agent):

        #Lets first create a list of variations for the stations loads:
        stationMaxLoads = [station.maximum for station in self.stations]
        stationLoadVariations = help.generate_variations(stationMaxLoads)

        rowIndex = 0
        for row in self.grid:
            columnIndex = 0
            for column in row:
                if column == 0 or column == 1:
                    for posAgentLoad in range(agent.maximum + 1):
                        for option in stationLoadVariations:
                            state = tuple([(rowIndex,columnIndex),posAgentLoad] + option)
                            self.stateSpacePlus.append(state)
                            if not(not any(option) and posAgentLoad == 0):
                                self.stateSpace.append(state)

                columnIndex+=1
            rowIndex+=1
        
    def isTerminalState(self,state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getState(self):
        stationLoads = [stationLoad.load for stationLoad in self.stations]
        return tuple([self.agent.position, self.agent.load] + stationLoads)

    def setState(self,state):
        # This function needs to update the grid and the state
        self.removeAgent()
        self.agent.position = state[0]
        self.agent.load = state[1]
        i = 0
        for station in self.stations:
            station.load = state[i+2]
            i+=1
        
        #Update the grid:
        self.addAgent()

    def randomAction(self):
        numberActions = len(self.possibleActions)
        randomInteger = random.randint(0,numberActions)
        return self.possibleActions[randomInteger-1]

    def action(self,action):
        if action == "Up":
            return self.move(0)
        elif action == "Right":
            return self.move(1)
        elif action == "Down":
            return self.move(2)
        elif action == "Left":
            return self.move(3)
        elif action == "Load":
            return self.load()
        elif action == "Unload":
            return self.unload()

    def move(self,key):
        currentX, currentY = self.agent.position
        if key == 0:
            newX = currentX - 1
            return tuple([(newX,currentY),self.agent.load] + [stationLoad.load for stationLoad in self.stations])

        elif key == 1:
            newY = currentY + 1
            return tuple([(currentX,newY),self.agent.load] + [stationLoad.load for stationLoad in self.stations])
        
        elif key == 2:
            newX = currentX + 1
            return tuple([(newX,currentY),self.agent.load] + [stationLoad.load for stationLoad in self.stations])
        
        elif key == 3:
            newY = currentY - 1
            return tuple([(currentX,newY),self.agent.load] + [stationLoad.load for stationLoad in self.stations])            

    def load(self):
        currentAgentPosition = self.agent.position
        currentAgentLoad = self.agent.load
        stationPositions = [station.position for station in self.stations]
        currentStationLoads = [station.load for station in self.stations]

        # Now figure out which station is the agent next to
        i = 0
        for possibleStationPosition in stationPositions:
            if help.manhattenDistance(currentAgentPosition, possibleStationPosition) == 1:
                newAgentLoad = currentAgentLoad + 1
                newStationLoad = currentStationLoads[i] - 1
                newStationLoads = currentStationLoads[:i] + [newStationLoad] + currentStationLoads[i+1:]
                return tuple([currentAgentPosition, newAgentLoad] + newStationLoads)
            i+=1
        
        return tuple([currentAgentPosition, currentAgentLoad] + currentStationLoads)
                

    def unload(self):
        currentAgentPosition = self.agent.position
        currentAgentLoad = self.agent.load
        stationPositions = [station.position for station in self.stations]
        currentStationLoads = [station.load for station in self.stations]

        if help.manhattenDistance(currentAgentPosition,self.deposit) == 1:
            return tuple([currentAgentPosition, currentAgentLoad-1] + currentStationLoads)

        # Now figure out which station is the agent next to
        i = 0
        for possibleStationPosition in stationPositions:
            if help.manhattenDistance(currentAgentPosition, possibleStationPosition) == 1:
                newAgentLoad = currentAgentLoad - 1
                newStationLoad = currentStationLoads[i] + 1
                newStationLoads = currentStationLoads[:i] + [newStationLoad] + currentStationLoads[i+1:]
                return tuple([currentAgentPosition, newAgentLoad] + newStationLoads)
            i+=1
        
        return tuple([currentAgentPosition, currentAgentLoad] + currentStationLoads)        


    def illegalState(self,newState):

        if newState not in self.stateSpacePlus:
            return True
        else:
            return False

    def step(self, action):
        trialState = self.action(action)
        if not self.illegalState(trialState):
            self.setState(trialState)
        newState = self.getState()

        if self.isTerminalState(newState):
            reward = 0
        else:
            reward = -1

        return newState, reward, self.isTerminalState(newState) 
    
    def reset(self):
        # In this function we need to rebuld the grid and initial state:
        self.agent = copy.deepcopy(self.initialAgent)
        self.stations = copy.deepcopy(self.initialStations)
        self.grid = np.zeros((self.width,self.height))
        self.addAgent()
        self.addDeposit()
        self.addObstacles()
        self.addStations()