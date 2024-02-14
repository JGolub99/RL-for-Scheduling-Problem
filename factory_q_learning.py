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

We also need to define our state space for our Q values. Note that the bigger our environment and greater load
that our agent(s) and stations can take will massively increase the size of the state-space.
We will pursue with this for now but it suggests that a more sophisticated method such as DQN might be necessary.
The state will be a list: [agentPosition, agentLoad, stationLoads]
Consequently, the state-space will be a list of these dictionaries.
How many possible states? : no.free_spaces*(agent_max+1)*Product(station_max+1)

For the load and unload actions, I will enforce that when the agent is at a station, it is not adjacent to any
other station. This means that when we perform this action, it only has one station to consider.
This assumption can be enforced by further discretising the factory floor if needed.

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
        self.grid = np.zeros((width,height))
        self.stateSpace = [] # Does not include terminal states
        self.stateSpacePlus = [] # Does include terminal 

        # These are state variables that need to be updates with each step
        self.agent = agent
        self.stations = listOfStations
        self.noStations = len(listOfStations)

        self.deposit = deposit
        self.obstacles = listOfObstacles

        # These members build the grid
        self.addAgent()
        self.addDeposit()
        self.addObstacles()
        self.addStations()

        self.createStateSpace(agent)
        print(self.grid)
        print("Initial state: ", self.getState())
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
        stationLoadVariations = generate_variations(stationMaxLoads)

        rowIndex = 0
        for row in self.grid:
            columnIndex = 0
            for column in row:
                if column == 0 or column == 1:
                    for posAgentLoad in range(agent.maximum + 1):
                        for option in stationLoadVariations:
                            state = [(rowIndex,columnIndex),posAgentLoad] + option
                            self.stateSpacePlus.append(state)
                            if not(not any(option) and posAgentLoad == 0):
                                self.stateSpace.append(state)

                columnIndex+=1
            rowIndex+=1
        
    def isTerminalState(self,state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getState(self):
        stationLoads = [stationLoad.load for stationLoad in self.stations]
        return [self.agent.position, self.agent.load] + stationLoads

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
            return [(newX,currentY),self.agent.load] + [stationLoad.load for stationLoad in self.stations]

        elif key == 1:
            newY = currentY + 1
            return [(currentX,newY),self.agent.load] + [stationLoad.load for stationLoad in self.stations]
        
        elif key == 2:
            newX = currentX + 1
            return [(newX,currentY),self.agent.load] + [stationLoad.load for stationLoad in self.stations]
        
        elif key == 3:
            newY = currentY - 1
            return [(currentX,newY),self.agent.load] + [stationLoad.load for stationLoad in self.stations]            

    def load(self):
        currentAgentPosition = self.agent.position
        currentAgentLoad = self.agent.load
        stationPositions = [station.position for station in self.stations]
        currentStationLoads = [station.load for station in self.stations]

        # Now figure out which station is the agent next to
        i = 0
        for possibleStationPosition in stationPositions:
            if manhattenDistance(currentAgentPosition, possibleStationPosition) == 1:
                newAgentLoad = currentAgentLoad + 1
                newStationLoad = currentStationLoads[i] - 1
                newStationLoads = currentStationLoads[:i] + [newStationLoad] + currentStationLoads[i+1:]
                return [currentAgentPosition, newAgentLoad] + newStationLoads
            i+=1
        
        return [currentAgentPosition, currentAgentLoad] + currentStationLoads
                

    def unload(self):
        currentAgentPosition = self.agent.position
        currentAgentLoad = self.agent.load
        stationPositions = [station.position for station in self.stations]
        currentStationLoads = [station.load for station in self.stations]

        if manhattenDistance(currentAgentPosition,self.deposit) == 1:
            return [currentAgentPosition, currentAgentLoad-1] + currentStationLoads

        # Now figure out which station is the agent next to
        i = 0
        for possibleStationPosition in stationPositions:
            if manhattenDistance(currentAgentPosition, possibleStationPosition) == 1:
                newAgentLoad = currentAgentLoad - 1
                newStationLoad = currentStationLoads[i] + 1
                newStationLoads = currentStationLoads[:i] + [newStationLoad] + currentStationLoads[i+1:]
                return [currentAgentPosition, newAgentLoad] + newStationLoads
            i+=1
        
        return [currentAgentPosition, currentAgentLoad] + currentStationLoads        


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

# This function helps with generating the variations of station loads (made by ChatGPT)
def generate_variations(lst):
    variations = set()

    # Helper function to recursively generate variations
    def generate_helper(current_lst):
        # Add the current list to variations
        variations.add(tuple(current_lst))

        # Iterate through the list elements
        for i in range(len(current_lst)):
            # If the element is greater than 0, decrement it and generate variations
            if current_lst[i] > 0:
                current_lst[i] -= 1
                generate_helper(current_lst)
                current_lst[i] += 1  # backtrack

    generate_helper(lst)
    return [list(variation) for variation in variations]

def manhattenDistance(pos1,pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2-x1) + abs(y2-y1)

myObstacles = [(2,1)]
station1 = Station(2,3,(0,1))
station2 = Station(1,1,(0,2))
myStations = [station1,station2]
myAgent = Agent(0,2,(3,0))
deposit = (3,2)
myFactory = Factory(3,4,myObstacles,myStations,myAgent,deposit)
myFactory.step("Up")
myFactory.step("Up")
myFactory.step("Right")
myFactory.step("Load")
print(myFactory.getState())
print(myFactory.grid)
