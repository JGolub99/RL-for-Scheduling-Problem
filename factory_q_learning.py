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
        #self.actionSpace = {"Up":self.move(0), }
        self.possibleActions = ["Up", "Down", "Left", "Right", "Load", "Unload"]
    

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
            station.load = state[i+1]
            i+=1
        
        #Update the grid:
        self.addAgent()

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



myObstacles = [(2,1)]
station1 = Station(2,3,(0,1))
station2 = Station(1,1,(0,2))
myStations = [station1,station2]
myAgent = Agent(0,2,(3,0))
deposit = (3,2)
myFactory = Factory(3,4,myObstacles,myStations,myAgent,deposit)
