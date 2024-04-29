class ObstacleSchedule():

    def __init__(self,list_of_schedules):    # [ {1:(1,1), 2:"Right", 3: "Down"} ]
        self.schedules = list_of_schedules
        self.counter = 0

    def clock(self):
        self.counter+=1
        output = []
        for i, schedule in enumerate(self.schedules):
            if self.counter in schedule:
                output.append((i,schedule[self.counter]))
        
        return output

    def reset(self):
        self.counter = 0

'''
mySchedule = ObstacleSchedule([{1:(1,1), 2:"Up", 3:"Down"}, {1:(4,5), 3:"Left", 4:"Load"}])
for _ in range(10):
    print(mySchedule.clock())
'''
