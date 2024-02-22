import numpy as np

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

# This one takes in a tuple and returns a flattened list (ChatGPT)
def flatten_tuple(input_tuple):
    result = []

    for item in input_tuple:
        if isinstance(item, tuple):
            result.extend(flatten_tuple(item))
        else:
            result.append(item)

    return result

def manhattenDistance(pos1,pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2-x1) + abs(y2-y1)

def bestAction(factory,q_values):
    state = factory.getState()
    values = np.array([q_values[state,action] for action in factory.possibleActions])
    best = np.argmax(values)
    return factory.possibleActions[best]

def bestActionDouble(factory,q_values1,q_values2):
    state = factory.getState()
    values1 = np.array([q_values1[state,action] for action in factory.possibleActions])
    values2 = np.array([q_values2[state,action] for action in factory.possibleActions])
    values = np.add(values1,values2)
    best = np.argmax(values)
    return factory.possibleActions[best]

def get_index(element,myList):
   
    index = 0
    for item in myList:
        if item == element:
            return index
        index+=1