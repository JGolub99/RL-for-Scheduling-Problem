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

def manhattenDistance(pos1,pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2-x1) + abs(y2-y1)

def bestAction(factory,q_values):
    state = factory.getState()
    values = np.array([q_values[state,action] for action in factory.possibleActions])
    best = np.argmax(values)
    return factory.possibleActions[best]