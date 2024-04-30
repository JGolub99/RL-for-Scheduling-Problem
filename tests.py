def performTestOne(myAgent,factories,optimals,thresh):
    final_scores = []
    old_epsilon = myAgent.epsilon
    myAgent.epsilon = 0.02
    for factory,optimal_score in zip(factories,optimals):
        score = 0
        done = False
        factory.reset()
        while (not done) and (abs(score)<thresh):
            state = factory.getState()
            action = myAgent.choose_action(help.flatten_tuple(state))
            actionString = factory.possibleActions[action]
            nextState, reward, done, _ = factory.step(actionString)
            score += reward
            #myAgent.store_transition(help.flatten_tuple(state),action,reward,help.flatten_tuple(nextState),done)    
            #print(actionString)
        print("Empirical Regret: ",abs(score-optimal_score))
        final_scores.append(abs(score-optimal_score))
    myAgent.epsilon = old_epsilon
    return final_scores

def performTestTwo(myAgent,factory,probabilities,freq,thresh):

    old_epsilon = myAgent.epsilon
    myAgent.epsilon = 0.02
    score = 0
    done = False
    factory.reset()
    while (not done) and (abs(score)<thresh):
        state = factory.getState()
        action = myAgent.choose_action(help.flatten_tuple(state))
        actionString = factory.possibleActions[action]
        nextState, reward, done, _ = factory.step(actionString)
        score += reward
    optimalScore = score

    scores = []

    for probability in probabilities:
        score = 0
        done = False
        factory.reset()
        step = 0
        while (not done) and (abs(score)<thresh):
            step +=1
            state = factory.getState()
            action = myAgent.choose_action(help.flatten_tuple(state))
            actionString = factory.possibleActions[action]
            nextState, reward, done, _ = factory.step(actionString)
            score += reward            
            if step % freq == 0:
                factory.perturb(probability)
        scores.append(abs(score-optimalScore))
    
    myAgent.epsilon = old_epsilon
    return scores

def performTestThree(myAgent,factory,schedules,thresh):

    #schedules is a list of MovingObstacle schedules
    final_scores = []
    old_epsilon = myAgent.epsilon
    myAgent.epsilon = 0.02
    for schedule in schedules:
        schedule.reset()
        crashes = 0
        score = 0
        done = False
        factory.reset()
        while (not done) and (abs(score)<thresh):
            state = factory.getState()
            action = myAgent.choose_action(help.flatten_tuple(state))
            actionString = factory.possibleActions[action]
            nextState, reward, done, crashed = factory.step(actionString)
            score += reward
            if crashed:
                crashes +=1
            moves = schedule.clock()
            if len(moves)!=0:
                for obstacle,move in moves:
                    if type(move) == tuple:
                        factory.addMovingObstacle(move)
                    elif move == "Remove":
                        factory.removeObstacle(obstacle)
                    else:
                        factory.moveObstacle(obstacle,move)
        print("Number of crashes: ", crashes)
        final_scores.append(crashes)

    myAgent.epsilon = old_epsilon
    return final_scores