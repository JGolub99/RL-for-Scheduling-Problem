import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import Rlclasses as RL
import numpy as np
import helpfunctions as help

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 8  # Nothing special with 8, feel free to change
        hidden_space2 = 16  # Nothing special with 16, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims),
            nn.Softmax()
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_probs = self.policy_mean_net(shared_features)
        #action_stddevs = torch.log(
        #    1 + torch.exp(self.policy_stddev_net(shared_features))
        #)

        return action_probs
    

def sample_action(environment,network) -> float:
    """Returns an action, conditioned on the policy and observation.

    Args:
        state: Observation from the environment

    Returns:
        action: Action to be performed
    """
    state = environment.getState()
    state = torch.tensor(np.array(help.flatten_tuple(state)))
    action_means = network(state)

    # create a normal distribution from the predicted
    #   mean and standard deviation and sample an action
    distrib = Categorical(action_means)
    action = distrib.sample()
    prob = distrib.log_prob(action)

    action = action.numpy()

    #self.probs.append(prob)

    return action, prob

def update(network, learning_rate, gamma, rewards, probs):
    """Updates the policy network's weights."""
    optimizer = torch.optim.AdamW(network.parameters(), learning_rate)
    running_g = 0
    gs = []

    # Discounted return (backwards) - [::-1] will return an array in reverse
    for R in rewards[::-1]:
        running_g = R + gamma * running_g
        gs.insert(0, running_g)

    deltas = torch.tensor(gs)

    loss = 0
    # minimize -1 * prob * reward obtained
    for log_prob, delta in zip(probs, deltas):
        loss += log_prob.mean() * delta * (-1)

    # Update the policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Instantiate the environment:
myObstacles = [(2,1)]
station1 = RL.Station(2,3,(0,1))
station2 = RL.Station(1,1,(0,2))
myStations = [station1,station2]
myAgent = RL.Agent(1,2,(3,0))
deposit = (3,2)
myFactory = RL.Factory(3,4,myObstacles,myStations,myAgent,deposit)
print(myFactory.grid)

# Set the hyperparameters:
ALPHA = 0.1
GAMMA = 1

network = Policy_Network(len(myFactory.getState())+1,len(myFactory.possibleActions))
numberEpisodes = 100
for i in range(numberEpisodes):
    episodeReward = 0
    doneFlag = False
    myFactory.reset()
    probabilities = []
    rewards = []

    while not doneFlag:
        action, prob = sample_action(myFactory,network)
        probabilities.append(prob)
        newState, reward, doneFlag = myFactory.step(myFactory.possibleActions[action.item()])
        rewards.append(reward)
        episodeReward+=reward
    print(episodeReward)
    update(network,ALPHA,GAMMA,rewards,probabilities)

#print(sample_action(myFactory,network))