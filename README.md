# Manual

## CMA
This file implements the Covariance Matrix Adaption Evolutionary Strategies algortihm. It uses a separate Args classes for the parameters of the weights and biasses. The file contains a function that runs one generation of CMA-ES: it samples new weights and biasses, sets these weights and biasses in a network and runs one episode with that agent. After all agents have run one episode, the fitnesses are sorted and the parameters of the weights and biasses are updated.

## DQN
This file implements the Deep Q-Network. It is epsilon-greedy and has a separate class that implements the replay memory that is used during training. 

## main
In the main, both agents are run for a fixed amount of time. The total rewards are stored. This file should be run to compare both algorithms. It contains a list with the network sizes we tested and one of these should be passed to the function that compares the algorithm.

## network
This file contains the neural networks that both agents. The network receives a list containing the sizes for each hidden layer and can build a neural network that matches these numbers. The file also contains a setter and a getter for the weights.

## plot
This file is used for plotting the moving average of the rewards of both algorithms.

## reinforcementLearning
This file contains the environment and also contains the code for running the agents on the reinforcement learning task. In this class, the total rewards are also computed, and returned as fitnesses for the CMA algorithm.

The running_the_main.png shows the output when running the main. It prints the reward every 100 epochs.
