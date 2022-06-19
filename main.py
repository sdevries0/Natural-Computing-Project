from time import time
import DQN
import CMA
import reinforcementLearning
import numpy as np
import time

path = 'results2/'

def compare_DQN_CMA(hidden_size):
    # Create the environment
    rl_env = reinforcementLearning.Reinforcement_learning()

    # Initialize the two agents
    DQN_agent = DQN.Agent(rl_env, hidden_size)
    CMA_agent = CMA.Agent(rl_env, hidden_size)

    DQN_rewards = []
    CMA_rewards = []

    # To keep track of run time
    start = time.time()
    end = time.time()
    dif = start-end
    counter = 0

    # Run for three hours
    while dif < 10800:
        # Play an episode
        reward = rl_env.play_episode_DQN(DQN_agent)
        DQN_rewards.append(reward)

        # Update time
        end = time.time()
        dif = end-start
        counter += 1

        # Print to keep track of progress
        if counter % 100 == 0:
            print(counter," reward is ", reward)

    np.savez(path + 'DQN_' + '_'.join(map(str,hidden_size)) + '.npz', np.array(DQN_rewards))

    # To keep track of run time
    start = time.time()
    end = time.time()
    dif = start-end
    counter = 0

    # Run for three hours
    while dif < 10800:
        try:
            # Run one generation of CMA algorithm
            reward = CMA_agent.run_generation()
            CMA_rewards.append(reward)

            # Update time
            end = time.time()
            dif = end-start
            counter += 1

            # Print to keep track of progress
            if counter % 100 == 0:
                print(counter," reward is ", reward)
        
        # Exception for exploding weights
        except Exception as e:
            print(e)
            np.savez(path + 'CMA_rewards_' + '_'.join(map(str,hidden_size)) + '.npz', np.array(CMA_rewards))
            break
            
    np.savez(path + 'CMA_rewards_' + '_'.join(map(str,hidden_size)) + '.npz', np.array(CMA_rewards))

# Different network sizes we tested
network_hiddens = [[4,3],[8,4],[16,8],[16,12,8],[32,16],[32,24,12],[48,24,12]]
compare_DQN_CMA(network_hiddens[0])
