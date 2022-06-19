import gym
import numpy as np

class Reinforcement_learning:
    def __init__(self):
        # Load the environment
        self.env = gym.make("Acrobot-v1")

        # Get observation and action space from environment
        self.input_neurons = self.env.observation_space.shape[0]
        self.output_neurons = self.env.action_space.n

    def play_episode_CMA(self, agent):
        current_observation = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # Run until task is completed or when 500 steps have been made
        while not done and steps < 500:
            # Get values for actions
            actions = agent.predict(current_observation)
            # Get best action
            action_index = np.argmax(actions)
            # Apply the action and get observations
            new_observation, reward, done, _ = self.env.step(action_index)
            # Update total reward
            total_reward = total_reward + reward

            # Update state
            current_observation = new_observation
            steps += 1
            
        return total_reward

    def play_episode_DQN(self, agent):
        current_observation = self.env.reset()
        total_reward = 0
        done = False
        steps = 0

        # Run until task is completed or when 500 steps have been made
        while not done and steps < 500:
            # Get an action
            action = agent.compute_action(current_observation)
            # Apply the action and get observations
            new_observation, reward, done, _ = self.env.step(action)
            # Add observations to the agent's memory
            agent.add_to_memory(current_observation, action, reward, new_observation, done)
            # Update total reward
            total_reward = total_reward + reward
            # Train the agent
            agent.train()

            # Update state
            current_observation = new_observation
            steps += 1

        return total_reward

    def fitness(self, agent):
        # Play an episode
        reward = self.play_episode_CMA(agent)

        return reward