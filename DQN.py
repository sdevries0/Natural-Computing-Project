import numpy as np
import warnings
import tensorflow as tf
from tensorflow import keras
import network
warnings.filterwarnings("ignore")

class Agent:
    def __init__(self, env, network_hiddens, gamma = 0.99, epsilon = 1, decay = 0.9999, min_epsilon = 0.01):
        self.environment = env
        # Set network parameters
        self.network_hiddens = network_hiddens
        self.network_input = self.environment.input_neurons
        self.network_out = self.environment.output_neurons

        # Initialize replay memory
        self.replay_memory = ReplayMemory(10000, 24, self.network_input)

        # Create network and target network
        self.network = network.Network()
        self.network.create_network(self.network_input, self.network_hiddens, self.network_out, 'linear')
        self.network.model.summary()
        self.target_network = network.Network()
        self.target_network.create_network(self.network_input, self.network_hiddens, self.network_out, 'linear')

        # Set hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.steps = 1

    def update_target_network(self):
        # Update the target network
        self.target_network.model.set_weights(self.network.model.get_weights())

    def compute_action(self, observation):
        # Use the network to determine the next action
        if self.epsilon < np.random.rand():
            action =  np.argmax(self.network.predict(observation)[0])
        else:
        # Choose a random action
            action = np.random.randint(0,2)
        # Update epsilon with the decay
        self.epsilon = max(self.decay*self.epsilon, self.min_epsilon)
        self.steps = self.steps + 1
        return action

    def train(self):
        # Update the parameters of the target network
        if self.steps % 200 == 0:
            self.update_target_network()
        # Train if there are enough samples in the memory
        if (self.replay_memory.reached_batch_size()):
            states, actions, rewards, new_states, done = self.replay_memory.sample()
            # Compute the Temporal Difference
            targets = self.target_network.model(new_states)
            td_targets = rewards + (1-done) * self.gamma * keras.backend.max(targets,axis=1)
            with tf.GradientTape() as tape:
                # Get Q-values for the current state
                Q = self.network.model(states) 
                a=tf.one_hot(actions, self.network_out)
                Q_values = tf.reduce_sum(Q*a, axis=1)
                # Compute the TDE as loss
                loss = self.network.loss(td_targets,Q_values)
                # Compute gradients and apply them
                gradients = tape.gradient(loss, self.network.model.trainable_weights)
                self.network.optimizer.apply_gradients(zip(gradients, self.network.model.trainable_weights))
            
    def add_to_memory(self, *args):
        # Add observations to the memory
        self.replay_memory.add_sample(*args)

class ReplayMemory:
    def __init__(self, capacity, batch_size, observation_size):
        # Initialize arrays to store observations
        self.state_memory = np.zeros((capacity,observation_size))
        self.action_memory = np.zeros((capacity))
        self.reward_memory = np.zeros((capacity))
        self.new_state_memory = np.zeros((capacity,observation_size))
        self.done_memory = np.zeros((capacity))
        self.counter = 0
        self.batch_size = batch_size
        self.capacity = capacity

    def add_sample(self, obs, action, reward, new_obs, done):
        # Add observations to the memory
        self.state_memory[self.counter%self.capacity] = obs
        self.action_memory[self.counter%self.capacity] = action
        self.reward_memory[self.counter%self.capacity] = reward
        self.new_state_memory[self.counter%self.capacity] = new_obs
        self.done_memory[self.counter%self.capacity] = done
        self.counter = self.counter + 1
        if self.counter == (self.batch_size*3):
            print("begin training")

    def reached_batch_size(self):
        # Determine if there are enough samples to start training
        return self.counter >= (self.batch_size*3)

    def sample(self):
        # Return randomly chosen samples from the memory
        batch = np.random.choice(range(0,min(self.counter,self.capacity)), self.batch_size)
        return self.state_memory[batch], self.action_memory[batch], self.reward_memory[batch], self.new_state_memory[batch], self.done_memory[batch]



