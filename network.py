import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

class Network:
    def __init__(self) -> None:
        # Initialize loss, learning rate and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.model = None
        self.model_weights_shape = None
        self.model_bias_shape = None

    def create_network(self, input, hidden_list, output, output_activation):
        # Acrobot: 3 actions, 6 observations
        self.model = tf.keras.Sequential()

        # Create input layer
        self.model.add(tf.keras.Input(shape=(input,)))

        # Add Dense layers according to the hidden sizes
        for i in range(len(hidden_list)):
            self.model.add(tf.keras.layers.Dense(hidden_list[i], activation = 'relu'))

        # Add output layer
        self.model.add(tf.keras.layers.Dense(output, activation = output_activation)) 

        self.model.compile(loss = self.loss, optimizer = self.optimizer)
        
        # Create array with shapes of weights and biasses
        self.model_weights_shape = [[input, hidden_list[0]]]
        for i in range(len(hidden_list)-1):
            self.model_weights_shape.append([hidden_list[i], hidden_list[i+1]])
        self.model_weights_shape.append([hidden_list[-1], output])
        self.model_bias_shape = []
        for i in range(len(hidden_list)):
            self.model_bias_shape.append([hidden_list[i]])
        self.model_bias_shape.append([output])

    def evaluate(self, y_data, prediction):
        # Evaluate the prediction
        return metrics.mean_squared_error(y_data, prediction)

    def get_weights(self):
        # Get weights and biasses
        weights = []
        biasses = []
        for l in self.model.layers:
            weight, bias = l.get_weights()
            weights.append(weight)
            biasses.append(bias)
        return weights, biasses

    def set_weights(self, weights, biasses):
        start_point_w = 0
        start_point_b = 0

        for i in range(0, len(self.model.layers)):
            # Get the weights and biasses from the vector
            layer_weights = weights[start_point_w : start_point_w + (self.model_weights_shape[i][0]*self.model_weights_shape[i][1])]
            layer_biasses = biasses[start_point_b : start_point_b + self.model_bias_shape[i][0]]

            # Reshape the weights to fit the shape of the layer
            layer_weights = layer_weights.reshape(self.model_weights_shape[i][0], self.model_weights_shape[i][1])

            # Set the weights of the network
            self.model.layers[i].set_weights([layer_weights, layer_biasses]) 

            # Update index points
            start_point_w += self.model_weights_shape[i][0]*self.model_weights_shape[i][1]
            start_point_b += self.model_bias_shape[i][0] 

    def predict(self,input):
        # Predict actions
        return self.model(input[np.newaxis,:])


