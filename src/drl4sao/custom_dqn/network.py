import tensorflow as tf


import tensorflow as tf

class DeepQNetwork(tf.keras.Model):
    """
    DeepQNetwork class represents a deep neural network for Q-learning in reinforcement learning.

    Attributes:
        - input_dims (int): Number of input dimensions for the neural network.
        - n_actions (int): Number of possible actions in the environment.
        - network_layers (list): List of integers representing the number of neurons in each hidden layer.

    Methods:
        - __init__(self, input_dims, n_actions, network_layers): Initializes the DeepQNetwork instance.
        - call(self, state): Defines the forward pass of the neural network.
    """

    def __init__(self, input_dims, n_actions, network_layers):
        """
        Initializes the DeepQNetwork instance.

        Parameters:
            - input_dims (int): Number of input dimensions for the neural network.
            - n_actions (int): Number of possible actions in the environment.
            - network_layers (list): List of integers representing the number of neurons in each hidden layer.
        """
        super(DeepQNetwork, self).__init__()

        # Define input layer
        self.inputs = tf.keras.Input(shape=(input_dims,))

        # Create hidden layers with ReLU activation
        self.core_layers = []
        for num_layer in network_layers:
            self.core_layers.append(
                tf.keras.layers.Dense(num_layer, activation='relu'))

        # Output layer with softmax activation for multiple actions
        self.fc3 = tf.keras.layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        """
        Defines the forward pass of the neural network.

        Parameters:
            - state (tf.Tensor): Input tensor representing the current state.

        Returns:
            tf.Tensor: Output tensor representing Q-values for each action.
        """
        x = self.core_layers[0](state)
        for layer in self.core_layers[1:]:
            x = layer(x)
        x = self.fc3(x)
        return x

