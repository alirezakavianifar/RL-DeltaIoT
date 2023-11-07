import tensorflow as tf


class DeepQNetwork(tf.keras.Model):
    def __init__(self, input_dims, n_actions, network_layers):
        super(DeepQNetwork, self).__init__()
        self.inputs = tf.keras.Input(shape=(input_dims,))
        self.core_layers = []
        for index, num_layer in enumerate(network_layers):
            self.core_layers.append(
                tf.keras.layers.Dense(num_layer, activation='relu'))
        self.fc3 = tf.keras.layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.core_layers[0](state)
        for index, layer in enumerate(self.core_layers[1:]):
            x = layer(x)
        x = self.fc3(x)
        return x
