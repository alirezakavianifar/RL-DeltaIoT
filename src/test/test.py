import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Generate synthetic data (e.g., 2D points)
np.random.seed(42)
data_size = 1000
original_data = np.random.rand(data_size, 2)

# Build the autoencoder model
input_layer = Input(shape=(2,))
encoded = Dense(1, activation='relu')(input_layer)  # Compressed representation
decoded = Dense(2, activation='sigmoid')(encoded)    # Reconstructed output

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(original_data, original_data, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Use the trained autoencoder to encode and decode the data
encoded_data = autoencoder.predict(original_data)

# Plot original and reconstructed data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(original_data[:, 0], original_data[:, 1], color='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], color='red', label='Reconstructed Data')
plt.title('Reconstructed Data (Encoded and Decoded)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()
