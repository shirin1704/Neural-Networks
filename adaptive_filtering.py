import numpy as np
import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

def generate_data(num_samples=1000, noise_factor=0.5):
    np.random.seed(42)
    t = np.linspace(0, 1, num_samples)
    s = np.sin(2 * np.pi * t) # Clean signal
    n = noise_factor * np.random.normal(size=num_samples) # Noise
    x = s + n # Noisy signal
    return x, s

X_train, S_train = generate_data(num_samples = 1000)
X_test, S_test = generate_data(num_samples = 300)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
S_train = S_train.reshape(-1, 1)
S_test = S_test.reshape(-1, 1)

model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, S_train, epochs=100, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, S_test)
print(f'Test Loss: {loss}')

S_pred = model.predict(X_test)

#Plot results
plt.figure(figsize=(12, 6))
plt.plot(S_test, label='Clean Signal', linewidth=2)
plt.plot(X_test, label='Noisy Signal', alpha=0.6)
plt.plot(S_pred, label='Predicted Clean Signal', linestyle='dashed', linewidth=2)
plt.legend()
plt.show()