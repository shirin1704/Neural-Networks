import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow import keras # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

#1. Assuming a trained model exists or is trained here
# For demonstration, let's create a simple regression model
model = keras.Sequential([
    keras.layers.Dense(units = 19, activation = 'relu', input_shape = (1,)),
    keras.layers.Dense(units = 1)
])
model.compile(optimizer = 'adam', loss = 'mse')

# Dummy training data
X_train = np.array([[1], [2], [3], [4], [5]])
Y_train = np.array([[2], [4], [6], [8], [10]])
model.fit(X_train, Y_train, epochs = 100)

plt.scatter(X_train, Y_train)
plt.show()

#2. Prepare new input data
X_new = np.array([[6], [7], [8]])

#3. Generate new predictions
predictions = model.predict(X_new)

print("Predictions for new data:")
print(predictions)

plt.scatter(X_new, predictions)
plt.show()
