import numpy as np # pyright: ignore[reportMissingImports]
from keras.models import Sequential # pyright: ignore[reportMissingImports]
from keras.layers import Dense # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

X = np.random.rand(1000, 20)
Y = ((X*3) + np.random.randn(1000, 20)*(0.8))

model = Sequential()
model.add(Dense(64, input_dim = 20, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(20, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size = 32, epochs = 10)

X_test = np.random.rand(1000, 20)
Y_test = ((X*3) + np.random.randn(1000, 20)*(0.8))

loss, accuracy = model.evaluate(X_test, Y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)