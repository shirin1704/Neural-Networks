#Building a Basic Neural Network
import numpy as np # pyright: ignore[reportMissingImports]
from keras.models import Sequential # pyright: ignore[reportMissingImports]
from keras.layers import Dense # pyright: ignore[reportMissingImports]
from sklearn.metrics import r2_score  #pyright: ignore[reportMissingModuleSource]

X = np.random.rand(1000, 20)
Y = ((X*3) + np.random.randn(1000, 20)* 0.8)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dense(32, activation='relu'))
model.add(Dense(20, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['root_mean_squared_error'])

model.fit(X, Y, batch_size = 32, epochs=50)

X_test = np.random.rand(1000, 20)
Y_test = ((X_test*3) + np.random.randn(1000, 20)*0.8)
Y_pred = model.predict(X_test)

loss, rmse = model.evaluate(X_test, Y_test)
print("Test Loss: ", loss)
print("Test RMSE: ", rmse)
print("R2 Score: ", r2_score(Y_test, Y_pred))