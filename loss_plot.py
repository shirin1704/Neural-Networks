import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn import preprocessing # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from keras.models import Sequential  # pyright: ignore[reportMissingImports]
from keras.layers import Dense # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

data = pd.read_csv('pima-indians-diabetes.csv')
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaled, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.3)

model = Sequential([
    Dense(32, activation = 'relu', input_shape = (8,)),
    Dense(32, activation = 'relu'),
    Dense(1, activation = 'sigmoid'),
])

model.compile(optimizer = 'sgd', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

hist = model.fit(X_train, Y_train,
                 batch_size = 32, epochs = 100,
                 validation_data = (X_val, Y_val))

model.evaluate(X_test, Y_test)[1]

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()
