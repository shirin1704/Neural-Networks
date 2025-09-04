import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("Housing.csv")
X = data[["area", "bedrooms"]].values
y = data["price"].values.reshape(-1, 1)

# Train model
model = LLSNeuralNetwork(n_hidden=30, activation="relu", random_state=42) # pyright: ignore[reportUndefinedVariable]
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
print("Training MSE:", mse)

# Plot Actual vs Predicted
plt.scatter(y, y_pred, c="blue", label="Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Prediction")
plt.xlabel("Actual Price ($1000s)")
plt.ylabel("Predicted Price ($1000s)")
plt.legend()
plt.show()