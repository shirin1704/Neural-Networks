import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression # pyright: ignore[reportMissingModuleSource]

data = pd.read_csv('pima-indians-diabetes.csv')

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train)