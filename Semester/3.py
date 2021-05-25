from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = pd.DataFrame(iris.target)
data.head()


model = MLPClassifier(
    hidden_layer_sizes=(12,),
    max_iter=5000,
    activation='logistic',
    solver='sgd',
    learning_rate_init=0.001)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

model_fit = model.fit(X, Y)

X_test = pd.DataFrame([
    [5, 2, 1, 1],
    [2, 7, 4, 1]
])
Y_test = model.predict(X_test)

for y in Y_test:
    print(iris.target_names[y])
