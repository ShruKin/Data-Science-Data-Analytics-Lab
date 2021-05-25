from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = pd.DataFrame(iris.target)
data.head()


model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=5000,
    activation='logistic',
    solver='sgd',
    learning_rate_init=0.001)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

model_fit = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_pred, Y_test).round(2)*100, "%")
