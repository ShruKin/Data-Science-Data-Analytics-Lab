from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = pd.DataFrame(iris.target)
# data.head()

model = LogisticRegression(max_iter=1000)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model_fit = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

scores = cross_val_score(model, X, Y, cv=10)

print("Average Score: %0.2f" % (scores.mean()))
