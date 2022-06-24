from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier


X = [
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0]
]

Y = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, Y)
result = clf.predict([[1, 1, 0, 0, 1, 1]])
print(f"Expected result of A_15 is 1, perceptron calculated {result[0]}")

clf = DecisionTreeClassifier()
clf.fit(X, Y)
result = clf.predict([[1, 1, 0, 0, 1, 1]])
print(f"Expected result of A_15 is 1, decision tree calculated {result[0]}")