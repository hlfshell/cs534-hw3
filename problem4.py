from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus


X = [
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0],
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

coefficients = clf.coef_[0:1][0]
intercept = clf.intercept_[0:1][0]
print("The linear equation for the perception is:")
print(f"T = {coefficients[0]}x1 + {coefficients[1]}x2 +  {coefficients[2]}x3 + {coefficients[3]}x4 + {coefficients[4]}x5 + {coefficients[5]}x6 + {intercept}")

clf = DecisionTreeClassifier()
clf.fit(X, Y)
result = clf.predict([[1, 1, 0, 0, 1, 1]])
print(f"Expected result of A_15 is 1, decision tree calculated {result[0]}")

dot_data = export_graphviz(
    clf,
    # out_file="tree.png",
    feature_names=["x1", "x2", "x3", "x4", "x5", "x6"],
    filled=True, rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')