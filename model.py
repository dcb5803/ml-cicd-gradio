from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_model():
    iris = load_iris()
    clf = RandomForestClassifier()
    clf.fit(iris.data, iris.target)
    return clf, iris
