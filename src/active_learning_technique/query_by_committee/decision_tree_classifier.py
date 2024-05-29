from sklearn import tree
from sklearn.metrics import accuracy_score
from src.support.support import clprint, Reason


class DecisionTreeClassifier:

    def __init__(self, dataset):
        self.tree = tree.DecisionTreeClassifier()
        self._train_model(dataset)

    def predict(self, x):
        x = x.reshape(1, -1)
        return self.tree.predict(x)

    def update(self, dataset):
        self._train_model(dataset)

    def _train_model(self, dataset):
        x, y = dataset.get_train_numpy()
        self.tree.fit(x, y)
        x, y = dataset.get_test_numpy()
        predictions = self.tree.predict(x)
        clprint("Decision Tree Classifier Accuracy {}%...".format((accuracy_score(y, predictions) * 100)), Reason.INFO_TRAINING, loggable=True)
