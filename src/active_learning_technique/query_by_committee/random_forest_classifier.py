from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src.support import clprint, Reason


class RandomForestClassifier:

    def __init__(self, dataset):
        self.classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)
        self._train_model(dataset)

    def predict(self, x):
        x = x.reshape(1, -1)
        return self.classifier.predict(x)

    def update(self, dataset):
        self._train_model(dataset)

    def _train_model(self, dataset):
        x, y = dataset.get_train_numpy()
        self.classifier.fit(x, y)
        x, y = dataset.get_test_numpy()
        predictions = self.classifier.predict(x)
        clprint("Random Forest Classifier Accuracy {}%...".format((accuracy_score(y, predictions) * 100)), Reason.INFO_TRAINING, loggable=True)