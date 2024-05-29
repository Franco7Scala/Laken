import numpy

from sklearn import svm
from sklearn.metrics import accuracy_score
from src.support import clprint, Reason


class SVM:

    def __init__(self, dataset, n_classes):
        self.n_classes = n_classes
        self.svm = svm.SVC(decision_function_shape="ovo")
        self._train_model(dataset)

    def predict(self, x):
        x = x.reshape(1, -1)
        return numpy.eye(self.n_classes)[int(self.svm.predict(x)[0])]

    def update(self, dataset):
        self._train_model(dataset)

    def _train_model(self, dataset):
        x, y = dataset.get_train_numpy()
        self.svm.fit(x, y)
        x, y = dataset.get_test_numpy()
        predictions = self.svm.predict(x)
        clprint("SVM Accuracy {}%...".format((accuracy_score(y, predictions) * 100)), Reason.INFO_TRAINING, loggable=True)
