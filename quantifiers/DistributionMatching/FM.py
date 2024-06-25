import QuantifiersLibrary.quantifiers.DistributionMatching.setup_paths as setup_paths
import pdb

import numpy as np
import cvxpy as cvx
import pandas as pd

from interface_class.Quantifier import Quantifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

class FM(Quantifier):

    def __init__(self, classifier, data_split='holdout'):
        self.classifier = classifier
        self.data_split = data_split

        self.train_scores = []
        self.test_scores = []
        
        self.train_labels = None
        self.nclasses = None

    def get_class_proportion(self):
        CM = np.zeros((self.nclasses, self.nclasses))
        y_cts = np.array([np.count_nonzero(self.train_labels == i) for i in range(self.nclasses)])

        p_yt = y_cts / self.train_labels.shape[0]
        
        for i in range(self.nclasses):
            idx = np.where(self.train_labels == i)[0]
            # pdb.set_trace()
            CM[:, i] += np.sum(self.train_scores[idx] > p_yt, axis=0)
        CM = CM / y_cts
        p_y_hat = np.sum(self.test_scores > p_yt, axis = 0) / self.test_scores.shape[0]
        
        p_hat = cvx.Variable(CM.shape[1])
        constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
        problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
        problem.solve()

        return np.round(p_hat.value, 2)

    def fit(self, X_train, y_train):
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size=0.33)

        self.classifier.fit(X_val_train, y_val_train)
        scores = self.classifier.predict_proba(X_val_test)
        self.train_scores = scores

        self.train_labels = y_val_test
        self.nclasses = len(np.unique(y_val_test))

        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        scores = self.classifier.predict_proba(X_test)
        self.test_scores = scores

        return self.get_class_proportion()

if __name__ == '__main__':

    data = datasets.load_breast_cancer()

    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, y, test_size=0.33)

    knn = KNeighborsClassifier(n_neighbors=7)

    fm = FM(knn)
    fm.fit(X_trainn, y_trainn)

    class_distribution = fm.predict(X_testt)

    print(fm.classifier.classes_)
    print(data.target_names)
    print(class_distribution)