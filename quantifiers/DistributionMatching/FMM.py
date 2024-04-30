import setup_paths

import numpy as np
import cvxpy
import pandas as pd

from interface_class.Quantifier import Quantifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

class FMM(Quantifier):

    def __init__(self, classifier, nbins = 64, ndescriptors = 1, distribution_function = 'CDF', data_split='holdout'):
        self.classifier = classifier
        self.data_split = data_split

        self.train_distrib = None
        self.train_scores = []
        self.test_scores = []
        
        self.train_labels = None
        self.nclasses = None

        self.nbins = nbins
        self.ndescriptors = ndescriptors

        self.distribution_function = distribution_function

    ############
    #  Functions for solving optimization problems with different loss functions
    ############
    def solve_l1(self, train_distrib, test_distrib, n_classes, solver='ECOS'):
        """ Solves AC, PAC, PDF and Friedman optimization problems for L1 loss function

            min   |train_distrib * prevalences - test_distrib|
            s.t.  prevalences_i >=0
                sum prevalences_i = 1

            Parameters
            ----------
            train_distrib : array, shape depends on the optimization problem
                Represents the distribution of each class in the training set
                PDF: shape (n_bins * n_classes, n_classes)
                AC, PAC, Friedman: shape (n_classes, n_classes)

            test_distrib : array, shape depends on the optimization problem
                Represents the distribution of the testing set
                PDF: shape shape (n_bins * n_classes, 1)
                AC, PAC, Friedman: shape (n_classes, 1)

            n_classes : int
                Number of classes

            solver : str, (default='ECOS')
                The solver used to solve the optimization problem. The following solvers have been tested:
                'ECOS', 'ECOS_BB', 'CVXOPT', 'GLPK', 'GLPK_MI', 'SCS' and 'OSQP', but it seems that 'CVXOPT' does not
                work

            Returns
            -------
            prevalences : array, shape=(n_classes, )
            Vector containing the predicted prevalence for each class
        """
        prevalences = cvxpy.Variable(n_classes)
        objective = cvxpy.Minimize(cvxpy.norm(np.squeeze(test_distrib) - train_distrib * prevalences, 1))

        contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

        prob = cvxpy.Problem(objective, contraints)
        prob.solve(solver=solver)

        return np.array(prevalences[0:n_classes].value).squeeze()

    def solve_l2cvx(self, train_distrib, test_distrib, n_classes, solver='ECOS'):
        prevalences = cvxpy.Variable(n_classes)
        objective = cvxpy.Minimize(cvxpy.sum_squares(train_distrib * prevalences - test_distrib))

        contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

        prob = cvxpy.Problem(objective, contraints)
        prob.solve(solver=solver)

        return np.array(prevalences[0:n_classes].value).squeeze()
    
    def get_class_proportion(self):
        test_distrib = np.zeros((self.nbins * self.ndescriptors, 1))

        for descr in range(self.ndescriptors):
            test_distrib[descr * self.nbins:(descr + 1) * self.nbins, 0] = \
                np.histogram(self.test_scores[:, descr], bins=self.nbins, range=(0., 1.))[0]
            
            test_distrib = test_distrib / len(self.test_scores)

        if self.distribution_function == 'CDF':
            test_distrib = np.cumsum(test_distrib, axis=0)

        prevalences = self.solve_l1(train_distrib=self.train_distrib, test_distrib=test_distrib,
                               n_classes=len(self.nclasses))

        return np.round([prevalences[1], prevalences[0]], 2)

    def fit(self, X_train, y_train):
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size=0.33)

        self.classifier.fit(X_val_train, y_val_train)
        self.train_scores = self.classifier.predict_proba(X_val_test)
        self.train_scores = [score[1] for score in self.train_scores]

        data = {'scores': self.train_scores, 'class': y_val_test}
        self.train_scores = pd.DataFrame(data)

        self.nclasses = np.unique(self.train_scores['class'])

        train_distrib = np.zeros((self.nbins * self.ndescriptors, len(self.nclasses)))

        for n_cls, cls in enumerate(self.nclasses):
            descr = 0
            train_distrib[descr * self.nbins:(descr + 1) * self.nbins, n_cls] = np.histogram(self.train_scores[self.train_scores['class'] == cls]['scores'], bins=self.nbins, range=(0., 1.))[0]
            train_distrib[:, n_cls] = train_distrib[:, n_cls] / (self.train_scores[self.train_scores['class'] == cls].shape[0])

        train_distrib = np.cumsum(train_distrib, axis=0)
        self.train_distrib = train_distrib

        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        self.test_scores = self.classifier.predict_proba(X_test)
        return self.get_class_proportion()

if __name__ == '__main__':

    data = datasets.load_breast_cancer()

    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, y, test_size=0.33)

    knn = KNeighborsClassifier(n_neighbors=7)

    fmm = FMM(knn)
    fmm.fit(X_trainn, y_trainn)

    class_distribution = fmm.predict(X_testt)

    print(fmm.classifier.classes_)
    print(data.target_names)
    print(class_distribution)