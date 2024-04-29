import setup_paths

from interface_class.Quantifier import Quantifier

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

class EMQ(Quantifier):

    def __init__(self, classifier, data_split='holdout'):
        self.classifier = classifier
        self.data_split = data_split

        self.test_scores = []
        self.train_labels = None
        self.nclasses = None

        self.max_it = 1000 # Max num of iterations
        self.eps = 1e-6 # Small constant for stopping criterium

    def class_dist(self, Y, nclasses):
        return np.array([np.count_nonzero(Y == i) for i in range(nclasses)]) / Y.shape[0]

    def get_class_proportion(self):
        print(self.train_labels)

        p_tr = self.class_dist(self.train_labels, self.nclasses)
        print(p_tr)

        p_s = np.copy(p_tr)
        print(p_s)

        p_cond_tr = np.array(self.test_scores)
        p_cond_s = np.zeros(p_cond_tr.shape)

        print(p_cond_tr)
        print(p_cond_s)


        # Perguntar sobre as divisões por 0!!
        for it in range(self.max_it):
            r = p_s / p_tr
            print(r)
            p_cond_s = p_cond_tr * r
            s = np.sum(p_cond_s, axis = 1)
            for c in range(self.nclasses):
                p_cond_s[:,c] = p_cond_s[:,c] / s
            p_s_old = np.copy(p_s)
            p_s = np.sum(p_cond_s, axis = 0) / p_cond_s.shape[0]
            if (np.sum(np.abs(p_s - p_s_old)) < self.eps):
                break

        prop = p_s/np.sum(p_s)
        
        return(np.round(prop, 2))


    def fit(self, X_train, y_train):
        # Provavel fazer validation aqui!!! (nao usar o treino real :P)
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size=0.33)

        self.classifier.fit(X_val_train, y_val_train)
        scores = self.classifier.predict_proba(X_val_test)

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

    emq = EMQ(knn)
    emq.fit(X_trainn, y_trainn)

    class_distribution = emq.predict(X_testt)

    print(emq.classifier.classes_)
    print(data.target_names)
    print(class_distribution)