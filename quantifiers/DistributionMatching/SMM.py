import QuantifiersLibrary.quantifiers.DistributionMatching.setup_paths as setup_paths

import numpy as np
import pandas as pd

from interface_class.Quantifier import Quantifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

class SMM(Quantifier):

    def __init__(self, classifier, data_split='holdout'):
        self.classifier = classifier
        self.data_split = data_split
        self.p_scores = []
        self.n_scores = []
        self.test_scores = []

    def get_class_proportion(self):
        mean_pos_scr = np.mean(self.p_scores)
        mean_neg_scr = np.mean(self.n_scores)  #calculating mean of pos & neg scores

        mean_te_scr = np.mean(self.test_scores)              #Mean of test scores
                
        # Perguntar andre, pode dar negativo essa divisao?
        alpha =  (mean_te_scr - mean_neg_scr)/(mean_pos_scr - mean_neg_scr)     #evaluating Positive class proportion
            
        if alpha <= 0:   #clipping the output between [0,1]
            pos_prop = 0
        elif alpha >= 1:
            pos_prop = 1
        else:
            pos_prop = alpha

        pos_prop = np.round(pos_prop, 2)

        neg_prop = np.round(1 - pos_prop, 2)

        return [neg_prop, pos_prop]


    def fit(self, X_train, y_train):
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size=0.33)

        self.classifier.fit(X_val_train, y_val_train)
        scores = self.classifier.predict_proba(X_val_test)

        p_scores = scores[y_val_test == 1]
        self.p_scores = [p_score[1] for p_score in p_scores]

        n_scores = scores[y_val_test == 0]
        self.n_scores = [n_score[1] for n_score in n_scores]

        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        scores = self.classifier.predict_proba(X_test)
        scores = [score[1] for score in scores]

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

    smm = SMM(knn)
    smm.fit(X_trainn, y_trainn)

    class_distribution = smm.predict(X_testt)

    print(smm.classifier.classes_)
    print(data.target_names)
    print(class_distribution)