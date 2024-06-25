import QuantifiersLibrary.quantifiers.ClassifyCountCorrect.setup_paths as setup_paths

import pdb

from interface_class.Quantifier import Quantifier
from utils.Quantifier_Utils import TPRandFPR

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np
import pandas as pd

import time

class MedianSweep2(Quantifier):

    def __init__(self, classifier):
        self.classifier = classifier
        self.tprfpr = None

    def get_class_proportion(self, scores):

        index = np.where(abs(self.tprfpr['tpr'] - self.tprfpr['fpr']) > (1/4))[0].tolist()
        if index == 0:
            index = np.where(abs(self.tprfpr['tpr'] - self.tprfpr['fpr']) >= 0)[0].tolist()

        prevalances_array = []    
        for i in index:
            
            threshold, fpr, tpr = self.tprfpr.loc[i]

            pos_scores = [score[1] for score in scores]
 
            estimated_positive_ratio = len([pos_score for pos_score in pos_scores if pos_score >= threshold])
            estimated_positive_ratio /= len(scores)

            diff_tpr_fpr = abs(float(tpr-fpr))  
        
            if diff_tpr_fpr == 0.0:            
                diff_tpr_fpr = 1     
        
            final_prevalence = abs(estimated_positive_ratio - fpr)/diff_tpr_fpr
            
            prevalances_array.append(final_prevalence)  
    
        pos_prop = np.median(prevalances_array)
        
        if pos_prop <= 0:                           #clipping the output between [0,1]
            pos_prop = 0
        elif pos_prop >= 1:
            pos_prop = 1
        else:
            pos_prop = np.round(pos_prop, 2)

        neg_prop = np.round(1-pos_prop, 2)

        return [neg_prop, pos_prop]


    def fit(self, X_train, y_train):
        # Validation split
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train)

        # Validation train
        self.classifier.fit(X_val_train, y_val_train)

        # Validation result_table
        pos_val_scores = self.classifier.predict_proba(X_val_test)[:, 1]

        # Generating the dataframe with the positive scores and its own class
        pos_val_scores = pd.DataFrame(pos_val_scores, columns=['score'])
        pos_val_labels = pd.DataFrame(y_val_test, columns=['class'])

        # Needed to reset the index, predict_proba result_table reset the indexes!
        pos_val_labels.reset_index(drop=True, inplace=True)

        val_scores = pd.concat([pos_val_scores, pos_val_labels], axis='columns', ignore_index=False)

        # Generating the tpr and fpr for thresholds between [0, 1] for the validation scores!
        self.tprfpr = TPRandFPR(val_scores)

        # Fit the classifier again but now with the whole train set
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        # Result from the test
        scores = self.classifier.predict_proba(X_test)

        # Class proportion generated through the main algorithm
        return self.get_class_proportion(scores)



def MS_method2(test_scores, tprfpr):
    """Median Sweep2

    It quantifies events based on their scores, applying Median Sweep (MS2) method, according to Forman (2006).
    
    Parameters
    ----------
    test scores: array
        A numeric vector of scores predicted from the test set.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
        
    Returns
    -------
    array
        the class distribution of the test. 
    """
    start = time.time()

    index = np.where(abs(tprfpr['tpr'] - tprfpr['fpr']) > (1/4) )[0].tolist()
    if index == 0:
        index = np.where(abs(tprfpr['tpr'] - tprfpr['fpr']) >=0 )[0].tolist()

    
    prevalances_array = []    
    for i in index:
        
        threshold, fpr, tpr = tprfpr.loc[i]
        estimated_positive_ratio = len(np.where(test_scores >= threshold)[0])/len(test_scores)
        
        diff_tpr_fpr = abs(float(tpr-fpr))  
    
        if diff_tpr_fpr == 0.0:            
            diff_tpr_fpr = 1     
    
        final_prevalence = abs(estimated_positive_ratio - fpr)/diff_tpr_fpr
        
        prevalances_array.append(final_prevalence)  
  
    pos_prop = np.median(prevalances_array)
    
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    stop = time.time()
    #return stop - start
    return pos_prop


if __name__ == '__main__':
    data = datasets.load_breast_cancer()

    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, y, test_size=0.33)

    knn = KNeighborsClassifier(n_neighbors=7)
    ms2 = MedianSweep2(knn)

    ms2.fit(X_trainn, y_trainn)
    class_distribution = ms2.predict(X_testt)

    print(ms2.classifier.classes_)
    print(data.target_names)
    print(class_distribution)