import time
start_time = time.time()

###Loading packages
import os
import numpy as np
import pandas as pd
import math
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from numpy.random import seed
seed(1)


import itertools

def measurements(y_test, y_pred, y_pred_prob):  
    acc = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    npv = TN/(TN+FN)       
    return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc]

def model_predict(X, y, y_index, model, col_name):
    y_pred_prob = model.predict_proba(X)
    # keep probabilities for the positive outcome only
    y_pred_prob = y_pred_prob[:, 1]
    y_pred_class = np.where(y_pred_prob > 0.5, 1, 0)

    ###create dataframe
    pred_result = pd.DataFrame()
    pred_result['id'] = y_index
    pred_result['y_true'] = y
    pred_result['prob_'+col_name] = y_pred_prob
    pred_result['class_'+col_name] = y_pred_class
    
    performance =measurements(y, y_pred_class, y_pred_prob)

    return pred_result, performance

def para_selection(var, X, X_test, y, y_test, base_path, descriptor):
    #for para
    lr = [0.3, 0.1, 0.01]
    n_estimators = [50, 100, 200, 300]
    max_depth = [5, 6, 7]
    subsample = [0.6, 0.7, 0.8]
    scale_pos_weight=[0.46]

    paras = [l for l in itertools.product(lr, n_estimators, max_depth, subsample, scale_pos_weight)]

    #for var in range(len(paras)):
    print(var)
    para = paras[int(var)]
    print(para)
    
    #initial performance dictionary
    test_results={}

    ### scale the input
    sc = MinMaxScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_test = sc.transform(X_test)


    ### define column name
    col_name = 'xgboost_'+'paras_'+var

    ###create classifier
    clf = XGBClassifier(learning_rate=para[0], n_estimators=para[1], max_depth=para[2], subsample=para[3], scale_pos_weight=para[4])
    clf.fit(X, np.array(y, dtype=np.int))

    ### predict test results
    test_class, test_result=model_predict(X_test, np.array(y_test, dtype=np.int), y_test.index, clf, col_name)

    test_results[col_name]=test_result

    test_class.to_csv(base_path+'/'+ descriptor +'_'+col_name+'_class.csv')

    ###save the result of validation results
    pd.DataFrame(data=test_results.items()).to_csv(base_path+'/'+ descriptor +'_'+col_name+'_performance.csv')

    
print("--- %s seconds ---" % (time.time() - start_time))    
    