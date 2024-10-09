import time
start_time = time.time()

import sys
var=sys.argv[1]

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

def para_selection(var, X, X_test, y, y_test, base_path):
    #for para
    lr = [0.0001, 0.001, 0.01]
    n_estimators = [100, 300, 500,700]
    max_depth = [7, 9, 11]
    subsample = [0.7, 0.8, 1]
    scale_pos_weight=[0.37]

    paras = [l for l in itertools.product(lr, n_estimators, max_depth, subsample, scale_pos_weight)]

    #for var in range(len(paras)):
    print(var)
    para = paras[int(var)]
    print(para)
    
    #create path
    path10 = base_path + '/training_performance'
    path20 = base_path + '/test_performance'

    path1 = base_path + '/training_class'
    path2 = base_path + '/test_class'

    ###make the directory
    os.mkdir(base_path)
    os.mkdir(path10)
    os.mkdir(path20)

    os.mkdir(path1)
    os.mkdir(path2)
    
    #initial performance dictionary
    train_results={}
    test_results={}
    pred_test_df = pd.DataFrame()

    for i in range(20):
        skf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)
        j = 0
        for train_index, validation_index in skf.split(X,np.array(y, dtype=np.int)):
            ###get train, validation dataset
            X_train, X_validation = X.iloc[train_index,:], X.iloc[validation_index,:]
            y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

            ### scale the input
            sc = MinMaxScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_validation = sc.transform(X_validation)
            X_test_s = sc.transform(X_test)

           ### define column name
            col_name = 'xgboost_'+'seed_'+str(i)+'_skf_'+str(j)+'_paras_'+var+'_K_'+str(para)
            col_name1 = 'xgboost_'+'seed_'+str(i)+'_paras_'+var+'_K_'+str(para)
            col_name2 = 'xgboost_'+'paras_'+var

            ###create classifier
            clf = XGBClassifier(learning_rate=para[0], n_estimators=para[1], max_depth=para[2], subsample=para[3], scale_pos_weight=para[4])
            clf.fit(X_train, np.array(y_train, dtype=np.int))

            ### predict validation results
            train_class, train_result=model_predict(X_validation, np.array(y_validation, dtype=np.int),y_validation.index, clf, col_name)
            train_results[col_name]=train_result


            ### predict test results
            test_class, test_result=model_predict(X_test_s, np.array(y_test, dtype=np.int),y_test.index, clf, col_name)

            test_results[col_name]=test_result

            pred_test_df = pd.concat([pred_test_df, test_class],axis=1, sort=False)
            j += 1
            train_class.to_csv(path1+'/train_'+col_name+'.csv')

    ###save the result of validation results
    pd.DataFrame(data=train_results.items()).to_csv(path10+'/train_'+col_name2+'.csv')
    pred_test_df.to_csv(path2+'/test_'+col_name2+'.csv')
    pd.DataFrame(data=test_results.items()).to_csv(path20+'/test_'+col_name2+'.csv')

### please update the following path 
#define path
base_path = '/compute01/yqu/dictrank/results/mold2'

#read data
DICT = pd.read_excel(r'/compute01/yqu/dictrank/data/DICTrank_Mold2_1006.xlsx')

## remove the "Ambiguous-DICT concern" drugs
print(DICT.shape)
DICT.drop(DICT[DICT['DICT']=='Ambiguous'].index,inplace=True)

## Classify the "Less- and Most- DICT concern " as cardiotox positive  and label "1"
## classify the "No-DICT concern" as cardiotox negtive and label "0"
DICT.loc[DICT['DICT']=="Less",'DICT']=1
DICT.loc[DICT['DICT']=="Most",'DICT']=1
DICT.loc[DICT['DICT']=="No",'DICT']=0

#get the feature columns
cols=DICT.columns[-700:]
data=DICT[['DICT',*cols]]
print(data.shape)

zero_cols = data.columns[(data == 0).all()]
data.drop(zero_cols, axis=1, inplace=True)
print(data.shape)

X1=data.iloc[:,1:]
y1=data["DICT"]
X1=X1.fillna(0)

## data and split(random and stratify)
X, X_test, y, y_test = train_test_split(X1, y1, test_size=.2,stratify=y1,random_state=42)
print('X_train shape:', X.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y.shape)
print('y_test shape:',y_test.shape)

para_selection(var, X, X_test, y, y_test, base_path+'/xgboost/xgboost'+ var)
    
print("--- %s seconds ---" % (time.time() - start_time)) 