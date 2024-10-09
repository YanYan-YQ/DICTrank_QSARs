#!/account/yqu/anaconda3/bin/python

import sys
#var=sys.argv[1]


import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed
seed(1)

### import scripts
import single_knn
import single_lr
import single_svm
import single_rf
import single_xgboost

### please update the following path 

#define path
base_path = '/compute01/yqu/dictrank/results/singleTest_prediction' 

#read the best hyperparameters data
bestParaDf = pd.read_csv('/compute01/yqu/dictrank/results/summarizedPerforamance_paraSelection/best_para_train_result.csv')

# each method prediction in single dataset 
def runSingleDataset(X, X_test, y, y_test, base_path, descriptor, bestParaDf):
    for method in ['knn', 'lr', 'svm', 'rf', 'xgboost']:
        var = bestParaDf[(bestParaDf['descriptor']==descriptor) & (bestParaDf['method']==method)]['feature'].str.split('_').str[1].values[0]    
        if method == 'knn':
            single_knn.para_selection(var, X, X_test, y, y_test, base_path, descriptor)
        elif method == 'lr':
            single_lr.para_selection(var, X, X_test, y, y_test, base_path, descriptor)
        elif method == 'svm':
            single_svm.para_selection(var, X, X_test, y, y_test, base_path, descriptor)
        elif method == 'rf':
            single_rf.para_selection(var, X, X_test, y, y_test, base_path, descriptor)
        elif method == 'xgboost':
            single_xgboost.para_selection(var, X, X_test, y, y_test, base_path, descriptor)
        
         
for descriptor in ['mold2']:
       if descriptor == 'mold2':
       DICT = pd.read_csv('/compute01/yqu/dictrank/data/dictmold2_777.csv')
       colsN = 777
    print(DICT.shape)
    
    #get the feature columns
    cols=DICT.columns[-colsN:]
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
    
    runSingleDataset(X, X_test, y, y_test, base_path, descriptor, bestParaDf)
        

print("--- %s seconds ---" % (time.time() - start_time))

