#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

spaces = [1, 1.2, 1.5, 1.7, 2]

# XGBoost Params
param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'multi:softmax','num_class':4 }

# specify validations set to watch performance

num_round = 50
results = []

for sep in spaces:
    X,y = datasets.make_classification(n_samples=20000, n_features=20, n_informative=15, n_classes=4,class_sep=sep)
    result = {}
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    # Training data is 60000x780
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)
    watchlist  = [(dtest,'Test'), (dtrain,'Train')]
    bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=10, evals_result=result)
    result['size'] = X_train.shape[0]
    preds = bst.predict(dtest)
    results.append(result)
    
# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()

print(results)

plt.bar([i for i in spaces],[i.get('Test').get('merror')[-1] for i in results],align='center',width=0.1)
plt.legend()
plt.xticks([i for i in spaces])
plt.xlabel('Class separation')
plt.ylabel('Error rate')
plt.show()
