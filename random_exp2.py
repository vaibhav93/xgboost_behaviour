#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_perc = [0.9, 0.8, 0.6, 0.4]
#Generate dataset
X,y = datasets.make_classification(n_samples=20000, n_features=20, n_informative=15, n_classes=4)


# XGBoost Params
param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'multi:softmax','num_class':4 }

# specify validations set to watch performance

num_round = 400
results = []

for train_size in train_perc:
    result = {}
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)
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
# for index,res in enumerate(results):
#     plt.plot(res.get('Test').get('merror'),label='training size='+str(train_perc[index]*res.get('size')))
print([i.get('Test').get('merror')[-1] for i in results])
print([i*50000 for i in train_perc])
plt.bar([i*50000 for i in train_perc],[i.get('Test').get('merror')[-1] for i in results],align='center',width=1000)
plt.legend()
plt.xticks([i*50000 for i in train_perc])
plt.xlabel('Number of traning instances')
plt.ylabel('Error rate')
plt.show()
