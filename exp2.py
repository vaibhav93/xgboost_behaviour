#!/usr/bin/python
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
mem = Memory("./mycache")

@mem.cache
def get_data(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def callback(iteration, booster, eval_results):
    print(eval_results)
    return
dtest, ltest = get_data("data/mnist.txt.test")
dtrain, ltrain = get_data("data/mnist.txt.train")

# Training data is 60000x780
dtrain = xgb.DMatrix(dtrain, label=ltrain)

# Test data has 778 features. It is reshaped to 780, since XGBoost throws error
# if features mismatch
dtest = scipy.sparse.csr_matrix((dtest.data, dtest.indices, dtest.indptr), shape=(10000, 780))
dtest = xgb.DMatrix(dtest,label=ltest);

# XGBoost Params
params = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'multi:softmax','num_class':10,'gamma':0.5 }
eta_list = [0.1, 0.3, 0.5, 0.8, 1]

# specify validations set to watch performance
watchlist  = [(dtest,'Test'), (dtrain,'Train')]
num_round = 200
results = []

for eta in eta_list:
    result = {}
    print(eta)
    params['eta'] = eta
    print(params)
    bst = xgb.train(params, dtrain, num_round, watchlist,early_stopping_rounds=10, evals_result=result)
    preds = bst.predict(dtest)
    results.append(result)
# this is prediction


labels = dtest.get_label()
#print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]!=labels[i])) /float(len(preds))))

print(results)
for index,res in enumerate(results):
    plt.plot(res.get('Test').get('merror'),label='s='+str(eta_list[index]))
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Error rate')
plt.show()
# trainPlt = plt.plot(results.get('Train').get('merror'),label='Train accuracy')
# testPlt = plt.plot(results.get('Test').get('merror'),label='Test accuracy')
# plt.legend()
# plt.xlabel('Number of iterations')
# plt.ylabel('Error rate')
# plt.show()
