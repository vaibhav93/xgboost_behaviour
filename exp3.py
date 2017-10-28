#!/usr/bin/python
import numpy as np
import scipy.sparse
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
print(dtrain.num_row())
# Splice training data in various sizes. Total size is 60000
trainingMatrices = []
trainingMatrices.append(dtrain.slice(np.random.randint(low=0,high=60000,size=1000)))
trainingMatrices.append(dtrain.slice(np.random.randint(low=0,high=60000,size=5000)))
trainingMatrices.append(dtrain.slice(np.random.randint(low=0,high=60000,size=10000)))
trainingMatrices.append(dtrain.slice(np.random.randint(low=0,high=60000,size=25000)))
trainingMatrices.append(dtrain.slice(np.random.randint(low=0,high=60000,size=50000)))

# Test data has 778 features. It is reshaped to 780, since XGBoost throws error
# if features mismatch
dtest = scipy.sparse.csr_matrix((dtest.data, dtest.indices, dtest.indptr), shape=(10000, 780))
dtest = xgb.DMatrix(dtest,label=ltest);

# XGBoost Params
params = {'max_depth':5, 'eta':0.3, 'silent':1, 'objective':'multi:softmax','num_class':10,'gamma':0.5 }

# specify validations set to watch performance
watchlist  = [(dtest,'Test'), (dtrain,'Train')]
num_round = 200
results = []

for mat in trainingMatrices:
    result = {}
    print(params)
    bst = xgb.train(params, mat, num_round, watchlist,early_stopping_rounds=10, evals_result=result)
    preds = bst.predict(dtest)
    results.append(result)

labels = dtest.get_label()

for index,res in enumerate(results):
    plt.plot(res.get('Test').get('merror'),label='Size='+str(trainingMatrices[index].num_row()))
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Error rate')
plt.show()
