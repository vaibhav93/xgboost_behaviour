#!/usr/bin/python
import numpy as np
import time
import scipy.sparse
import xgboost as xgb
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
if __name__ == "__main__": 
    os.environ["OMP_NUM_THREADS"] = "2"
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

    tree_methods = ['exact','approx']
    # XGBoost Params
    params = {'max_depth':5, 'eta':0.3, 'silent':1, 'objective':'multi:softmax','num_class':10,'n_jobs':2, 'gamma':0.5}

    # specify validations set to watch performance
    watchlist  = [(dtest,'Test')]
    num_round = 20
    results = []

    for method in tree_methods:
        result = {}
        params['tree_method'] = method
        start = time.time()
        bst = xgb.train(params,dtrain, num_round, watchlist,early_stopping_rounds=10, evals_result=result)
        preds = bst.predict(dtest)
        elapsed = time.time() - start
        print('Elapsed ' + str(elapsed) + ' seconds')
        results.append(result)

    labels = dtest.get_label()

    for index,res in enumerate(results):
        plt.plot(res.get('Test').get('merror'),label='Method= ' + tree_methods[index])
        plt.legend()
        plt.xlabel('Number of iterations')
        plt.ylabel('Error rate')
        plt.show()

        
