import os

if __name__ == "__main__":
    # NOTE: on posix systems, this *has* to be here and in the
    # `__name__ == "__main__"` clause to run XGBoost in parallel processes
    # using fork, if XGBoost was built with OpenMP support. Otherwise, if you
    # build XGBoost without OpenMP support, you can use fork, which is the
    # default backend for joblib, and omit this.
    try:
        from multiprocessing import set_start_method
    except ImportError:
        raise ImportError("Unable to import multiprocessing.set_start_method."
                          " This example only runs on Python 3.4")
    set_start_method("forkserver")

    from sklearn.externals.joblib import Memory
    from sklearn.datasets import load_svmlight_file
    import scipy.sparse

    mem = Memory("./mycache")
    @mem.cache
    def get_data(path):
        data = load_svmlight_file(path)
        return data[0], data[1]


    def callback(iteration, booster, eval_results):
        print(eval_results)
        return
    dtest, ltest = get_data("data/mnist.txt.test")
    dtest = scipy.sparse.csr_matrix((dtest.data, dtest.indices, dtest.indptr), shape=(10000, 780))
    dtrain, ltrain = get_data("data/mnist.txt.train")
    
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import load_boston
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    import time
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(31337)

    print("Parallel Parameter optimization")

    threads = [1,2,3,4]
    os.environ["OMP_NUM_THREADS"] = "1"  # or to whatever you want

    start = time.time()
    results = []
    for num_threads in threads:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        xgb_model = xgb.XGBClassifier(max_depth=5,n_jobs=num_threads,objective='multi:softmax', n_estimators=3,num_class=10).fit(dtrain, ltrain, eval_set=[(dtest,ltest)],early_stopping_rounds=10)
        predictions = xgb_model.predict(dtest)
        elapsed = time.time() - start
        print('Time Elapsed ' + str(elapsed) + ' seconds')
        results.append({'threads':num_threads,'time':elapsed})
        actuals = ltest
        print(accuracy_score(actuals,predictions))
        
    print(results)
    plt.bar(threads, [result['time'] for result in results], align='center')
    #plt.plot(threads,[result['time'] for result in results])
    plt.legend()
    plt.xticks(threads)
    plt.xlabel('Number of threads')
    plt.ylabel('Training time in seconds')
    plt.show()
    

         
