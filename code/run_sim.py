#####################################################
# Overview
#
# Script to create data, train anomaly detection 
# models and summarize performance.
#
# # Outputs:
#   A dataframe with performance metrics.
#   A dataframe of example data.
#####################################################

# %%
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score

# %%
# Labels completely driven by first two columns
# Four two dimential random varaibles.
# Means at corners of a square. 
# Can append uninformative varaibles too.
# y equal to 1 means outlier. y equal to 0 means typical.
def generate_data(n_samples, n_col_extra = 0):
    SIGMA = np.array([[.15, 0], [0, .15]])
    N_per = int(np.floor(n_samples/4))

    # Quadrants
    Q1 = np.random.multivariate_normal(mean  = (1, 1), cov = SIGMA, size = N_per)
    Q2 = np.random.multivariate_normal(mean  = (-1, 1), cov = SIGMA, size = N_per)
    Q3 = np.random.multivariate_normal(mean  = (-1, -1), cov = SIGMA, size = N_per)
    Q4 = np.random.multivariate_normal(mean  = (1, -1), cov = SIGMA, size = N_per)

    X = np.vstack([Q1, Q2, Q3, Q4])

    if n_col_extra > 0:
        meanVec = np.repeat(0, n_col_extra)
        SIGMA = np.zeros(shape = (n_col_extra, n_col_extra))
        for index in range(0, n_col_extra):
            SIGMA[index,index] = .15
        X_extra = np.random.multivariate_normal(meanVec, SIGMA, n_samples)
        X = np.hstack([X, X_extra])


    y = np.hstack([np.repeat(0, Q1.shape[0]), np.repeat(1, Q2.shape[0]), np.repeat(0, Q3.shape[0]), np.repeat(1, Q4.shape[0])])

    return X, y

# runs sim varying n_col_extra
def run_sim(n_col_extra):
    resultDFs = []
    for iteration in range(0, 10):
        N = 50000

        # train models
        X, y = generate_data(N, n_col_extra)
        X_train = X[y == 0, :]
 
        models = []
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernels:
            model = OneClassSVM(kernel = kernel)
            models.append(model)
        
        # Docs say to model needs to standardization.
        model = make_pipeline(StandardScaler(), SGDOneClassSVM())
        models.append(model)

        distances = ['minkowski', 'euclidean', 'cosine']
        for distance in distances:
            models.append(LocalOutlierFactor(novelty=True, metric = distance, n_jobs = 8))

        for model in models:
            model.fit(X_train)

        # test models
        X_test, y_test = generate_data(N, n_col_extra)
        PPVs = []
        RECALLs = []
        for model in models:
            y_hat = model.predict(X_test)
            y_hat[y_hat == 1] = 0
            y_hat[y_hat == -1] = 1

            if y_hat.max() == 0:
                PPVs.append(0)
            else:
                PPVs.append(precision_score(y_test, y_hat))
            RECALLs.append(recall_score(y_test, y_hat))

        modelNames = ['svm-linear', 'svm-poly', 'svm-rbf', 'svm-sigmoid', 'svm-rbf-approx', 'LOF-minkowski', 'LOF-euclidean', 'LOF-cosine']
        resultDF = {'model':modelNames, 'percision':PPVs, 'recall':RECALLs}
        resultDF = pd.DataFrame(data = resultDF)
        resultDF['n_col_extra'] = n_col_extra
        resultDF['iteration'] = iteration
        resultDF = resultDF[['n_col_extra', 'iteration', 'model', 'percision', 'recall']]
        resultDFs.append(resultDF)

    finalResultDF = pd.concat(resultDFs)
    return finalResultDF

# %%
np.random.seed(0)
pieces = []
for n_col_extra in range(0, 11):
    print(n_col_extra)
    pieces.append(run_sim(n_col_extra))
resultDF = pd.concat(pieces)
resultDF.head()

# %%
resultDF.to_csv(path_or_buf = 'S:\\Python\\projects\\anomaly_detection\\data\\results.csv', index = False)

# %%
np.random.seed(42)
X_test, y_test = generate_data(50000, 0)
exampleDF = {'LABEL':y_test, 'X1':X_test[:,0], 'X2':X_test[:,1]}
exampleDF = pd.DataFrame(data = exampleDF)
exampleDF.head()

# %%
exampleDF.to_csv(path_or_buf = 'S:\\Python\\projects\\anomaly_detection\\data\\exampleDataDF.csv', index = False)

# %%
