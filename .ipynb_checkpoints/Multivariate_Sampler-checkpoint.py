# Importing libraries and packages
# We will import scipy linear algebra to do Cholesky decomposition of the covariance matrix to transform standard
# Gaussian noise into samples from multivariate normal distribution with an arbitrary mean and variance.

import numpy as np
import scipy.linalg as scla
import time
from scipy.spatial.distance import cdist

def mult_normal_sampl(mu,covariance,no_of_samples):
    L = scla.cholesky(covariance)    # cholesky decomposition of covariance matrix.
    Z = np.random.normal(size=(no_of_samples, covariance.shape[0]))    # std normal matrix of size (no_of_samples x no_of_row_of_cov_matrix)
    return Z.dot(L) + mu    # Returns multivariate samples witn mean=mu and covariance.

mu = np.array([1., -5.])
covariance = np.array([[1., 0.3], [0.3, 2.]])
X = mult_normal_sampl(mu,covariance,no_of_samples=100)
print(X.mean(axis=0))
print(np.cov(X.T))

X = np.random.normal(size=(500,1))
K = np.exp(-cdist(X, X, "sqeuclidean")) + 1e-6 * np.eye(X.shape[0])
mu = np.zeros((X.shape[0],))

time.sleep(1.)
start = time.time()
samples = np.random.multivariate_normal(mu, K, size = (10000,))
end = time.time()
print(f"Time Elapsed using built-in numpy function: {end-start}")

time.sleep(1.)
start = time.time()
samples = mult_normal_sampl(mu, K, 10000)
end = time.time()
print(f"Time Elapsed using our function: {end-start}")

print(K.shape)