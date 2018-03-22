"""Machine Learning Related Functions.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import numpy as np


def pca(X, n_components=None, p_var_retained=None):
    """Perform PCA on X.

    # Parameters
    X : np.ndarray
        an array of data with (N x num of features)
    n_components : int
        number of components retained
    p_var_retained : float
        percentage of variance retained
        (0 <= p_var_retained <= 1)

    # Returns
    X_reduce : np.ndarray
        the array with reduced dimension.
    R : np.ndarray
        the rotation matrix
    n_retained : np.ndarray
        number of dimensions retained
    """
    n_dim = X.shape[1]
    n_samples = X.shape[0]
    if n_components is not None:
        assert 1 <= n_components <= n_dim
    if p_var_retained is not None:
        assert 0 < p_var_retained <= 1
    else:
        p_var_retained = 0.99

    # compute covariance matrix
    sigma = 1./n_samples*(X.T.dot(X))

    # SVD - find eigen values and vectors
    u, s, v = np.linalg.svd(sigma)

    # transpose
    X_trans = X.dot(u)

    # determine number of components retained
    if n_components is not None:
        n_retained = n_components
    else:
        acc_eig_vals = np.cumsum(s)/np.sum(s)
        n_retained = np.where(acc_eig_vals > p_var_retained)[0][0]
        n_retained = n_retained+1 if n_retained == 0 else n_retained

    # select
    X_reduce = X_trans[:, :n_retained]

    return X_reduce, u, n_retained


def pca_fit(X, R, n_components=None):
    """Perform PCA on X with rotation matrix R.

    # Parameters
    X : np.ndarray
        an array of data with (N x num of features)
    R : np.ndarray
        an rotation matrix
    n_components : int
        number of components retained

    # Returns
    X_reduce : np.ndarray
        the array with reduced dimension.
    """
    n_dim = X.shape[1]
    if n_components is not None:
        assert n_components <= n_dim

    # transpose
    X_trans = X.dot(R)

    # select
    X_reduce = X_trans[:, :n_components]

    return X_reduce
