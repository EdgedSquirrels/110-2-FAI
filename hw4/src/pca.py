from pandas import DataFrame
from torch import Tensor
from data_transformer import DataTransformer
import numpy as np
import torch


class PCA(DataTransformer):
    """
    Add more of your code here if you want to
    Parameters
    ----------
    n_components : int
        Number of components to keep.
    """
    def __init__(self, args):
        self.n_components = args.latent_space_dim

    def fit(self, X:Tensor):
        self.Xmean = X.mean(axis = 0)
        X2 = X - self.Xmean
        # print(X.shape, self.Xmean.shape)
        L, V = torch.linalg.eigh(X2.T @ X2)
        # print(L)
        self.transform_matrix = V[:, -self.n_components:].T
        # print(V.shape, self.transform_matrix.shape)

        # plot_component(X[0], "0")
        # raise NotImplementedError
    
    def transform(self, X):
        # project onto the eigenface
        A = self.transform_matrix
        tmp = (X - self.Xmean) @ A.T
        return tmp

        # raise NotImplementedError

    
    def reconstruct(self, X_transformed):
        """
        Reconstruct the transformed X
        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_components)
            of reconstructed values.
        """
        return X_transformed @ self.transform_matrix + self.Xmean
        # raise NotImplementedError
