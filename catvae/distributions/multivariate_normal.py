import math

import torch
from catvae.distributions import constraints
from torch.distributions.distribution import Distribution, MultivariateNormal
from torch.distributions.utils import _standard_normal, lazy_property


class MultivariateNormalFactor(MultivariateNormal):

    def __init__(self, loc, U, diag, V):
        """

        Parameters
        ----------
        loc : torch.Tensor
            Mean of the distribution
        U : torch.Tensor
            Left orthonormal factor matrix for decomposing covariance matrix.
        diag : torch.Tensor
            Diagonal matrix of eigenvalues for covariance decomposition
        V : torch.Tensor
            Right orthonormal factor matrix for decomposing covariance matrix.
        """
        arg_constraints = {'loc': constraints.real_vector,
                           'U': constraints.left_orthonormal,
                           'diag': constraints.real_vector
                           'V': constraints.left_orthonormal}

    @property
    def covariance_matrix(self):
        pass

    @property
    def precision_matrix(self):
        pass

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def rsample(self, sample_shape):
        """ Eigenvalue decomposition can also be used for sampling
        https://stats.stackexchange.com/a/179275/79569
        """
        pass

    def log_prob(self, value):
        pass

    def entropy(self):
        pass


class MultivariateNormalFactorSum(MultivariateNormal):

    def __init__(self, loc,
                 U1, diag1, V1,
                 U2, diag2, V2
    ):
        """

        Parameters
        ----------
        loc : torch.Tensor
            Mean of the distribution
        U : torch.Tensor
            Left orthonormal factor matrix for decomposing covariance matrix.
        diag : torch.Tensor
            Diagonal matrix of eigenvalues for covariance decomposition
        V : torch.Tensor
            Right orthonormal factor matrix for decomposing covariance matrix.
        """
        arg_constraints = {'loc': constraints.real_vector,
                           'U1': constraints.left_orthonormal,
                           'diag1': constraints.real_vector
                           'V1': constraints.left_orthonormal,
                           'U2': constraints.real_vector,
                           'diag2': constraints.real_vector
                           'V2': constraints.real_vector
        }

    @property
    def covariance_matrix(self):
        pass

    @property
    def precision_matrix(self):
        pass

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def rsample(self, sample_shape):
        """ Eigenvalue decomposition can also be used for sampling
        https://stats.stackexchange.com/a/179275/79569
        """
        pass

    def log_prob(self, value):
        pass

    def entropy(self):
        pass
