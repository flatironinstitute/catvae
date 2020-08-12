import math

import torch
from catvae.distributions import constraints
from torch.distributions.distribution import Distribution, MultivariateNormal
from torch.distributions.utils import _standard_normal, lazy_property


class MultivariateNormalFactor(MultivariateNormal):

    def __init__(self, loc, U, diag):
        """

        Parameters
        ----------
        loc : torch.Tensor
            Mean of the distribution
        U : torch.Tensor
            Orthonormal factor matrix for decomposing covariance matrix.
        diag : torch.Tensor
            Diagonal matrix of eigenvalues for covariance decomposition
        """
        arg_constraints = {'loc': constraints.real_vector,
                           'U': constraints.left_orthonormal,
                           'diag': constraints.positive}
        self.loc = loc
        self.U = U
        self.S = S

    @property
    def covariance_matrix(self):
        return self.U @ torch.diag(self.S) @ self.U.t()

    @property
    def precision_matrix(self):
        return self.U @ torch.diag(1 / self.S) @ self.U.t()

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        # TODO: not sure what this should be
        pass

    def rsample(self, sample_shape=):
        """ Eigenvalue decomposition can also be used for sampling
        https://stats.stackexchange.com/a/179275/79569
        """
        D = torch.diag(torch.sqrt(S))
        L = self.U @ D
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(L, eps)

    def log_prob(self, value):
        # TODO
        pass

    def entropy(self):
        # TODO
        pass


class MultivariateNormalFactorSum(MultivariateNormal):

    def __init__(self, loc, U1, diag1, U2, diag2
    ):
        """ Multivariate normal distribution parameterized as the
            sum of two normal distributions whose covariances can be
            represented as an eigenvalue decomposition.

        Parameters
        ----------
        loc : torch.Tensor
            Mean of the distribution
        U1 : torch.Tensor
            Left orthonormal factor matrix for decomposing the
            first covariance matrix.
        diag1 : torch.Tensor
            Diagonal matrix of eigenvalues for the
            first covariance decomposition
        U2 : torch.Tensor
            Left orthonormal factor matrix for decomposing the
            second covariance matrix.
        diag2 : torch.Tensor
            Diagonal matrix of eigenvalues for the
            second covariance decomposition
        """
        arg_constraints = {'loc': constraints.real_vector,
                           'U1': constraints.left_orthonormal,
                           'diag1': constraints.positive,
                           'U2': constraints.real_vector,
                           'diag2': constraints.positive
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
