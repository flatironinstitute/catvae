import math

import torch
from catvae.distributions import constraints
import torch.distributions.constraints as torch_constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal, lazy_property
import numpy as np
import functools


class MultivariateNormalFactor(Distribution):

    def __init__(self, mu, U, diag, n, validate_args=None):
        """ Asymptotic approximation of the multinomial distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        U : torch.Tensor
            Orthonormal factor matrix for decomposing covariance matrix.
        diag : torch.Tensor
            Diagonal matrix of eigenvalues for covariance decomposition
        n : torch.Tensor
            Number of multinomial observations

        Notes
        -----
        Can incorporate the number of samples in the diagonal
        """
        arg_constraints = {'mu': torch_constraints.real_vector,
                           'U': constraints.left_orthonormal,
                           'diag': torch_constraints.positive,
                           'n': torch_constraints.positive}
        if mu.dim() < 1:
            raise ValueError("`mu` must be at least one-dimensional.")
        d = U.shape[-1]
        if mu.shape[-1] != d - 1:
            raise ValueError(f"The last dimension of `mu` must be {d-1}")

        self.mu = mu
        self.U = U
        self.S = diag
        self.n = n
        batch_shape, event_shape = self.mu.shape[:-1], self.mu.shape[-1:]
        super(MultivariateNormalFactor, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    @property
    def covariance_matrix(self):
        return (1 / self.n) * self.U @ torch.diag(self.S) @ self.U.t()

    @property
    def precision_matrix(self):
        return (self.n) * self.U @ torch.diag(1 / self.S) @ self.U.t()

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        raise NotImplementedError('`variance` is not implemented.')

    @functools.cached_property
    def cholesky(self):
        cov = self.covariance_matrix
        return torch.cholesky(cov)

    @property
    def log_det(self):
        d = self.U.shape[-1]
        return torch.sum(torch.log(self.S)) + d * np.log(1 / self.n)

    def rsample(self, sample_shape):
        """ Eigenvalue decomposition can also be used for sampling
        https://stats.stackexchange.com/a/179275/79569
        """
        L = self.cholesky
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.mu.dtype, device=self.mu.device)
        return self.mu + _batch_mv(L, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.mu
        L = self.cholesky
        M = _batch_mahalanobis(L, diff)
        half_log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        p = - half_log_det - 0.5 * (
            self._event_shape[0] * math.log(2 * math.pi) + M
        )
        return p

    def entropy(self):
        raise NotImplementedError('`entropy` is not implemented.')


class MultivariateNormalFactorSum(Distribution):

    def __init__(self, mu, U1, diag1, U2, diag2, n, validate_args=None):
        """ Multivariate normal distribution parameterized as the
            sum of two normal distributions whose covariances can be
            represented as an eigenvalue decomposition.

        Parameters
        ----------
        mu1 : torch.Tensor
            Mean of the first distribution
        U1 : torch.Tensor
            Left orthonormal factor matrix for decomposing the
            first covariance matrix.
        diag1 : torch.Tensor
        `    Diagonal matrix of eigenvalues for the
            first covariance decomposition
        mu2 : torch.Tensor
            Mean of the second distribution
        U2 : torch.Tensor
            Left orthonormal factor matrix for decomposing the
            second covariance matrix.
        diag2 : torch.Tensor
            Diagonal matrix of eigenvalues for the
            second covariance decomposition
        n : torch.Tensor
            Number of multinomial observations.
        """
        arg_constraints = {'mu': torch_constraints.real_vector,
                           'U1': constraints.left_orthonormal,
                           'diag1': torch_constraints.positive,
                           'U2': torch_constraints.real_vector,
                           'diag2': torch_constraints.positive,
                           'n': torch_constraints.positive
        }

        if mu.dim() < 1:
            raise ValueError("`mu` must be at least one-dimensional.")
        d = U1.shape[-1]
        if mu.shape[-1] != d - 1:
            raise ValueError(f"The last dimension of `mu` must be {d-1}")

        self.mu = mu
        self.U1 = U1
        self.S1 = diag1
        self.U2 = U2
        self.S2 = diag2
        self.n = n
        batch_shape, event_shape = self.mu.shape[:-1], self.mu.shape[-1:]
        super(MultivariateNormalFactorSum, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    @property
    def covariance_matrix(self):
        sigmaU1 = (1 / self.n) * self.U1 @ torch.diag(self.S1) @ self.U1.t()
        sigmaU2 = self.U2 @ torch.diag(self.S2) @ self.U2.t()
        return sigmaU1 + sigmaU2

    @property
    def precision_matrix(self):
        invS1 = self.n * self.U1 @ torch.diag(1 / self.S1) @ self.U1.t()
        W = self.U2
        invD = torch.diag(1 / self.S2)

        # Woodbury identity
        C = torch.inverse(invD + W.t() @ invS1 @ W)
        invS = invS1 - invS1 @ W @ C @ W.t() @ invS1
        return invS

    @property
    def log_det(self):
        # Matrix determinant lemma, similar to the Woodbury identity
        invS1 = self.n * self.U1 @ torch.diag(1 / self.S1) @ self.U1.t()
        W = self.U2
        d = self.U1.shape[-1]
        invD = torch.diag(1 / self.S2)
        logdet_A = torch.sum(torch.log(self.S1)) + d * np.log(1 / self.n)
        logdet_C = torch.log(torch.det(invD + W.t() @ invS1 @ W))
        logdet_D = torch.sum(torch.log(self.S2))
        return logdet_A + logdet_C + logdet_D

    @property
    def mean(self):
        return self.mu

    @functools.cached_property
    def cholesky(self):
        cov = self.covariance_matrix
        return torch.cholesky(cov)

    @property
    def variance(self):
        raise NotImplementedError('`variance` is not implemented.')

    def rsample(self, sample_shape):
        """ Eigenvalue decomposition can also be used for sampling
        https://stats.stackexchange.com/a/179275/79569
        """
        L = self.cholesky
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.mu.dtype, device=self.mu.device)
        return self.mu + _batch_mv(L, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.mu
        L = self.cholesky
        M = _batch_mahalanobis(L, diff)
        half_log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        p = - half_log_det - 0.5 * (
            self._event_shape[0] * math.log(2 * math.pi) + M
        )
        return p

    def entropy(self):
        raise NotImplementedError('`entropy` is not implemented.')
