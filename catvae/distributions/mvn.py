import math

import torch
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal
import numpy as np
import functools


torch.pi = torch.Tensor([torch.acos(torch.zeros(1)).item() * 2])


def _batch_mahalanobis_factor(L_inv, x):
    r"""
    Computes the squared Mahalanobis distance
    :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.
    """
    xL = (x @ L_inv)
    xLxt = (xL * x).sum(-1)
    return xLxt


def sparse_identity(d):
    # i = torch.arange(d)
    # idx = torch.stack((i, i))
    # v = torch.ones(d)
    # Id = torch.sparse_coo_tensor(idx, v, requires_grad=False)
    Id = torch.eye(d)
    return Id


class MultivariateNormalFactor(Distribution):

    def __init__(self, mu, U, diag, n, validate_args=False):
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

        Important : this cannot handle batching
        """
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
        eps = _standard_normal(shape, dtype=self.mu.dtype,
                               device=self.mu.device)
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

    def __init__(self, mu, U1, diag1, U2, diag2, n, validate_args=False):
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
        if len(self.S1.shape) == 1:
            P = torch.diag(self.S1)
            invN = (1 / self.n)
        elif len(self.S1.shape) == 2:
            P = torch.stack([
                torch.diag(self.S1[i, :].squeeze())
                for i in range(self.S1.shape[0])
            ], dim=0)
            invN = (1 / self.n).unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f'Cannot handle dimensions {self.S1.shape}')

        sigmaU1 = invN * (self.U1 @ P @ self.U1.t())
        sigmaU2 = self.U2 @ torch.diag(self.S2) @ self.U2.t()
        return sigmaU1 + sigmaU2

    @property
    def precision_matrix(self):
        if len(self.S1.shape) == 1:
            invP = torch.diag(1 / self.S1)
            n = (1 / self.n)
        elif len(self.S1.shape) == 2:
            invP = torch.stack([
                torch.diag(1 / self.S1[i, :].squeeze())
                for i in range(self.S1.shape[0])
            ], dim=0)
            n = (1 / self.n).unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f'Cannot handle dimensions {self.S1.shape}')

        invS1 = n * self.U1 @ invP @ self.U1.t()
        W = self.U2
        invD = torch.diag(1 / self.S2)

        # Woodbury identity
        C = torch.inverse(invD + W.t() @ invS1 @ W)
        invS = invS1 - invS1 @ W @ C @ W.t() @ invS1
        return invS

    @property
    def log_det(self):
        if len(self.S1.shape) == 1:
            invP = torch.diag(1 / self.S1)
            n = (1 / self.n)
        elif len(self.S1.shape) == 2:
            invP = torch.stack([
                torch.diag(1 / self.S1[i, :].squeeze())
                for i in range(self.S1.shape[0])
            ], dim=0)
            n = (1 / self.n).unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f'Cannot handle dimensions {self.S1.shape}')

        # Matrix determinant lemma, similar to the Woodbury identity
        invS1 = n * self.U1 @ invP @ self.U1.t()
        W = self.U2
        d = self.U1.shape[-1]
        invD = torch.diag(1 / self.S2)
        logdet_A = torch.sum(torch.log(self.S1)) + d * torch.log(1 / self.n)
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
        eps = _standard_normal(shape, dtype=self.mu.dtype,
                               device=self.mu.device)
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
        d = self.mu.shape[-1]
        half_logdet = self.log_det / 2
        return half_logdet + (d / 2) * (torch.log(2 * torch.pi) + 1)


class MultivariateNormalFactorIdentity(Distribution):

    def __init__(self, mu, sigma2, D, W, validate_args=False):
        """ Multivariate normal distribution with the form
            N(mu, sigma*I + W'DW)"""
        if mu.dim() < 1:
            raise ValueError("`mu` must be at least one-dimensional.")
        d = W.shape[0]
        self.mu = mu
        self.sigma2 = sigma2
        self.D = D
        self.W = W
        self.d = d
        batch_shape, event_shape = self.mu.shape[:-1], self.mu.shape[-1:]
        super(MultivariateNormalFactorIdentity, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    @property
    def covariance_matrix(self):
        wdw = self.W @ torch.diag(self.D) @ self.W.t()
        idx = torch.arange(self.d)
        wdw[idx, idx] += self.sigma2
        return wdw

    @property
    def precision_matrix(self):
        # Woodbury identity
        # inv(A + WDWt) = invA - invA @ W inv(invD + Wt invA W) Wt invA
        W, D = self.W, self.D
        invD = torch.diag(1 / D)
        invAW = W / self.sigma2
        C = invD + W.t() @ invAW
        invC = torch.inverse(C)
        cor = invAW @ invC @ invAW.t()
        idx = torch.arange(self.d)
        cor[idx, idx] -= (1 / self.sigma2)
        return -cor

    @property
    def log_det(self):
        # Matrix determinant lemma
        # det(A + WDWt) = det(invD + Wt invA W) det(D) det (A)
        W = self.W
        invD = torch.diag(1 / self.D)
        invAW = W / self.sigma2
        logdet_A = torch.log(self.sigma2) * self.d
        logdet_C = torch.slogdet(invD + W.t() @ invAW)[1]
        logdet_D = torch.sum(torch.log(self.D))
        res = logdet_A + logdet_C + logdet_D
        return res

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
        L = self.cholesky
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.mu.dtype,
                               device=self.mu.device)
        return self.mu + _batch_mv(L, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.mu
        # L = self.W @ torch.diag(torch.sqrt(self.D))
        M = _batch_mahalanobis_factor(self.precision_matrix, diff)
        p = - 0.5 * self.log_det - 0.5 * (
            self._event_shape[0] * math.log(2 * math.pi) + M
        )
        return p

    def entropy(self):
        pass
