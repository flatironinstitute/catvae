import unittest
from catvae.distributions.mvn import MultivariateNormalFactor
from catvae.distributions.mvn import MultivariateNormalFactorSum
import torch
import torch.testing as tt
from gneiss.balances import _balance_basis
from gneiss.cluster import random_linkage
import math


class TestMultivariateNormalFactor(unittest.TestCase):
    def setUp(self):
        n = 100
        torch.manual_seed(0)
        self.U1 = torch.randn((n, n))
        self.D1 = torch.rand(n)
        self.D1 = self.D1 / torch.sum(self.D1)
        self.n = n

        psi = _balance_basis(random_linkage(self.n))[0]
        self.psi = torch.Tensor(psi.copy())

    def test_precision_matrix(self):
        # tests how accurately the inverse covariance matrix can be computed
        loc = torch.zeros(self.n)
        dist = MultivariateNormalFactor(loc, self.psi, self.D1)
        exp = torch.inverse(self.psi @ torch.diag(1 / self.D1) @ self.psi.t())
        tt.assert_allclose(exp, dist.precision_matrix,
                           rtol=1,
                           atol=1/(self.n * math.sqrt(self.n)))

    def test_rsample(self):
        pass

    def test_log_prob(self):
        pass

    def test_entropy(self):
        pass


class TestMultivariateNormalFactorSum(unittest.TestCase):
    def setUp(self):
        pass

    def test_mean(self):
        pass

    def test_variance(self):
        pass

    def test_covariance_matrix(self):
        pass

    def test_precision_matrix(self):
        pass

    def test_rsample(self):
        pass

    def test_log_prob(self):
        pass

    def test_entropy(self):
        pass


if __name__ == '__main__':
    unittest.main()
