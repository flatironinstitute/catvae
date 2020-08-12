import unittest
from catvae.distributions.multivariate_normal import MultivariateNormalFactor
from catvae.distributions.multivariate_normal import MultivariateNormalFactorSum
import torch
import torch.testing as tt
from gneiss.balances import _balance_basis
from gneiss.cluster import random_linkage



class TestMultivariateNormalFactor(unittest.TestCase):
    def setUp(self):
        n = 10
        torch.manual_seed(0)
        self.U1 = torch.random.randn((n, n))
        self.D1 = torch.random.random(n)
        self.n = n

        psi = _balance_basis(random_linkage(self.n))[0]
        self.psi = torch.Tensor(psi)

    def test_precision_matrix(self):
        # tests how accurately the inverse covariance matrix can be computed
        dist = MultivariateNormalFactor()
        exp = torch.inverse(self.psi @ torch.diag(D1) @ self.psi.t())
        tt.assert_allclose(exp, dist.precision_matrix)

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
