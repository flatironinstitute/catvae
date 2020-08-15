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
        n = 200
        d = 100
        torch.manual_seed(0)
        self.D1 = torch.rand(d)
        self.D1 = self.D1 / torch.sum(self.D1)
        self.n = n
        self.d = d

        psi = _balance_basis(random_linkage(self.d))[0]
        self.psi = torch.Tensor(psi.copy())

    def test_covariance_matrix(self):
        loc = torch.zeros(self.d - 1)
        dist = MultivariateNormalFactor(loc, self.psi, 1 / self.D1, self.n)
        cov = dist.covariance_matrix
        self.assertEqual(cov.shape, (self.d -1 , self.d - 1))

    def test_precision_matrix(self):
        # tests how accurately the inverse covariance matrix can be computed
        loc = torch.zeros(self.d - 1)
        dist = MultivariateNormalFactor(loc, self.psi, 1 / self.D1, self.n)
        exp = torch.inverse(
            (1 / self.n) * self.psi @ torch.diag(1 / self.D1) @ self.psi.t())
        tt.assert_allclose(exp, dist.precision_matrix,
                           rtol=1, atol=1 / (math.sqrt(self.d)))

    def test_log_det(self):
        loc = torch.zeros(self.d - 1)
        dist = MultivariateNormalFactor(loc, self.psi, 1 / self.D1, self.n)
        cov = dist.covariance_matrix
        exp = torch.log(torch.det(cov))
        res = dist.log_det
        tt.assert_allclose(int(res), int(exp))

    def test_rsample(self):
        loc = torch.ones(self.d - 1)
        dist = MultivariateNormalFactor(loc, self.psi, 1 / self.D1, self.n)
        samples = dist.rsample([10000])
        self.assertAlmostEqual(float(samples.mean()), 1, places=2)

    def test_log_prob(self):
        loc = torch.ones(self.d - 1)
        dist = MultivariateNormalFactor(loc, self.psi, 1 / self.D1, self.n)
        samples = dist.rsample([100])
        logp = dist.log_prob(samples)
        #self.assertEqual(logp.shape, [100])
        self.assertAlmostEqual(int(logp.mean()), -123, places=3)

    def test_entropy(self):
        pass


class TestMultivariateNormalFactorSum(unittest.TestCase):
    def setUp(self):
        n = 200
        d = 100
        k = 4
        torch.manual_seed(0)
        self.W = torch.randn((d - 1, k))
        self.D = torch.rand(k)
        self.P = torch.rand(d)
        self.P = self.P / torch.sum(self.P)
        self.n = n
        self.d = d

        psi = _balance_basis(random_linkage(self.d))[0]
        self.psi = torch.Tensor(psi.copy())

    def test_covariance_matrix(self):
        loc = torch.zeros(self.d - 1)
        dist = MultivariateNormalFactorSum(
            loc, self.psi, 1 / self.P,
            self.W, self.D, self.n)
        cov = dist.covariance_matrix
        self.assertEqual(cov.shape, (self.d -1 , self.d - 1))

    def test_precision_matrix(self):
        # tests how accurately the inverse covariance matrix can be computed
        loc = torch.zeros(self.d - 1)
        dist = MultivariateNormalFactorSum(
            loc, self.psi, 1 / self.P,
            self.W, self.D, self.n)
        sigmaU1 = (1 / self.n) * self.psi @ torch.diag(1 / self.P) @ self.psi.t()
        sigmaU2 = self.W @ torch.diag(self.D) @ self.W.t()
        exp = torch.inverse(sigmaU1 + sigmaU2)
        tt.assert_allclose(exp, dist.precision_matrix,
                           rtol=1, atol=1 / (math.sqrt(self.d)))
    def test_log_det(self):
        loc = torch.zeros(self.d - 1)
        dist = MultivariateNormalFactorSum(
            loc, self.psi, 1 / self.P,
            self.W, self.D, self.n)
        cov = dist.covariance_matrix
        exp = torch.log(torch.det(cov))
        res = dist.log_det
        tt.assert_allclose(res, exp, rtol=1, atol=1)

    def test_rsample(self):
        loc = torch.ones(self.d - 1)
        dist = MultivariateNormalFactorSum(
            loc, self.psi, 1 / self.P,
            self.W, self.D, self.n)
        samples = dist.rsample([10000])
        self.assertAlmostEqual(float(samples.mean()), 1, places=1)

    def test_log_prob(self):
        loc = torch.ones(self.d - 1)
        dist = MultivariateNormalFactorSum(
            loc, self.psi, 1 / self.P,
            self.W, self.D, self.n)
        samples = dist.rsample([100])
        logp = dist.log_prob(samples)
        self.assertEqual(int(logp.mean()), -386)

    def test_entropy(self):
        pass


if __name__ == '__main__':
    unittest.main()
