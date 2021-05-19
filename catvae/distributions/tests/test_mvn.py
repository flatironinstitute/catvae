import unittest
from catvae.distributions.mvn import MultivariateNormalFactorIdentity
from torch.distributions import MultivariateNormal
from catvae.distributions.utils import seed_all
import torch
import torch.testing as tt
import math


class TestMultivariateNormalFactorIdentity(unittest.TestCase):
    def setUp(self):
        n = 200
        d = 100
        k = 4
        seed_all(0)
        self.W = torch.randn((d, k))
        self.D = torch.rand(k)
        self.n = n
        self.d = d
        self.s2 = torch.Tensor([3])
        self.Id = torch.eye(self.d)

    def test_covariance_matrix(self):
        loc = torch.zeros(self.d)
        exp = (self.W @ torch.diag(self.D) @ self.W.t() +
               self.s2 * self.Id)
        dist = MultivariateNormalFactorIdentity(
            loc, self.s2, self.D, self.W)
        cov = dist.covariance_matrix
        self.assertEqual(cov.shape, (self.d, self.d))
        tt.assert_allclose(exp, cov)

    def test_precision_matrix(self):
        # tests how accurately the inverse covariance matrix can be computed
        loc = torch.zeros(self.d)
        dist = MultivariateNormalFactorIdentity(
            loc, self.s2, self.D, self.W)
        exp = torch.inverse(self.W @ torch.diag(self.D) @ self.W.t() +
                            self.s2 * self.Id)
        tt.assert_allclose(exp, dist.precision_matrix,
                           rtol=1, atol=1 / (math.sqrt(self.d)))

    def test_log_det(self):
        loc = torch.zeros(self.d)
        dist = MultivariateNormalFactorIdentity(
            loc, self.s2, self.D, self.W)
        cov = dist.covariance_matrix
        res = dist.log_det
        exp = torch.slogdet(cov)[1]
        tt.assert_allclose(res, exp)

    def test_rsample(self):
        loc = torch.ones(self.d)
        dist = MultivariateNormalFactorIdentity(
            loc, self.s2, self.D, self.W)
        samples = dist.rsample([10000])
        self.assertAlmostEqual(float(samples.mean()), 1, places=2)

    def test_log_prob(self):
        loc = torch.ones(self.d)

        wdw = self.W @ torch.diag(self.D) @ self.W.t()
        sI = self.s2 * self.Id
        sigma = sI + wdw
        dist2 = MultivariateNormal(loc, covariance_matrix=sigma)
        samples = dist2.rsample([10000])
        exp_logp = dist2.log_prob(samples)

        dist1 = MultivariateNormalFactorIdentity(
            loc, self.s2, self.D, self.W)
        res_logp = dist1.log_prob(samples)

        self.assertAlmostEqual(float(exp_logp.mean()),
                               float(res_logp.mean()), places=3)

    def test_entropy(self):
        pass


if __name__ == '__main__':
    unittest.main()
