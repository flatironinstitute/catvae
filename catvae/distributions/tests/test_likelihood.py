from catvae.distributions.likelihood import (
    expectation_mvn_factor_sum_multinomial,
    expectation_joint_mvn_factor_mvn_factor_sum,
    expectation_mvn_factor_sum_mvn_factor_sum
)
import torch
from torch.distributions import Multinomial, MultivariateNormal
from catvae.distributions.mvn import MultivariateNormalFactor
from catvae.distributions.mvn import MultivariateNormalFactorSum
from gneiss.balances import _balance_basis
from gneiss.cluster import random_linkage

import torch.testing as tt
import unittest


class TestExpectations(unittest.TestCase):
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

    def test_expectation_mvn_factor_sum_multinomial(self):
        torch.manual_seed(0)
        loc = torch.ones(self.d - 1)
        q = MultivariateNormalFactorSum(
            loc, self.psi, 1 / self.P,
            self.W, self.D, self.n)

        x = torch.ones((1, self.d))
        # MC sampling to get an estimate of true expectation
        samples = 10000
        eta = q.rsample([samples])
        logits = eta @ self.psi
        p = Multinomial(total_count=self.n, logits=logits)
        lp = p.log_prob(x)
        exp = lp.mean()
        gam = torch.Tensor([1.])
        res = expectation_mvn_factor_sum_multinomial(
            q, self.psi.t(), x, gam)
        self.assertGreater(float(exp), float(res))

    def test_expectation_joint_mvn_factor_mvn_factor_sum(self):
        pass

    def test_expectation_mvn_factor_sum_mvn_factor_sum(self):
        pass


if __name__ == '__main__':
    unittest.main()
