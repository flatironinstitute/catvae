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
import numpy as np
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
        np.random.seed(0)
        torch.manual_seed(0)
        torch.random.initial_seed()

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
        gam = torch.Tensor([25000.])
        res = expectation_mvn_factor_sum_multinomial(
            q, self.psi.t(), x, gam)
        self.assertFalse(np.isinf(float(res)))
        self.assertGreater(float(exp), float(res))

    def test_expectation_joint_mvn_factor_mvn_factor_sum(self):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.random.initial_seed()
        s2 = 2
        samples = 10000
        loc = torch.ones(self.d - 1)  # logit units
        std = torch.ones(self.d - 1) * s2
        qeta = MultivariateNormalFactorSum(
            self.W @ self.V @ self.hx,
            self.psi, 1 / self.P,
            self.W, self.D, self.n)
        #qz = MultivariateNormalFactor(loc, self.W, self.D, 1)
        print(torch.diag(self.D))
        qz = MultivariateNormal(
            self.V @ self.hx,
            scale_tril=torch.diag(torch.sqrt(self.D)))
        loc = torch.zeros(self.d - 1)
        s2 = torch.Tensor([s2])

        # MC samples for validation
        eta = qeta.rsample([samples])
        z = qz.rsample([samples])
        lp = Normal(z @ self.W.t(), std).log_prob(eta)  # p log likelihood
        exp = torch.mean(lp)
        res = expectation_joint_mvn_factor_mvn_factor_sum(qeta, qz, s2)
        self.assertAlmostEqual(float(exp), float(res))

    def test_expectation_mvn_factor_sum_mvn_factor_sum(self):
        pass


if __name__ == '__main__':
    unittest.main()
