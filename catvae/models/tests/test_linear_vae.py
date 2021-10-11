import unittest
import torch
from catvae.models import LinearVAE, LinearBatchVAE
import numpy as np
import numpy.testing as npt


class TestLinearVAE(unittest.TestCase):

    def test_sample(self):
        k, D = 10, 40
        model = LinearVAE(D, k)
        x = torch.zeros(2, D)
        s = model.sample(x).detach().numpy()
        npt.assert_allclose(s.shape, np.array([2, k]))


class TestLinearBatchVAE(unittest.TestCase):
    @unittest.skip('This is currently not supported.')
    def test_sample(self):
        k, C, D = 10, 3, 40
        prior = torch.randn(C, D)
        model = LinearBatchVAE(D, k, k, C, batch_prior=prior)
        x = torch.zeros(2, D)
        b = torch.Tensor([0, 1]).long()
        s = model.sample(x, b).detach().numpy()
        npt.assert_allclose(s.shape, np.array([2, k]))


if __name__ == '__main__':
    unittest.main()
