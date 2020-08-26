import torch
import torch.nn as nn
from gneiss.cluster import random_linkage
from gneiss.balances import _balance_basis
from catvae.composition import closure, ilr
from catvae.distributions.likelihood import (
    expectation_mvn_factor_sum_multinomial_taylor,
    expectation_mvn_factor_sum_multinomial_bound,
    expectation_joint_mvn_factor_mvn_factor_sum,
)
import numpy as np
from torch.distributions import Multinomial, MultivariateNormal, Normal
from catvae.distributions.mvn import MultivariateNormalFactorSum


LOG_2_PI = np.log(2.0 * np.pi)


class LinearCatVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_scale=0.001,
                 basis=None,  imputer=None,
                 batch_size=10):
        super(LinearCatVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Psi must be dimension D - 1 x D
        if basis is None:
            tree = random_linkage(self.input_dim)
            Psi = torch.Tensor(_balance_basis(tree)[0].copy())
        else:
            Psi = torch.Tensor(basis.copy())

        if imputer is None:
            self.imputer = lambda x: x + 1

        self.encoder = nn.Linear(input_dim - 1, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim - 1, bias=False)
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.zeros(batch_size, self.input_dim - 1))
        self.encoder.weight.data.normal_(0.0, init_scale)
        self.decoder.weight.data.normal_(0.0, init_scale)

        Id = torch.eye(input_dim - 1).to(self.eta.device)
        zI = torch.ones(self.hidden_dim).to(self.eta.device)
        zm = torch.zeros(self.hidden_dim).to(self.eta.device)

        self.register_buffer('Psi', Psi)
        self.register_buffer('Id', Id)
        self.register_buffer('zI', zI)
        self.register_buffer('zm', zm)

    def forward(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        mu = self.decoder(z_mean)
        W = self.decoder.weight
        # penalties
        D = torch.exp(0.5 * self.variational_logvars)
        var = torch.exp(self.log_sigma_sq)
        d = W.shape[-1] + 1
        sI = var * self.Id
        # TODO replace this with a factor MVN distribution later
        wdw = W @ torch.diag(D) @ W.t()
        sigma = sI + wdw
        qdist = MultivariateNormal(mu, covariance_matrix=sigma)
        prior_loss = Normal(self.zm, self.zI).log_prob(z_mean).mean()
        logit_loss = qdist.log_prob(self.eta).mean()
        mult_loss = Multinomial(logits=self.eta @ self.Psi).log_prob(x).mean()
        loglike = mult_loss + logit_loss + prior_loss
        return -loglike

    def get_reconstruction_loss(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        eta = self.decoder(z_mean)
        mult_loss = Multinomial(logits=eta @ self.Psi).log_prob(x).mean()
        return - mult_loss
