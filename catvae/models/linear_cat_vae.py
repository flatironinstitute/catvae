import torch
import torch.nn as nn
from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from scipy.sparse import coo_matrix
from catvae.composition import closure, ilr
from catvae.distributions.likelihood import (
    expectation_mvn_factor_sum_multinomial_taylor,
    expectation_mvn_factor_sum_multinomial_bound,
    expectation_joint_mvn_factor_mvn_factor_sum,
)
import numpy as np
from torch.distributions import Multinomial, MultivariateNormal, Normal
from catvae.distributions.mvn import MultivariateNormalFactorIdentity
from typing import Callable

LOG_2_PI = np.log(2.0 * np.pi)


class LinearCatVAE(nn.Module):

    def __init__(self, input_dim : int, hidden_dim : int,
                 init_scale : float = 0.001,
                 basis : coo_matrix = None,
                 imputer : Callable[[torch.Tensor], torch.Tensor]=None,
                 batch_size : int =10):
        super(LinearCatVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Psi must be dimension D - 1 x D
        if basis is None:
            tree = random_linkage(self.input_dim)
            basis = sparse_balance_basis(tree)[0].copy()
        indices = np.vstack((basis.row, basis.col))
        Psi = torch.sparse_coo_tensor(
            indices.copy(), basis.data.astype(np.float32).copy(),
            requires_grad=False)
        # Psi.requires_grad = False

        if imputer is None:
            self.imputer = lambda x: x + 1

        self.encoder = nn.Linear(input_dim - 1, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim - 1, bias=False)
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.zeros(batch_size, self.input_dim - 1))
        self.encoder.weight.data.normal_(0.0, init_scale)
        self.decoder.weight.data.normal_(0.0, init_scale)

        Id = torch.eye(input_dim - 1).to(self.eta.device).to_sparse()
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
        D = torch.exp(self.variational_logvars)
        var = torch.exp(self.log_sigma_sq)
        qdist = MultivariateNormalFactorIdentity(mu, var, D, W)
        prior_loss = Normal(self.zm, self.zI).log_prob(z_mean).mean(0).sum()
        logit_loss = qdist.log_prob(self.eta).mean(0).sum()
        mult_loss = Multinomial(
            logits=(self.Psi.t() @ self.eta.t()).t()).log_prob(x).mean(0).sum()
        loglike = mult_loss + logit_loss + prior_loss
        return -loglike

    def get_reconstruction_loss(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        eta = self.decoder(z_mean)
        mult_loss = Multinomial(
            logits=(self.Psi.t() @ self.eta.t()).t()).log_prob(x).mean()
        return - mult_loss
