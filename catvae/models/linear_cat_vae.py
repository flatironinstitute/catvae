import torch
import torch.nn as nn
from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from scipy.sparse import coo_matrix
import numpy as np
from torch.distributions import Multinomial, MultivariateNormal, Normal
from catvae.composition import closure, ilr
from catvae.distributions.mvn import MultivariateNormalFactorIdentity
from typing import Callable

LOG_2_PI = np.log(2.0 * np.pi)


class LinearCatVAE(nn.Module):

    def __init__(self, input_dim : int, hidden_dim : int,
                 init_scale : float = 0.001,
                 basis : coo_matrix = None,
                 encoder_depth : int = 1,
                 imputer : Callable[[torch.Tensor], torch.Tensor]=None,
                 batch_size : int =10, bias=True):
        super(LinearCatVAE, self).__init__()
        self.initialize(input_dim, hidden_dim, init_scale,
                        basis, encoder_depth, imputer,
                        batch_size, bias)

    def initialize(self, input_dim : int, hidden_dim : int,
                   init_scale : float = 0.001,
                   basis : coo_matrix = None,
                   encoder_depth : int = 1,
                   imputer : Callable[[torch.Tensor], torch.Tensor]=None,
                   batch_size : int =10, bias=True):
        self.hidden_dim = hidden_dim
        self.bias = bias
        # Psi must be dimension D - 1 x D
        if basis is None:
            tree = random_linkage(input_dim)
            basis = sparse_balance_basis(tree)[0].copy()
        indices = np.vstack((basis.row, basis.col))
        Psi = torch.sparse_coo_tensor(
            indices.copy(), basis.data.astype(np.float32).copy(),
            requires_grad=False)

        # Psi.requires_grad = False
        self.input_dim = Psi.shape[0]
        if imputer is None:
            self.imputer = lambda x: x + 1

        if encoder_depth > 1:
            self.first_encoder = nn.Linear(
                self.input_dim, hidden_dim, bias=self.bias)
            num_encoder_layers = encoder_depth
            layers = []
            layers.append(self.first_encoder)
            for layer_i in range(num_encoder_layers - 1):
                layers.append(nn.Softplus())
                layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=self.bias))
            self.encoder = nn.Sequential(*layers)

            # initialize
            for encoder_layer in self.encoder:
                if isinstance(encoder_layer, nn.Linear):
                    encoder_layer.weight.data.normal_(0.0, init_scale)

        else:
            self.encoder = nn.Linear(self.input_dim, hidden_dim, bias=self.bias)
            self.encoder.weight.data.normal_(0.0, init_scale)

        self.decoder = nn.Linear(hidden_dim, self.input_dim, bias=False)
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.01))
        self.eta = nn.Parameter(torch.zeros(batch_size, self.input_dim))
        self.eta.data.normal_(0.0, init_scale)
        self.decoder.weight.data.normal_(0.0, init_scale)
        zI = torch.ones(self.hidden_dim).to(self.eta.device)
        zm = torch.zeros(self.hidden_dim).to(self.eta.device)
        self.register_buffer('Psi', Psi)
        self.register_buffer('zI', zI)
        self.register_buffer('zm', zm)

    def encode(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z = self.encoder(hx)
        return z

    def forward(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        mu = self.decoder(z_mean)
        W = self.decoder.weight
        # penalties
        D = torch.exp(self.variational_logvars)
        var = torch.exp(self.log_sigma_sq)
        qdist = MultivariateNormalFactorIdentity(mu, var, D, W)
        logp = self.Psi.t() @ self.eta.t()
        prior_loss = Normal(self.zm, self.zI).log_prob(z_mean).mean()
        logit_loss = qdist.log_prob(self.eta).mean()
        mult_loss = Multinomial(logits=logp.t()).log_prob(x).mean()
        loglike = mult_loss + logit_loss + prior_loss
        return -loglike

    def reset(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        self.eta.data = hx.data

    def get_reconstruction_loss(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        eta = self.decoder(z_mean)
        logp = self.Psi.t() @ eta.t()
        mult_loss = Multinomial(logits=logp.t()).log_prob(x).mean()
        return - mult_loss


class LinearBatchCatVAE(LinearCatVAE):
    def __init__(self, input_dim : int, hidden_dim : int,
                 init_scale : float = 0.001,
                 basis : coo_matrix = None,
                 encoder_depth : int = 1,
                 imputer : Callable[[torch.Tensor], torch.Tensor]=None,
                 batch_size : int =10, bias=True):
        super(LinearBatchCatVAE, self).__init__(
            input_dim, hidden_dim, init_scale,
            basis, encoder_depth, imputer,
            batch_size, bias
        )

    def encode(self, x):
        #B = B.sum(axis=0) + 1
        #B = B.unsqueeze(0)
        #batch_effects = (self.Psi @ B.t()).t()
        hx = ilr(self.imputer(x), self.Psi)
        #hx -= batch_effects  # Subtract out batch effects
        z = self.encoder(hx)
        return z

    def forward(self, x, B):
        hx = ilr(self.imputer(x), self.Psi)
        batch_effects = (self.Psi @ B.t()).t()
        hx -= batch_effects  # Subtract out batch effects
        z_mean = self.encoder(hx)
        mu = self.decoder(z_mean)
        mu += batch_effects  # Add batch effects back in
        W = self.decoder.weight
        # penalties
        D = torch.exp(self.variational_logvars)
        var = torch.exp(self.log_sigma_sq)
        qdist = MultivariateNormalFactorIdentity(mu, var, D, W)
        logp = self.Psi.t() @ self.eta.t()
        prior_loss = Normal(self.zm, self.zI).log_prob(z_mean).mean()
        logit_loss = qdist.log_prob(self.eta).mean()
        mult_loss = Multinomial(logits=logp.t()).log_prob(x).mean()
        loglike = mult_loss + logit_loss + prior_loss
        return -loglike

    def get_reconstruction_loss(self, x, B):
        hx = ilr(self.imputer(x), self.Psi)
        batch_effects = (self.Psi @ B.t()).t()
        hx -= batch_effects  # Subtract out batch effects
        z_mean = self.encoder(hx)
        eta = self.decoder(z_mean)
        eta += batch_effects   # Add batch effects back in
        logp = self.Psi.t() @ eta.t()
        mult_loss = Multinomial(logits=logp.t()).log_prob(x).mean()
        return - mult_loss
