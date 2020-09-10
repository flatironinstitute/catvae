import torch
import torch.nn as nn
from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from scipy.sparse import coo_matrix
import numpy as np
from torch.distributions import Multinomial, MultivariateNormal, Normal
from catvae.composition import closure, ilr
from catvae.distributions.mvn import MultivariateNormalFactorIdentity
from catvae.sparse import SparseMatrix
from typing import Callable

LOG_2_PI = np.log(2.0 * np.pi)


class LinearCatVAE(nn.Module):

    def __init__(self, input_dim : int, hidden_dim : int,
                 init_scale : float = 0.001,
                 basis : coo_matrix = None,
                 encoder_depth : int = 1,
                 imputer : Callable[[torch.Tensor], torch.Tensor]=None,
                 batch_size : int =10):
        super(LinearCatVAE, self).__init__()
        self.hidden_dim = hidden_dim
        # Psi must be dimension D - 1 x D
        if basis is None:
            tree = random_linkage(input_dim)
            basis = sparse_balance_basis(tree)[0].copy()

        self.Psi = SparseMatrix.fromcoo(basis)
        self.Psi.requires_grad = False

        self.input_dim = self.Psi.shape[0]
        if imputer is None:
            self.imputer = lambda x: x + 1

        if encoder_depth > 1:
            self.first_encoder = nn.Linear(
                self.input_dim, hidden_dim, bias=False)
            num_encoder_layers = encoder_depth
            layers = []
            layers.append(self.first_encoder)
            for layer_i in range(num_encoder_layers - 1):
                layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*layers)

            # initialize
            for encoder_layer in self.encoder:
                if isinstance(encoder_layer, nn.Linear):
                    encoder_layer.weight.data.normal_(0.0, init_scale)
        else:
            self.encoder = nn.Linear(self.input_dim, hidden_dim, bias=False)
            self.encoder.weight.data.normal_(0.0, init_scale)

        self.decoder = nn.Linear(hidden_dim, self.input_dim, bias=False)
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.01))
        self.eta = nn.Parameter(torch.zeros(batch_size, self.input_dim))
        self.eta.data.normal_(0.0, init_scale)
        #self.encoder.weight.data.normal_(0.0, init_scale)
        self.decoder.weight.data.normal_(0.0, init_scale)
        zI = torch.ones(self.hidden_dim).to(self.eta.device)
        zm = torch.zeros(self.hidden_dim).to(self.eta.device)
        # self.register_buffer('Psi', Psi)
        self.register_buffer('zI', zI)
        self.register_buffer('zm', zm)

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.log_sigma_sq = self.log_sigma_sq.to(device)
        self.variational_logvars = self.variational_logvars.to(device)
        self.Psi = self.Psi.to(device)
        self.eta = self.eta.to(device)
        return self

    def cuda(self, device):
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        self.log_sigma_sq = self.log_sigma_sq.cuda()
        self.variational_logvars = self.variational_logvars.cuda()
        self.Psi = self.Psi.cuda()
        self.eta = self.eta.cuda()
        return self

    def forward(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        mu = self.decoder(z_mean)
        W = self.decoder.weight
        # penalties
        D = torch.exp(self.variational_logvars)
        var = torch.exp(self.log_sigma_sq)
        qdist = MultivariateNormalFactorIdentity(mu, var, D, W)
        logp = ilr_inv(self.eta, self.Psi)
        prior_loss = Normal(self.zm, self.zI).log_prob(z_mean).mean()
        logit_loss = qdist.log_prob(self.eta).mean()
        mult_loss = Multinomial(logits=logp).log_prob(x).mean()
        loglike = mult_loss + logit_loss + prior_loss
        return -loglike

    def reset(self, x):
        with torch.no_grad():
            hx = ilr(self.imputer(x), self.Psi)
            self.eta = nn.Parameter(hx)

    def get_reconstruction_loss(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        eta = self.decoder(z_mean)
        logp = ilr_inv(self.eta, self.Psi)
        mult_loss = Multinomial(logits=logp).log_prob(x).mean()
        return - mult_loss
