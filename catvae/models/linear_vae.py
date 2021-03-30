"""
Author : XuchanBao
This code was adapted from
https://github.com/XuchanBao/linear-ae
"""
import torch
import torch.nn as nn
from catvae.composition import ilr
from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from torch.distributions import Multinomial, Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import geotorch

LOG_2_PI = np.log(2.0 * np.pi)


class LinearVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_scale=0.001,
                 encoder_depth=1,
                 basis=None, bias=False):
        super(LinearVAE, self).__init__()
        self.bias = bias
        self.hidden_dim = hidden_dim

        if basis is None:
            tree = random_linkage(input_dim)
            basis = sparse_balance_basis(tree)[0].copy()
        indices = np.vstack((basis.row, basis.col))
        Psi = torch.sparse_coo_tensor(
            indices.copy(), basis.data.astype(np.float32).copy(),
            requires_grad=False).coalesce()
        # note this line corresponds to the true input dim
        self.input_dim = Psi.shape[0]
        self.register_buffer('Psi', Psi)

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

        self.decoder = nn.Linear(hidden_dim, self.input_dim, bias=self.bias)
        geotorch.grassmannian(self.decoder, 'weight')
        self.imputer = lambda x: x + 1
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

    def gaussian_kl(self, z_mean, z_logvar):
        return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))

    def gaussian_kl2(self, m1, s1, m2, s2):
        x = Normal(m1, s1)
        y = Normal(m2, s2)
        return - kl_divergence(x, y)

    def recon_model_loglik(self, x_in, x_out):
        logp = (self.Psi.t() @ x_out.t()).t()
        mult_loss = Multinomial(logits=logp).log_prob(x_in).mean()
        return mult_loss

    def encode(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z = self.encoder(hx)
        return z

    def forward(self, x):
        x_ = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(x_)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        x_out = self.decoder(z_sample)
        kl_div = (-self.gaussian_kl(
            z_mean, self.variational_logvars)).mean(0).sum()
        recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
        loss = kl_div + recon_loss
        return loss

    def get_reconstruction_loss(self, x):
        x_ = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(x_)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        x_out = self.decoder(z_sample)
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss


class LinearBatchVAE(LinearVAE):
    def __init__(self, input_dim, hidden_dim, batch_dim, batch_prior,
                 init_scale=0.001, encoder_depth=1,
                 basis=None, bias=False):
        """ Account for batch effects.

        Parameters
        ----------
        input_dim : int
           Number of dimensions for input counts
        hidden_dim : int
           Number of hidden dimensions
        batch_dim : int
           Number of batches (i.e. studies) to do batch correction
        batch_priors : np.array of float
           Normal variance priors for batch effects of shape D
        """
        super(LinearBatchVAE, self).__init__(
            input_dim, hidden_dim, init_scale,
            basis=basis, encoder_depth=encoder_depth,
            bias=bias)
        # Need to create a separate matrix, since abs doesn't work on
        # sparse matrices :(
        posPsi = torch.sparse.FloatTensor(
            self.Psi.indices(), torch.abs(self.Psi.values()))
        batch_prior = ilr(batch_prior, posPsi)
        self.register_buffer('batch_prior', batch_prior)
        self.batch_logvars = nn.Parameter(torch.zeros(self.input_dim))
        self.batch_embed = nn.Embedding(batch_dim, batch_dim)
        self.beta = nn.Embedding(batch_dim, self.input_dim)

    def encode(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z = self.encoder(hx)
        return z

    def forward(self, x, b):
        """ Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input counts of shape b x D
        b : torch.Tensor
            Batch indices of shape C
        """
        bx = self.batch_embed(b)
        hx = ilr(self.imputer(x), self.Psi)
        hbx = torch.cat((bx, hx), dim=1)
        z_mean = self.encoder(hx)
        # sample z
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        # sample beta
        eps = torch.normal(torch.zeros_like(bx), 1.0)
        batch_effects = self.beta(b)
        b_sample = batch_effects + eps * torch.exp(0.5 * self.batch_logvars)

        x_out = self.decoder(z_sample)
        x_out += b_sample  # Add batch effects back in
        # Weight by latent prior
        kl_div_z = (-self.gaussian_kl(
            z_mean, self.variational_logvars)).mean(0).sum()
        # Weight by batch prior
        kl_div_b = (-self.gaussian_kl2(
            batch_effects, torch.exp(self.batch_logvars),
            torch.zeros_like(self.batch_prior), self.batch_prior
        )).mean(0).sum()
        recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
        loss = recon_loss + kl_div_z + kl_div_b
        return loss, recon_loss, kl_div_z, kl_div_b

    def get_reconstruction_loss(self, x, b):
        bx = self.batch_embed(b)
        hx = ilr(self.imputer(x), self.Psi)
        hbx = torch.cat((bx, hx), dim=1)
        z_mean = self.encoder(hbx)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        batch_effects = self.beta(b)
        x_out = self.decoder(z_sample)
        x_out += batch_effects  # Add batch effects back in
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss
