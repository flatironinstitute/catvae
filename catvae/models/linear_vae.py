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
from torch.distributions import Multinomial, Categorical
from torch.distributions.kl import kl_divergence
import numpy as np
import geotorch

LOG_2_PI = np.log(2.0 * np.pi)


def get_basis(input_dim, basis=None):
    if basis is None:
        tree = random_linkage(input_dim)
        basis = sparse_balance_basis(tree)[0].copy()
    indices = np.vstack((basis.row, basis.col))
    Psi = torch.sparse_coo_tensor(
        indices.copy(), basis.data.astype(np.float32).copy(),
        requires_grad=False).coalesce()
    return Psi


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=1, init_scale=0.001):
        if depth > 1:
            self.first_encoder = nn.Linear(
                self.input_dim, hidden_dim, bias=self.bias)
            num_encoder_layers = depth
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
            self.encoder = nn.Linear(
                self.input_dim, hidden_dim, bias=self.bias)
            self.encoder.weight.data.normal_(0.0, init_scale)

    def forward(self, x):
        return self.encoder(x)


class LinearVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_scale=0.001,
                 encoder_depth=1,
                 basis=None, bias=False):
        super(LinearVAE, self).__init__()
        self.bias = bias
        self.hidden_dim = hidden_dim
        Psi = get_basis(input_dim, basis).coalesce()
        # note this line corresponds to the true input dim
        self.input_dim = Psi.shape[0]
        self.register_buffer('Psi', Psi)
        self.encoder = Encoder(self.input_dim, hidden_dim, init_scale)
        self.decoder = nn.Linear(hidden_dim, self.input_dim, bias=self.bias)
        geotorch.grassmannian(self.decoder, 'weight')
        self.imputer = lambda x: x + 1
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

    def gaussian_kl(self, z_mean, z_logvar):
        return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))

    def multinomial_kl(self, y, p):
        return kl_divergence(Categorical(y), Categorical(p))

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
    def __init__(self, input_dim, hidden_dim, batch_dim,
                 batch_priors, class_priors=None,
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
            input_dim + batch_dim, hidden_dim + batch_dim,
            init_scale, basis=basis, encoder_depth=encoder_depth,
            bias=bias)
        Psi = get_basis(input_dim, basis).coalesce()
        # note this line corresponds to the true input dim
        self.input_dim = Psi.shape[0]
        self.register_buffer('Psi', Psi)

        self.encoder = Encoder(self.input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.input_dim, bias=self.bias)
        geotorch.grassmannian(self.decoder, 'weight')

        self.imputer = lambda x: x + 1
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))
        # Need to create a separate matrix, since abs doesn't work on
        # sparse matrices :(
        posPsi = torch.sparse.FloatTensor(
            self.Psi.indices(), torch.abs(self.Psi.values()))
        batch_priors = ilr(batch_priors, posPsi)
        self.batch_classifier = nn.Sequential(
            nn.Linear(batch_dim, batch_dim),
            nn.Softmax())
        if class_priors is None:
            class_priors = torch.ones(batch_dim) / batch_dim
        self.register_buffer('batch_priors', batch_priors)
        self.register_buffer('class_priors', class_priors)
        self.batch_dim = batch_dim
        self.batch_embed = nn.Embedding(batch_dim, batch_dim)

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
        hx = ilr(self.imputer(x), self.Psi)
        bx = self.batch_embed(b)
        hbx = torch.cat(bx, hx)
        z_mean = self.encoder(hbx)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        x_out = self.decoder(z_sample)
        # Weight by latent prior
        kl_div_z = (-self.gaussian_kl(
            z_mean, self.variational_logvars)).mean(0).sum()
        # Weight by batch differential prior
        Wb = self.decoder.weight[:self.batch_dim]
        zb = z_mean[:self.batch_dim]
        batch_effects = Wb @ zb
        kl_div_b = (-self.gaussian_kl(
            batch_effects, self.batch_priors)).mean(0).sum()
        # Weight by batch class prior
        batch_pred = self.classifier(z_mean[:self.batch_dim])
        kl_div_y = (-self.multinomial_kl(
            self.class_priors, batch_pred))
        recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
        loss = kl_div_z + kl_div_b + kl_div_y + recon_loss
        # something is wrong with kl_div_b
        return loss

    def get_reconstruction_loss(self, x, b):
        batch_effects = self.batch_embed(b)
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        x_out = self.decoder(z_sample)
        x_out += batch_effects  # Add batch effects back in
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss
