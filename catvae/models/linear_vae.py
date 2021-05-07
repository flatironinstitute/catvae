import torch
import torch.nn as nn
from catvae.composition import ilr, closure
from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from torch.distributions import Multinomial, Normal
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


class ArcsineEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ArcsineEmbed, self).__init__()
        self.embed = nn.Parameter(
            torch.zeros(input_dim, hidden_dim))
        self.ffn_weights = nn.Parameter(torch.zeros(input_dim - 1, hidden_dim, 1))
        self.bias = nn.Parameter(torch.zeros(input_dim - 1))

    def forward(self, x, Psi):
        a = torch.arcsin(torch.sqrt(closure(x)))  # B x D
        x_ = a[:, :, None] * self.embed     # B x D x H
        x_ = (self.Psi @ x_.T).T            # B x D-1
        fx = torch.einsum('bih,ihk -> bik', x_, self.ffn_weights).squeeze()
        return fx + self.bias


class CLREmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ArcsineEmbed, self).__init__()
        self.embed = nn.Parameter(
            torch.zeros(input_dim, hidden_dim))
        self.ffn_weights = nn.Parameter(torch.zeros(input_dim - 1, hidden_dim, 1))
        self.bias = nn.Parameter(torch.zeros(input_dim - 1))

    def forward(self, x, Psi):
        a = torch.arcsin(torch.sqrt(closure(x)))  # B x D
        a = torch.log(closure(x + 1))
        a = a - a.mean(axis=1).reshape(-1, 1)     # center around mean
        x_ = a[:, :, None] * self.embed     # B x D x H
        x_ = (self.Psi @ x_.T).T            # B x D-1
        fx = torch.einsum('bih,ihk -> bik', x_, self.ffn_weights).squeeze()
        return fx + self.bias


class Encoder(nn.Module):
    def __init__(self, input_dim : int,
                 hidden_dim : int,
                 latent_dim : int, bias : bool=False,
                 depth : int = 1, init_scale : float = 0.001):
        super(Encoder, self).__init__()
        if depth > 1:
            first_encoder = nn.Linear(
                input_dim, hidden_dim, bias=bias)
            num_encoder_layers = depth
            layers = []
            layers.append(first_encoder)
            for layer_i in range(num_encoder_layers - 2):
                layers.append(nn.Softplus())
                layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(nn.Softplus())
            layers.append(nn.Linear(hidden_dim, latent_dim, bias=bias))
            self.encoder = nn.Sequential(*layers)

            # initialize
            for encoder_layer in self.encoder:
                if isinstance(encoder_layer, nn.Linear):
                    encoder_layer.weight.data.normal_(0.0, init_scale)
        elif depth == 2:
            layers = nn.Sequential(*[
                nn.Linear(input_dim, hidden_dim, bias=bias),
                nn.Softplus(),
                nn.Linear(hidden_dim, latent_dim, bias=bias)
            ])
            self.encoder = nn.Sequential(*layers)
            for encoder_layer in self.encoder:
                if isinstance(encoder_layer, nn.Linear):
                    encoder_layer.weight.data.normal_(0.0, init_scale)
        elif depth == 1:
            self.encoder = nn.Linear(
                input_dim, latent_dim, bias=bias)
            self.encoder.weight.data.normal_(0.0, init_scale)
        else:
            raise ValueError(f'Depth of {depth} is not appropriate.')

    def forward(self, x):
        return self.encoder(x)


class LinearVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim=None,
                 init_scale=0.001, encoder_depth=1,
                 basis=None, bias=False, transform='arcsine'):
        super(LinearVAE, self).__init__()
        if latent_dim is None:
            latent_dim = hidden_dim
        self.bias = bias
        self.hidden_dim = hidden_dim
        Psi = get_basis(input_dim, basis).coalesce()
        # note this line corresponds to the true input dim
        self.input_dim = Psi.shape[0]
        self.register_buffer('Psi', Psi)
        self.encoder = Encoder(
            self.input_dim, hidden_dim, latent_dim,
            bias=bias, depth=encoder_depth, init_scale=init_scale)
        self.decoder = nn.Linear(
            latent_dim, self.input_dim, bias=self.bias)
        geotorch.grassmannian(self.decoder, 'weight')
        self.imputer = lambda x: x + 1
        self.variational_logvars = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))
        self.transform = transform
        if self.transform == 'arcsine':
            self.input_embed = ArcsineEmbed(self.input_dim + 1, hidden_dim)
        if self.transform == 'clr':
            self.input_embed = CLREmbed(self.input_dim + 1, hidden_dim)

    def gaussian_kl(self, z_mean, z_logvar):
        return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))
        # x = Normal(0, 1)
        # y = Normal(z_mean, torch.exp(z_logvar))
        # return - kl_divergence(x, y)

    def gaussian_kl2(self, m1, s1, m2, s2):
        x = Normal(m1, torch.exp(0.5 * s1))
        y = Normal(m2, torch.exp(0.5 * s2))
        return - kl_divergence(x, y)

    def recon_model_loglik(self, x_in, x_out):
        logp = (self.Psi.t() @ x_out.t()).t()
        mult_loss = Multinomial(logits=logp).log_prob(x_in).mean()
        return mult_loss

    def encode(self, x):
        if self.transform in {'arcsine', 'clr'}:
            hx = self.input_embed(x, self.Psi)
        elif self.transform == 'pseudocount':
            fx = torch.log(x + 1)                 # ILR transform for testing
            hx = (self.Psi @ fx.T).T              # B x D-1
        z = self.encoder(hx)
        return z

    def forward(self, x):
        z_mean = self.encode(x)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        x_out = self.decoder(z_sample)
        kl_div = self.gaussian_kl(
            z_mean, self.variational_logvars).mean(0).sum()
        recon_loss = self.recon_model_loglik(x, x_out).mean(0).sum()
        elbo = kl_div + recon_loss
        loss = - elbo
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
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 batch_dim, batch_prior,
                 init_scale=0.001, encoder_depth=1,
                 basis=None, bias=False, transform='arcsine'):
        """ Account for batch effects.

        Parameters
        ----------
        input_dim : int
           Number of dimensions for input counts
        hidden_dim : int
           Number of hidden dimensions within encoder
        latent_dim : int
           Number of hidden dimensions within latent space
        batch_dim : int
           Number of batches (i.e. studies) to do batch correction
        batch_priors : np.array of float
           Normal variance priors for batch effects of shape D
        transform : str
           Choice of input transform.  Can choose from
           arcsine, pseudocount and rclr (TODO).
        """
        super(LinearBatchVAE, self).__init__(
            input_dim, hidden_dim, latent_dim,
            init_scale, basis=basis, encoder_depth=encoder_depth,
            bias=bias, transform=transform)
        self.batch_dim = batch_dim
        self.ilr_dim = input_dim - 1
        batch_prior = batch_prior
        self.register_buffer('batch_prior', batch_prior)
        self.batch_logvars = nn.Parameter(torch.zeros(self.ilr_dim))
        self.beta = nn.Embedding(batch_dim, self.ilr_dim)

    def encode(self, x, b):
        # TODO: call super.encode()
        if self.transform == 'arcsine':
            hx = self.input_embed(x, self.Psi)
        elif self.transform == 'pseudocount':
            fx = torch.log(x + 1)                     # ILR transform for testing
            hx = (self.Psi @ fx.T).T                      # B x D-1
        batch_effects = self.beta(b)                  # B x D-1
        hx = hx - batch_effects
        z = self.encoder(hx)
        return z

    def forward(self, x, b):
        z_mean = self.encode(x, b)
        batch_effects = self.beta(b)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        eps = torch.normal(torch.zeros_like(batch_effects), 1.0)
        b_sample = batch_effects + eps * torch.exp(0.5 * self.batch_logvars)
        x_out = self.decoder(z_sample) + b_sample
        kl_div_z = self.gaussian_kl(
            z_mean, self.variational_logvars).mean(0).sum()
        kl_div_b = self.gaussian_kl2(
            batch_effects, self.batch_logvars,
            torch.zeros_like(self.batch_prior), self.batch_prior
        ).mean(0).sum()
        recon_loss = self.recon_model_loglik(x, x_out).mean(0).sum()
        elbo = kl_div_z + kl_div_b + recon_loss
        loss = - elbo
        return loss, -recon_loss, -kl_div_z, -kl_div_b

    def get_reconstruction_loss(self, x, b):
        z_mean = self.encode(x, b)
        eps = torch.normal(torch.zeros_like(z_mean), 1.0)
        z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
        batch_effects = self.beta(b)
        x_out = self.decoder(z_sample)
        x_out += batch_effects  # Add batch effects back in
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss
