import torch
import torch.nn as nn
from catvae.composition import closure
from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from torch.distributions import Multinomial, Normal, Gamma
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
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


def rnormalgamma(mu_dist : Normal, std_dist : Gamma):
    mu = mu_dist.rsample()
    prec = std_dist.rsample()
    std = torch.sqrt(1 / prec)
    eps = torch.normal(torch.zeros_like(std), 1.0)
    return mu + std * eps


class ArcsineEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(ArcsineEmbed, self).__init__()
        self.embed = nn.Parameter(
            torch.zeros(input_dim, hidden_dim))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=True),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, 1, bias=True),
        )

    def forward(self, x, Psi):
        a = torch.arcsin(torch.sqrt(closure(x)))  # B x D
        x_ = a[:, :, None] * self.embed           # B x D x H
        fx = self.ffn(x_).squeeze()
        fx = (Psi @ fx.T).T                         # B x D-1
        return fx


class CLREmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(CLREmbed, self).__init__()
        self.embed = nn.Parameter(
            torch.zeros(input_dim, hidden_dim))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=True),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, 1, bias=True),
        )

    def forward(self, x, Psi):
        a = torch.arcsin(torch.sqrt(closure(x)))  # B x D
        a = torch.log(closure(x + 1))
        a = a - a.mean(axis=1).reshape(-1, 1)     # center around mean
        x_ = a[:, :, None] * self.embed           # B x D x H
        fx = self.ffn(x_).squeeze()
        fx = (Psi @ fx.T).T                       # B x D-1
        return fx


class Encoder(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 latent_dim: int, bias: bool = False,
                 dropout: float = 0, batch_norm: bool = True,
                 depth: int = 1, init_scale: float = 0.001):
        super(Encoder, self).__init__()
        if depth > 1:
            first_encoder = nn.Linear(
                input_dim, hidden_dim, bias=bias)
            num_encoder_layers = depth
            layers = []
            layers.append(first_encoder)
            layers.append(nn.Softplus())
            layers.append(nn.Dropout(dropout))
            for layer_i in range(num_encoder_layers - 2):
                layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(nn.Softplus())
                layers.append(nn.Dropout(dropout))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
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
                 basis=None, bias=True,
                 transform='pseudocount', distribution='multinomial',
                 dropout=0, batch_norm=False, grassmannian=True):
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
            bias=bias, depth=encoder_depth, init_scale=init_scale,
            dropout=dropout, batch_norm=batch_norm)
        self.decoder = nn.Linear(
            latent_dim, self.input_dim, bias=self.bias)
        if grassmannian:
            geotorch.grassmannian(self.decoder, 'weight')
        self.variational_logvars = nn.Parameter(torch.zeros(latent_dim))
        self.transform = transform
        self.distribution = distribution
        if self.transform == 'arcsine':
            self.input_embed = ArcsineEmbed(self.input_dim + 1,
                                            hidden_dim, dropout)
        if self.transform == 'clr':
            self.input_embed = CLREmbed(self.input_dim + 1,
                                        hidden_dim, dropout)

    def recon_model_loglik(self, x_in, x_out):
        logp = (self.Psi.t() @ x_out.t()).t()
        if self.distribution == 'multinomial':
            dist_loss = Multinomial(
                logits=logp, validate_args=False  # weird ...
            ).log_prob(x_in).mean()
        elif self.distribution == 'gaussian':
            # MSE loss based out on DeepMicro
            # https://www.nature.com/articles/s41598-020-63159-5
            dist_loss = Normal(
                loc=logp, scale=1, validate_args=False  # weird ...
            ).log_prob(x_in).mean()
        else:
            raise ValueError(
                f'Distribution {self.distribution} is not supported.')
        return dist_loss

    def impute(self, x):
        if self.transform in {'arcsine', 'clr'}:
            hx = self.input_embed(x, self.Psi)
        elif self.transform == 'pseudocount':
            fx = torch.log(x + 1)                 # ILR transform for testing
            hx = (self.Psi @ fx.T).T              # B x D-1
        elif self.transform == 'none':
            hx = x
        else:
            raise ValueError(f'Unrecognzied transform {self.transform}')
        return hx

    def sample(self, x):
        z_mean = self.encode(x)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        z_sample = qz.rsample()
        return z_sample

    def encode(self, x):
        hx = self.impute(x)
        z = self.encoder(hx)
        return z

    def forward(self, x):
        z_mean = self.encode(x)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        z_sample = qz.rsample()
        x_out = self.decoder(z_sample)
        kl_div = kl_divergence(qz, Normal(0, 1)).mean(0).sum()
        recon_loss = self.recon_model_loglik(x, x_out).mean(0).sum()
        elbo = recon_loss - kl_div
        loss = - elbo
        return loss

    def get_reconstruction_loss(self, x):
        z_mean = self.encode(x)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        z_sample = qz.sample()
        x_out = self.decoder(z_sample)
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss


class LinearDLRVAE(LinearVAE):

    def __init__(self, input_dim, hidden_dim, latent_dim=None,
                 init_scale=0.001, encoder_depth=1,
                 basis=None, bias=True,
                 transform='pseudocount', distribution='multinomial',
                 dropout=0, batch_norm=False, grassmannian=True):
        super(LinearDLRVAE, self).__init__(
            input_dim, hidden_dim, latent_dim,
            init_scale=init_scale, basis=basis,
            encoder_depth=encoder_depth,
            bias=bias, transform=transform, dropout=dropout,
            batch_norm=batch_norm, grassmannian=grassmannian)
        self.log_sigma_sq = nn.Parameter(torch.ones(input_dim - 1))

    def sample(self, x):
        z_mean = self.encode(x)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        z_sample = qz.sample()
        return z_sample

    def forward(self, x):
        z_mean = self.encode(x)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        ql = Normal(0, torch.exp(0.5 * self.log_sigma_sq))
        z_sample = qz.rsample()
        l_sample = ql.rsample()
        x_out = self.decoder(z_sample) + l_sample
        kl_div = kl_divergence(qz, Normal(0, 1)).mean(0).sum()
        recon_loss = self.recon_model_loglik(x, x_out).mean(0).sum()
        elbo = recon_loss - kl_div
        loss = - elbo
        return loss

    def get_reconstruction_loss(self, x):
        z_sample = self.sample(x)
        x_out = self.decoder(z_sample)
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss


class LinearBatchVAE(LinearVAE):
    def __init__(self, input_dim, hidden_dim, latent_dim, batch_dim,
                 beta_prior, gam_prior, phi_prior,
                 init_scale=0.001, encoder_depth=1,
                 basis=None, bias=True,
                 transform='pseudocount',
                 distribution='multinomial',
                 batch_norm=False, dropout=0,
                 grassmannian=True):
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
        beta_prior : np.array of float
           Normal variance priors for batch effects of shape D - 1.
           Note that these priors are assumed to be in ILR coordinates.
        gam_prior : np.array of float
           Alpha for Gamma prior on batch over-dispersion.
        phi_prior : np.array of float
           Beta for Gamma prior on batch over-dispersion.
        transform : str
           Choice of input transform.  Can choose from
           arcsine, pseudocount and rclr (TODO).
        """
        super(LinearBatchVAE, self).__init__(
            input_dim, hidden_dim, latent_dim,
            init_scale=init_scale, basis=basis,
            encoder_depth=encoder_depth,
            bias=bias, transform=transform, dropout=dropout,
            batch_norm=batch_norm, grassmannian=grassmannian)
        self.log_sigma_sq = nn.Parameter(torch.ones(input_dim - 1))
        self.batch_dim = batch_dim
        self.ilr_dim = input_dim - 1
        # define batch priors
        beta_prior = torch.Tensor(beta_prior).float()
        self.register_buffer('bpr', beta_prior)
        self.register_buffer('gpr', gam_prior)
        self.register_buffer('ppr', phi_prior)
        # define batch posterior vars
        self.beta = nn.Embedding(batch_dim, self.ilr_dim)
        self.beta_logvars = nn.Embedding(batch_dim, self.ilr_dim)
        self.loggamma = nn.Embedding(batch_dim, self.ilr_dim)
        self.logphi = nn.Embedding(batch_dim, self.ilr_dim)
        # initialize posterior weights
        self.beta.weight.data.fill_(0.0)
        self.beta_logvars.weight.data.fill_(-8)  # exp(-7) = 0.001
        self.loggamma.weight.data.fill_(2.5)     # roughly e
        self.logphi.weight.data.fill_(0.3)       # roughly 1 / e
        # define encoder batch vars
        self.batch_embed = nn.Embedding(batch_dim, latent_dim)

    def pretrained_parameters(self):
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        params += [self.log_sigma_sq, self.variational_logvars]
        return params

    def batch_parameters(self):
        params = list(self.beta.parameters())
        params += list(self.beta_logvars.parameters())
        params += list(self.loggamma.parameters())
        params += list(self.logphi.parameters())
        params += list(self.batch_embed.parameters())
        return params

    def encode(self, x, b):
        hx = self.impute(x)
        zb = self.batch_embed(b)
        z = self.encoder(hx) - zb
        return z

    def encode_marginalized(self, x, b):
        """ Marginalize over batch_effects given predictions

        This will compute the expected batch effect given the
        batch classification probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Counts of interest (B x D)
        b : torch.Tensor
            Batch effect prediction log probabilities (B x K)

        Notes
        -----
        This assumes that the batch classifier is well-calibrated.
        """
        # obtain expected batch effect
        beta_ = self.beta.weight
        m = nn.Softmax()
        batch_effects = b @ m(beta_)
        hx = self.impute(x)
        hx = hx - batch_effects
        z = self.encoder(hx)
        return z

    def sample(self, x, b, size=None):
        # obtain mean of latent distribution
        z_mean = self.encode(x, b)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        return qz.sample(size)

    def forward(self, x, b):
        z_mean = self.encode(x, b)
        gam = F.softplus(self.loggamma(b), beta=0.1)
        phi = F.softplus(self.logphi(b), beta=0.1)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        ql = Normal(0, torch.exp(0.5 * self.log_sigma_sq))
        qb = Normal(self.beta(b), torch.exp(0.5 * self.beta_logvars(b)))
        qS = Gamma(gam, phi)
        # draw differentiable MC samples
        z_sample = qz.rsample()
        b_sample = rnormalgamma(qb, qS)
        l_sample = ql.rsample()
        # compute KL divergence + reconstruction loss
        x_out = self.decoder(z_sample) + b_sample + l_sample
        zb = torch.zeros_like(self.bpr)
        kl_div_z = kl_divergence(qz, Normal(0, 1)).mean(0).sum()
        kl_div_b = kl_divergence(qb, Normal(0, self.bpr)).mean(0).sum()
        kl_div_S = kl_divergence(qS, Gamma(self.gpr, self.ppr)).mean(0).sum()
        recon_loss = self.recon_model_loglik(x, x_out).mean(0).sum()
        elbo = recon_loss - kl_div_z - kl_div_b - kl_div_S
        loss = - elbo
        return loss, -recon_loss, kl_div_z, kl_div_b, kl_div_S

    def get_reconstruction_loss(self, x, b):
        z_mean = self.encode(x, b)
        batch_effects = self.beta(b)
        qz = Normal(z_mean, torch.exp(0.5 * self.variational_logvars))
        qb = Normal(batch_effects, torch.exp(0.5 * self.batch_logvars))
        z_sample = qz.sample()
        b_sample = qb.sample()
        x_out = self.decoder(z_sample)
        x_out += batch_effects  # Add batch effects back in
        recon_loss = -self.recon_model_loglik(x, x_out)
        return recon_loss
