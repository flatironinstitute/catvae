import torch
import torch.nn as nn
from gneiss.balances import _balance_basis
from gneiss.cluster import random_linkage
from gneiss.balances import _balance_basis
from catvae.utils import closure, ilr, ilr_inv
from skbio.stats.composition import closure
import numpy as np

LOG_2_PI = np.log(2.0 * np.pi)


def K(x):
    """ Log normalization constant for multinomial"""
    return (torch.lgamma(1 + torch.sum(x, dim=-1)) -
            torch.sum(torch.lgamma(1 + x), dim=-1))


class AsymptoticCovariance:
    """ Methods for manipulating the covariance matrix of the
    asymptotic multinomial distribution."""
    def __init__(self, Psi):
        self.Psi = Psi

    def cov(self, x):
        """ Obtain covariance estimate for nonzero counts x."""
        p = closure(x)
        return self.Psi.T @ torch.diag(1 / p) self.Psi

    def inv_cov(self, x):
        """ Obtain inverse covariance estimate for nonzero counts x."""
        p = closure(x)
        return self.Psi @ torch.diag(p) self.Psi.T

    def det_cov(self, x):
        """ Obtain determinant of covariance estimate for nonzero counts x."""
        p = closure(x)
        return torch.prod(p)


class LinearCatVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_scale=0.001,
                 basis=None, use_analytic_elbo=True, use_batch_norm=False,
                 deep_decoder=False, decoder_depth=1):
        super(LinearVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Psi must be dimension D - 1 x D
        if basis is None:
            tree = random_linkage(self.input_dim)
            self.Psi = _balance_basis(tree)[0]
        else:
            self.Psi = basis
        self.Sigma = AsymptoticCovariance(self.Psi)

        self.use_analytic_elbo = use_analytic_elbo
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            assert not use_analytic_elbo, "When using batch norm, need use_analytic_elbo to be False"
            self.bn = nn.BatchNorm1d(num_features=hidden_dim)
        else:
            self.bn = None

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))

        self.use_deep_decoder = deep_decoder
        if deep_decoder:
            assert not use_analytic_elbo, "When using deep decoder, need use_analytic_elbo to be False"
            self.final_decoder = nn.Linear(hidden_dim, input_dim, bias=False)

            num_decoder_layers = decoder_depth
            if use_batch_norm:
                layers = []
                for layer_i in range(num_decoder_layers - 1):
                    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                    layers.append(nn.BatchNorm1d(num_features=hidden_dim))
                layers.append(self.final_decoder)
            else:
                layers = []
                for layer_i in range(num_decoder_layers - 1):
                    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(self.final_decoder)

            self.decoder = nn.Sequential(*layers)
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

        self.encoder.weight.data.normal_(0.0, init_scale)

        if isinstance(self.decoder, nn.Sequential):
            for decoder_layer in self.decoder:
                if isinstance(decoder_layer, nn.Linear):
                    decoder_layer.weight.data.normal_(0.0, init_scale)
        else:
            self.decoder.weight.data.normal_(0.0, init_scale)

    def multinomial_loglike(self, x, eta):
        """ The expected multinomial reconstruction loss (i)

        Parameters
        ----------
        x : torch.Tensor
           The input counts of dimension (B, N, D)
        eta : torch.Tensor

        """
        b, n, d = x.shape
        denom = x @ (self.Psi @ eta).sum(dim=-1).view(b, n, 1)
        return K(x) + x @ self.Psi @ eta - denom

    def multinomial_kl(self, x, eta):
        """ KL divergence between asymptotic multinomial and decoding normal"""
        p = closure(x)
        x = ilr(x, self.Psi)

        z_logvar = self.variational_logvars
        W = self.decoder.weight
        V = self.encoder.weight
        tr_wdw = torch.trace(
            torch.mm(W, torch.mm(torch.diag(torch.exp(z_logvar)), W.t())))
        wv = torch.mm(W, V)
        vtwtwv = wv.t().mm(wv)
        xtvtwtwvx = (x * torch.mm(x, vtwtwv)).mean(0).sum()
        xtwvx = 2.0 * (x * x.mm(wv)).mean(0).sum()
        xtx = (x * x).mean(0).sum()

        S = self.Sigma.cov(p)
        xSx = 2.0 * (x * x.mm(X)).mean(0).sum()

        d = x.shape[-1]
        log_2pi_s2 = (LOG_2_PI + self.log_sigma_sq) / 2.0
        s2 = torch.exp(self.log_sigma_sq)
        # TODO: where tf are the extra terms coming from?
        loss = - (1/s2) * (tr_wdw + xtvtwtwvx - xtwvx + xtx + xSx) - (d / 2) * log_2pi_s2
        return loss

    def gaussian_kl(self, z_mean, z_logvar):
        """ KL divergence between latent posterior and latent prior (iii)"""
        return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))
        # return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))

    def recon_model_loglik(self, x_in, x_out):
        diff = (x_in - x_out) ** 2
        sigma_sq = torch.exp(self.log_sigma_sq)
        # No dimension constant as we sum after
        return 0.5 * (-diff / sigma_sq - LOG_2_PI - self.log_sigma_sq)

    def analytic_exp_recon_loss(self, x):
        z_logvar = self.variational_logvars
        tr_wdw = torch.trace(
            torch.mm(self.decoder.weight, torch.mm(torch.diag(torch.exp(z_logvar)), self.decoder.weight.t())))

        wv = torch.mm(self.decoder.weight, self.encoder.weight)
        vtwtwv = wv.t().mm(wv)
        xtvtwtwvx = (x * torch.mm(x, vtwtwv)).mean(0).sum()

        xtwvx = 2.0 * (x * x.mm(wv)).mean(0).sum()

        xtx = (x * x).mean(0).sum()

        exp_recon_loss = -(tr_wdw + xtvtwtwvx - xtwvx + xtx) / (2.0 * torch.exp(self.log_sigma_sq)) - x.shape[-1] * (
                LOG_2_PI + self.log_sigma_sq) / 2.0
        return exp_recon_loss

    def analytic_elbo(self, x, z_mean):
        """Computes the analytic ELBO for a linear VAE.
        """
        z_logvar = self.variational_logvars
        kl_div = (-self.gaussian_kl(z_mean, z_logvar)).mean(0).sum()
        exp_recon_loss = self.analytic_exp_recon_loss(x)

        return kl_div - exp_recon_loss

    def forward(self, x):
        z_mean = self.encoder(x)

        if not self.use_analytic_elbo:
            eps = torch.normal(torch.zeros_like(z_mean), 1.0)

            z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)

            if self.use_batch_norm:
                z_sample = self.bn(z_sample)

            x_out = self.decoder(z_sample)

            kl_div = (-self.gaussian_kl(z_mean, self.variational_logvars)).mean(0).sum()
            recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
            loss = kl_div + recon_loss
        else:
            loss = self.analytic_elbo(x, z_mean)
        elbo = - loss
        return elbo

    def get_reconstruction_loss(self, x):
        if self.use_analytic_elbo:
            return - self.analytic_exp_recon_loss(x)
        else:
            z_mean = self.encoder(x)
            eps = torch.normal(torch.zeros_like(z_mean), 1.0)

            z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)

            if self.use_batch_norm:
                z_sample = self.bn(z_sample)

            x_out = self.decoder(z_sample)

            recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
            return recon_loss
