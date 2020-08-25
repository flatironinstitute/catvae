import torch
import torch.nn as nn
from gneiss.cluster import random_linkage
from gneiss.balances import _balance_basis
from catvae.composition import closure, ilr
from catvae.distributions.likelihood import (
    expectation_mvn_factor_sum_multinomial_bound,
    expectation_joint_mvn_factor_mvn_factor_sum,
)
import numpy as np
from torch.distributions import Multinomial, MultivariateNormal
from catvae.distributions.mvn import MultivariateNormalFactorSum


LOG_2_PI = np.log(2.0 * np.pi)


class LinearCatVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_scale=0.001,
                 basis=None, use_analytic_elbo=True, use_batch_norm=False,
                 deep_decoder=False, decoder_depth=1, imputer=None,
                 mc_samples=1):
        super(LinearCatVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Psi must be dimension D - 1 x D
        if basis is None:
            tree = random_linkage(self.input_dim)
            Psi = torch.Tensor(_balance_basis(tree)[0].copy())
        else:
            Psi = torch.Tensor(basis.copy())
        self.register_buffer('Psi', Psi)
        self.mc_samples = mc_samples

        self.use_analytic_elbo = use_analytic_elbo
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            assert not use_analytic_elbo, (
                "When using batch norm, "
                "need use_analytic_elbo to be False")
            self.bn = nn.BatchNorm1d(num_features=hidden_dim)
        else:
            self.bn = None

        self.encoder = nn.Linear(input_dim - 1, hidden_dim, bias=False)
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.gamma = nn.Parameter(torch.zeros(hidden_dim))

        if imputer is None:
            self.imputer = lambda x: x + 1

        self.use_deep_decoder = deep_decoder
        if deep_decoder:
            assert not use_analytic_elbo, (
                "When using deep decoder, "
                "need use_analytic_elbo to be False")
            self.final_decoder = nn.Linear(
                hidden_dim, input_dim, bias=False)

            num_decoder_layers = decoder_depth
            if use_batch_norm:
                layers = []
                for layer_i in range(num_decoder_layers - 1):
                    layers.append(
                        nn.Linear(hidden_dim, hidden_dim, bias=False))
                    layers.append(nn.BatchNorm1d(num_features=hidden_dim))
                layers.append(self.final_decoder)
            else:
                layers = []
                for layer_i in range(num_decoder_layers - 1):
                    layers.append(
                        nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(self.final_decoder)

            self.decoder = nn.Sequential(*layers)
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim - 1, bias=False)

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
           The estimated logit mean outputted by the decoder
        """
        W = self.decoder.weight
        n = x.sum(axis=-1)
        p = closure(self.imputer(x))
        D = torch.exp(self.variational_logvars)
        qdist = MultivariateNormalFactorSum(
            eta, self.Psi, 1 / p,
            W, D, n)
        return expectation_mvn_factor_sum_multinomial_bound(
            qdist, self.Psi, x)

    def multinomial_kl(self, x, eta, z):
        """ KL divergence between asymptotic multinomial and decoding normal

        x : torch.Tensor
            Observed counts.
        eta : torch.Tensor
            The expected value of q(eta|z)
        z : torch.Tensor
            The expected value or q(z|x)
        """
        n = x.sum(axis=-1)
        p = closure(self.imputer(x))
        D = torch.exp(0.5 * self.variational_logvars)
        W = self.decoder.weight

        #print(W.shape, self.Psi.shape, p.shape, D.shape)
        qeta = MultivariateNormalFactorSum(
            eta, self.Psi, 1 / p,
            W, D, n)
        qz = MultivariateNormal(z, scale_tril=torch.diag(D))

        std = torch.exp(0.5 * self.log_sigma_sq)
        expp = expectation_joint_mvn_factor_mvn_factor_sum(
            qeta=qeta, qz=qz, std=std)

        expq = qeta.entropy()
        return expp - expq

    def gaussian_kl(self, z_mean, z_logvar):
        """ KL divergence between latent posterior and latent prior (iii)"""
        return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))

    def recon_model_loglik(self, x_in, eta):
        return Multinomial(eta @ self.Psi).log_prob(x_in).mean()

    def analytic_elbo(self, x, eta, z_mean):
        """Computes the analytic ELBO for a categorical VAE."""
        z_logvar = self.variational_logvars
        exp_gauss = (-self.gaussian_kl(z_mean, z_logvar)).mean(0).sum()
        exp_mult = self.multinomial_kl(x, eta, z_mean).mean()
        exp_recon_loss = self.multinomial_loglike(x, eta).mean()
        return exp_recon_loss + exp_mult + exp_gauss

    def forward(self, x):
        hx = ilr(self.imputer(x), self.Psi)
        z_mean = self.encoder(hx)
        if not self.use_analytic_elbo:
            eps = torch.normal(torch.zeros_like(z_mean), self.mc_samples)
            z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)
            if self.use_batch_norm:
                z_sample = self.bn(z_sample)
            eta = self.decoder(z_sample)
            kl_div = (
                -self.gaussian_kl(z_mean, self.variational_logvars)
            ).mean(0).sum()
            recon_loss = (-self.recon_model_loglik(x, eta)).mean(0).sum()
            loss = kl_div + recon_loss
        else:
            eta = self.decoder(z_mean)
            loss = self.analytic_elbo(x, eta, z_mean)
        elbo = - loss
        return elbo

    def get_reconstruction_loss(self, x):
        if self.use_analytic_elbo:
            hx = ilr(elf.imputer(x), self.Psi)
            z_mean = self.encoder(hx)
            eta = self.decoder(z_mean)
            return - self.multinomial_loglike(x, eta)
        else:
            hx = ilr(self.imputer(x), self.Psi)
            z_mean = self.encoder(hx)
            eps = torch.normal(torch.zeros_like(z_mean), 1.0)

            z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)

            if self.use_batch_norm:
                z_sample = self.bn(z_sample)

            eta = self.decoder(z_sample)

            recon_loss = (-self.recon_model_loglik(x, eta)).mean(0).sum()
            return recon_loss
