"""
Author : XuchanBao
This code was taken from
https://github.com/XuchanBao/linear-ae
"""
import torch
import torch.nn as nn
from catvae.composition import ilr
import numpy as np
from torch_sparse import coalesce

LOG_2_PI = np.log(2.0 * np.pi)


class LinearVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_scale=0.001,
                 use_analytic_elbo=True, encoder_depth=1,
                 likelihood='gaussian', basis=None):
        super(LinearVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.likelihood = likelihood
        self.use_analytic_elbo = use_analytic_elbo

        if basis is None:
            tree = random_linkage(input_dim)
            basis = sparse_balance_basis(tree)[0].copy()


        indices = torch.Tensor(np.vstack((basis.row, basis.col)))
        values = torch.Tensor(basis.data.astype(np.float32).copy())
        m, n = basis.shape
        Psi = coalesce(indices, values, m, n)
        Psi.requires_grad = False
        self.input_dim = m
        self.register_buffer('Psi', Psi)

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
        self.imputer = lambda x: x + 1
        self.variational_logvars = nn.Parameter(torch.zeros(hidden_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

    def gaussian_kl(self, z_mean, z_logvar):
        return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))
        # return 0.5 * (1 + z_logvar - z_mean * z_mean - torch.exp(z_logvar))

    def recon_model_loglik(self, x_in, x_out):
        if self.likelihood == 'gaussian':
            x_in = ilr(torch.log(self.imputer(x)), self.Psi)
            diff = (x_in - x_out) ** 2
            sigma_sq = torch.exp(self.log_sigma_sq)
            # No dimension constant as we sum after
            return 0.5 * (-diff / sigma_sq - LOG_2_PI - self.log_sigma_sq)
        elif self.likelihood == 'multinomial':
            x_out = ilr(x_out, self.Psi)
            logp = F.softmax(x_out)
            mult_loss = Multinomial(logits=logp).log_prob(x).mean()
            return mult_loss

    def analytic_exp_recon_loss(self, x):
        z_logvar = self.variational_logvars
        tr_wdw = torch.trace(
            torch.mm(self.decoder.weight,
                     torch.mm(torch.diag(torch.exp(z_logvar)),
                              self.decoder.weight.t())))

        wv = torch.mm(self.decoder.weight, self.encoder.weight)
        vtwtwv = wv.t().mm(wv)
        xtvtwtwvx = (x * torch.mm(x, vtwtwv)).mean(0).sum()

        xtwvx = 2.0 * (x * x.mm(wv)).mean(0).sum()

        xtx = (x * x).mean(0).sum()

        exp_recon_loss = -(
            tr_wdw + xtvtwtwvx - xtwvx + xtx) / (
                2.0 * torch.exp(self.log_sigma_sq)) - x.shape[-1] * (
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
        x_ = ilr(torch.log(self.imputer(x)), self.Psi)
        z_mean = self.encoder(x_)

        if not self.use_analytic_elbo:
            eps = torch.normal(torch.zeros_like(z_mean), 1.0)

            z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)

            if self.use_batch_norm:
                z_sample = self.bn(z_sample)

            x_out = self.decoder(z_sample)

            kl_div = (-self.gaussian_kl(
                z_mean, self.variational_logvars)).mean(0).sum()
            recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
            loss = kl_div + recon_loss
        else:
            loss = self.analytic_elbo(x_, z_mean)
        return loss

    def get_reconstruction_loss(self, x):
        x_ = ilr(torch.log(self.imputer(x)), self.Psi)
        if self.use_analytic_elbo:
            return - self.analytic_exp_recon_loss(x_)
        else:
            z_mean = self.encoder(x_)
            eps = torch.normal(torch.zeros_like(z_mean), 1.0)

            z_sample = z_mean + eps * torch.exp(0.5 * self.variational_logvars)

            if self.use_batch_norm:
                z_sample = self.bn(z_sample)

            x_out = self.decoder(z_sample)

            recon_loss = (-self.recon_model_loglik(x, x_out)).mean(0).sum()
            return recon_loss
