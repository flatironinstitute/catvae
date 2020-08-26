import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Multinomial, Normal
from catvae.distributions.mvn import MultivariateNormalFactorSum


# TODO make pi also cpu compatible
torch.pi = torch.Tensor([torch.acos(torch.zeros(1)).item() * 2]).cuda()


def K(x):
    """ Log normalization constant for multinomial"""
    return (torch.lgamma(1 + torch.sum(x, dim=-1)) -
            torch.sum(torch.lgamma(1 + x), dim=-1))


def expectation_mvn_factor_sum_multinomial_taylor(
        q: MultivariateNormalFactorSum, psi: torch.Tensor, x: torch.Tensor,
        gamma: torch.Tensor):
    r""" Lower bound for th efirst expectation involving
        multinomial reconstruction error

    Parameters
    ----------
    q : MultivariateNormalFactorSum
       q(eta | x) = N(W V(h(x)), WDW^T + \frac{1}{nf} Psi^T diag(x)^{-1} Psi)
    psi : torch.Tensor
       ILR basis of dimension D x (D - 1)
    x : torch.Tensor
       Input counts of dimension N x D
    gamma : torch.Tensor
       Auxiliary variational parameter to be jointly optimized.

    Notes
    -----
    We can't get closed-form updates here, so we need to obtain another
    lower bound for this expectation.  See Blei and Lafferty et al.
    """
    mu_eta = q.mean
    cov = q.covariance_matrix
    cov_eta = torch.stack([torch.diagonal(psi.t() @ cov[i] @ psi)
                           for i in range(cov.shape[0])])
    logits = mu_eta @ psi
    exp_ln = torch.exp(logits + cov_eta)
    # approximation for normalization factor
    gamma = F.softplus(gamma).repeat(exp_ln.shape[0])
    denom = (1 / gamma) * exp_ln.sum(axis=-1) + torch.log(gamma) - 1
    denom = denom.unsqueeze(1)
    logits = logits - denom
    return K(x) + (x * logits).sum(dim=-1)


def expectation_mvn_factor_sum_multinomial_bound(
        q: MultivariateNormalFactorSum, psi: torch.Tensor, x: torch.Tensor):
    r""" Lower bound for the first expectation involving
         multinomial reconstruction error

    Parameters
    ----------
    q : MultivariateNormalFactorSum
       q(eta | x) = N(W V(h(x)), WDW^T + \frac{1}{nf} Psi^T diag(x)^{-1} Psi)
    psi : torch.Tensor
       ILR basis of dimension D x (D - 1)
    x : torch.Tensor
       Input counts of dimension N x D
    gamma : torch.Tensor
       Auxiliary variational parameter to be jointly optimized.

    Notes
    -----
    We can't get closed-form updates here, so we need to obtain another
    approximation for this expectation.  We will use the fact that
    I - 1dd converges to the identity matrix with increasing d.
    """
    mu_eta = q.mean
    device = mu_eta
    logits = mu_eta @ psi
    # Approach 1 : Silva et al 2017
    denom = torch.logsumexp(logits, dim=-1)
    exp_logits = logits - denom

    # Approach 2:
    # d = mu_eta.shape[-1] + 1
    # Id = torch.eye(d).to(device)
    # dd = (1 / d) * torch.ones((d, d)).to(device)
    # A = Id - dd
    # exp_logits = A @ logits
    return K(x) + (x * exp_logits).sum(-1)

def mean_trace(X):
    return sum(torch.trace(X[i]) for i in range(X.shape[0])) / X.shape[0]

def expectation_joint_mvn_factor_mvn_factor_sum(
        qeta: MultivariateNormalFactorSum,
        qz: Normal, std: torch.Tensor):
    """ Part of the second expectation KL(q||p)

    Parameters
    ----------
    q1 : MultivariateNormalFactorSum
       q(eta | x) = N(W V(h(x)), WDW^T + \frac{1}{n} Psi^T diag(x)^{-1} Psi)
    q2 : MultivariateNormal
       q(z | x) = N(V(h(x)), WDW^T)
    std : torch.Tensor
       Standard deviation of p(eta | z)
    """
    W = qeta.U2
    d = W.shape[-2] + 1
    sigma = qeta.covariance_matrix
    D = qz.covariance_matrix
    wdw = W @ D @ W.t()
    tr_wdw = mean_trace(wdw)
    tr_S = mean_trace(sigma)
    s2 = std ** 2
    norm_s2 = (-1 / (2 * s2))

    half_logdet = - (d / 2) * (torch.log(2 * torch.pi) + torch.log(s2))
    res = norm_s2 * (tr_wdw + tr_S) + half_logdet
    return res
