import torch
from torch.distributions import MultivariateNormal
from catvae.distributions.mvn import MultivariateNormalFactor
from catvae.distributions.mvn import MultivariateNormalFactorSum


torch.pi = torch.Tensor([torch.acos(torch.zeros(1)).item() * 2])


def K(x):
    """ Log normalization constant for multinomial"""
    return (torch.lgamma(1 + torch.sum(x, dim=-1)) -
            torch.sum(torch.lgamma(1 + x), dim=-1))


def expectation_mvn_factor_sum_multinomial(
        q : MultivariateNormalFactorSum, psi : torch.Tensor, x : torch.Tensor,
        gamma : torch.Tensor):
    """ Lower bound for th efirst expectation involving
        multinomial reconstruction error
    q : MultivariateNormalFactorSum
       q(\eta | x) = N(W V(h(x)), WDW^T + \frac{1}{n} \Psi^T diag(x)^{-1} \Psi)
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
    cov_eta = torch.diagonal(psi @ q.covariance_matrix @ psi.t())
    logits = psi @ mu_eta
    exp_ln = torch.exp(psi @ mu_eta + 0.5 * cov_eta)
    # approximation for normalization factor
    denom = (1 / gamma) * exp_ln.sum(axis=-1) + torch.log(gamma) - 1
    return K(x) + x @ (logits - denom)


def expectation_joint_mvn_factor_mvn_factor_sum(
        qeta : MultivariateNormalFactorSum,
        qz : MultivariateNormal, s2 : torch.Tensor):
    """ Part of the second expectation KL(q||p)

    Parameters
    ----------
    q1 : MultivariateNormalFactorSum
       q(\eta | x) = N(W V(h(x)), WDW^T + \frac{1}{n} \Psi^T diag(x)^{-1} \Psi)
    q2 : MultivariateNormal
       q(z | x) = N(V(h(x)), WDW^T)
    s2 : torch.Tensor
       Standard deviation of p(\eta | z)
    """
    W = qeta.U2
    d = W.shape[-2] + 1
    sigma = qeta.covariance_matrix
    tr_wdw = torch.trace(W @ qz.covariance_matrix @ W.t())
    tr_S = torch.trace(sigma)

    norm_s2 = (-1 / (2 * s2))
    half_logdet = - (d / 2) * (torch.log(2 * torch.pi) + torch.log(s2))
    # print(tr_wdw, tr_S, half_logdet, norm_s2)
    res = norm_s2 * (tr_wdw + tr_S) + half_logdet
    return res


def expectation_mvn_factor_sum_mvn_factor_sum(
        q : MultivariateNormalFactorSum, x : torch.Tensor):
    """ Part of the second expectation KL(q||p)

    Parameters
    ----------
    q1 : MultivariateNormalFactorSum
       q(\eta | x) = N(W V(h(x)), WDW^T + \frac{1}{n} \Psi^T diag(x)^{-1} \Psi)
    x : torch.Tensor
       ILR transformed input data.
    """
    Eeta = q.mean
    S = q.covariance_matrix
    Sinv = q.precision_matrix
    xtSinvx = x.t() @ Sinv @ x
    d = S.shape[-1]
    xSE = x.t() @ Sinv @ Eeta

    sigma = psi.T @ torch.diag(1 / P) @ psi
    tr_S = torch.trace(sigma)

    ESE = Eeta.t() @ S @ Eeta
    logdetS = q.log_det
    res = -0.5 * (xtSinvx - 2 * xSE  + tr_S + ESE) - \
          (d / 2) * (torch.log(2 * torch.pi) - logdetS)
    return res
