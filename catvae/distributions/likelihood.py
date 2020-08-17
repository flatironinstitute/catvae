import torch
from torch.distributions import Multinomial, MultivariateNormal
from catvae.distributions.mvn import MultivariateNormalFactor
from catvae.distributions.mvn import MultivariateNormalFactorSum


torch.pi = torch.acos(torch.zeros(1)).item() * 2


def K(x):
    """ Log normalization constant for multinomial"""
    return (torch.lgamma(1 + torch.sum(x, dim=-1)) -
            torch.sum(torch.lgamma(1 + x), dim=-1))


def expectation_mvn_factor_sum_multinomial(
        q : MultivariateNormalFactorSum, psi : torch.Tensor, x : torch.Tensor):
    """ Lower bound for th efirst expectation involving
        multinomial reconstruction error
    q : MultivariateNormalFactorSum
       q(\eta | x) = N(W V(h(x)), WDW^T + \frac{1}{n} \Psi^T diag(x)^{-1} \Psi)
    psi : torch.Tensor
       ILR basis
    x : torch.Tensor
       Input counts
    """
    b, n, d = x.shape
    exp_eta = q.mean
    denom = x @ (psi @ exp_eta)
    return K(x) + x @ psi @ exp_eta - denom


def expectation_joint_mvn_factor_mvn_factor_sum(
        qeta : MultivariateNormalFactorSum,
        qz : MultivariateNormal,
        p : MultivariateNormalFactor):
    """ Part of the second expectation KL(q||p)

    Parameters
    ----------
    q1 : MultivariateNormalFactorSum
       q(\eta | x) = N(W V(h(x)), WDW^T + \frac{1}{n} \Psi^T diag(x)^{-1} \Psi)
    q2 : MultivariateNormal
       q(z | x) = N(V(h(x)), WDW^T)
    p : MultivariateNormalFactor
       p(\eta | z) = N(Wz, \sigma^2 I )
    """
    wvx = q1.mean
    d = wvx.shape[-1]   # TODO: make sure this dimension is correct
    sigma = q1.covariance_matrix
    s2 = torch.diagonal(p.scale_tril).mean(1)
    tr_wdw = torch.trace(q2.covariance_matrix)
    xtvtwtwvx = wvx.t() @ wvx
    tr_S = torch.trace(sigma)
    hxtShx = wvx.t() @ sigma @ wvx
    res = - (tr_wdw  - xtvtwtwvx + tr_S + hxtShx) / (2 * s2) + \
          - (d / 2) * (torch.log(2 * torch.pi) + torch.log(s2))
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
