import torch
from torch.distributions import Multinomial
from catvae.distributions.mvn import MultivariateNormalFactor
from catvae.distributions.mvn import MultivariateNormalFactorSum


torch.pi = torch.acos(torch.zeros(1)).item() * 2


def expectation_mvn_factor_sum_multinomial(
        q : MultivariateNormalFactorSum, p : Multinomial):
    """ The first expectation involving multinomial reconstruction error"""
    pass


def expectation_joint_mvn_factor_mvn_factor_sum(
        q1 : MultivariateNormalFactor,
        q2 : MultivariateNormalFactorSum,
        p : MultivariateNormalFactor):
    """ Part of the second expectation KL(q||p)"""
    pass


def expectation_mvn_factor_sum_mvn_factor_sum(
        q : MultivariateNormalFactorSum,
        p : MultivariateNormalFactorSum):
    """ Part of the second expectation KL(q||p)"""
    tr_wdw = torch.trace(W torch @ torch.diag(D) @ W.t())
    wv = W @ V
    vtwtwv = wv.t() @ wv
    xtvtwtwvx = x.t() @ vtwtwv @ x
    sigma = psi.T @ torch.diag(1 / P) @ psi
    tr_S = torch.trace(sigma)
    hxShx = wvx.t() @ sigma @ wvx
    log_det = torch.log(P).sum(-1)
    d = W.shape[0]
    res = - (tr_wdw  - xtvtwtwvx + tr_S + hxShx) / (2 * s2) + \
          - (d / 2) * (torch.log(2 * torch.pi) + torch.log(s2))
    # ??? Something is off here.
    return res
