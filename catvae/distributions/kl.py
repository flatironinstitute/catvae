"""
Encodes KL divergences between normal distributions
"""
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2


def _exp_p_eta_z(V, W, D, psi, P, x, s2):
    """ This is mainly for debugging"""
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
    return res


@register_kl(MultivariateNormalFactor, MultivariateNormalFactorSum)
def _kl_multivariate_normal_factor_multivariate_normal_factor_sum(p, q):
    """ """
    # see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    #Multivariate_normal_distributions
    Sp = p.covariance_matrix
    Sq = q.covariance_matrix
    invSp = p.precision_matrix
    invSq = q.precision_matrix
    pmu = p.mean
    qmu = q.mean
    log_pdet = torch.sum(torch.log(p.S))
    # q = 0, p = 1

    trpq = torch.trace(invSp @ sq)
    diff = pmu - qmu
    psi, P = q.U1, 1 / q.S1
    W, D = q.U2, q.S2
    s2 = torch.diagonal(p.scale_tril)


    # first expectation E_q[p]

    # second expectation E_q[q]
    pass
