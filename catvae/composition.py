import numpy as np
import torch
from torch.sparse import mm


def closure(x):
    denom = torch.sum(x, dim=-1)

    if len(x.size()) == 3:
        b, n, d = x.size()
        denom = denom.reshape(b, n, 1)
        return x / denom
    elif len(x.size()) == 2:
        n, d = x.size()
        denom = denom.reshape(n, 1)
        return x / denom
    elif len(x.size()) == 1:
        d = x.size()
        return x / denom
    else:
        raise ValueError(f'`x` has dimensions {x.shape}, which are too big')

def ilr(p, basis):
    return mm(basis, torch.log(p).T).T


def ilr_inv(eta, basis):
    return torch.nn.Softmax(eta @ basis, dim=-1)


def alr_basis(D, denom=0):
    """ Computes alr basis (in numpy) """
    basis = np.eye(D-1)
    z = - np.ones((D-1, 1))
    basis = np.hstack((
        basis[:, :denom], z, basis[:, denom:]))
    return basis


def identity_basis(D):
    basis = np.eye(D)
    return basis
