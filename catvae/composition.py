import numpy as np
import torch
import torch.nn as nn
from torch.sparse import mm
from gneiss.util import match_tips
from gneiss.balances import sparse_balance_basis
from skbio import TreeNode


class pseudoCLR(nn.Module):
    def __init__(self):
        super(pseudoCLR, self).__init__()

    def forward(self, x):
        y = torch.log(x + 1)
        y = y - y.mean(axis=1).view(-1, 1)
        return y


class pseudoALR(nn.Module):
    def __init__(self):
        super(pseudoALR, self).__init__()

    def forward(self, x):
        y = torch.log(x + 1)
        y = y[:, 1:] - y[:, 0].view(-1, 1)
        return y


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
        denom = x.size()
        return x / denom
    else:
        raise ValueError(f'`x` has dimensions {x.shape}, which are too big')


def ilr(p, basis):
    return mm(basis, torch.log(p).T).T


def ilr_inv(eta, basis):
    return torch.nn.Softmax(eta @ basis, dim=-1)


def ilr_basis(nwk, table):
    tree = TreeNode.read(nwk)
    t = tree.copy()
    t.bifurcate()
    table, t = match_tips(table, t)
    basis = sparse_balance_basis(tree)[0]
    return basis


def alr_basis(D, denom=0):
    """ Computes alr basis (in numpy) """
    basis = np.eye(D - 1)
    z = - np.ones((D - 1, 1))
    basis = np.hstack((
        basis[:, :denom], z, basis[:, denom:]))
    return basis


def identity_basis(D):
    basis = np.eye(D)
    return basis
