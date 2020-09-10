import torch
import unittest
import time
import numpy as np
import pandas as pd
from gneiss.balances import _balance_basis
from gneiss.balances import sparse_balance_basis
from gneiss.cluster import random_linkage, rank_linkage
from catvae.sparse import SparseMatrix


class TestSparse(unittest.TestCase):

    def setUp(self):
        pass

    def test_basis_multiplication(self):
        k = 30
        dims = [50, 100, 250, 500, 1000, 5000]
        for d in dims:
            #psi = _balance_basis(random_linkage(d))[0]
            psi = sparse_balance_basis(rank_linkage(pd.Series(np.arange(d))))[0]
            sp_psi = SparseMatrix.fromcoo(psi)
            d_psi = torch.Tensor(psi.todense())
            nnz = psi.nnz
            density = (nnz / (d * (d-1)))
            W = torch.randn((d, k))

            start = time.time()
            res = d_psi @ W
            end = time.time()
            dense_time = end - start
            start = time.time()
            res = sp_psi @ W
            end = time.time()
            sparse_time = end - start

            print(f'Dimension : {d}, Elements : {d * (d - 1)}, Density : {density}, '
                  f'Dense {dense_time}, Sparse {sparse_time} ')


if __name__ == '__main__':
    unittest.main()
