import torch
import torch.nn as nn
from torch_sparse import transpose, spmm, spspmm
import numpy as np


class SparseMatrix(nn.Module):
    def __init__(self, index, values, m, n):
        """ Creates sparse matrix from scipy coo matrix.

        Parameters
        ----------
        x : scipy.sparse.coo_matrix
        """
        super(SparseMatrix, self).__init__()
        self.index = torch.LongTensor(index)
        self.values = torch.Tensor(values)
        self._m, self._n = m, n

    @staticmethod
    def fromcoo(x):
        index = np.vstack((x.row, x.col)).astype(np.long)
        values = x.data.astype(np.float32).copy()
        m, n = x.shape
        return SparseMatrix(index, values, m, n)

    def t(self):
        """ Transpose operation."""
        index, value = transpose(self.index, self.values, self._m, self._n)
        out = SparseMatrix(index, value, self._n, self._m)
        device = self.values.device
        out = out.to(device)
        return out

    def to(self, device):
        self.index = self.index.to(device)
        self.values = self.values.to(device)
        return self

    def cuda(self):
        self.index = self.index.cuda()
        self.values = self.values.cuda()
        return self

    @property
    def shape(self):
        return (self._m, self._n)

    def __matmul__(self, y):
        if isinstance(y, torch.Tensor):
            k = y.shape[1]
            return spmm(self.index, self.values, self._m, self._n, y)
        elif isinstance(y, SparseMatrix):
            k, n = y.shape
            index, value = spspmm(self.index, self.values,
                                  y.index, y.values, self._m, k, n)
            return SparseMatrix(index, value, self._m, n)
        else:
            raise ValueError('Sparse matrix multiplication with type '
                             f'{type(y)} not allowed.')

    def __rmatmul__(self, y):
        if isinstance(y, torch.Tensor):
            other = self.t()
            m, n = other.shape
            out = spmm(other.index, other.values, m, n, y.t())
            return out.t()
        elif isinstance(y, SparseMatrix):
            n, k = y.shape
            index, value = spspmm(y.index, y.values,
                                  self.index, self.values,
                                  n, k, self._n)
        else:
            raise ValueError('Sparse matrix multiplication with type '
                             f'{type(y)} not allowed.')
