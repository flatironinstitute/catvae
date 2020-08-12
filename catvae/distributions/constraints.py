r"""
The following constraints are implemented

- ``constraints.orthonormal``
- ``constraints.right_orthonormal``
- ``constraints.left_orthonormal``
"""
from torch.distributions.constraints import Constraint
import torch


__all__ = ['orthonormal', 'right_orthonormal', 'left_orthonormal']


class _Orthonormal(Constraint):
    """ Make sure that matrix is orthonormal
        Specifically: `U @ UT = I` and `UT @ U = I`
    """
    def check(self, value):
        n, d = value.shape[-2], value.shape[-1]
        In = torch.eye(n)
        Id = torch.eye(n)
        right_ortho = torch.allclose(value @ value.T, In)
        left_ortho = torch.allclose(value.T @ value, Id)
        return right_ortho and left_ortho

class _RightOrthonormal(Constraint):
    """ Make sure that matrix is right orthonormal
        Specifically: `U @ UT = I`
    """
    def check(self, value):
        n, d = value.shape[-2], value.shape[-1]
        In = torch.eye(n)
        right_ortho = torch.allclose(value @ value.T, In)
        return right_ortho

class _LeftOrthonormal(Constraint):
    """ Make sure that matrix is left orthonormal
        Specifically: `UT @ U = I`
    """
    def check(self, value):
        n, d = value.shape[-2], value.shape[-1]
        Id = torch.eye(n)
        left_ortho = torch.allclose(value.T @ value, Id)
        return left_ortho


# Public interface
orthonormal = _Orthonormal()
right_orthonormal = _RightOrthonormal()
left_orthonormal = _LeftOrthonormal()
