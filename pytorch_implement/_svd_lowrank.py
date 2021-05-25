from torch import Tensor
import torch
from . import _linalg_utils as _utils
from .overrides import has_torch_function, handle_torch_function
from typing import Optional, Tuple
from get_approximate_basis import get_approximate_basis

def _svd_lowrank(A: Tensor, q: Optional[int] = 6, niter: Optional[int] = 2,
                 M: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is None:
        M_t = None
    else:
        M_t = _utils.transpose(M)
    A_t = _utils.transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in
        # order to keep B shape minimal (the m < n case) or the V
        # shape small (the n > q case)
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.conj().transpose(-2, -1)
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = _utils.transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.conj().transpose(-2, -1)
        U = Q.matmul(U)

    return U, S, V