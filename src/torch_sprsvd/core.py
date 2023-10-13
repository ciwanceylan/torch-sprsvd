from typing import Union, Literal

import torch
import torch_sparse as tsp
from torch import nn as nn, optim as optim

TORCH_MATRIX = Union[torch.Tensor, tsp.SparseTensor]


def _minimize_semi_linear_form(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                               num_iter: int = 10):
    """
    Aims to minimize ||X @ A - B|| + ||C @ X - D|| w.r.t X.
    Args:
        A: Coefficients of semi-linear equation
        B: Coefficients of semi-linear equation
        C: Coefficients of semi-linear equation
        D: Coefficients of semi-linear equation
        num_iter: Number of LBFGS steps

    Returns: X

    """
    num_rows = B.shape[0]
    num_cols = A.shape[1]
    X = nn.Parameter(torch.zeros(num_rows, num_cols, dtype=A.dtype, device=A.device))

    optimizer = optim.LBFGS(params=[X])

    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            loss1 = torch.pow(torch.linalg.matrix_norm(X @ A - B), 2)
            loss2 = torch.pow(torch.linalg.matrix_norm(C @ X - D), 2)
            loss = loss1 + loss2
            loss.backward()
            # print(f"Loss: {loss.item()}, loss1: {loss1.item()}, loss2: {loss2.item()}")
            return loss

        optimizer.step(closure)
    return X.detach()


def rsvd_basic(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, num_iter: int = 0):
    """
    Algorithm 1 from 'https://doi.org/10.24963/ijcai.2017/468'.
    Args:
        input_matrix:
        k: number of singular values and vectors
        num_oversampling: oversampling can improve accuracy at a small cost of computing a larger approximation matrix.
        num_iter: Number of iterations over the input_matrix. More iterations can improve accuracy.

    Returns: U, singular_values, V

    """
    dtype = input_matrix.dtype() if isinstance(input_matrix, tsp.SparseTensor) else input_matrix.dtype
    device = input_matrix.device() if isinstance(input_matrix, tsp.SparseTensor) else input_matrix.device

    num_rows, num_cols = input_matrix.shape  # [m, n]
    omega = torch.randn(num_cols, k + num_oversampling, dtype=dtype, device=device)  # [n, k + p]
    sample_mat = input_matrix @ omega  # [m, k + p]
    sample_mat, _ = torch.linalg.qr(sample_mat, mode='reduced')  # [m, k + p]

    for i in range(num_iter):
        omega, _ = torch.linalg.qr(input_matrix.t() @ sample_mat, mode='reduced')  # [n, k + p]
        sample_mat, _ = torch.linalg.qr(input_matrix @ omega, mode='reduced')  # [m, k + p]

    omega = input_matrix.t() @ sample_mat  # [n, k + p]
    U1, singular_values, Vh = torch.linalg.svd(omega.t())  # [k + p, k + p], [k + p, k + p], [k + p, n]
    U = sample_mat @ U1[:, :k]  # [m, k]
    singular_values = singular_values[:k]  # [k]
    Vh = Vh[:k, :]  # [n, k]
    return U, singular_values, Vh


HALKO_RSVD_MODES = Literal['col_projection', 'row_projection', 'combined']


def sp_rsvd_halko(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10,
                  mode: HALKO_RSVD_MODES = 'col_projection'):
    """
    Algorithm 2 from 'https://doi.org/10.24963/ijcai.2017/468' adapted from https://arxiv.org/pdf/0909.4061.pdf (algorithm 5).
    Args:
        input_matrix:
        k: number of singular values and vectors
        num_oversampling: oversampling can improve accuracy at a small cost of computing a larger approximation matrix.
        mode: Mode for obtaining the small approximation matrix.
            'col_projection' focuses on projection of the column space
            'row_projection' focuses on projection of the row space
            'combined' uses both projections at the cost of gradient based optimization which may be slow (and not support autograd). TODO test autograd

    Returns: U, singular_values, V

    """
    dtype = input_matrix.dtype() if isinstance(input_matrix, tsp.SparseTensor) else input_matrix.dtype
    device = input_matrix.device() if isinstance(input_matrix, tsp.SparseTensor) else input_matrix.device

    num_rows, num_cols = input_matrix.shape  # [m, n]
    omega_cols = torch.randn(num_cols, k + num_oversampling, dtype=dtype, device=device)  # [n, k + p]
    omega_rows = torch.randn(num_rows, k + num_oversampling, dtype=dtype, device=device)  # [m, k + p]

    sample_mat_cols = input_matrix @ omega_cols  # [m, k + p]
    sample_mat_rows = input_matrix.t() @ omega_rows  # [n, k + p]

    sample_mat_orth_cols, _ = torch.linalg.qr(sample_mat_cols, mode='reduced')  # [m, k + p]
    sample_mat_orth_rows, _ = torch.linalg.qr(sample_mat_rows, mode='reduced')  # [n, k + p]

    if mode == 'row_projection':
        # B = (Omgt'*Q)\(Yt' * Qt);
        input_approx = torch.linalg.lstsq(
            omega_rows.t() @ sample_mat_orth_cols,
            sample_mat_rows.t() @ sample_mat_orth_rows
        ).solution  # [k+p, k+p]
        U1, singular_values, V1h = torch.linalg.svd(input_approx)  # [k + p, k + p], [k + p, k + p], [k + p, k + p]

        U = sample_mat_orth_cols @ U1  # [m, k + p]
        U = U[:, :k]  # [m, k]
        Vh = V1h @ sample_mat_orth_rows.t()  # [n, k + p]
        Vh = Vh[:k, :]  # [n, k]
        singular_values = singular_values[:k]  # [k]

    elif mode == 'col_projection':
        # B = (Omg'*Qt)\(Y' * Q); % A  ' ~ Qt*B*Q'
        input_approx_t = torch.linalg.lstsq(
            omega_cols.t() @ sample_mat_orth_rows,
            sample_mat_cols.t() @ sample_mat_orth_cols
        ).solution  # [k+p, k+p]
        U1, singular_values, V1h = torch.linalg.svd(
            input_approx_t.t())  # [k + p, k + p], [k + p, k + p], [k + p, k + p]

        U = sample_mat_orth_cols @ U1  # [m, k + p]
        U = U[:, :k]  # [m, k]
        Vh = V1h @ sample_mat_orth_rows.t()  # [k + p, n]
        Vh = Vh[:k, :]  # [n, k]
        singular_values = singular_values[:k]  # [k]

    elif mode == 'combined':
        # Aims to minimize ||X @ A - B|| + ||C @ X - D|| w.r.t X
        input_approx = _minimize_semi_linear_form(
            A=sample_mat_orth_rows.t() @ omega_cols,
            B=sample_mat_orth_cols.t() @ sample_mat_cols,
            C=omega_rows.t() @ sample_mat_orth_cols,
            D=sample_mat_rows.t() @ sample_mat_orth_rows,
        )
        U1, singular_values, V1h = torch.linalg.svd(input_approx)  # [k + p, k + p], [k + p, k + p], [k + p, k + p]

        U = sample_mat_orth_cols @ U1  # [m, k + p]
        U = U[:, :k]  # [m, k]
        Vh = V1h @ sample_mat_orth_rows.t()  # [k + p, n]
        Vh = Vh[:k, :]  # [n, k]
        singular_values = singular_values[:k]  # [k]
    else:
        raise ValueError(f"Unknown Halko single-pass rSVD mode {mode}.")

    return U, singular_values, Vh


def _ensure_compatible_batch_size_and_oversampling(k: int, num_oversampling: int, batch_size: int):
    """
    k+num_oversampling must be a multiple of batch_size.
    Args:
        k:
        num_oversampling:
        batch_size:

    Returns:

    """
    multiple = (k + num_oversampling) // batch_size
    num_oversampling = (multiple + 1) * batch_size - k
    # print(f"Adjusted oversampling to {num_oversampling} to comply with batch_size {batch_size}")

    return num_oversampling


def sp_rsvd_block(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, block_size: int = 10):
    dtype = input_matrix.dtype() if isinstance(input_matrix, tsp.SparseTensor) else input_matrix.dtype
    device = input_matrix.device() if isinstance(input_matrix, tsp.SparseTensor) else input_matrix.device

    num_oversampling = _ensure_compatible_batch_size_and_oversampling(k=k,
                                                                      num_oversampling=num_oversampling,
                                                                      batch_size=block_size)

    num_rows, num_cols = input_matrix.shape  # [m, n]
    num_oversampling = min(num_cols - k, num_oversampling)

    omega_cols = torch.randn(num_cols, k + num_oversampling, dtype=dtype, device=device)  # [n, k + p]

    G = input_matrix @ omega_cols
    H = input_matrix.t() @ G
    Q = torch.zeros((num_rows, 0), dtype=G.dtype, device=G.device)
    B = torch.zeros((0, num_cols), dtype=G.dtype, device=G.device)

    num_blocks = k // block_size
    assert k == num_blocks * block_size  # Sanity check

    for i in range(num_blocks):
        temp = B @ omega_cols[:, i * block_size:(i + 1) * block_size]
        Yi = G[:, i * block_size:(i + 1) * block_size] - Q @ temp
        Qi, Ri = torch.linalg.qr(Yi, mode='reduced')
        Qi, Rit = torch.linalg.qr(Qi - Q @ (Q.t() @ Qi), mode='reduced')
        Ri = Rit @ Ri
        Bi = torch.linalg.lstsq(
            Ri.t(),
            H[:, i * block_size:(i + 1) * block_size].t() - Yi.t() @ Q @ B - temp.t() @ B
        ).solution
        Q = torch.cat((Q, Qi), dim=1)  # [m, (i+1) * b]
        B = torch.cat((B, Bi), dim=0)  # [(i+1) * b, n]

    U1, singular_values, Vh = torch.linalg.svd(B)  # [k + p, k + p], [k + p, k + p], [k + p, n]

    U = Q @ U1  # [m, k + p]
    U = U[:, :k]  # [m, k]
    Vh = Vh[:k, :]  # [n, k]
    singular_values = singular_values[:k]  # [k]

    return U, singular_values, Vh
