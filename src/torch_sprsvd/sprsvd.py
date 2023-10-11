from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim


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
    loss_fn = nn.MSELoss()

    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(torch.matmul(X, A) - B) + loss_fn(torch.matmul(C, X) - D)
            loss.backward()
            return loss

        optimizer.step(closure)
    return A.detach()


def _rsvd_basic(input_matrix: torch.Tensor, k: int, num_oversampling: int = 10, num_iter: int = 0):
    """
    Algorithm 1 from 'https://doi.org/10.24963/ijcai.2017/468'.
    Args:
        input_matrix:
        k: number of singular values and vectors
        num_oversampling: oversampling can improve accuracy at a small cost of computing a larger approximation matrix.
        num_iter: Number of iterations over the input_matrix. More iterations can improve accuracy.

    Returns: U, singular_values, V

    """
    num_rows, num_cols = input_matrix.shape  # [m, n]
    omega = torch.randn(num_cols, k + num_oversampling)  # [n, k + p]
    sample_mat = input_matrix @ omega  # [m, k + p]
    sample_mat, _ = torch.linalg.qr(sample_mat, mode='reduced')  # [m, k + p]

    for i in range(num_iter):
        omega, _ = torch.linalg.qr(input_matrix.T @ sample_mat, mode='reduced')  # [n, k + p]
        sample_mat, _ = torch.linalg.qr(input_matrix @ omega, mode='reduced')  # [m, k + p]

    omega = input_matrix.T @ sample_mat  # [n, k + p]
    U1, singular_values, V = torch.linalg.svd(omega.T)  # [k + p, k + p], [k + p, k + p], [n, k + p]
    U = sample_mat @ U1[:, :k]  # [m, k]
    singular_values = singular_values[:k]  # [k]
    V = V[:, :k]  # [n, k]
    return U, singular_values, V


HALKO_RSVD_MODES = Literal['col_projection', 'row_projection', 'combined']


def _sp_rsvd_halko(input_matrix: torch.Tensor, k: int, num_oversampling: int = 10,
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

    num_rows, num_cols = input_matrix.shape  # [m, n]
    omega_cols = torch.randn(num_cols, k + num_oversampling)  # [n, k + p]
    omega_rows = torch.randn(num_rows, k + num_oversampling)  # [m, k + p]

    sample_mat_cols = input_matrix @ omega_cols  # [m, k + p]
    sample_mat_rows = input_matrix.t() @ omega_rows  # [n, k + p]

    sample_mat_orth_cols, _ = torch.linalg.qr(sample_mat_cols, mode='reduced')  # [m, k + p]
    sample_mat_orth_rows, _ = torch.linalg.qr(sample_mat_rows, mode='reduced')  # [n, k + p]

    if mode == 'row_projection':
        # B = (Omgt'*Q)\(Yt' * Qt);
        input_approx = torch.linalg.lstsq(A=omega_rows.t() @ sample_mat_orth_cols,
                                          B=sample_mat_rows.t() @ sample_mat_orth_rows)  # [k+p, k+p]
        U1, singular_values, V1 = torch.linalg.svd(input_approx)  # [k + p, k + p], [k + p, k + p], [k + p, k + p]

        U = sample_mat_orth_cols @ U1  # [m, k + p]
        U = U[:, :k]  # [m, k]
        V = sample_mat_orth_rows @ V1  # [n, k + p]
        V = V[:, :k]  # [n, k]
        singular_values = singular_values[:k]  # [k]

    elif mode == 'col_projection':
        # B = (Omg'*Qt)\(Y' * Q); % A  ' ~ Qt*B*Q'
        input_approx_t = torch.linalg.lstsq(A=omega_cols.t() @ sample_mat_orth_rows,
                                            B=sample_mat_cols.t() @ sample_mat_orth_cols)  # [k+p, k+p]
        V1, singular_values, U1 = torch.linalg.svd(input_approx_t)  # [k + p, k + p], [k + p, k + p], [k + p, k + p]

        U = sample_mat_orth_cols @ U1  # [m, k + p]
        U = U[:, :k]  # [m, k]
        V = sample_mat_orth_rows @ V1  # [n, k + p]
        V = V[:, :k]  # [n, k]
        singular_values = singular_values[:k]  # [k]

    elif mode == 'combined':
        # Aims to minimize ||X @ A - B|| + ||C @ X - D|| w.r.t X
        input_approx = _minimize_semi_linear_form(
            A=sample_mat_orth_rows.t() @ omega_cols,
            B=sample_mat_orth_cols.t() @ sample_mat_cols,
            C=omega_rows.t() @ sample_mat_orth_cols,
            D=sample_mat_rows.t() @ sample_mat_orth_rows,
        )
        U1, singular_values, V1 = torch.linalg.svd(input_approx)  # [k + p, k + p], [k + p, k + p], [k + p, k + p]

        U = sample_mat_orth_cols @ U1  # [m, k + p]
        U = U[:, :k]  # [m, k]
        V = sample_mat_orth_rows @ V1  # [n, k + p]
        V = V[:, :k]  # [n, k]
        singular_values = singular_values[:k]  # [k]

    else:
        raise ValueError(f"Unknown Halko single-pass rSVD mode {mode}.")

    return U, singular_values, V
