import pytest
import torch
import torch_sparse as tsp
import tests.utils as tutils
import torch_sprsvd.sprsvd as tsprsvd
import torch_sprsvd.core as tsprsvd_core


def main():
    dtype = 'float'
    k = 50
    block_size = 50
    type_ = 6
    mat_type = 'dense'

    dtype = torch.double if dtype == 'double' else torch.float
    num_rows = num_cols = 50
    true_singular_values = tutils.synthetic_singular_values(type_=type_, p=num_cols).to(dtype=dtype)
    A = tutils.generate_synthetic_matrix(num_cols, num_rows, type_=type_, dtype=dtype)

    input_matrix = tsp.SparseTensor.from_dense(A) if mat_type == "sparse" else A

    num_oversampling = 10
    omega = torch.randn(num_cols, k + num_oversampling, dtype=dtype, device=torch.device('cpu'))
    G = input_matrix @ omega
    H = input_matrix.t() @ G

    U, sig_values, Vh = tsprsvd_core.gh_sp_rsvd_block(omega_cols=omega, G=G, H=H, k=k, block_size=block_size)
    U_stream, sig_values_stream, Vh_stream = tsprsvd_core.gh_sp_rsvd_block(omega_cols=omega, G=G, H=H, k=k,
                                                                           block_size=block_size)

    assert torch.allclose(sig_values, sig_values_stream)
    print("done")


if __name__ == "__main__":
    main()
