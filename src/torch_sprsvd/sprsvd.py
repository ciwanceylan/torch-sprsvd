import torch_sprsvd.core as core
from torch_sprsvd.core import TORCH_MATRIX, HALKO_RSVD_MODES


def multi_pass_rsvd(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, num_iter: int = 0):
    return core.rsvd_basic(input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, num_iter=num_iter)


def single_pass_rsvd(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10,
                     mode: HALKO_RSVD_MODES = 'col_projection'):
    return core.sp_rsvd_halko(input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, mode=mode)


def single_pass_block_rsvd(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, block_size: int = 10):
    return core.sp_rsvd_block(input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, block_size=block_size)
