from typing import List, Optional
import torch
import torch_sprsvd.core as core
from torch_sprsvd.core import TORCH_MATRIX, HALKO_RSVD_MODES


class StreamSPBlockRSVD:
    _g_tensors: List[torch.Tensor]
    _G: Optional[torch.Tensor]
    _H: torch.Tensor

    def __init__(self, omega: torch.Tensor, k: int, block_size: int = 10):
        super().__init__()
        self.omega = omega
        self.k = k
        self.block_size = block_size
        self._g_tensors = []
        self._G = None
        self._H = torch.zeros((omega.shape[1], omega.shape[1]), dtype=omega.dtype, device=omega.device)

    @classmethod
    def create(cls, num_cols: int, k: int, num_oversampling: int = 10, block_size: int = 10,
               dtype: torch.dtype = None, device: torch.device = None):
        num_oversampling = min(num_cols - k, num_oversampling)
        omega = torch.randn(num_cols, k + num_oversampling, dtype=dtype, device=device)
        return cls(omega=omega, k=k, block_size=block_size)

    def update(self, tensor_batch: TORCH_MATRIX):
        G = tensor_batch @ self.omega  # [ num_rows x (k+p) ]
        self._g_tensors.append(G)
        self._H = self._H + tensor_batch.t() @ G  # [ (k+p) x (k+p) ]

    def merge_g(self):
        self._G = torch.cat(self._g_tensors, dim=0)
        return self._G

    def set_G_and_H(self, G: torch.Tensor, H: torch.Tensor):
        self._G = G
        self._H = H

    def compute_block_rsvd(self):
        if self._G is None:
            self.merge_g()
        return core.gh_sp_rsvd_block(omega_cols=self.omega, G=self._G, H=self._H, k=self.k, block_size=self.block_size)


def multi_pass_rsvd(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, num_iter: int = 0):
    return core.rsvd_basic(input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, num_iter=num_iter)


def single_pass_rsvd(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10,
                     mode: HALKO_RSVD_MODES = 'col_projection'):
    return core.sp_rsvd_halko(input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, mode=mode)


def single_pass_block_rsvd(input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, block_size: int = 10):
    return core.sp_rsvd_block(input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, block_size=block_size)
