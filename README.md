# Single-pass rSVD in PyTorch

Sometimes it is necessary to compute the singular value decomposition (SVD) in a single pass through the data, for example, when the data is streamed and too large to store in memory.
For this, single-pass randomized SVD (sp-rSVD) is incredibly useful! 
The sp-rSVD algorithm overcomes memory limitations by projecting the incoming data into a lower dimensional space, which perseveres the subspaces with the largest singular values.
Then, the SVD can be calculated via additional processing of the down-projected matrix.

This is a PyTorch implementation of single-pass rSVD algorithms found in [https://doi.org/10.1137/090771806](https://doi.org/10.1137/090771806) and [https://doi.org/10.24963/ijcai.2017/468](https://doi.org/10.24963/ijcai.2017/468).
Previous MatLab and C implementation by the authors of the second paper can be found here: [https://github.com/THU-numbda/rSVD-single-pass](https://github.com/THU-numbda/rSVD-single-pass).


## Requirements and installation

This implementation uses PyTorch together with [torch_sparse](https://github.com/rusty1s/pytorch_sparse) for sparse matrix support.
Pip requirements can be found in [requirements_cuda.txt](requirements_cuda.txt) and [requirements_cpu.txt](requirements_cpu.txt) for GPU and CPU only respectively.
Full conda environments can be found in [cuda_environment.yml](cuda_environment.yml) and [cpu_environment.yml](cpu_environment.yml).

### Installation

First install the dependencies, e.g.
```bash
conda env create --file cpu_environment.yml
conda activate torch_sp_rsvd_cuda_env
```
Then install the package
```commandline
python -m pip install -e <path/to/repo/root>
```

## Run tests

To run tests, install [pytest](https://docs.pytest.org/en/stable/) and invoke
```commandline
pytest tests/
```
The tests use random sampling for the data matrices. 
This means that tests occasionally fail due to small violations of the required relative error.
If this happens, please rerun the test and check if it fails repeatedly.

## Algorithms

This package implements three algorithms: Standard multi-pass rSVD, single-pass rSVD by [Halko](https://doi.org/10.1137/090771806), and the single-pass rSVD by [Yu et al.](https://doi.org/10.24963/ijcai.2017/468).
Each of these functions are defined in [src/torch_sprsvd/sprsvd.py](src/torch_sprsvd/sprsvd.py), while their core logic is implemented in [src/torch_sprsvd/core.py](src/torch_sprsvd/core.py).

### Standard multi-pass rSVD
Implemented as `multi_pass_rsvd`. This algorithm can achieve very high accuracy (in terms of the singular values), while having good time complexity for large *sparse* matrices.
However, it requires the full data matrix to be present at once, meaning it cannot be used for streaming data.

This implementation is useful for comparing numerical accuracy, as its precision can be increased through the number of over-sample dimensions and algorithm iterations.

### Halko single-pass rSVD
Implemented as `single_pass_rsvd`. This algorithm is only included for comparison and should not be used for applications.
It cannot be used for streams as it requires the whole data matrix at once, even though it only uses a single pass.
While this means it should be faster than the multi-pass version, it also has significantly lower accuracy. 
In several cases, the accuracy is much lower also compared to the next algorithm.

### Streaming single-pass rSVD
Implemented both as the class object `StreamSPBlockRSVD` and the function `single_pass_block_rsvd`. 
This algorithm is able to compute rSVD for streaming data via is class object implementation, see the usage example below.
For testing, the computations can also be performed with a single function call using `single_pass_block_rsvd`.

## Usage example: streaming rSVD

```python
import torch
import torch_sparse as tsp
import tests.utils as tutils
import torch_sprsvd.sprsvd as tsprsvd

# Setup
dtype = 'float'
k = 40
block_size = 8
num_oversampling = 8
type_ = 6
mat_type = 'dense'
dtype = torch.double if dtype == 'double' else torch.float
num_rows = num_cols = 60
true_singular_values = tutils.synthetic_singular_values(type_=type_, p=num_cols).to(dtype=dtype)
A = tutils.generate_synthetic_matrix(num_cols, num_rows, type_=type_, dtype=dtype)
input_matrix = tsp.SparseTensor.from_dense(A) if mat_type == "sparse" else A

# Streaming demo
sprsvd_pipeline = tsprsvd.StreamSPBlockRSVD.create(num_cols=num_cols, k=k, num_oversampling=num_oversampling,
                                                   block_size=block_size, dtype=dtype)
batch_size = 10
for inx in range(0, num_cols, batch_size):
    batch = input_matrix[inx:inx+batch_size]
    sprsvd_pipeline.update(batch)
U, singular_values, Vh = sprsvd_pipeline.compute_block_rsvd()
print(torch.abs(true_singular_values[:k] - singular_values))
```
This also works with autograd. For example, we can wrap `input_matrix` in a `nn.Parameter` object:
```python
input_matrix = nn.Parameter(input_matrix)
sprsvd_pipeline = tsprsvd.StreamSPBlockRSVD.create(num_cols=num_cols, k=k, num_oversampling=num_oversampling,
                                                   block_size=block_size, dtype=dtype)

batch_size = 10
for inx in range(0, num_cols, batch_size):
    batch = input_matrix[inx:inx+batch_size]
    sprsvd_pipeline.update(batch)
U, singular_values, Vh = sprsvd_pipeline.compute_block_rsvd()

loss = torch.sum(singular_values)
loss.backward()
print(input_matrix.grad)
```

See `demo.py` for more usage examples.

## Citing

Please consider citing this repository and the original the authors of the algorithms if you use this work:
```bibtex
@misc{ceylan2024torchsprsvd,
  author = {Ceylan, Ciwan},
  title = {Single-pass rSVD in PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ciwanceylan/torch-sprsvd}}
}


@article{halko2011finding,
  title={Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions},
  author={Halko, Nathan and Martinsson, Per-Gunnar and Tropp, Joel A},
  journal={SIAM review},
  volume={53},
  number={2},
  pages={217--288},
  year={2011},
  publisher={SIAM}
}


@inproceedings{ijcai2017p468,
  author    = {Wenjian Yu and Yu Gu and Jian Li and Shenghua Liu and Yaohang Li},
  title     = {Single-Pass PCA of Large High-Dimensional Data},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {3350--3356},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/468},
  url       = {https://doi.org/10.24963/ijcai.2017/468},
}
```