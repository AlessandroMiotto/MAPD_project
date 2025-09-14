# Distributed algorithms

The goal of this project was to implement a **QR decomposition algorithm** for tall-and-skinny matrices that leverages Dask’s parallel computing capabilities on a cluster of machines (CloudVeneto). Following [this article](https://arxiv.org/abs/1301.1071), we implemented three versions of the QR decomposition algorithm:
1) QR decomposition via Cholesky decomposition
2) QR indirect method
3) QR direct method

An extensive benchmark section is provided, where we analyzed how the algorithms respond to variations in some cluster hyperparameters, such as the number of partitions and the number of workers, and even the complexity of the task (i.e., the size of the matrix).

## Index
- `Cholesky.ipynb` Definition and implementation of the parallel Cholesky QR algorithm
- `Indirect.ipynb` Definition and implementation of the Indirect QR method
- `Direct.ipynb` Definition and implementation of the Direct QR method
- `functions.py` A pure Python module containing three functions, each implementing one of the three parallel QR algorithms in their optimized form.
- `Benchmark_partitions.ipynb` Benchmark notebook exploring how changing the number of _partitions_ affects the performance of the algorithms
- `Benchmark_workers.ipynb` Benchmark notebook exploring how changing the number of _workers_ affects the performance of the algorithms
- `Benchmark_size.ipynb` Benchmark notebook exploring how the size of the matrix affects the performance of the algorithms
- `Benchmark_cond.ipynb` Benchmark notebook exploring how the algorithms behave with ill-conditioned matrices.
- `data/` Data folder
- `fig/` Figures folder