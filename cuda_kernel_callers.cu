#include <cublas_v2.h>
#pragma comment (lib, "cublas.lib")

#include "cuda_kernel_callers.cuh"

void cuda_simple_dgemm(
   const real* a, const real* b, real* c,
   size_t m, size_t n, size_t k)
{
	const size_t thread_per_block_1_dim = 32;
	dim3 thread_per_block(thread_per_block_1_dim, thread_per_block_1_dim);	
	dim3 blocks(
		std::ceil(static_cast<float>(n) / thread_per_block_1_dim),
		std::ceil(static_cast<float>(m) / thread_per_block_1_dim));

	cuda_simple_dgemm_kernel<<<blocks, thread_per_block>>>(a, b, c, m, n, k);
}

void cublas_dgemm(
   const real* a, const real* b, real* c,
   size_t m, size_t n, size_t k)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	const real alpha = 1;
	const real beta = 0;

	cublasSgemm_v2(
		handle, CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		&alpha,
		a, m,
		b, k,
		&beta,
		c, m);

	cublasDestroy(handle);
} 