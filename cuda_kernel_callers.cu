#include <cublas_v2.h>
#pragma comment (lib, "cublas.lib")

#include "cuda_kernel_callers.cuh"

void cuda_simple_dgemm(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K)
{
	const size_t thread_per_block_1_dim = 32;
	dim3 thread_per_block(thread_per_block_1_dim, thread_per_block_1_dim);	
	dim3 blocks(
		std::ceil(static_cast<float>(N) / thread_per_block_1_dim),
		std::ceil(static_cast<float>(M) / thread_per_block_1_dim));

	cuda_simple_dgemm_kernel<<<blocks, thread_per_block>>>(a, b, c, M, N, K);
}

void cublas_dgemm(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	const real alpha = 1;
	const real beta = 0;

	cublasSgemm_v2(
		handle, CUBLAS_OP_N, CUBLAS_OP_N,
		M, N, K,
		&alpha,
		a, M,
		b, K,
		&beta,
		c, M);

	cublasDestroy(handle);
} 