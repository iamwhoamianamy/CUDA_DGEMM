#include "cuda_kernels.cuh"

__global__ void cuda_simple_dgemm_kernel(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K)
{
	real sum = 0;
	const size_t m = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t n = blockDim.x * blockIdx.x + threadIdx.x;

	if (m < M && n < N)
	{
		for (size_t k = 0; k < K; k++)
		{
			sum += get_elem(a, m, k, M) * get_elem(b, k, n, K);
		}

		set_elem(c, m, n, M, sum);
	}
}