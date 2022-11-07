#include "cuda_kernels.cuh"

__global__ void cuda_simple_dgemm_kernel(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K)
{
	real sum = 0;
	const size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < M && j < N)
	{
		for (size_t k = 0; k < K; k++)
		{
			sum += get_elem(a, i, k, M) * get_elem(b, k, j, K);
		}

		set_elem(c, i, j, M, sum);
	}
}
